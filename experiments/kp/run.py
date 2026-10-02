"""Run GenoContext-AMR on the K. pneumoniae annotation, label, and split inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.data.inputs import RecordStore, Split, parsed_cache_dir, read_manifest, read_phenotypes, read_split
from src.evaluation.fusion import summarize_fusion
from src.training.pipeline import collect_metrics, run_group
from src.training.fusion import validate_fusion_run
from src.training.parallel import validate_records
from src.training.train import select_device

LOG = logging.getLogger(__name__)


def split_root(config: dict, seed: int) -> Path:
    root = Path(config["paths"]["amrgnn_root"])
    if seed == 0:
        return root / "exact_seed0_64_16_20/amrgnn/splits"
    return root / f"fixed_test_multiseed/seed{seed}/amrgnn/splits"


def antibiotic_names(config: dict) -> list[str]:
    return sorted(
        path.name for path in split_root(config, 0).iterdir()
        if path.is_dir() and (path / "train.ids").exists()
    )


def prepare(config: dict, seeds: list[int], antibiotics: list[str]) -> tuple[RecordStore, dict, dict]:
    manifest = read_manifest(config["paths"]["manifest"])
    labels = read_phenotypes(config["paths"]["phenotypes"])
    store = RecordStore(
        manifest, parsed_cache_dir(config),
        Path(config["bakta"]["db"]), config["bakta"]["threads"],
    )
    splits: dict[tuple[int, str], Split] = {}
    all_ids: set[str] = set()
    for antibiotic in antibiotics:
        reference_test = set(read_split(split_root(config, 0), antibiotic, labels).test)
        for seed in seeds:
            split = read_split(split_root(config, seed), antibiotic, labels)
            if set(split.test) != reference_test:
                raise ValueError(f"Fixed test IDs differ for {antibiotic}, seed {seed}")
            splits[(seed, antibiotic)] = split
            all_ids.update(split.train)
            all_ids.update(split.val)
            all_ids.update(split.test)
    missing = all_ids - set(manifest)
    if missing:
        raise ValueError(f"{len(missing)} split isolates missing from manifest: {sorted(missing)[:5]}")
    workers = config["training"].get("preprocess_workers", 1)
    LOG.info("Checking %d K. pneumoniae annotations with %d workers", len(all_ids), workers)
    for index, _sample in enumerate(validate_records(store, tuple(sorted(all_ids)), workers), 1):
        if index % 100 == 0:
            LOG.info("Checked %d/%d annotations", index, len(all_ids))
    return store, labels, splits


def main() -> None:
    parser = argparse.ArgumentParser(description="GenoContext-AMR K. pneumoniae experiment")
    parser.add_argument("command", choices=("preflight", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/kp.yaml"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--antibiotic", action="append")
    parser.add_argument("--full", action="store_true", help="Explicitly launch all 17 antibiotics/seeds after the fusion pilot gate")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifacts = Path(config["paths"]["artifacts"])
    fusion = config.get("fusion", {}).get("enabled", False)
    if args.full and (not fusion or args.command != "run" or args.seed or args.antibiotic):
        raise ValueError("--full requires fusion run without seed/antibiotic filters")
    if args.command == "summarize":
        metrics = collect_metrics(artifacts)
        LOG.info("Wrote %d metric rows to %s", len(metrics), artifacts / "results/metrics.csv")
        if fusion:
            gate = summarize_fusion(artifacts, metrics)
            LOG.info("Fusion pilot: %s; eligible modes=%s", gate["status"], gate["eligible_feature_modes"])
        return

    if args.full:
        # Recompute the gate from predictions/metrics rather than trusting a stale JSON flag.
        gate = summarize_fusion(artifacts, collect_metrics(artifacts))
        if not gate["allow_full_run"]:
            raise ValueError(f"Full fusion run blocked by validation pilot gate: {gate['status']}")
        pilot_store, pilot_labels, pilot_splits = prepare(config, [0, 1, 2], ["ciprofloxacin"])
        for seed in (0, 1, 2):
            for mode in ("raw", "masked"):
                validate_fusion_run(config, pilot_store, pilot_labels, pilot_splits[(seed, "ciprofloxacin")],
                                    "ciprofloxacin", seed, mode)
    seeds = list(range(10)) if args.full else sorted(set(args.seed or config["experiment"]["seeds"]))
    if any(seed not in range(10) for seed in seeds):
        raise ValueError("K. pneumoniae seeds must be 0 through 9")
    configured_antibiotics = config["experiment"].get("antibiotics")
    antibiotics = antibiotic_names(config) if args.full else list(dict.fromkeys(
        args.antibiotic or configured_antibiotics or antibiotic_names(config)))
    if (args.full or (args.antibiotic is None and not configured_antibiotics)) and len(antibiotics) != 17:
        raise ValueError(f"Expected 17 K. pneumoniae antibiotics, found {len(antibiotics)}")
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    LOG.info("Threads: neural=%d RF=%d preprocessing_workers=%d",
             torch.get_num_threads(), config["training"].get("rf_threads", config["training"]["cpu_threads"]),
             config["training"].get("preprocess_workers", 1))
    store, labels, splits = prepare(config, seeds, antibiotics)
    if args.command == "preflight":
        LOG.info("Preflight passed")
        return

    device = select_device()
    LOG.info("Training on %s", device)
    for seed in seeds:
        for antibiotic in antibiotics:
            for feature_mode in ("raw", "masked"):
                LOG.info("seed=%d antibiotic=%s mode=%s", seed, antibiotic, feature_mode)
                run_group(
                    config, store, labels, splits[(seed, antibiotic)],
                    antibiotic, seed, feature_mode, device,
                )
    metrics = collect_metrics(artifacts)
    LOG.info("Wrote %d metric rows to %s", len(metrics), artifacts / "results/metrics.csv")
    if fusion:
        gate = summarize_fusion(artifacts, metrics)
        LOG.info("Fusion pilot: %s; eligible modes=%s", gate["status"], gate["eligible_feature_modes"])


if __name__ == "__main__":
    main()
