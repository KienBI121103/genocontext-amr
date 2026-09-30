"""Run GenoContext-AMR on the K. pneumoniae annotation, label, and split inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.data.inputs import RecordStore, Split, read_manifest, read_phenotypes, read_split
from src.training.pipeline import collect_metrics, run_group
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
    artifacts = Path(config["paths"]["artifacts"])
    store = RecordStore(
        manifest, artifacts / "features/parsed_gff",
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
    LOG.info("Checking %d K. pneumoniae annotations", len(all_ids))
    for index, sample in enumerate(sorted(all_ids), 1):
        store.get(sample)
        if index % 100 == 0:
            LOG.info("Checked %d/%d annotations", index, len(all_ids))
    return store, labels, splits


def main() -> None:
    parser = argparse.ArgumentParser(description="GenoContext-AMR K. pneumoniae experiment")
    parser.add_argument("command", choices=("preflight", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/kp.yaml"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--antibiotic", action="append")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifacts = Path(config["paths"]["artifacts"])
    if args.command == "summarize":
        metrics = collect_metrics(artifacts)
        LOG.info("Wrote %d metric rows to %s", len(metrics), artifacts / "results/metrics.csv")
        return

    seeds = args.seed or list(config["experiment"]["seeds"])
    if any(seed not in range(10) for seed in seeds):
        raise ValueError("K. pneumoniae seeds must be 0 through 9")
    antibiotics = args.antibiotic or antibiotic_names(config)
    if args.antibiotic is None and len(antibiotics) != 17:
        raise ValueError(f"Expected 17 K. pneumoniae antibiotics, found {len(antibiotics)}")
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
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


if __name__ == "__main__":
    main()
