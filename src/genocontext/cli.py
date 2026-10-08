"""Command-line entry point: ``genocontext <command> --config experiments/kp/config.yaml``.

Commands (see experiments/kp/README.md for the full run book):
  preflight   check inputs, annotations and split files
  profiles    build per-isolate gene-state profiles (label-free, parallel)
  e0          dev-only check of 80/20 (protocol B) vs 64/16/20 (protocol A)
  run         train/evaluate models on split tiers (resumable; one task per scheme × seed × model × drug)
  stack       logistic stacking of finished runs (fitted on selection/out-of-fold predictions only)
  explain     mechanism decomposition and token attributions of finished GenoContext-KG runs
  candidates  cross-seed evidence ladder of putative determinants
  compare     metric tables, E0 decision, pre-registered endpoints, six-method and low-data tables
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from genocontext.config import Config, load_config
from genocontext.data.cohort import Cohort, annotation_paths, load_cohort
from genocontext.evaluation.ensemble import stack
from genocontext.evaluation.report import collect_runs, e0_decision, load_test_predictions
from genocontext.evaluation.tables import (
    ENDPOINTS,
    endpoint_tests,
    low_data_table,
    pooled_cluster_bootstrap,
    seed_summary,
    six_methods_table,
)
from genocontext.explain.report import candidates as evidence_ladder
from genocontext.explain.report import explain_run
from genocontext.features.lexicon import Knowledge, load_knowledge
from genocontext.features.profile import build_profiles, load_profile
from genocontext.features.store import ProfileStore
from genocontext.pipeline import REGISTRY, Task, load_store, run_e0, run_task

LOG = logging.getLogger("genocontext")
SCHEMES = ["amrgnn_fixed", "random_grouped", "cg_blocked"]
REFERENCE = Path("/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/fusion_v1/reports/"
                 "six_methods_seed0_20261008/reference_values.json")


def _inputs(cohort: Cohort) -> dict[str, tuple[Path, Path]]:
    return {str(isolate): annotation_paths(path) for isolate, path in cohort.assemblies.items()}


def _preflight(config: Config) -> Cohort:
    cohort = load_cohort(config.paths.labels, config.paths.manifest)
    load_knowledge(config.paths.knowledge)
    inputs = _inputs(cohort)
    with ThreadPoolExecutor(32) as pool:
        exists = list(pool.map(lambda pair: pair[0].is_file() and pair[1].is_file(), inputs.values()))
    missing = [isolate for isolate, ok in zip(inputs, exists, strict=True) if not ok]
    if missing:
        raise SystemExit(f"{len(missing)} isolates lack GFF3/FAA, e.g. {missing[:3]}")
    if not (config.paths.splits / "split_protocol.json").is_file():
        raise SystemExit(f"Run experiments/kp/make_splits.py first: {config.paths.splits}")
    LOG.info("Preflight passed: %d isolates, %d drugs", len(cohort.labels), len(cohort.drugs))
    return cohort


def preflight(config: Config, args: argparse.Namespace) -> None:
    _preflight(config)


def profiles(config: Config, args: argparse.Namespace) -> None:
    cohort = _preflight(config)
    inputs = _inputs(cohort)
    for count, _ in enumerate(build_profiles(config, inputs), 1):
        if count % 250 == 0:
            LOG.info("Built %d profiles", count)
    qc = pd.DataFrame({i: load_profile(config.paths.profiles / f"{i}.json.gz").qc for i in inputs}).T
    qc.rename_axis("isolate_id").to_csv(config.paths.profiles.parent / "assembly_qc.tsv", sep="\t")
    LOG.info("Assembly QC: median contigs %.0f, >500 contigs: %d isolates", qc["n_contigs"].median(),
             int((qc["n_contigs"] > 500).sum()))


def _context(config: Config) -> tuple[Cohort, Knowledge, ProfileStore]:
    cohort = load_cohort(config.paths.labels, config.paths.manifest)
    knowledge = load_knowledge(config.paths.knowledge)
    return cohort, knowledge, load_store(config, cohort, knowledge)


def _tasks(args: argparse.Namespace, drugs: tuple[str, ...], models: list[str]) -> list[Task]:
    """Tier 1 has one partition per target drug; isolate-level tiers serve all drugs in one task."""
    tasks = []
    for scheme in args.scheme:
        targets = (args.drug or list(drugs)) if scheme == "amrgnn_fixed" else [None]
        tasks += [Task(scheme, seed, model, args.protocol, drug, variant)
                  for seed in args.seed for model in models for drug in targets for variant in args.variant]
    return tasks


def e0(config: Config, args: argparse.Namespace) -> None:
    cohort, knowledge, store = _context(config)
    for scheme in args.scheme:
        for model in args.model:
            run_e0(config, scheme, model, cohort, knowledge, store)


def run(config: Config, args: argparse.Namespace) -> None:
    cohort, knowledge, store = _context(config)
    for task in _tasks(args, cohort.drugs, args.model):
        LOG.info("Finished %s", run_task(config, task, cohort, knowledge, store))


def stack_runs(config: Config, args: argparse.Namespace) -> None:
    """``--model a b`` -> runs/.../a+b/<protocol>/predictions.tsv.gz (members must be finished)."""
    cohort = load_cohort(config.paths.labels, config.paths.manifest)
    name = "+".join(args.model)
    for task in _tasks(args, cohort.drugs, [args.model[0]]):
        members = {m: Task(task.scheme, task.seed, m, task.protocol, task.drug).directory(config.paths.artifacts)
                   for m in args.model}
        out = Task(task.scheme, task.seed, name, task.protocol, task.drug).directory(config.paths.artifacts)
        if (out / "task.json").exists() or not all((d / "task.json").exists() for d in members.values()):
            continue
        frames = {m: pd.read_csv(d / "predictions.tsv.gz", sep="\t", dtype={"isolate_id": str})
                  for m, d in members.items()}
        out.mkdir(parents=True, exist_ok=True)
        stack(frames).to_csv(out / "predictions.tsv.gz", sep="\t", index=False)
        (out / "task.json").write_text(json.dumps({"stacked": {m: str(d) for m, d in members.items()}}, indent=1))
        LOG.info("Stacked %s", out)


def explain(config: Config, args: argparse.Namespace) -> None:
    cohort, knowledge, store = _context(config)
    for task in _tasks(args, cohort.drugs, args.model):
        if (task.directory(config.paths.artifacts) / "models.pt").exists():
            LOG.info("Explained %s", explain_run(config, task, cohort, knowledge, store))


def candidates(config: Config, args: argparse.Namespace) -> None:
    cohort, knowledge, store = _context(config)
    out = config.paths.artifacts / "results"
    out.mkdir(parents=True, exist_ok=True)
    for scheme in args.scheme:
        table = evidence_ladder(config, scheme, args.protocol, knowledge, store, cohort)
        table.to_csv(out / f"candidates_{scheme}_{args.protocol}.tsv", sep="\t", index=False)
        LOG.info("%d candidate determinants for %s", len(table), scheme)


def compare(config: Config, args: argparse.Namespace) -> None:
    out = config.paths.artifacts / "results"
    out.mkdir(parents=True, exist_ok=True)
    if (config.paths.artifacts / "e0").is_dir():
        decision = e0_decision(config.paths.artifacts)
        LOG.info("E0 decision: adopt protocol B (80/20) = %s", decision["adopt_protocol_B"])
    if not (config.paths.artifacts / "runs").is_dir():
        return
    metrics = collect_runs(config.paths.artifacts)
    metrics.to_csv(out / "metrics.tsv", sep="\t", index=False)
    seed_summary(metrics).to_csv(out / "seed_summary.tsv", sep="\t", index=False)
    low_data_table(metrics).to_csv(out / "low_data.tsv", sep="\t", index=False)
    if args.reference.is_file():
        six_methods_table(metrics, args.reference, ["kg", "kg+lgbm_genestate", "lgbm_genestate"]).to_csv(
            out / "six_methods_tier1.tsv", sep="\t", index=False)
    for protocol in ("A", "B"):
        tests = pd.concat([endpoint_tests(metrics, protocol, m) for m in ("auroc", "auprc", "balanced_accuracy")])
        tests.to_csv(out / f"endpoints_protocol{protocol}.tsv", sep="\t", index=False)
    groups = pd.read_csv(config.paths.splits / "clonal_groups.tsv", sep="\t", dtype=str).set_index("isolate_id")
    predictions = load_test_predictions(config.paths.artifacts)
    boot = [pooled_cluster_bootstrap(predictions, groups["group"], scheme, protocol, candidate, reference)
            for comparisons in ENDPOINTS.values() for scheme, candidate, reference in comparisons
            for protocol in ("A", "B")]
    pd.concat(boot).to_csv(out / "cluster_bootstrap.tsv", sep="\t", index=False)
    LOG.info("Wrote result tables to %s", out)


COMMANDS = {"preflight": preflight, "profiles": profiles, "e0": e0, "run": run, "stack": stack_runs,
            "explain": explain, "candidates": candidates, "compare": compare}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="genocontext", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--config", type=Path, default=Path("experiments/kp/config.yaml"))
    parser.add_argument("--scheme", nargs="+", default=["cg_blocked"], choices=SCHEMES)
    parser.add_argument("--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--model", nargs="+", default=["kg"], choices=sorted(REGISTRY))
    parser.add_argument("--protocol", default="B", choices=["A", "B"])
    parser.add_argument("--drug", nargs="+", help="tier-1 target drugs (default: all 17)")
    parser.add_argument("--variant", nargs="+", default=[""],
                        help="'' | rep<r> | shuffled_labels | zeroshot=<drug> | n<count>-<drug>-r<rep>")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="six-method reference_values.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    COMMANDS[args.command](load_config(args.config), args)


if __name__ == "__main__":
    main()
