"""Collect predictions into metric tables and the E0 protocol decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from genocontext.evaluation.metrics import metric_row
from genocontext.evaluation.stats import bootstrap_mean_ci

E0_METRICS = ("auroc", "auprc", "balanced_accuracy", "f1")


def per_drug_metrics(predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(keys + ["drug"], sort=True):
        if group["label"].nunique() < 2:
            continue
        row = metric_row(group["label"].astype(int), group["probability"], float(group["threshold"].iloc[0]))
        rows.append(dict(zip(keys + ["drug"], key, strict=True)) | row)
    return pd.DataFrame(rows)


RUN_KEYS = ["scheme", "seed", "model", "protocol"]


def load_test_predictions(artifacts: Path) -> pd.DataFrame:
    """Test predictions of every finished run: runs/<scheme>/seed<k>/<drug|all>/<model>/<protocol>."""
    frames = []
    for path in sorted((artifacts / "runs").glob("*/seed*/*/*/*/predictions.tsv.gz")):
        if not (path.parent / "task.json").exists():  # interrupted run
            continue
        scheme, seed, _, model, protocol = path.parts[-6:-1]
        frame = pd.read_csv(path, sep="\t", dtype={"isolate_id": str})
        frames.append(frame[frame["part"] == "test"].assign(scheme=scheme, seed=int(seed[4:]), model=model,
                                                           protocol=protocol))
    if not frames:
        raise SystemExit(f"No finished runs under {artifacts / 'runs'}")
    return pd.concat(frames, ignore_index=True)


def collect_runs(artifacts: Path) -> pd.DataFrame:
    return per_drug_metrics(load_test_predictions(artifacts), RUN_KEYS)


def e0_decision(artifacts: Path, margin: float = -0.01) -> dict[str, object]:
    """Adopt protocol B (80/20) if its discrimination is not worse and thresholded metrics are non-inferior."""
    frames = []
    for path in sorted((artifacts / "e0").glob("*/*/predictions.tsv.gz")):
        frame = pd.read_csv(path, sep="\t", dtype={"isolate_id": str})
        frames.append(frame.assign(scheme=path.parts[-3], model=path.parts[-2]))
    metrics = per_drug_metrics(pd.concat(frames), ["scheme", "model", "held_fold", "protocol"])
    wide = metrics.pivot_table(index=["scheme", "model", "held_fold", "drug"], columns="protocol",
                               values=list(E0_METRICS))
    summary: list[dict[str, object]] = []
    decisions: list[bool] = []
    for (scheme, model), group in wide.groupby(level=["scheme", "model"]):
        row: dict[str, object] = {"scheme": scheme, "model": model, "n_pairs": len(group)}
        for metric in E0_METRICS:
            mean, low, high = bootstrap_mean_ci(group[(metric, "B")] - group[(metric, "A")])
            row |= {f"delta_{metric}": mean, f"delta_{metric}_low": low, f"delta_{metric}_high": high}
        deltas = {k: float(v) for k, v in row.items() if k.startswith("delta_")}  # type: ignore[arg-type]
        adopt = (deltas["delta_auroc"] >= 0 and deltas["delta_auprc"] >= 0
                 and deltas["delta_balanced_accuracy_low"] >= margin and deltas["delta_f1_low"] >= margin)
        summary.append(row | {"adopt_B": adopt})
        decisions.append(adopt)
    table = pd.DataFrame(summary)
    out = artifacts / "e0"
    table.to_csv(out / "e0_summary.tsv", sep="\t", index=False)
    metrics.to_csv(out / "e0_metrics.tsv", sep="\t", index=False)
    decision = {"adopt_protocol_B": all(decisions), "rule": "all scheme×model: mean ΔAUROC>=0, ΔAUPRC>=0, "
                f"95% CI lower bound of Δbalanced accuracy and ΔF1 >= {margin}", "per_model": summary}
    (out / "e0_decision.json").write_text(json.dumps(decision, indent=1, default=float))
    return decision
