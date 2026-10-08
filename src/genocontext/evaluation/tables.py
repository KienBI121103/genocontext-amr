"""Publication tables: seed summaries, pre-registered endpoints, six-method comparison and low-data curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genocontext.evaluation.stats import benjamini_hochberg, cluster_bootstrap, corrected_repeated_kfold_ttest

METRICS = ("auroc", "auprc", "f1", "balanced_accuracy", "mcc", "accuracy", "vme", "me", "brier")
# Pre-registered endpoints (docs/protocol.md): family -> list of (scheme, candidate, reference)
ENDPOINTS = {
    "P1_vs_panka_like": [("cg_blocked", "kg", "lgbm_presence")],
    "P2_value_of_prior": [("cg_blocked", "kg", "kg_dense"), ("cg_blocked", "kg", "deepsets"),
                          ("cg_blocked", "kg", "kg_rewired")],
    "P4_gene_state": [("cg_blocked", "kg", "kg_no_target_res"), ("cg_blocked", "lgbm_genestate", "lgbm_presence")],
}


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds of the per-seed macro average (equal weight per drug)."""
    macro = metrics.groupby(["scheme", "model", "protocol", "seed"])[list(METRICS)].mean()
    summary = macro.groupby(["scheme", "model", "protocol"]).agg(["mean", "std"])
    summary.columns = ["_".join(map(str, column)) for column in summary.columns]
    return summary.reset_index().assign(n_seeds=macro.groupby(["scheme", "model", "protocol"]).size().to_numpy())


def endpoint_tests(metrics: pd.DataFrame, protocol: str, metric: str = "auroc") -> pd.DataFrame:
    """Per-drug and macro paired differences across seeds with the corrected repeated K-fold t-test (BH per family)."""
    rows = []
    for family, comparisons in ENDPOINTS.items():
        family_rows = []
        for scheme, candidate, reference in comparisons:
            sub = metrics[(metrics["scheme"] == scheme) & (metrics["protocol"] == protocol)]
            wide = sub.pivot_table(index=["seed", "drug"], columns="model", values=metric)
            if candidate not in wide or reference not in wide:
                continue
            delta = (wide[candidate] - wide[reference]).dropna()
            per_seed_macro = delta.groupby(level="seed").mean()
            for drug, values in [("MACRO", per_seed_macro), *delta.groupby(level="drug")]:
                t, p = corrected_repeated_kfold_ttest(values.to_numpy(), n_folds=5) if len(values) > 1 \
                    else (np.nan, np.nan)
                family_rows.append({"family": family, "scheme": scheme, "candidate": candidate,
                                    "reference": reference, "metric": metric, "drug": drug, "n_seeds": len(values),
                                    "delta_mean": float(values.mean()), "delta_sd": float(values.std()),
                                    "t": t, "p": p})
        if family_rows:
            frame = pd.DataFrame(family_rows)
            frame["q_bh"] = benjamini_hochberg(frame["p"].fillna(1.0))
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def macro_auroc_delta(frame: pd.DataFrame) -> float:
    """Macro (equal-weight drugs) AUROC of probability_a minus probability_b."""
    deltas = []
    for _, group in frame.groupby("drug"):
        if group["label"].nunique() == 2:
            deltas.append(roc_auc_score(group["label"], group["probability_a"])
                          - roc_auc_score(group["label"], group["probability_b"]))
    return float(np.mean(deltas)) if deltas else float("nan")


def pooled_cluster_bootstrap(predictions: pd.DataFrame, partition_groups: pd.Series, scheme: str, protocol: str,
                             candidate: str, reference: str, n_boot: int = 1000) -> pd.DataFrame:
    """Per repeat, pool the five outer folds (each isolate predicted once) and bootstrap clonal groups."""
    rows = []
    sub = predictions[(predictions["scheme"] == scheme) & (predictions["protocol"] == protocol)]
    for repeat in (0, 1):
        seeds = range(5 * repeat, 5 * repeat + 5)
        pooled = [sub[(sub["model"] == m) & sub["seed"].isin(seeds)][["isolate_id", "drug", "label", "probability"]]
                  for m in (candidate, reference)]
        merged = pooled[0].merge(pooled[1], on=["isolate_id", "drug", "label"], suffixes=("_a", "_b"))
        if merged.empty:
            continue
        merged["group"] = merged["isolate_id"].map(partition_groups)
        estimate, low, high = cluster_bootstrap(merged, "group", macro_auroc_delta, n_boot)
        rows.append({"scheme": scheme, "protocol": protocol, "candidate": candidate, "reference": reference,
                     "repeat": repeat, "macro_delta_auroc": estimate, "low": low, "high": high,
                     "n_clusters": merged["group"].nunique()})
    return pd.DataFrame(rows)


def six_methods_table(metrics: pd.DataFrame, reference_json: Path, models: list[str]) -> pd.DataFrame:
    """Tier-1 seed-0 results next to the earlier six-method report (descriptive; different training budgets)."""
    reference = json.loads(reference_json.read_text())
    rows = []
    for method, values in reference["reported"].items():
        for metric in ("accuracy", "f1", "auroc", "auprc"):
            for drug, value in zip(reference["drugs"], values[metric], strict=True):
                rows.append({"method": method, "drug": drug, "metric": metric, "value": value})
    ours = metrics[(metrics["scheme"] == "amrgnn_fixed") & (metrics["seed"] == 0) & metrics["model"].isin(models)]
    for row in ours.itertuples():
        for metric in ("accuracy", "f1", "auroc", "auprc"):
            rows.append({"method": f"{row.model} ({row.protocol})", "drug": row.drug, "metric": metric,
                         "value": getattr(row, metric)})
    table = pd.DataFrame(rows).pivot_table(index=["metric", "drug"], columns="method", values="value")
    return table.reset_index()


def low_data_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Label-subsampling (``model@n<count>-<drug>-r<rep>``) and zero-shot (``model@zeroshot=<drug>``) results."""
    variant = metrics["model"].str.extract(r"^(?P<base>[^@]+)@(?:n(?P<n_count>\d+)-(?P<target>\w+)-r(?P<rep>\d+)"
                                           r"|zeroshot=(?P<zero>\w+))$")
    data = metrics.join(variant).dropna(subset=["base"])
    data["target"] = data["target"].fillna(data["zero"])
    data = data[data["drug"] == data["target"]]
    data["n_labels"] = pd.to_numeric(data["n_count"]).fillna(0).astype(int)
    return data.groupby(["scheme", "protocol", "base", "target", "n_labels"])[["auroc", "auprc", "mcc"]].agg(
        ["mean", "std", "count"]).reset_index()
