"""Summarize this project's fusion pilot against its own RF branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.fusion import align_probabilities


def summarize_fusion(artifacts: Path, metrics: pd.DataFrame, pilot_antibiotic: str = "ciprofloxacin",
                     pilot_seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    # Import lazily to avoid a dependency cycle with the training dispatcher.
    from src.training.pipeline import write_csv

    rows = []
    keys = ["antibiotic", "seed", "feature_mode", "split"]
    rf = metrics[metrics["model"] == "random_forest"]
    if rf.duplicated(keys).any():
        raise ValueError("Duplicate RF metric rows")
    for _, fusion in metrics[metrics["model"].isin(
            ("genocontext_fusion_real", "genocontext_fusion_shuffled"))].iterrows():
        matches = rf
        for key in keys:
            matches = matches[matches[key] == fusion[key]]
        if len(matches) != 1:
            raise ValueError("Fusion metrics require one matching RF cohort")
        baseline = matches.iloc[0]
        output = artifacts / "results" / f"seed{int(fusion['seed'])}" / fusion["antibiotic"] / fusion["feature_mode"]
        frame = pd.read_csv(output / f"{fusion['model']}_predictions.csv", dtype={"isolate_id": str})
        frame = frame[frame["split"] == fusion["split"]]
        rf_frame = pd.read_csv(output / "random_forest_predictions.csv", dtype={"isolate_id": str})
        rf_frame = rf_frame[rf_frame["split"] == fusion["split"]]
        ids = tuple(frame["isolate_id"])
        baseline_scores = align_probabilities(ids, tuple(rf_frame["isolate_id"]), rf_frame["probability_R"])
        truth = frame["true_label"].eq("R").to_numpy()
        rf_labels = rf_frame.set_index("isolate_id").loc[list(ids), "true_label"].eq("R").to_numpy()
        if not np.array_equal(truth, rf_labels):
            raise ValueError("Fusion and RF prediction labels differ")
        if not np.allclose(baseline_scores, frame["probability_rf"], rtol=0, atol=1e-12):
            raise ValueError("Fusion probabilities do not match the RF branch")
        rf_wrong = (baseline_scores >= float(baseline["threshold"])) != truth
        fusion_wrong = (frame["probability_R"].to_numpy() >= float(fusion["threshold"])) != truth
        row = {key: fusion[key] for key in keys}
        row.update(model=fusion["model"], edge_mode=fusion["edge_mode"], alpha=fusion["alpha"],
                   rf_f1=baseline["f1"], fusion_f1=fusion["f1"],
                   delta_f1=fusion["f1"] - baseline["f1"],
                   rf_probability_std=float(np.std(baseline_scores)),
                   gnn_probability_std=float(np.std(frame["probability_gnn"])),
                   fusion_probability_std=float(np.std(frame["probability_R"])),
                   rf_errors_corrected=int((rf_wrong & ~fusion_wrong).sum()),
                   rf_errors_introduced=int((~rf_wrong & fusion_wrong).sum()))
        for metric in ("auroc", "auprc", "specificity", "recall"):
            row[f"delta_{metric}"] = fusion[metric] - baseline[metric]
        rows.append(row)
    if not rows:
        raise ValueError("No fusion results to summarize")
    comparison = pd.DataFrame(rows)
    write_csv(comparison, artifacts / "results/fusion_vs_rf.csv")
    aggregate = comparison.groupby(["antibiotic", "feature_mode", "split", "edge_mode"], as_index=False).agg(
        n_seeds=("seed", "nunique"), mean_delta_f1=("delta_f1", "mean"),
        std_delta_f1=("delta_f1", "std"), mean_fusion_f1=("fusion_f1", "mean"),
        mean_rf_f1=("rf_f1", "mean"), mean_alpha=("alpha", "mean"))
    write_csv(aggregate, artifacts / "results/fusion_summary.csv")
    modes = {}
    for mode in ("raw", "masked"):
        validation = comparison[(comparison["antibiotic"] == pilot_antibiotic) &
                                (comparison["feature_mode"] == mode) &
                                (comparison["split"] == "val") &
                                (comparison["edge_mode"] == "real") &
                                (comparison["seed"].isin(pilot_seeds))]
        complete = set(validation["seed"]) == set(pilot_seeds)
        improvements = int((validation["delta_f1"] > 1e-12).sum())
        mean_gain = float(validation["delta_f1"].mean()) if len(validation) else None
        modes[mode] = {"complete": complete, "improved_seeds": improvements,
                       "mean_validation_delta_f1": mean_gain,
                       "eligible": complete and improvements >= 2 and mean_gain > 1e-12}
    eligible = [mode for mode, result in modes.items() if result["eligible"]]
    gate = {"pilot_antibiotic": pilot_antibiotic, "pilot_seeds": list(pilot_seeds),
            "feature_modes": modes, "eligible_feature_modes": eligible,
            "allow_full_run": bool(eligible),
            "status": "eligible_for_full_run" if eligible else
                      "pilot_incomplete" if not all(row["complete"] for row in modes.values()) else
                      "no_demonstrated_fusion_benefit_retain_rf"}
    path = artifacts / "results/fusion_gate.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    temporary.replace(path)
    return gate
