"""Validation-selected probability blending; no learned stacking head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score

from src.evaluation.metrics import select_f1_threshold


def align_probabilities(
    expected_ids: tuple[str, ...], sample_ids: tuple[str, ...], probabilities,
) -> np.ndarray:
    """Align a branch to the exact expected cohort, rejecting drops and duplicates."""
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected isolate IDs must be nonempty and unique")
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(expected_ids):
        raise ValueError("Branch isolate IDs do not match the expected cohort")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(sample_ids):
        raise ValueError("Branch probabilities and isolate IDs must have equal length")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Branch probabilities must be finite and within [0, 1]")
    positions = {sample: index for index, sample in enumerate(sample_ids)}
    return values[[positions[sample] for sample in expected_ids]]


def blend_probabilities(rf_probabilities, gnn_probabilities, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Fusion alpha must be finite and within [0, 1]")
    rf = np.asarray(rf_probabilities, dtype=float)
    gnn = np.asarray(gnn_probabilities, dtype=float)
    if rf.ndim != 1 or not len(rf) or rf.shape != gnn.shape:
        raise ValueError("RF and GNN probabilities must be nonempty vectors of equal length")
    for values in (rf, gnn):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Branch probabilities must be finite and within [0, 1]")
    return (1 - alpha) * rf + alpha * gnn


@dataclass(frozen=True)
class FusionSelection:
    alpha: float
    threshold: float
    validation_f1: float

    def predict(self, rf_probabilities, gnn_probabilities) -> np.ndarray:
        return blend_probabilities(rf_probabilities, gnn_probabilities, self.alpha)


def select_fusion(y_val, rf_val, gnn_val) -> tuple[FusionSelection, list[dict]]:
    """Search validation only; exact F1 ties keep the smallest graph weight."""
    best = None
    search = []
    for index in range(21):
        alpha = index / 20
        probabilities = blend_probabilities(rf_val, gnn_val, alpha)
        threshold = select_f1_threshold(y_val, probabilities)
        score = float(f1_score(y_val, probabilities >= threshold))
        search.append({"alpha": alpha, "threshold": threshold, "validation_f1": score})
        if best is None or score > best.validation_f1:
            best = FusionSelection(alpha, threshold, score)
    return best, search
