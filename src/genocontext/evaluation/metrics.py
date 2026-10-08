"""Thresholds and per-drug binary resistance metrics (1 = resistant)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _validate(y_true: ArrayLike, scores: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    labels = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != probs.shape or not len(labels):
        raise ValueError("Labels and scores must be non-empty vectors of equal length")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("Labels must be binary")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Scores must be probabilities")
    return labels, probs


def select_threshold(y_true: ArrayLike, scores: ArrayLike, objective: str = "balanced_accuracy") -> float:
    """Pick the observed score that maximises balanced accuracy (default) or resistant-class F1.

    Ties resolve to the candidate closest to 0.5.
    """
    labels, probs = _validate(y_true, scores)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection needs both classes")
    candidates = np.unique(probs)
    order = np.sort(probs)
    sorted_labels = labels[np.argsort(probs, kind="stable")]
    below_pos = np.concatenate(([0], np.cumsum(sorted_labels)))
    cut = np.searchsorted(order, candidates, side="left")
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    tp = n_pos - below_pos[cut]
    fp = (len(labels) - cut) - tp
    if objective == "balanced_accuracy":
        score = 0.5 * (tp / n_pos + (n_neg - fp) / n_neg)
    elif objective == "f1":
        score = 2 * tp / (2 * tp + fp + (n_pos - tp))
    else:
        raise ValueError(f"Unknown threshold objective: {objective}")
    best = np.flatnonzero(np.isclose(score, score.max()))
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def calibration(labels: NDArray[np.int64], probs: NDArray[np.float64]) -> tuple[float, float]:
    """Calibration intercept and slope from a logistic recalibration of logit(p)."""
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metric_row(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, float]:
    """Discrimination, error-rate and calibration metrics at a fixed threshold."""
    labels, probs = _validate(y_true, scores)
    predicted = (probs >= threshold).astype(np.int64)
    tp = int(((labels == 1) & (predicted == 1)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    both = tp + fn > 0 and tn + fp > 0
    intercept, slope = calibration(labels, probs) if both else (float("nan"), float("nan"))
    return {
        "n": float(len(labels)), "n_resistant": float(tp + fn), "n_susceptible": float(tn + fp),
        "threshold": threshold, "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)) if both else float("nan"),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probs)) if both else float("nan"),
        "auprc": float(average_precision_score(labels, probs)) if both else float("nan"),
        "vme": fn / (tp + fn) if tp + fn else float("nan"),  # resistant called susceptible
        "me": fp / (tn + fp) if tn + fp else float("nan"),  # susceptible called resistant
        "brier": float(brier_score_loss(labels, probs)),
        "calibration_intercept": intercept, "calibration_slope": slope,
    }
