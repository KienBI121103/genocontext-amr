"""Validation threshold selection and binary resistance metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty vectors of equal length")
    if not set(labels).issubset({0, 1}):
        raise ValueError("Labels must be binary (0=S, 1=R)")
    labels = labels.astype(int)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction probabilities must be finite and within [0, 1]")
    return labels, scores


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the observed validation score that gives the highest resistant-class F1."""
    labels, scores = _validate(y_true, probabilities)
    if set(labels) != {0, 1}:
        raise ValueError("Validation labels must contain both S and R")
    thresholds = np.unique(scores)
    order = np.argsort(scores)
    prefix_positive = np.concatenate(([0], np.cumsum(labels[order])))
    positions = np.searchsorted(scores[order], thresholds, side="left")
    true_positive = labels.sum() - prefix_positive[positions]
    false_positive = len(labels) - positions - true_positive
    false_negative = labels.sum() - true_positive
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return float(thresholds[np.argmax(f1)])


def metric_row(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float,
    antibiotic: str, model: str, split: str, seed: int,
    feature_mode: str, edge_mode: str,
) -> dict[str, object]:
    labels, scores = _validate(y_true, probabilities)
    predicted = (scores >= threshold).astype(int)
    tp = int(((labels == 1) & (predicted == 1)).sum())
    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    n_resistant = tp + fn
    n_susceptible = tn + fp
    return {
        "antibiotic": antibiotic, "model": model, "split": split, "seed": seed,
        "feature_mode": feature_mode, "edge_mode": edge_mode,
        "n": len(labels), "n_resistant": n_resistant, "n_susceptible": n_susceptible,
        "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": tn / n_susceptible if n_susceptible else float("nan"),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, scores)) if n_resistant and n_susceptible else float("nan"),
        "auprc": float(average_precision_score(labels, scores)) if n_resistant and n_susceptible else float("nan"),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }


def prediction_rows(
    sample_ids: tuple[str, ...], y_true: np.ndarray, probabilities: np.ndarray,
    threshold: float, antibiotic: str, model: str, split: str, seed: int,
    feature_mode: str, edge_mode: str,
) -> list[dict[str, object]]:
    labels, scores = _validate(y_true, probabilities)
    if len(sample_ids) != len(labels):
        raise ValueError("Isolate IDs and predictions must have equal length")
    return [
        {
            "isolate_id": sample, "antibiotic": antibiotic, "model": model,
            "split": split, "seed": seed, "feature_mode": feature_mode,
            "edge_mode": edge_mode, "true_label": "R" if label else "S",
            "predicted_label": "R" if score >= threshold else "S",
            "probability_R": float(score), "threshold": threshold,
        }
        for sample, label, score in zip(sample_ids, labels, scores, strict=True)
    ]
