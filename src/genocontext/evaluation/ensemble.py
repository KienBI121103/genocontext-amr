"""Stacked ensembles fitted on selection-set probabilities only (out-of-fold under protocol B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from genocontext.evaluation.metrics import select_threshold

KEYS = ["isolate_id", "drug", "part", "label"]


def _logit(p: pd.Series | pd.DataFrame) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out: NDArray[np.float64] = np.log(clipped / (1 - clipped))
    return out


def stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per drug, a logistic stacker over member logits; its threshold is chosen on the selection rows."""
    merged = None
    for name, frame in frames.items():
        part = frame[KEYS + ["probability"]].rename(columns={"probability": name})
        merged = part if merged is None else merged.merge(part, on=KEYS, how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping predictions to stack")
    names, rows = list(frames), []
    for _drug, group in merged.groupby("drug"):
        selection = group[group["part"] == "selection"]
        if selection["label"].nunique() < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(selection[names]), selection["label"])
        probability = model.predict_proba(_logit(group[names]))[:, 1]
        threshold = select_threshold(selection["label"].astype(int), probability[group["part"] == "selection"])
        rows.append(group[KEYS].assign(probability=probability, threshold=threshold,
                                       **{f"weight_{n}": w for n, w in zip(names, model.coef_[0], strict=True)}))
    return pd.concat(rows, ignore_index=True)
