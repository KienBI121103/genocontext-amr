"""Inference helpers: bootstrap CIs, corrected repeated K-fold t-test, Benjamini–Hochberg, Wilson intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats


def bootstrap_mean_ci(values: ArrayLike, n_boot: int = 2000, seed: int = 0, level: float = 0.95
                      ) -> tuple[float, float, float]:
    """Mean and percentile CI of the mean over independent units (e.g. fold × drug differences)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def cluster_bootstrap(frame: pd.DataFrame, cluster: str, statistic: Callable[[pd.DataFrame], float],
                      n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``statistic`` resampling whole clusters (e.g. clonal groups)."""
    groups = {name: index.to_numpy() for name, index in frame.groupby(cluster).groups.items()}
    names = list(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([groups[names[i]] for i in rng.integers(0, len(names), len(names))])
        draws.append(statistic(frame.loc[rows]))
    alpha = (1 - level) / 2
    finite = np.array(draws)[np.isfinite(draws)]
    return statistic(frame), float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))


def corrected_repeated_kfold_ttest(differences: ArrayLike, n_folds: int, test_fraction: float = 0.2
                                   ) -> tuple[float, float]:
    """Bouckaert & Frank (2004) corrected t-test for r×K cross-validation differences.

    Variance is inflated by (1/n + n_test/n_train) to account for overlapping training sets.
    Returns (t statistic, two-sided p-value).
    """
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if n < 2 or n_folds < 2:
        raise ValueError("Need at least two paired differences")
    ratio = test_fraction / (1 - test_fraction)
    variance = d.var(ddof=1) * (1 / n + ratio)
    if variance == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt(variance)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def wilson_interval(successes: float, total: float, level: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = successes / total
    centre = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return float(centre - half), float(centre + half)
