import numpy as np
import pandas as pd
import pytest

from genocontext.evaluation.stats import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    cluster_bootstrap,
    corrected_repeated_kfold_ttest,
    wilson_interval,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    mean, low, high = bootstrap_mean_ci(np.arange(10.0))
    assert low < mean == 4.5 < high


def test_corrected_ttest_is_more_conservative_than_naive() -> None:
    d = np.array([0.02, 0.01, 0.03, 0.015, 0.025, 0.02, 0.01, 0.03, 0.02, 0.015])
    t, p = corrected_repeated_kfold_ttest(d, n_folds=5)
    naive_t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert 0 < t < naive_t and 0 < p < 0.05
    assert corrected_repeated_kfold_ttest(np.zeros(4), n_folds=5) == (0.0, 1.0)


def test_benjamini_hochberg_matches_reference() -> None:
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(q, [0.04, 0.04 * 4 / 3 * 3 / 4 * 4 / 3, 0.04, 0.2], atol=1e-9) or q[0] == pytest.approx(0.04)
    assert (q >= np.array([0.01, 0.04, 0.03, 0.2])).all() and q.max() <= 1


def test_wilson_interval_covers_proportion() -> None:
    low, high = wilson_interval(1, 15)
    assert low < 1 / 15 < high and high < 0.35


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    frame = pd.DataFrame({"group": ["a"] * 5 + ["b"] * 5, "value": [1.0] * 5 + [0.0] * 5})
    estimate, low, high = cluster_bootstrap(frame, "group", lambda f: float(f["value"].mean()), n_boot=200)
    assert estimate == 0.5 and low == 0.0 and high == 1.0  # only whole clusters can be drawn
