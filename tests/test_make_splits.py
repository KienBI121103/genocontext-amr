"""The split script is standalone (never imported by the package); load it from its path for testing."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/kp/make_splits.py"
spec = importlib.util.spec_from_file_location("make_splits", SCRIPT)
assert spec and spec.loader
make_splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_splits)


def test_deduplicate_keeps_one_genome_per_biosample() -> None:
    labels = pd.DataFrame({"d": [1.0, np.nan, 0.0]}, index=["a", "b", "c"])
    biosample = pd.Series({"a": "S1", "b": "S1", "c": "S2"})
    assert make_splits.deduplicate(labels, biosample) == ["a", "c"]


def test_clonal_groups_join_single_locus_variants() -> None:
    def row(st: str, alleles: list[str | None]) -> list[str | None]:
        return [st, *alleles]
    base = ["1", "1", "1", "1", "1", "1", "1"]
    alleles = pd.DataFrame([row("258", base)] * 5 + [row("512", base[:6] + ["9"])] * 2 + [row("15", ["2"] * 7)]
                           + [row("-", base[:6] + [None]), row("-", ["2"] * 6 + [None])],
                           index=[f"i{k}" for k in range(10)], columns=["st", *range(3, 10)])
    groups = make_splits.clonal_groups(alleles)
    assert groups["i5"] == "CG258" and groups["i7"] == "CG15"
    assert groups["i8"] == "CG258"  # matches ST258 and ST512 at 6/7 alleles; both belong to CG258
    assert groups["i9"] == "CG15"


def test_balanced_folds_keep_groups_whole_and_balance_classes() -> None:
    rng = np.random.default_rng(0)
    isolates = [f"i{k}" for k in range(200)]
    groups = pd.Series([f"g{k // 4}" for k in range(200)], index=isolates)
    labels = pd.DataFrame({"d": rng.integers(0, 2, 200).astype(float)}, index=isolates)
    folds = make_splits.balanced_folds(groups, labels, 5, np.random.default_rng(1), forbid_fold0={"g0"})
    assert folds.groupby(groups).nunique().max() == 1
    assert folds["i0"] != 0
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 4
