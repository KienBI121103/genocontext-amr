import numpy as np
import pandas as pd
import pytest
import torch

from genocontext.evaluation.ensemble import stack
from genocontext.explain.report import integrated_gradients
from genocontext.pipeline import Task, apply_variant
from genocontext.training.fit import collate
from tests.test_models import graph, kg

DRUGS = ("a", "b")


def labels_and_train() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, (40, 2)).astype(float)
    labels[::4, 1] = np.nan
    return labels, np.arange(30)


def test_variants_only_touch_training_labels() -> None:
    labels, train = labels_and_train()
    test = np.arange(30, 40)
    for variant in ("", "rep2", "shuffled_labels", "zeroshot=b", "n5-a-r1"):
        out = apply_variant(variant, labels, train, DRUGS, seed=3)
        assert np.array_equal(out.labels[test], labels[test], equal_nan=True), variant
    assert apply_variant("rep2", labels, train, DRUGS, 3).train_seed == 2003
    zero = apply_variant("zeroshot=b", labels, train, DRUGS, 3)
    assert zero.zero_shot == "b" and np.isnan(zero.labels[train, 1]).all()
    sub = apply_variant("n5-a-r1", labels, train, DRUGS, 3)
    assert (~np.isnan(sub.labels[train, 0])).sum() == 5
    shuffled = apply_variant("shuffled_labels", labels, train, DRUGS, 3)
    assert np.nansum(shuffled.labels[train]) == np.nansum(labels[train])  # permutation keeps class counts
    with pytest.raises(ValueError):
        apply_variant("bogus", labels, train, DRUGS, 3)


def test_task_directory_encodes_variant(tmp_path: object) -> None:
    from pathlib import Path
    root = Path(str(tmp_path))
    assert Task("cg_blocked", 0, "kg", "B").directory(root).parts[-3:] == ("all", "kg", "B")
    assert Task("amrgnn_fixed", 1, "kg", "A", "imipenem", "zeroshot=imipenem").directory(root).parts[-3:] == (
        "imipenem", "kg@zeroshot=imipenem", "A")


def test_integrated_gradients_are_complete() -> None:
    model = kg()
    model.eval()
    batch = collate([{k: torch.from_numpy(v) for k, v in graph(t).items()} for t in ([0, 1, 3], [2, 4, 5])])
    attr = integrated_gradients(model, batch, n_drugs=2, steps=64)
    with torch.no_grad():
        gap = model(batch) - model(batch, torch.zeros(len(batch.token)))
    summed = torch.zeros_like(gap).index_add_(0, batch.node_graph, attr)
    assert torch.allclose(summed, gap, atol=2e-2)


def test_stacking_uses_selection_rows_only() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for part, n in (("selection", 200), ("test", 50)):
        y = rng.integers(0, 2, n)
        rows.append(pd.DataFrame({"isolate_id": [f"{part}{i}" for i in range(n)], "drug": "a", "part": part,
                                  "label": y, "good": np.clip(y * 0.6 + rng.random(n) * 0.4, 0.01, 0.99),
                                  "noise": rng.random(n)}))
    frame = pd.concat(rows)
    frames = {m: frame.rename(columns={m: "probability"})[["isolate_id", "drug", "part", "label", "probability"]]
              for m in ("good", "noise")}
    stacked = stack(frames)
    assert stacked["weight_good"].iloc[0] > abs(stacked["weight_noise"].iloc[0])
    poisoned = {m: f.assign(label=np.where(f["part"] == "test", 1 - f["label"], f["label"])) for m, f in frames.items()}
    restacked = stack(poisoned)
    assert np.allclose(restacked["threshold"], stacked["threshold"])  # test labels never influence the stacker
