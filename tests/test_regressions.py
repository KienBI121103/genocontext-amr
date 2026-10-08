"""Regression tests for issues found in the 2026-10-08 code review."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from genocontext.config import FeatureConfig, ModelConfig, TrainConfig
from genocontext.features.lexicon import Knowledge
from genocontext.features.space import FeatureSpace
from genocontext.models.baselines import QueryAttention
from genocontext.pipeline import e0_inner
from genocontext.training.fit import LightGBMLearner, collate, masked_bce, protocol_b
from tests.test_features import _store, node
from tests.test_models import graph, kg


def test_mutually_exclusive_alleles_are_not_merged(tmp_path: Path, knowledge: Knowledge) -> None:
    profiles = [[node("fam:A" if r % 2 else "fam:B"), node("fam:C" if r < 4 else "fam:D")] for r in range(8)]
    space = FeatureSpace.fit(_store(tmp_path, knowledge, profiles), np.arange(8),
                             FeatureConfig(min_prevalence=1, max_families=10))
    blocks = [set(m) for m in space.members]
    assert {"fam:A"} in blocks and {"fam:B"} in blocks  # r = -1: two blocks, not one constant block


def test_lightgbm_skips_drugs_without_training_labels() -> None:
    rng = np.random.default_rng(0)
    x = sparse.csr_matrix(rng.integers(0, 2, (60, 5)).astype(np.float32))
    labels = np.column_stack([x[:, 0].toarray().ravel(), np.full(60, np.nan)])  # drug 1 masked (tier 1)
    learner = LightGBMLearner(x, labels, threads=1)
    outcome = protocol_b(learner, labels, np.arange(45), np.arange(45) % 5, np.arange(45, 60), seed=0, n_seeds=1)
    assert np.isfinite(outcome.test[:, 0]).all() and np.isnan(outcome.test[:, 1]).all()


def test_masked_loss_is_finite_for_unlabelled_batch() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = masked_bce(logits, torch.full((2, 3), float("nan")))
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(logits.grad).all()


def test_query_attention_is_stable_per_isolate_and_drug() -> None:
    model = QueryAttention(6, 6, 3, 2, ModelConfig(hidden=8, dropout=0.0)).eval()
    with torch.no_grad():
        model.query.mul_(1e3)  # extreme scores would underflow a single global max
        model.out.normal_()
    batch = collate([{k: torch.from_numpy(v) for k, v in graph(t).items()} for t in ([0, 1], [2, 5])])
    alone = collate([{k: torch.from_numpy(v) for k, v in graph([2, 5]).items()}])
    out = model(batch)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1], model(alone)[0], atol=1e-4)  # independent of batch companions


def test_e0_protocol_a_validates_on_fold0_or_a_fold_without_big_groups() -> None:
    groups = ["BIG"] * 40 + [f"g{i}" for i in range(60)]
    folds = [1] * 40 + [i % 5 for i in range(60)]
    part = pd.DataFrame({"group": groups, "outer": "train", "inner_fold": folds})
    _, a_folds, _, _ = e0_inner(part, held=2)
    train = np.flatnonzero(part["inner_fold"].to_numpy() != 2)
    assert set(part["inner_fold"].to_numpy()[train][a_folds == 0]) == {0}
    _, a_folds, _, _ = e0_inner(part, held=0)
    train = np.flatnonzero(part["inner_fold"].to_numpy() != 0)
    assert 1 not in set(part["inner_fold"].to_numpy()[train][a_folds == 0])  # fold 1 holds the big group


def test_zero_shot_intercept_comes_from_class_features() -> None:
    model = kg()
    with torch.no_grad():
        model.bias.fill_(5.0)
        model.drug_embedding_mask[1] = 0.0
    intercepts = model.intercepts()
    class_term = model.class_bias(model.drug_features).squeeze(-1)
    assert torch.isclose(intercepts[1], class_term[1]) and torch.isclose(intercepts[0], class_term[0] + 5.0)


def test_train_config_defaults_are_valid() -> None:
    assert TrainConfig().ensemble_seeds >= 1
