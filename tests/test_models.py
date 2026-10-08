from dataclasses import replace

import numpy as np
import torch

from genocontext.config import ModelConfig, TrainConfig
from genocontext.models.baselines import DeepSets, QueryAttention, knn_placement
from genocontext.models.kg import GenoContextKG
from genocontext.training.fit import NeuralLearner, collate, masked_bce, protocol_a, protocol_b, thresholds_from

N_TOKENS, N_STATES, N_FLAGS, N_MECH, N_DRUGS = 6, 6, 3, 3, 2
MECH_OF_TOKEN = {0: 2, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2}  # token 1 -> mechanism 0 (resistance), token 2 -> mechanism 1


def graph(tokens: list[int]) -> dict[str, np.ndarray]:
    n = len(tokens)
    edges = np.array([[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]).T.reshape(2, -1)
    return {"token": np.array(tokens), "state": np.zeros(n, dtype=np.int64), "flags": np.zeros((n, 3), np.float32),
            "pair_node": np.arange(n), "pair_mech": np.array([MECH_OF_TOKEN[t] for t in tokens]),
            "edge_index": edges.astype(np.int64)}


def kg(config: ModelConfig | None = None) -> GenoContextKG:
    config = config or ModelConfig(hidden=16, query=8, dropout=0.0)
    prior = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # mechanism 0 -> drug 0, mechanism 1 -> drug 1
    return GenoContextKG(N_TOKENS, N_STATES, N_FLAGS, torch.eye(N_DRUGS), prior, config)


def test_kg_decomposition_is_exact_and_dense_control_runs() -> None:
    batch = collate([{k: torch.from_numpy(v) for k, v in graph(t).items()} for t in ([0, 1, 3], [2, 4], [5])])
    for model in (kg(), kg(ModelConfig(hidden=16, query=8, use_prior=False, context_layer=False))):
        model.eval()
        c, pairs = model.decompose(batch)
        assert torch.allclose(model(batch), model.intercepts() + c.sum(-1) + pairs.sum((-1, -2)), atol=1e-6)
        assert model.penalty() >= 0
    assert torch.equal(kg().edge_weights()[0], torch.tensor([1.0, kg().edge_weights()[0, 1].item()]))


def test_baselines_forward_shapes() -> None:
    batch = collate([{k: torch.from_numpy(v) for k, v in graph(t).items()} for t in ([0, 1], [2])])
    for model in (DeepSets(N_TOKENS, N_STATES, N_FLAGS, N_DRUGS, ModelConfig(hidden=8)),
                  QueryAttention(N_TOKENS, N_STATES, N_FLAGS, N_DRUGS, ModelConfig(hidden=8))):
        assert model(batch).shape == (2, N_DRUGS)


def test_masked_bce_ignores_missing_labels() -> None:
    logits = torch.tensor([[2.0, 0.0], [-2.0, 5.0]])
    full = masked_bce(logits, torch.tensor([[1.0, float("nan")], [0.0, float("nan")]]))
    single = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([1.0, 0.0]))
    assert torch.isclose(full, single)


def synthetic(n: int = 160, seed: int = 0) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Token 1 confers resistance to drug 0, token 2 to drug 1; drug 1 is untested for half the isolates."""
    rng = np.random.default_rng(seed)
    graphs, labels = [], np.full((n, N_DRUGS), np.nan)
    for i in range(n):
        has1, has2 = rng.random() < 0.5, rng.random() < 0.5
        graphs.append(graph([0] + [1] * has1 + [2] * has2 + [int(rng.integers(3, 6))]))
        labels[i, 0] = float(has1)
        if i % 2 == 0:
            labels[i, 1] = float(has2)
    return graphs, labels


def test_kg_learns_planted_mechanisms_and_protocols_respect_test() -> None:
    graphs, labels = synthetic()
    learner = NeuralLearner(kg, graphs, labels, TrainConfig(epochs=40, patience=10, batch_size=32,
                                                            learning_rate=1e-2))
    train, test = np.arange(120), np.arange(120, 160)
    inner = np.arange(120) % 5
    outcome_a = protocol_a(learner, labels, train, inner, test, seed=0)
    acc = ((outcome_a.test[:, 0] >= outcome_a.thresholds[0]) == labels[test, 0]).mean()
    assert acc > 0.9
    model = learner.model(outcome_a.fitted[0])
    batch = collate([learner.tensors[i] for i in test])
    c, _ = model.decompose(batch)
    has1 = torch.tensor([1 in g["token"] for g in (graphs[i] for i in test)])
    assert c[has1, 0, 0].mean() > c[~has1, 0, 0].mean()  # mechanism 0 carries drug-0 resistance
    poisoned = labels.copy()
    poisoned[test] = 1 - poisoned[test]  # protocols must not look at test labels
    outcome_b = protocol_b(NeuralLearner(kg, graphs, poisoned, TrainConfig(epochs=15, batch_size=32,
                                                                            learning_rate=1e-2)),
                           poisoned, train, inner, test, seed=0, n_seeds=2)
    assert len(outcome_b.fitted) == 2 and not np.isnan(outcome_b.selection).any()  # out-of-fold for every row
    assert np.allclose(outcome_b.thresholds, thresholds_from(poisoned[train], outcome_b.selection))


def test_knn_placement_uses_neighbours() -> None:
    from scipy import sparse
    train = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]))
    probs = knn_placement(train, np.array([1.0, 1.0, 0.0, 0.0]), sparse.csr_matrix(np.array([[1, 1, 0]])), k=2)
    assert probs[0] > 0.5


def test_model_config_overrides_are_frozen() -> None:
    assert replace(ModelConfig(), use_prior=False).use_prior is False
