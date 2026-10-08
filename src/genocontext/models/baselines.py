"""Baselines: mean-pool DeepSets, per-drug query attention (GL-HAT-style), LightGBM and kNN placement."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import torch
from numpy.typing import NDArray
from scipy import sparse
from torch import Tensor, nn

from genocontext.config import ModelConfig
from genocontext.models.kg import AMRModel, Batch, TokenEncoder


class DeepSets(AMRModel):
    """Mean-pooled token embeddings with one output per drug (no knowledge, no gene order)."""

    def __init__(self, n_tokens: int, n_states: int, n_flags: int, n_drugs: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = TokenEncoder(n_tokens, n_states, n_flags, config.hidden, config.dropout)
        self.head = nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.GELU(), nn.Dropout(config.dropout),
                                  nn.Linear(config.hidden, n_drugs))

    def forward(self, batch: Batch) -> Tensor:
        h = self.encoder(batch)
        total = torch.zeros(batch.size, h.shape[1]).index_add_(0, batch.node_graph, h)
        count = torch.bincount(batch.node_graph, minlength=batch.size).clamp(min=1).unsqueeze(-1)
        out: Tensor = self.head(total / count)
        return out


class QueryAttention(AMRModel):
    """A free learned query per drug attends over the isolate's tokens (GL-HAT-style readout)."""

    def __init__(self, n_tokens: int, n_states: int, n_flags: int, n_drugs: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = TokenEncoder(n_tokens, n_states, n_flags, config.hidden, config.dropout)
        self.query = nn.Parameter(torch.randn(n_drugs, config.hidden) / config.hidden**0.5)
        self.out = nn.Parameter(torch.zeros(n_drugs, config.hidden))
        self.bias = nn.Parameter(torch.zeros(n_drugs))

    def forward(self, batch: Batch) -> Tensor:
        h = self.encoder(batch)  # (N, H)
        scores = h @ self.query.T  # (N, D)
        index = batch.node_graph.unsqueeze(-1).expand_as(scores)
        peak = torch.full((batch.size, scores.shape[1]), -torch.inf).scatter_reduce(0, index, scores, "amax")
        scores = scores - peak.detach()[batch.node_graph]  # per-(isolate, drug) stabilisation
        weights = torch.exp(scores)
        norm = torch.zeros(batch.size, weights.shape[1]).index_add_(0, batch.node_graph, weights)
        pooled = torch.zeros(batch.size, weights.shape[1], h.shape[1]).index_add_(
            0, batch.node_graph, weights.unsqueeze(-1) * h.unsqueeze(1))
        pooled = pooled / norm.clamp(min=1e-12).unsqueeze(-1)
        return (pooled * self.out).sum(-1) + self.bias


LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10,
              "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "deterministic": True, "force_col_wise": True}


def fit_lightgbm(x: sparse.csr_matrix, y: NDArray[np.float64], val: tuple[sparse.csr_matrix, NDArray[np.float64]]
                 | None, rounds: int | None, seed: int, threads: int) -> tuple[lgb.Booster, int]:
    """One binary booster; early-stops on ``val`` log-loss (max 2000 rounds) or trains exactly ``rounds``."""
    params = LGB_PARAMS | {"seed": seed, "num_threads": threads}
    train = lgb.Dataset(x, label=y, free_raw_data=False)
    if val is None:
        booster = lgb.train(params, train, num_boost_round=max(int(rounds or 100), 1))
        return booster, booster.current_iteration()
    valid = lgb.Dataset(val[0], label=val[1], reference=train)
    booster = lgb.train(params, train, num_boost_round=2000, valid_sets=[valid],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster, booster.best_iteration


def knn_placement(train_x: sparse.csr_matrix, train_y: NDArray[np.float64], query_x: sparse.csr_matrix,
                  k: int = 5) -> NDArray[np.float64]:
    """Shrunken resistance rate among the k most Jaccard-similar training isolates (lineage placement)."""
    a, b = query_x.astype(np.float32), train_x.astype(np.float32)
    inter = (a @ b.T).toarray()
    union = np.asarray(a.sum(1)) + np.asarray(b.sum(1)).T - inter
    similarity = inter / np.maximum(union, 1)
    nearest = np.argpartition(-similarity, kth=min(k, similarity.shape[1] - 1), axis=1)[:, :k]
    prior = float(train_y.mean())
    rates: NDArray[np.float64] = (train_y[nearest].sum(1) + prior) / (k + 1)
    return rates
