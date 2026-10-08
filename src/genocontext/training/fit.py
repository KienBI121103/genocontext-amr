"""Learners and the two data-budget protocols.

Protocol A (64/16/20): fit on inner folds 1-4, early-stop/select and set thresholds on inner fold 0.
Protocol B (80/20):    inner 5-fold CV gives the training budget (median best epoch / boosting rounds) and
                       out-of-fold probabilities for thresholds; then refit on all training isolates
                       (averaging ``ensemble_seeds`` refits).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import sparse
from torch import Tensor, nn

from genocontext.config import TrainConfig
from genocontext.evaluation.metrics import select_threshold
from genocontext.features.space import Graph
from genocontext.models.baselines import fit_lightgbm, knn_placement
from genocontext.models.kg import AMRModel, Batch

LOG = logging.getLogger(__name__)
Index = NDArray[np.int64]
Labels = NDArray[np.float64]  # (n_isolates, n_drugs), NaN = untested


class Learner(Protocol):
    def fit(self, train: Index, val: Index | None, seed: int, budget: NDArray[np.float64] | None
            ) -> tuple[Any, NDArray[np.float64]]: ...

    def predict(self, fitted: Any, rows: Index) -> NDArray[np.float64]: ...


# ---- neural ---------------------------------------------------------------------------------------
def collate(graphs: list[dict[str, Tensor]]) -> Batch:
    offsets = np.cumsum([0] + [len(g["token"]) for g in graphs])
    return Batch(
        token=torch.cat([g["token"] for g in graphs]), state=torch.cat([g["state"] for g in graphs]),
        flags=torch.cat([g["flags"] for g in graphs]),
        node_graph=torch.cat([torch.full((len(g["token"]),), i, dtype=torch.long) for i, g in enumerate(graphs)]),
        pair_node=torch.cat([g["pair_node"] + int(o) for g, o in zip(graphs, offsets, strict=False)]),
        pair_mech=torch.cat([g["pair_mech"] for g in graphs]),
        edge_index=torch.cat([g["edge_index"] + int(o) for g, o in zip(graphs, offsets, strict=False)], dim=1),
        size=len(graphs),
    )


def masked_bce(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean over drugs (with >= 1 label in the batch) of the per-drug mean BCE."""
    mask = ~torch.isnan(labels)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(labels), reduction="none")
    if not mask.any():  # e.g. a final batch of isolates unlabelled under a subsampling variant
        return (logits * 0).sum()
    per_drug = (loss * mask).sum(0) / mask.sum(0).clamp(min=1)
    return per_drug[mask.any(0)].mean()


@dataclass
class NeuralLearner:
    build: Callable[[], AMRModel]
    graphs: list[Graph]
    labels: Labels
    config: TrainConfig

    def __post_init__(self) -> None:
        self.tensors = [{k: torch.from_numpy(v) for k, v in g.items()} for g in self.graphs]
        self.targets = torch.from_numpy(self.labels.astype(np.float32))

    def _batches(self, rows: Index, rng: np.random.Generator | None) -> list[Index]:
        order = rng.permutation(rows) if rng is not None else rows
        return [order[i:i + self.config.batch_size] for i in range(0, len(order), self.config.batch_size)]

    def _loss(self, model: AMRModel, rows: Index) -> float:
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(collate([self.tensors[i] for i in b])) for b in self._batches(rows, None)])
        return float(masked_bce(logits, self.targets[rows]))

    def fit(self, train: Index, val: Index | None, seed: int, budget: NDArray[np.float64] | None
            ) -> tuple[dict[str, Tensor], NDArray[np.float64]]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = self.build()
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        epochs = self.config.epochs if budget is None else max(int(budget[0]), 1)
        best, best_state, best_epoch, stale = float("inf"), copy.deepcopy(model.state_dict()), epochs, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for rows in self._batches(train, rng):
                optimiser.zero_grad(set_to_none=True)
                loss = masked_bce(model(collate([self.tensors[i] for i in rows])), self.targets[rows])
                torch.autograd.backward(loss + model.penalty())
                optimiser.step()
            if val is None:
                continue
            score = self._loss(model, val)
            if score < best - 1e-4:
                best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
            elif (stale := stale + 1) >= self.config.patience:
                break
        state = best_state if val is not None else model.state_dict()
        return state, np.full(self.labels.shape[1], float(best_epoch))

    def model(self, state: dict[str, Tensor]) -> AMRModel:
        model = self.build()
        model.load_state_dict(state)
        return model.eval()

    def predict(self, fitted: dict[str, Tensor], rows: Index) -> NDArray[np.float64]:
        model = self.model(fitted)
        with torch.no_grad():
            logits = torch.cat([model(collate([self.tensors[i] for i in b])) for b in self._batches(rows, None)])
        probs: NDArray[np.float64] = torch.sigmoid(logits).double().numpy()
        return probs


# ---- tabular --------------------------------------------------------------------------------------
@dataclass
class LightGBMLearner:
    x: sparse.csr_matrix
    labels: Labels
    threads: int

    def fit(self, train: Index, val: Index | None, seed: int, budget: NDArray[np.float64] | None
            ) -> tuple[list[Any], NDArray[np.float64]]:
        boosters: list[Any] = []
        rounds: list[float] = []
        for d in range(self.labels.shape[1]):
            rows = train[~np.isnan(self.labels[train, d])]
            if len(np.unique(self.labels[rows, d])) < 2:  # drug masked (tier 1) or single-class: no model
                boosters.append(None)
                rounds.append(np.nan)
                continue
            held = None
            if val is not None:
                vrows = val[~np.isnan(self.labels[val, d])]
                held = (self.x[vrows], self.labels[vrows, d]) if len(vrows) else None
            rounds_d = None if budget is None or np.isnan(budget[d]) else int(budget[d])
            booster, best = fit_lightgbm(self.x[rows], self.labels[rows, d], held, rounds_d, seed, self.threads)
            boosters.append(booster)
            rounds.append(float(best))
        return boosters, np.array(rounds, dtype=float)

    def predict(self, fitted: list[Any], rows: Index) -> NDArray[np.float64]:
        return np.column_stack([np.full(len(rows), np.nan) if b is None else
                                b.predict(self.x[rows], num_iteration=b.best_iteration or None) for b in fitted])


@dataclass
class KNNLearner:
    x: sparse.csr_matrix
    labels: Labels
    k: int = 5

    def fit(self, train: Index, val: Index | None, seed: int, budget: NDArray[np.float64] | None
            ) -> tuple[Index, NDArray[np.float64]]:
        return train, np.zeros(self.labels.shape[1])

    def predict(self, fitted: Index, rows: Index) -> NDArray[np.float64]:
        out = np.full((len(rows), self.labels.shape[1]), np.nan)
        for d in range(self.labels.shape[1]):
            ref = fitted[~np.isnan(self.labels[fitted, d])]
            out[:, d] = knn_placement(self.x[ref], self.labels[ref, d], self.x[rows], self.k)
        return out


# ---- protocols ------------------------------------------------------------------------------------
@dataclass
class Outcome:
    test: NDArray[np.float64]  # (n_test, D)
    selection: NDArray[np.float64]  # validation (A) or out-of-fold (B) probabilities, (n_train, D), NaN elsewhere
    thresholds: NDArray[np.float64]
    budget: NDArray[np.float64]
    fitted: list[Any]


def thresholds_from(labels: Labels, probs: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.full(labels.shape[1], 0.5)
    for d in range(labels.shape[1]):
        ok = ~np.isnan(labels[:, d]) & ~np.isnan(probs[:, d])
        if ok.any() and len(np.unique(labels[ok, d])) == 2:
            out[d] = select_threshold(labels[ok, d], probs[ok, d])
    return out


def protocol_a(learner: Learner, labels: Labels, train: Index, inner_fold: Index, test: Index, seed: int) -> Outcome:
    fit_rows, val_rows = train[inner_fold != 0], train[inner_fold == 0]
    fitted, budget = learner.fit(fit_rows, val_rows, seed, None)
    selection = np.full((len(train), labels.shape[1]), np.nan)
    selection[inner_fold == 0] = learner.predict(fitted, val_rows)
    return Outcome(learner.predict(fitted, test), selection, thresholds_from(labels[train], selection), budget,
                   [fitted])


def protocol_b(learner: Learner, labels: Labels, train: Index, inner_fold: Index, test: Index, seed: int,
               n_seeds: int) -> Outcome:
    selection = np.full((len(train), labels.shape[1]), np.nan)
    budgets = []
    for fold in np.unique(inner_fold):
        fitted, budget = learner.fit(train[inner_fold != fold], train[inner_fold == fold], seed + int(fold), None)
        selection[inner_fold == fold] = learner.predict(fitted, train[inner_fold == fold])
        budgets.append(budget)
    stacked = np.array(budgets, dtype=float)
    budget = np.full(stacked.shape[1], np.nan)
    finite = ~np.isnan(stacked).all(axis=0)
    budget[finite] = np.nanmedian(stacked[:, finite], axis=0)
    refits = [learner.fit(train, None, seed + 100 + s, budget)[0] for s in range(n_seeds)]
    test_probs = np.mean([learner.predict(f, test) for f in refits], axis=0)
    return Outcome(test_probs, selection, thresholds_from(labels[train], selection), budget, refits)
