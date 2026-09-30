"""Bounded parallel feature and graph construction for large isolate sets."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import torch
from scipy import sparse
from torch_geometric.data import Data

from src.data.inputs import RecordStore
from src.features.gene_features import GeneFeatureEncoder
from src.graph.builder import build_graph, neighbor_edges

_WORKER_STORE: RecordStore
_WORKER_ENCODER: GeneFeatureEncoder
_WORKER_LABELS: dict[tuple[str, str], int]
_WORKER_ANTIBIOTIC: str
_WORKER_NEIGHBOR_K: int


def _init_worker(
    store: RecordStore, encoder: GeneFeatureEncoder,
    labels: dict[tuple[str, str], int], antibiotic: str, neighbor_k: int,
) -> None:
    global _WORKER_STORE, _WORKER_ENCODER, _WORKER_LABELS, _WORKER_ANTIBIOTIC, _WORKER_NEIGHBOR_K
    _WORKER_STORE = store
    _WORKER_ENCODER = encoder
    _WORKER_LABELS = labels
    _WORKER_ANTIBIOTIC = antibiotic
    _WORKER_NEIGHBOR_K = neighbor_k
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _vector(sample: str) -> sparse.csr_matrix:
    return _WORKER_ENCODER.genome_vector(_WORKER_STORE.get(sample))


def _graph(sample: str) -> Data:
    return build_graph(
        _WORKER_STORE.get(sample), _WORKER_ENCODER,
        _WORKER_LABELS[(sample, _WORKER_ANTIBIOTIC)], _WORKER_NEIGHBOR_K,
    )


def _shuffled_edges(item: tuple[str, int]) -> torch.Tensor:
    sample, seed = item
    return neighbor_edges(
        _WORKER_STORE.get(sample), _WORKER_NEIGHBOR_K, shuffled=True, seed=seed,
    )


class FeatureBuilder:
    """Build one isolate at a time, using separate processes for Python-heavy work."""

    def __init__(
        self, store: RecordStore, encoder: GeneFeatureEncoder,
        labels: dict[tuple[str, str], int], antibiotic: str,
        neighbor_k: int, workers: int = 1,
    ) -> None:
        if workers < 1:
            raise ValueError("preprocess_workers must be positive")
        self.store = store
        self.encoder = encoder
        self.labels = labels
        self.antibiotic = antibiotic
        self.neighbor_k = neighbor_k
        self.workers = workers
        self.executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "FeatureBuilder":
        if self.workers > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    self.store, self.encoder, self.labels,
                    self.antibiotic, self.neighbor_k,
                ),
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def _map(self, worker, local, items: tuple) -> list:
        if self.executor is None:
            return [local(item) for item in items]
        result = []
        for start in range(0, len(items), 32):
            result.extend(self.executor.map(worker, items[start:start + 32], chunksize=4))
        return result

    def genome_matrix(self, samples: tuple[str, ...]) -> sparse.csr_matrix:
        rows = self._map(
            _vector, lambda sample: self.encoder.genome_vector(self.store.get(sample)),
            samples,
        )
        return sparse.vstack(rows, format="csr")

    def graphs(self, samples: tuple[str, ...]) -> list[Data]:
        return self._map(
            _graph,
            lambda sample: build_graph(
                self.store.get(sample), self.encoder,
                self.labels[(sample, self.antibiotic)], self.neighbor_k,
            ),
            samples,
        )

    def shuffle_edges(self, samples: tuple[str, ...], graphs: list[Data], seed: int) -> None:
        if len(samples) != len(graphs):
            raise ValueError("Sample and graph counts differ")
        items = tuple((sample, seed) for sample in samples)
        edges = self._map(
            _shuffled_edges,
            lambda item: neighbor_edges(
                self.store.get(item[0]), self.neighbor_k, shuffled=True, seed=item[1],
            ),
            items,
        )
        for graph, edge_index in zip(graphs, edges, strict=True):
            graph.edge_index = edge_index
