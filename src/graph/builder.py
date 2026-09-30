from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

from src.features.gene_features import GeneFeatureEncoder, ordered_genes
from src.parsing.gff import GeneRecord


def neighbor_edges(
    records: list[GeneRecord], neighbor_k: int = 1,
    shuffled: bool = False, seed: int = 0,
) -> torch.Tensor:
    if neighbor_k < 1:
        raise ValueError("neighbor_k must be positive")
    records = ordered_genes(records)
    by_contig: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_contig[record.contig_id].append(index)
    edges: list[tuple[int, int]] = []
    for contig, indices in by_contig.items():
        if shuffled:
            digest = hashlib.sha256(f"{seed}:{records[0].sample_id}:{contig}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            indices = rng.permutation(indices).tolist()
        for i, source in enumerate(indices):
            for target in indices[i + 1 : i + neighbor_k + 1]:
                edges.extend(((source, target), (target, source)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def build_graph(
    records: list[GeneRecord], encoder: GeneFeatureEncoder,
    label: int, neighbor_k: int = 1, shuffled: bool = False, seed: int = 0,
) -> Data:
    records = ordered_genes(records)
    if label not in (0, 1):
        raise ValueError("Graph label must be 0 (S) or 1 (R)")
    numbers, categories = encoder.transform(records)
    graph = Data(
        x_num=torch.from_numpy(numbers), x_cat=torch.from_numpy(categories),
        edge_index=neighbor_edges(records, neighbor_k, shuffled, seed),
        y=torch.tensor([label], dtype=torch.float32),
    )
    graph.sample_id = records[0].sample_id
    graph.num_nodes = len(records)
    return graph
