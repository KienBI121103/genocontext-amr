from pathlib import Path

import numpy as np
import torch

from src.explain.attribution import attribute_nodes
from src.features.gene_features import GeneFeatureEncoder, structural_features
from src.graph.builder import build_graph, neighbor_edges
from src.model.graphsage import GenoContextGNN
from src.parsing.gff import GeneRecord


def genes(sample="a"):
    return [
        GeneRecord(sample, "c1", "id1", 10, 100, "+", "CDS", "abc", "ordinary protein", (), (), 1000),
        GeneRecord(sample, "c1", "id2", 105, 150, "+", "CDS", "abc", "beta-lactamase", (), (), 1000),
        GeneRecord(sample, "c2", "id3", 10, 80, "-", "CDS", None, None, (), (), 500),
    ]


def test_train_only_vocabulary_and_masking():
    encoder = GeneFeatureEncoder(mask_amr_terms=True, min_category_frequency=1).fit([genes("train")])
    assert "beta-lactamase" not in encoder.vocabularies["product"]
    test = [GeneRecord("test", "c", "random-id", 1, 9, "+", "CDS", "unseen", None)]
    _, categories = encoder.transform(test)
    assert categories[0, 0] == 0  # unseen gene -> UNK
    assert categories[0, 1] == 1  # missing product -> MISSING
    assert "random-id" not in str(encoder.vocabularies)


def test_gene_distances_and_no_cross_contig_edges():
    records = genes()
    numbers = structural_features(records)
    assert numbers[0, 4] > 0
    assert numbers[1, 3] > 0
    assert numbers[1, 7] == 1
    edges = {tuple(edge) for edge in neighbor_edges(records).T.tolist()}
    assert edges == {(0, 1), (1, 0)}
    assert all(u != 2 and v != 2 for u, v in edges)


def test_graph_serialization_and_attribution(tmp_path: Path):
    records = genes()
    encoder = GeneFeatureEncoder(min_category_frequency=1).fit([records])
    graph = build_graph(records, encoder, 1)
    path = tmp_path / "graph.pt"
    torch.save(graph, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded.num_nodes == 3
    model = GenoContextGNN(encoder.category_sizes, hidden_dim=16, dropout=0)
    rows = attribute_nodes(model, loaded, records, torch.device("cpu"), top_n=1, neighborhood_k=1)
    assert rows
    assert {"isolate_id", "contig", "start", "end", "feature_id", "importance_score"} <= rows[0].keys()
    assert np.isfinite([row["importance_score"] for row in rows]).all()
