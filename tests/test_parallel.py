from pathlib import Path

import torch

from src.data.inputs import Isolate, RecordStore
from src.features.gene_features import GeneFeatureEncoder
from src.training.parallel import FeatureBuilder


def test_parallel_feature_builder_matches_serial(tmp_path: Path):
    samples = tuple("abcdefgh")
    manifest = {}
    for sample in samples:
        gff = tmp_path / f"{sample}.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "ctg\tBakta\tCDS\t1\t30\t.\t+\t0\tID=g1;gene=abc\n"
            "ctg\tBakta\tCDS\t40\t70\t.\t-\t0\tID=g2;gene=def\n"
            "ctg\tBakta\tCDS\t80\t110\t.\t+\t0\tID=g3;gene=ghi\n"
        )
        manifest[sample] = Isolate(sample, gff)
    store = RecordStore(manifest, tmp_path / "cache", tmp_path / "db")
    encoder = GeneFeatureEncoder(min_category_frequency=1).fit(
        store.get(sample) for sample in samples
    )
    labels = {(sample, "drug"): index % 2 for index, sample in enumerate(samples)}

    with FeatureBuilder(store, encoder, labels, "drug", 1, workers=1) as builder:
        serial_vectors = builder.genome_matrix(samples)
        serial_graphs = builder.graphs(samples)
        builder.shuffle_edges(samples, serial_graphs, seed=2)

    with FeatureBuilder(store, encoder, labels, "drug", 1, workers=2) as builder:
        parallel_vectors = builder.genome_matrix(samples)
        parallel_graphs = builder.graphs(samples)
        builder.shuffle_edges(samples, parallel_graphs, seed=2)

    assert (serial_vectors != parallel_vectors).nnz == 0
    for serial, parallel in zip(serial_graphs, parallel_graphs, strict=True):
        assert torch.equal(serial.x_num, parallel.x_num)
        assert torch.equal(serial.x_cat, parallel.x_cat)
        assert torch.equal(serial.edge_index, parallel.edge_index)
        assert torch.equal(serial.y, parallel.y)
