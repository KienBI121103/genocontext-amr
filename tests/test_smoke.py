from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd

from src.data.inputs import Split
from src.training.pipeline import MODELS, run_group
from src.parsing.gff import GeneRecord


class MemoryStore:
    def get(self, sample: str):
        return [
            GeneRecord(sample, "contig1", f"{sample}-1", 1, 50, "+", "CDS", "geneA", "protein A", (), (), 500),
            GeneRecord(sample, "contig1", f"{sample}-2", 55, 95, "-", "CDS", "geneB", "protein B", (), (), 500),
            GeneRecord(sample, "contig2", f"{sample}-3", 10, 80, "+", "CDS", None, None, (), (), 200),
        ]


def test_small_end_to_end_run(tmp_path: Path):
    samples = ("a", "b", "c", "d", "e", "f")
    labels = {(sample, "drug"): int(index % 2 == 0) for index, sample in enumerate(samples)}
    config = {
        "paths": {"artifacts": str(tmp_path)},
        "features": {"min_category_frequency": 1},
        "graph": {"neighbor_k": 1},
        "model": {"hidden_dim": 16, "dropout": 0.0},
        "training": {"epochs": 2, "patience": 2, "batch_size": 2,
                     "learning_rate": 0.01, "weight_decay": 0.0, "cpu_threads": 1},
        "explain": {"test_isolates_per_run": 1, "top_genes": 1, "neighborhood_k": 1},
    }
    torch.set_num_threads(1)
    run_group(config, MemoryStore(), labels, Split(samples[:2], samples[2:4], samples[4:]),
              "drug", 0, "raw", torch.device("cpu"))
    output = tmp_path / "results/seed0/drug/raw"
    for model in MODELS:
        assert (output / f"{model}_metrics.csv").is_file()
        assert (output / f"{model}_predictions.csv").is_file()
    assert (output / "genocontext_gnn_attribution.csv").is_file()
    metrics = pd.read_csv(output / "genocontext_gnn_real_metrics.csv")
    assert set(metrics["split"]) == {"val", "test"}
    assert {"f1", "auroc", "auprc", "threshold"} <= set(metrics)
    assert "threshold_policy" not in metrics

    (output / "random_forest_metrics.csv").unlink()
    with patch("src.training.pipeline.GeneFeatureEncoder.fit", side_effect=AssertionError("refit")):
        run_group(
            config, MemoryStore(), labels,
            Split(samples[:2], samples[2:4], samples[4:]),
            "drug", 0, "raw", torch.device("cpu"),
        )
    assert (output / "random_forest_metrics.csv").is_file()
