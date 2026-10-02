import json
from pathlib import Path

import pandas as pd
import torch

from scripts.explain import case_table, explain_run
from src.data.inputs import RecordStore, Split, read_manifest, read_phenotypes
from src.training.pipeline import run_group


def test_case_group_selection():
    index = pd.Index(["a", "b", "c", "d"], name="isolate_id")
    rf = pd.DataFrame({"true_label": ["R", "S", "R", "S"], "predicted_label": ["S", "S", "R", "R"]}, index=index)
    fusion = pd.DataFrame({"true_label": ["R", "S", "R", "S"], "predicted_label": ["R", "R", "R", "R"]}, index=index)
    assert case_table(rf, fusion)["case_group"].tolist() == ["rf_corrected", "error_introduced", "both_correct", "both_wrong"]


def test_posthoc_explanation_keeps_completed_run_compatible(tmp_path: Path):
    samples = tuple(f"s{index}" for index in range(8))
    manifest_rows, label_rows = [], []
    for index, sample in enumerate(samples):
        path = tmp_path / f"{sample}.gff3"
        gene = "qnrB1" if index % 2 == 0 else "ordinary"
        path.write_text("##gff-version 3\n"
                        f"ctg\tBakta\tCDS\t1\t90\t.\t+\t0\tID=g1;gene={gene};product=protein\n"
                        "ctg\tBakta\tCDS\t100\t180\t.\t+\t0\tID=g2;gene=gyrA;product=DNA%20gyrase\n"
                        "ctg\tBakta\tCDS\t200\t300\t.\t-\t0\tID=g3;product=hypothetical%20protein\n")
        manifest_rows.append({"isolate_id": sample, "gff3_path": str(path)})
        label_rows.append({"isolate_id": sample, "antibiotic": "drug", "label": "R" if index % 2 == 0 else "S"})
    manifest = tmp_path / "manifest.csv"
    phenotypes = tmp_path / "phenotypes.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    pd.DataFrame(label_rows).to_csv(phenotypes, index=False)
    config = {
        "paths": {"artifacts": str(tmp_path / "artifacts"), "manifest": str(manifest), "phenotypes": str(phenotypes)},
        "bakta": {"db": str(tmp_path / "db"), "threads": 1}, "fusion": {"enabled": True},
        "features": {"min_category_frequency": 1}, "graph": {"neighbor_k": 1},
        "model": {"hidden_dim": 8, "dropout": 0., "pooling": "mean_max_count"},
        "training": {"epochs": 1, "patience": 1, "batch_size": 2, "learning_rate": .01,
                     "weight_decay": 0., "cpu_threads": 1, "preprocess_workers": 1,
                     "checkpoint_metric": "auroc"},
        "explain": {"test_isolates_per_run": 1, "top_genes": 1, "neighborhood_k": 1},
    }
    torch.set_num_threads(1)
    store = RecordStore(read_manifest(manifest), tmp_path / "cache", tmp_path / "db")
    labels = read_phenotypes(phenotypes)
    split = Split(samples[:4], samples[4:6], samples[6:])
    run_group(config, store, labels, split, "drug", 0, "raw", torch.device("cpu"))
    run = tmp_path / "artifacts/results/seed0/drug/raw"
    before = {path: path.read_bytes() for path in run.iterdir() if path.is_file()}
    output = explain_run(config, "drug", 0, "raw", "all", top_genes=3, neighborhood_k=1, threads=1)
    assert all(path.read_bytes() == content for path, content in before.items())
    rows = pd.read_csv(output / "gene_neighborhoods.csv")
    assert set(rows["isolate_id"]) == set(split.test)
    assert set(rows["explanation_scope"]) == {"graph_branch"}
    assert {"db_xrefs", "ec_numbers", "gff3_path", "case_group", "probability_fused"} <= set(rows)
    summary = pd.read_csv(output / "gene_summary.csv")
    assert "qnrB1" in set(summary["gene"])
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["n_explained"] == 2 and not metadata["retrained"]
    assert (output / "rf_global_features.csv").exists()
    # Interpretation exports neither alter training signatures nor invalidate resume.
    run_group(config, store, labels, split, "drug", 0, "raw", torch.device("cpu"))
