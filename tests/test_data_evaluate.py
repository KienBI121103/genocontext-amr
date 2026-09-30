from pathlib import Path

import numpy as np
import pytest

from src.data.inputs import Isolate, RecordStore, read_manifest, read_phenotypes, read_split
from src.evaluation.metrics import metric_row, select_f1_threshold


def test_kp_manifest_and_binary_labels(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,assembly_path\n12.3,/a/12.3_genomic.fna\n")
    assert read_manifest(manifest)["12.3"].gff3_path.name == "12.3_genomic.gff3"
    labels = tmp_path / "labels.csv"
    labels.write_text("id,Biosample,drug\n12.3,bio,1\n13.3,bio,0\n")
    assert read_phenotypes(labels) == {("12.3", "drug"): 1, ("13.3", "drug"): 0}


def test_conflicting_labels_and_overlap(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    labels.write_text("isolate_id,antibiotic,label\na,drug,R\na,drug,S\n")
    with pytest.raises(ValueError, match="Conflicting"):
        read_phenotypes(labels)
    labels.write_text("isolate_id,antibiotic,label\na,drug,R\nb,drug,S\nc,drug,R\nd,drug,S\n")
    directory = tmp_path / "splits/drug"
    directory.mkdir(parents=True)
    (directory / "train.ids").write_text("a\nb\n")
    (directory / "val.ids").write_text("a\nd\n")
    (directory / "test.ids").write_text("c\nd\n")
    with pytest.raises(ValueError, match="Overlapping"):
        read_split(directory.parent, "drug", read_phenotypes(labels))


def test_thresholds_and_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_f1_threshold(y, p)
    assert threshold == 0.8
    row = metric_row(y, p, threshold, "drug", "model", "test", 0, "raw", "real")
    assert row["split"] == "test"
    assert row["f1"] == row["auroc"] == row["auprc"] == row["mcc"] == 1.0
    assert row["threshold"] == threshold



def test_validation_f1_selects_best_threshold():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.55, 0.4, 0.1])
    assert select_f1_threshold(y, p) == 0.4


def test_changed_gff3_is_revalidated(tmp_path: Path):
    gff = tmp_path / "sample.gff3"
    gff.write_text("##gff-version 3\nctg\tBakta\tCDS\t1\t9\t.\t+\t0\tID=gene1\n")
    store = RecordStore({"sample": Isolate("sample", gff)}, tmp_path / "cache", tmp_path / "db")
    assert len(store.get("sample")) == 1
    gff.write_text("not a valid GFF3\n")
    with pytest.raises(ValueError, match="Malformed GFF3"):
        store.get("sample")
