from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genocontext.config import load_config
from genocontext.data.cohort import read_labels, read_partition
from genocontext.data.faa import read_proteins
from genocontext.data.gff import parse_gff3
from genocontext.evaluation.metrics import metric_row, select_threshold
from tests.conftest import gff_line, write_annotation


def test_parse_gff3_keeps_xrefs_pseudo_and_contig_lengths(tmp_path: Path) -> None:
    gff, faa = write_annotation(tmp_path, "a", {"contig_2": 900, "contig_1": 500}, [
        gff_line("contig_2", 10, 300, "X_2", xrefs="UniRef:UniRef90_A,IS:ISKpn6"),
        gff_line("contig_1", 50, 200, "X_1", gene="ompC", pseudo=True),
    ], {"X_1": "MKV", "X_2": "MAA"})
    ann = parse_gff3(gff)
    assert [g.locus_tag for g in ann.genes] == ["X_1", "X_2"]  # sorted by contig, start
    assert ann.genes[0].pseudo and not ann.genes[1].pseudo
    assert ann.genes[1].xref("IS:") == "ISKpn6"
    assert ann.contig_lengths == {"contig_2": 900, "contig_1": 500}
    assert read_proteins(faa, {"X_2"}) == {"X_2": "MAA"}


def test_read_labels_rejects_non_binary(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("id,Biosample,drugA\ni1,B1,1\ni2,B2,2\n")
    with pytest.raises(ValueError):
        read_labels(path)
    path.write_text("id,Biosample,drugA,drugB\ni1,B1,1,\ni2,B2,0,1\n")
    labels, biosample = read_labels(path)
    assert np.isnan(labels.loc["i1", "drugB"]) and biosample["i2"] == "B2"


def test_partition_reader_prefers_per_drug_files(tmp_path: Path) -> None:
    seed = tmp_path / "amrgnn_fixed" / "seed0"
    (seed / "cipro").mkdir(parents=True)
    pd.DataFrame({"isolate_id": ["a", "b"], "group": ["g", "h"], "outer": ["train", "test"], "inner_fold": [0, -1]}
                 ).to_csv(seed / "cipro" / "partition.tsv", sep="\t", index=False)
    assert read_partition(tmp_path, "amrgnn_fixed", 0, "cipro").loc["b", "outer"] == "test"


def test_config_rejects_unknown_keys_and_resolves_paths(tmp_path: Path) -> None:
    paths = "\n".join(f"  {k}: data/{k}" for k in ("labels", "manifest", "splits", "profiles", "artifacts",
                                                   "knowledge", "references"))
    (tmp_path / "c.yaml").write_text(f"paths:\n{paths}\nmodel:\n  hidden: 8\n")
    assert load_config(tmp_path / "c.yaml").paths.labels == tmp_path / "data/labels"
    (tmp_path / "bad.yaml").write_text(f"paths:\n{paths}\nmodel:\n  hiden: 8\n")
    with pytest.raises(ValueError, match="hiden"):
        load_config(tmp_path / "bad.yaml")


def test_threshold_and_error_rates() -> None:
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.1])
    t = select_threshold(y, p)
    assert t == pytest.approx(0.7)  # balanced accuracy 0.875 beats 0.6 (0.75) and 0.3 (0.75)
    row = metric_row(y, p, t)
    assert row["vme"] == pytest.approx(0.25) and row["me"] == pytest.approx(0.0)
    assert row["auroc"] == pytest.approx(7 / 8)
