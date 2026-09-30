from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.annotation.bakta import ensure_bakta_annotation
from src.parsing.gff import parse_gff3


GFF = """##gff-version 3
##sequence-region ctg1 1 1000
ctg1\tBakta\tCDS\t10\t100\t.\t+\t0\tID=id1;gene=abc;product=transporter;Dbxref=COG:COG0001,EC:1.1.1.1
ctg1\tBakta\tCDS\t120\t200\t.\t-\t0\tID=id2;product=hypothetical protein
##sequence-region ctg2 1 500
ctg2\tBakta\tCDS\t20\t80\t.\t+\t0\tID=id3;gene=abc
##FASTA
>ctg2
ACGT
"""


def test_parse_gff3_multiple_contigs_and_missing_fields(tmp_path: Path):
    path = tmp_path / "sample.gff3"
    path.write_text(GFF)
    records = parse_gff3(path, "sample")
    assert len(records) == 3
    assert records[0].gene == records[2].gene == "abc"
    assert records[1].gene is None
    assert records[2].product is None
    assert records[0].ec_numbers == ("1.1.1.1",)
    assert records[2].contig_length == 500


def test_parse_malformed_selected_row(tmp_path: Path):
    path = tmp_path / "bad.gff3"
    path.write_text("##gff-version 3\nctg\tBakta\tCDS\t0\t1\t.\t+\t0\tID=x\n")
    with pytest.raises(ValueError, match="Invalid feature"):
        parse_gff3(path, "bad")


def test_bakta_skips_existing_without_subprocess(tmp_path: Path):
    genome = tmp_path / "sample.fna"
    genome.write_text(">c\nACGT\n")
    gff = tmp_path / "ann/sample.gff3"
    gff.parent.mkdir()
    gff.write_text(GFF)
    with patch("subprocess.run") as run:
        assert ensure_bakta_annotation(genome, gff.parent, tmp_path / "nonexistent") == gff
        run.assert_not_called()


def test_bakta_existing_invalid_stops_without_subprocess(tmp_path: Path):
    genome = tmp_path / "sample.fna"
    genome.write_text(">c\nACGT\n")
    gff = tmp_path / "ann/sample.gff3"
    gff.parent.mkdir()
    gff.write_text("")
    with patch("subprocess.run") as run, pytest.raises(ValueError, match="Missing or empty"):
        ensure_bakta_annotation(genome, gff.parent, tmp_path)
    run.assert_not_called()


def test_bakta_runs_only_when_missing(tmp_path: Path):
    genome = tmp_path / "sample.fna"
    genome.write_text(">c\nACGT\n")
    database = tmp_path / "db"
    database.mkdir()
    output = tmp_path / "ann"

    def fake_run(command, **kwargs):
        output.mkdir(exist_ok=True)
        (output / "sample.gff3").write_text(GFF)
        return Mock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=fake_run) as run:
        assert ensure_bakta_annotation(genome, output, database) == output / "sample.gff3"
        assert run.call_count == 1
