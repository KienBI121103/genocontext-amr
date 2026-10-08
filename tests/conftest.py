"""Shared synthetic fixtures: tiny Bakta-like annotations, knowledge and references."""

from __future__ import annotations

from pathlib import Path

import pytest

from genocontext.features.lexicon import Knowledge, load_knowledge

ROOT = Path(__file__).resolve().parents[1]


def gff_line(contig: str, start: int, end: int, locus: str, strand: str = "+", gene: str | None = None,
             product: str = "hypothetical protein", xrefs: str = "", pseudo: bool = False) -> str:
    attrs = [f"ID={locus}", f"product={product}"]
    if gene:
        attrs.append(f"gene={gene}")
    if xrefs:
        attrs.append(f"Dbxref={xrefs}")
    if pseudo:
        attrs.append("pseudo=True")
    return "\t".join([contig, "Bakta", "CDS", str(start), str(end), ".", strand, "0", ";".join(attrs)]) + "\n"


def write_annotation(directory: Path, name: str, contigs: dict[str, int], cds: list[str],
                     proteins: dict[str, str]) -> tuple[Path, Path]:
    gff = directory / f"{name}.gff3"
    header = "##gff-version 3\n" + "".join(f"##sequence-region {c} 1 {n}\n" for c, n in contigs.items())
    gff.write_text(header + "".join(cds) + "##FASTA\n>contig_1\nACGT\n")
    faa = directory / f"{name}.faa"
    faa.write_text("".join(f">{k} protein\n{v}\n" for k, v in proteins.items()))
    return gff, faa


@pytest.fixture(scope="session")
def knowledge() -> Knowledge:
    return load_knowledge(ROOT / "knowledge/mechanisms.yaml")
