"""Bakta GFF3 parsing: ordered CDS records with cross-references and contig lengths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True, slots=True)
class Gene:
    contig: str
    locus_tag: str
    start: int
    end: int
    strand: str
    gene: str | None
    product: str | None
    xrefs: tuple[str, ...]
    pseudo: bool

    def xref(self, prefix: str) -> str | None:
        """Return the first cross-reference value with this prefix, e.g. ``xref("UniRef:UniRef90_")``."""
        for value in self.xrefs:
            if value.startswith(prefix):
                return value[len(prefix):]
        return None


@dataclass(frozen=True)
class Annotation:
    genes: tuple[Gene, ...]
    contig_lengths: dict[str, int]


def _attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if text == ".":
        return result
    for item in filter(None, text.split(";")):
        if "=" not in item:
            raise ValueError(f"Invalid GFF3 attribute: {item!r}")
        key, value = item.split("=", 1)
        result[unquote(key)] = unquote(value)
    return result


def parse_gff3(path: str | Path) -> Annotation:
    """Read CDS features, sorted by contig and start; stops at the embedded ``##FASTA`` block."""
    path = Path(path)
    lengths: dict[str, int] = {}
    genes: list[Gene] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if line.startswith("##sequence-region "):
                parts = line.split()
                lengths[parts[1]] = int(parts[3])
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Expected 9 GFF3 columns at {path}:{number}")
            contig, _, kind, start, end, _, strand, _, raw = fields
            if kind != "CDS":
                continue
            attrs = _attributes(raw)
            genes.append(Gene(
                contig=contig, locus_tag=attrs.get("ID") or f"{contig}:{start}-{end}",
                start=int(start), end=int(end), strand=strand,
                gene=attrs.get("gene") or None, product=attrs.get("product") or None,
                xrefs=tuple(filter(None, attrs.get("Dbxref", "").split(","))),
                pseudo=attrs.get("pseudo", "").lower() == "true",
            ))
    if not genes:
        raise ValueError(f"No CDS features in {path}")
    genes.sort(key=lambda g: (g.contig, g.start, g.end, g.locus_tag))
    return Annotation(tuple(genes), lengths)
