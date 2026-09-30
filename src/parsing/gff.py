from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class GeneRecord:
    sample_id: str
    contig_id: str
    feature_id: str
    start: int
    end: int
    strand: str
    feature_type: str
    gene: str | None
    product: str | None
    db_xrefs: tuple[str, ...] = ()
    ec_numbers: tuple[str, ...] = ()
    contig_length: int | None = None


def _attributes(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if text == ".":
        return result
    for item in text.split(";"):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid GFF3 attribute: {item!r}")
        key, value = item.split("=", 1)
        result[unquote(key)] = unquote(value)
    return result


def parse_gff3(
    path: str | Path,
    sample_id: str,
    feature_types: tuple[str, ...] = ("CDS",),
) -> list[GeneRecord]:
    path = Path(path)
    regions: dict[str, int] = {}
    records: list[GeneRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##FASTA"):
                break
            if line.startswith("##sequence-region "):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid sequence-region at {path}:{line_number}")
                regions[parts[1]] = int(parts[3])
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Expected 9 GFF3 columns at {path}:{line_number}")
            contig, _, kind, raw_start, raw_end, _, strand, _, raw_attrs = fields
            if kind not in feature_types:
                continue
            try:
                start, end = int(raw_start), int(raw_end)
                attrs = _attributes(raw_attrs)
                if start < 1 or end < start or strand not in {"+", "-", "."}:
                    raise ValueError("invalid coordinates or strand")
            except ValueError as exc:
                raise ValueError(f"Invalid feature at {path}:{line_number}: {exc}") from exc
            xrefs = tuple(filter(None, attrs.get("Dbxref", "").split(",")))
            ec = tuple(filter(None, attrs.get("EC_number", "").split(",")))
            ec += tuple(x[3:] for x in xrefs if x.startswith("EC:"))
            records.append(GeneRecord(
                sample_id=sample_id, contig_id=contig,
                feature_id=attrs.get("ID") or f"{contig}:{start}-{end}:{line_number}",
                start=start, end=end, strand=strand, feature_type=kind,
                gene=attrs.get("gene") or None,
                product=attrs.get("product") or attrs.get("Name") or None,
                db_xrefs=xrefs, ec_numbers=tuple(dict.fromkeys(ec)),
                contig_length=regions.get(contig),
            ))
    if not records:
        raise ValueError(f"No selected features in GFF3: {path}")
    return sorted(records, key=lambda r: (r.contig_id, r.start, r.end, r.feature_id))
