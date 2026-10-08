"""Protein FASTA access by locus tag (Bakta FAA headers start with the GFF3 ``ID``)."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path


def read_proteins(path: str | Path, wanted: Collection[str] | None = None) -> dict[str, str]:
    """Return ``{locus_tag: sequence}``; restrict to ``wanted`` locus tags when given."""
    proteins: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    proteins[name] = "".join(chunks)
                tag = line[1:].split(maxsplit=1)[0]
                name = tag if wanted is None or tag in wanted else None
                chunks = []
            elif name is not None:
                chunks.append(line.strip())
    if name is not None:
        proteins[name] = "".join(chunks)
    return proteins
