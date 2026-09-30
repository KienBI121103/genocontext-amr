from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log1p
from typing import Iterable

import numpy as np
from scipy import sparse

from src.parsing.gff import GeneRecord

FIELDS = ("gene", "product", "cog", "ec")
MISSING = "<MISSING>"
UNK = "<UNK>"
AMR_TERMS = re.compile(
    r"\b(?:antibiotic|antimicrobial|resistan\w*|beta[- ]?lactamase|carbapenemase|"
    r"cephalosporinase|aminoglycoside[- ]?modifying|quinolone resistance|"
    r"multidrug efflux|methicillin resistance)\b|^bla[A-Za-z0-9_-]+$",
    re.IGNORECASE,
)


def ordered_genes(records: Iterable[GeneRecord]) -> list[GeneRecord]:
    return sorted(records, key=lambda r: (r.contig_id, r.start, r.end, r.feature_id))


def _category(record: GeneRecord, field: str, mask_amr_terms: bool) -> str:
    if field == "gene":
        value = record.gene
    elif field == "product":
        value = record.product
    elif field == "cog":
        value = next((x for x in record.db_xrefs if x.startswith("COG:COG")), None)
    else:
        value = record.ec_numbers[0] if record.ec_numbers else None
    if not value:
        return MISSING
    value = value.strip().lower()
    if mask_amr_terms and field in {"gene", "product"}:
        value = AMR_TERMS.sub("<AMR>", value)
    return value or MISSING


def structural_features(records: list[GeneRecord]) -> np.ndarray:
    """Fixed biological scaling avoids learning numerical statistics from held-out data."""
    records = ordered_genes(records)
    result = np.zeros((len(records), 9), dtype=np.float32)
    contig_max: dict[str, int] = {}
    for record in records:
        contig_max[record.contig_id] = max(contig_max.get(record.contig_id, 0), record.end)
    for index, gene in enumerate(records):
        prev = records[index - 1] if index and records[index - 1].contig_id == gene.contig_id else None
        nxt = records[index + 1] if index + 1 < len(records) and records[index + 1].contig_id == gene.contig_id else None
        previous_gap = gene.start - prev.end - 1 if prev else 0
        next_gap = nxt.start - gene.end - 1 if nxt else 0
        denominator = gene.contig_length or contig_max[gene.contig_id]
        result[index] = (
            log1p(gene.end - gene.start + 1) / 10,
            1.0 if gene.strand == "+" else -1.0 if gene.strand == "-" else 0.0,
            (gene.start + gene.end) / (2 * denominator),
            log1p(max(previous_gap, 0)) / 10,
            log1p(max(next_gap, 0)) / 10,
            float(previous_gap < 0) if prev else 0.0,
            float(next_gap < 0) if nxt else 0.0,
            float(prev is not None and gene.strand == prev.strand),
            float(nxt is not None and gene.strand == nxt.strand),
        )
    return result


@dataclass
class GeneFeatureEncoder:
    mask_amr_terms: bool = True
    min_category_frequency: int = 5
    vocabularies: dict[str, dict[str, int]] | None = None

    def fit(self, training_records: Iterable[list[GeneRecord]]) -> "GeneFeatureEncoder":
        counts = {field: Counter() for field in FIELDS}
        for records in training_records:
            for record in records:
                for field in FIELDS:
                    counts[field][_category(record, field, self.mask_amr_terms)] += 1
        self.vocabularies = {}
        for field in FIELDS:
            vocabulary = {UNK: 0, MISSING: 1}
            for token, count in sorted(counts[field].items()):
                if token not in vocabulary and count >= self.min_category_frequency:
                    vocabulary[token] = len(vocabulary)
            self.vocabularies[field] = vocabulary
        return self

    def transform(self, records: list[GeneRecord]) -> tuple[np.ndarray, np.ndarray]:
        if self.vocabularies is None:
            raise RuntimeError("Fit the gene vocabulary on training isolates first")
        records = ordered_genes(records)
        categories = np.zeros((len(records), len(FIELDS)), dtype=np.int64)
        for index, record in enumerate(records):
            for field_index, field in enumerate(FIELDS):
                categories[index, field_index] = self.vocabularies[field].get(
                    _category(record, field, self.mask_amr_terms), 0
                )
        return structural_features(records), categories

    @property
    def category_sizes(self) -> tuple[int, ...]:
        if self.vocabularies is None:
            raise RuntimeError("Encoder is not fitted")
        return tuple(len(self.vocabularies[field]) for field in FIELDS)

    def genome_vector(self, records: list[GeneRecord]) -> sparse.csr_matrix:
        numbers, categories = self.transform(records)
        sizes = self.category_sizes
        offsets = np.cumsum((0,) + sizes[:-1])
        cat_columns = (categories + offsets).ravel()
        cat_values = np.full(cat_columns.shape, 1 / len(records), dtype=np.float32)
        counts = sparse.csr_matrix(
            (cat_values, (np.zeros(len(cat_columns), dtype=int), cat_columns)),
            shape=(1, sum(sizes)), dtype=np.float32,
        )
        summary = sparse.csr_matrix(
            np.concatenate((numbers.mean(axis=0), [log1p(len(records)) / 10])).reshape(1, -1)
        )
        return sparse.hstack((summary, counts), format="csr")
