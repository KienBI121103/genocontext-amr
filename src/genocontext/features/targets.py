"""Locate curated chromosomal targets by protein similarity and call their state and residues.

Targets are found by a 5-mer prefilter over the whole proteome followed by local alignment to pinned
references (``knowledge/references/kp_targets.faa``); Bakta product names are not trusted for identity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices

from genocontext.config import FeatureConfig
from genocontext.data.gff import Annotation, Gene
from genocontext.features.lexicon import Target

K = 5
STATES = ("intact", "truncated", "is_disrupted", "edge_unknown", "absent")


@dataclass(frozen=True)
class Hit:
    target: str
    gene: Gene
    identity: float
    ref_blocks: tuple[tuple[int, int], ...]
    query_blocks: tuple[tuple[int, int], ...]
    query: str


@dataclass
class TargetCall:
    name: str
    state: str
    hits: list[Hit] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)  # e.g. "GyrA_S83I", "OmpK36_ins183_LSP"


def read_references(path: str | Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for record in Path(path).read_text(encoding="utf-8").split(">")[1:]:
        header, *lines = record.splitlines()
        references[header.split()[0]] = "".join(lines)
    return references


class TargetCaller:
    def __init__(self, references: dict[str, str], targets: dict[str, Target], config: FeatureConfig) -> None:
        missing = set(targets) - set(references)
        if missing:
            raise ValueError(f"No reference sequence for targets: {sorted(missing)}")
        self.references = {name: references[name] for name in targets}
        self.targets = targets
        self.config = config
        self.index: dict[str, set[str]] = defaultdict(set)
        for name, sequence in self.references.items():
            for i in range(len(sequence) - K + 1):
                self.index[sequence[i:i + K]].add(name)
        self.aligner = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5)
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    def _candidates(self, protein: str) -> list[str]:
        shared: dict[str, int] = defaultdict(int)
        for i in range(len(protein) - K + 1):
            for name in self.index.get(protein[i:i + K], ()):
                shared[name] += 1
        return [n for n, c in shared.items() if c >= (4 if len(self.references[n]) < 100 else 10)]

    def _align(self, gene: Gene, protein: str, name: str) -> Hit | None:
        reference = self.references[name]
        alignment = self.aligner.align(protein, reference)[0]
        query_blocks, ref_blocks = alignment.aligned
        columns = sum(int(e - s) for s, e in ref_blocks)
        same = sum(protein[qs + i] == reference[rs + i]
                   for (qs, qe), (rs, _) in zip(query_blocks, ref_blocks, strict=True) for i in range(int(qe - qs)))
        if columns < min(25, 0.4 * len(reference)) or same / columns < self.config.min_identity:
            return None
        return Hit(name, gene, same / columns, tuple((int(s), int(e)) for s, e in ref_blocks),
                   tuple((int(s), int(e)) for s, e in query_blocks), protein)

    def locate(self, genes: tuple[Gene, ...], proteins: dict[str, str]) -> dict[str, list[Hit]]:
        """Assign each protein to at most one target (best identity)."""
        found: dict[str, list[Hit]] = defaultdict(list)
        for gene in genes:
            protein = proteins.get(gene.locus_tag, "")
            if len(protein) < 20:
                continue
            hits = [h for name in self._candidates(protein) if (h := self._align(gene, protein, name))]
            if hits:
                best = max(hits, key=lambda h: (h.identity, sum(e - s for s, e in h.ref_blocks)))
                found[best.target].append(best)
        return found

    def call(self, annotation: Annotation, proteins: dict[str, str], mobile: set[str]) -> list[TargetCall]:
        located = self.locate(annotation.genes, proteins)
        return [self._state(name, located.get(name, []), annotation, mobile) for name in self.targets]

    def _state(self, name: str, hits: list[Hit], annotation: Annotation, mobile: set[str]) -> TargetCall:
        if not hits:
            return TargetCall(name, "absent")
        length = len(self.references[name])
        complete = [h for h in hits if not h.gene.pseudo and h.ref_blocks[0][0] <= 10
                    and sum(e - s for s, e in h.ref_blocks) >= self.config.min_coverage * length]
        call = TargetCall(name, "intact", hits)
        if not complete:
            call.state = self._broken_state(hits, annotation, mobile)
        best = max(complete or hits, key=lambda h: sum(e - s for s, e in h.ref_blocks))
        call.variants = self._variants(name, best)
        return call

    def _broken_state(self, hits: list[Hit], annotation: Annotation, mobile: set[str]) -> str:
        bp = self.config.promoter_bp
        for hit in hits:
            g = hit.gene
            if any(o.locus_tag in mobile and o.contig == g.contig and o.start <= g.end + bp and o.end >= g.start - bp
                   for o in annotation.genes):
                return "is_disrupted"
        edge = self.config.contig_edge_bp
        for hit in hits:
            g = hit.gene
            contig_length = annotation.contig_lengths.get(g.contig, g.end)
            if g.start <= edge or contig_length - g.end <= edge:
                return "edge_unknown"
        return "truncated"

    def _variants(self, name: str, hit: Hit) -> list[str]:
        target = self.targets[name]
        position = _reference_to_query(hit)
        variants = []
        for ref_pos, wild in sorted(target.residues.items()):
            index = position.get(ref_pos - 1)
            if index is not None and hit.query[index] != wild:
                variants.append(f"{name}_{wild}{ref_pos}{hit.query[index]}")
        if target.indels:
            variants += [f"{name}_{event}" for event in _indels(hit)]
        return variants


def _reference_to_query(hit: Hit) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for (qs, qe), (rs, _) in zip(hit.query_blocks, hit.ref_blocks, strict=True):
        for i in range(qe - qs):
            mapping[rs + i] = qs + i
    return mapping


def _indels(hit: Hit, max_length: int = 10) -> list[str]:
    """Insertions/deletions between aligned blocks, in 1-based reference coordinates."""
    events = []
    blocks = list(zip(hit.query_blocks, hit.ref_blocks, strict=True))
    for ((_, q_end), (_, r_end)), ((q_next, _), (r_next, _)) in zip(blocks, blocks[1:], strict=False):
        inserted, deleted = q_next - q_end, r_next - r_end
        if 0 < inserted <= max_length:
            events.append(f"ins{r_end}_{hit.query[q_end:q_next]}")
        if 0 < deleted <= max_length:
            events.append(f"del{r_end + 1}" + (f"-{r_next}" if deleted > 1 else ""))
    return events
