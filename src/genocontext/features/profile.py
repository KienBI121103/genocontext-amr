"""Label-free per-isolate gene-state profiles: nodes with state, mechanisms, flags and genomic positions."""

from __future__ import annotations

import gzip
import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from genocontext.config import Config, FeatureConfig
from genocontext.data.faa import read_proteins
from genocontext.data.gff import Annotation, parse_gff3
from genocontext.features.lexicon import Knowledge, amr_names, family_name, load_knowledge, mobile_name
from genocontext.features.targets import TargetCaller, read_references

LOG = logging.getLogger(__name__)
FLAGS = ("multi_copy", "contig_edge", "near_mobile")
PROFILE_VERSION = 1


@dataclass
class Node:
    token: str  # e.g. "amr:blaKPC-3", "target:OmpK36", "res:GyrA_S83I", "mge:ISKpn6", "fam:UniRef90_X"
    group: str  # amr | target | res | mge | fam
    state: str  # present | intact | truncated | is_disrupted | edge_unknown | absent
    mechanisms: list[str]
    flags: list[int]
    positions: list[list[int]]  # [contig_index, cds_index] of every copy
    loci: list[str]
    family: str | None = None  # amr family token (e.g. "amrfam:blaKPC")


@dataclass
class Profile:
    isolate: str
    qc: dict[str, float]
    nodes: list[Node]


def _qc(annotation: Annotation) -> dict[str, float]:
    lengths = np.sort(np.array(list(annotation.contig_lengths.values()) or [0]))[::-1]
    half = np.searchsorted(np.cumsum(lengths), lengths.sum() / 2)
    return {"n_contigs": float(len(annotation.contig_lengths)), "total_bp": float(lengths.sum()),
            "n50": float(lengths[min(half, len(lengths) - 1)]), "n_cds": float(len(annotation.genes))}


def _copy_flags(indices: list[int], annotation: Annotation, mobile_idx: set[int], cfg: FeatureConfig) -> list[int]:
    genes = annotation.genes
    edge = any(genes[i].start <= cfg.contig_edge_bp
               or annotation.contig_lengths.get(genes[i].contig, genes[i].end) - genes[i].end <= cfg.contig_edge_bp
               for i in indices)
    near = any(j in mobile_idx and j != i and genes[j].contig == genes[i].contig
               for i in indices for j in range(i - cfg.near_mobile_cds, i + cfg.near_mobile_cds + 1))
    return [int(len(indices) > 1), int(edge), int(near)]


def extract_profile(isolate: str, gff3: Path, faa: Path, knowledge: Knowledge, caller: TargetCaller,
                    cfg: FeatureConfig) -> Profile:
    annotation = parse_gff3(gff3)
    genes = annotation.genes
    contig_index = {c: i for i, c in enumerate(dict.fromkeys(g.contig for g in genes))}
    cds_index: list[int] = []
    counter: dict[str, int] = defaultdict(int)
    for g in genes:
        cds_index.append(counter[g.contig])
        counter[g.contig] += 1
    mobile = {i: name for i, g in enumerate(genes) if (name := mobile_name(g))}
    members: dict[tuple[str, str], list[int]] = defaultdict(list)
    meta: dict[tuple[str, str], tuple[list[str], str | None]] = {}
    for i, g in enumerate(genes):
        names = amr_names(g)
        fam = family_name(g)
        if names:
            key = (f"amr:{names[0]}", "amr")
            meta[key] = (list(knowledge.amr_mechanisms(names[0], g.product or "")), f"amrfam:{names[1]}")
        elif i in mobile:
            key = (f"mge:{mobile[i]}", "mge")
            meta[key] = (["other"], None)
        elif fam:
            key = (f"fam:{fam}", "fam")
            meta[key] = (["other"], None)
        else:
            continue
        members[key].append(i)

    def position(i: int) -> list[int]:
        return [contig_index[genes[i].contig], cds_index[i]]

    mobile_idx = set(mobile)
    nodes = [Node(token, group, "present", meta[(token, group)][0], _copy_flags(idx, annotation, mobile_idx, cfg),
                  [position(i) for i in idx], [genes[i].locus_tag for i in idx], meta[(token, group)][1])
             for (token, group), idx in members.items()]
    proteins = read_proteins(faa)
    row_of = {g.locus_tag: i for i, g in enumerate(genes)}
    for call in caller.call(annotation, proteins, {genes[i].locus_tag for i in mobile}):
        idx = [row_of[h.gene.locus_tag] for h in call.hits]
        mechanisms = list(knowledge.targets[call.name].mechanisms)
        flags = _copy_flags(idx, annotation, mobile_idx, cfg) if idx else [0, 0, 0]
        where = [position(i) for i in idx]
        nodes.append(Node(f"target:{call.name}", "target", call.state, mechanisms, flags, where,
                          [genes[i].locus_tag for i in idx]))
        nodes += [Node(f"res:{v}", "res", "present", mechanisms, [0, 0, 0], where, [genes[i].locus_tag for i in idx])
                  for v in call.variants]
    return Profile(isolate, _qc(annotation), nodes)


def save_profile(profile: Profile, path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump({"version": PROFILE_VERSION, **asdict(profile)}, handle, separators=(",", ":"))
    temporary.replace(path)


def load_profile(path: Path) -> Profile:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.pop("version") != PROFILE_VERSION:
        raise ValueError(f"Stale profile version: {path}")
    return Profile(raw["isolate"], raw["qc"], [Node(**n) for n in raw["nodes"]])


_WORKER: tuple[Knowledge, TargetCaller, FeatureConfig] | None = None


def _init_worker(knowledge_path: Path, references_path: Path, cfg: FeatureConfig) -> None:
    global _WORKER
    knowledge = load_knowledge(knowledge_path)
    _WORKER = (knowledge, TargetCaller(read_references(references_path), knowledge.targets, cfg), cfg)


def _build_one(item: tuple[str, str, str, str]) -> str:
    isolate, gff3, faa, out = item
    assert _WORKER is not None
    save_profile(extract_profile(isolate, Path(gff3), Path(faa), *_WORKER), Path(out))
    return isolate


def build_profiles(config: Config, inputs: dict[str, tuple[Path, Path]]) -> Iterator[str]:
    """Build missing profiles in parallel; refuses to mix profiles made from different knowledge/references."""
    root = config.paths.profiles
    root.mkdir(parents=True, exist_ok=True)
    stamp = {"version": PROFILE_VERSION, "knowledge_sha256": load_knowledge(config.paths.knowledge).sha256,
             "references": read_references(config.paths.references), "features": asdict(config.features)}
    stamp_path = root / "profiles_stamp.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise ValueError(f"Profiles in {root} were built with different knowledge/settings; use a new directory")
    stamp_path.write_text(json.dumps(stamp, sort_keys=True))
    todo = [(i, str(g), str(f), str(root / f"{i}.json.gz")) for i, (g, f) in sorted(inputs.items())
            if not (root / f"{i}.json.gz").exists()]
    LOG.info("Building %d/%d profiles with %d workers", len(todo), len(inputs), config.training.workers)
    with ProcessPoolExecutor(config.training.workers, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(config.paths.knowledge, config.paths.references, config.features)) as pool:
        yield from pool.map(_build_one, todo, chunksize=4)
