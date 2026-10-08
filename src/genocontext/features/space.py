"""Training-only feature space over a ProfileStore: vocabulary, prevalence filters, collinear blocks,
graph encodings (neural models) and sparse matrices (tree / kNN baselines)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from genocontext.config import FeatureConfig
from genocontext.features.profile import FLAGS
from genocontext.features.store import GROUPS, NODE_STATES, ProfileStore
from genocontext.features.targets import STATES

UNKNOWN_AMR = "unk:amr"
Graph = dict[str, NDArray[Any]]
Index = NDArray[np.int64]


@dataclass
class FeatureSpace:
    lookup: NDArray[np.int64]  # store token id -> block id; -1 = dropped, 0 = unknown AMR determinant
    blocks: list[str]  # representative token per block
    members: list[list[str]]
    block_group: list[str]
    mechanisms: tuple[str, ...]
    prevalence: dict[str, float]

    @classmethod
    def fit(cls, store: ProfileStore, train: Index, config: FeatureConfig,
            drop_groups: tuple[str, ...] = ()) -> FeatureSpace:
        n_tok, n = len(store.tokens), len(train)
        presence = _presence(store, train)  # (n, n_tokens) sparse 0/1, tokens and amr families
        counts = np.asarray(presence.sum(0)).ravel()
        group = np.array(GROUPS)[store.token_group]
        frequent = counts >= config.min_prevalence
        keep = np.flatnonzero(frequent & np.isin(group, ["amr", "res", "mge"]) | (group == "target"))
        fam = np.flatnonzero((group == "fam") & frequent & (counts <= n - config.min_prevalence))
        p = counts[fam] / n
        fam = fam[np.lexsort((fam, -p * (1 - p)))][:config.max_families]
        keep = np.concatenate([keep, fam])
        keep = keep[~np.isin(group[keep], list(drop_groups))]
        priority = np.array([GROUPS.index(g) for g in group[keep]])
        keep = keep[np.lexsort((np.array(store.tokens, dtype=object)[keep].astype(str), -counts[keep], priority))]
        members = _collinear_blocks(keep, presence, group, config.block_correlation)
        lookup = np.full(n_tok, -1, dtype=np.int64)
        lookup[group == "amr"] = 0 if "amr" not in drop_groups else -1
        for b, block in enumerate(members, start=1):
            lookup[block] = b
        return cls(lookup, [UNKNOWN_AMR] + [store.tokens[m[0]] for m in members],
                   [[UNKNOWN_AMR]] + [[store.tokens[t] for t in m] for m in members],
                   ["amr"] + [str(group[m[0]]) for m in members], store.mechanisms,
                   {store.tokens[t]: float(counts[t] / n) for t in keep})

    @property
    def n_tokens(self) -> int:
        return len(self.blocks)

    def _node_blocks(self, store: ProfileStore, row: int) -> tuple[slice, NDArray[np.int64]]:
        nodes = store.nodes(row)
        block = self.lookup[store.node_token[nodes]]
        family = store.node_family[nodes]
        family_block = np.where(family >= 0, self.lookup[np.maximum(family, 0)], -1)
        return nodes, np.where((block <= 0) & (family_block > 0), family_block, block)

    def graph(self, store: ProfileStore, row: int, k: int, shuffle_seed: int | None = None) -> Graph:
        """Nodes = retained blocks present in the isolate; edges = co-localisation within ±k CDS."""
        nodes, block = self._node_blocks(store, row)
        kept = np.flatnonzero(block >= 0)
        order, local = np.unique(block[kept], return_inverse=True)
        state = np.zeros(len(order), dtype=np.int64)
        np.maximum.at(state, local, store.node_state[nodes][kept].astype(np.int64))
        flags = np.zeros((len(order), len(FLAGS)), dtype=np.float32)
        np.maximum.at(flags, local, store.node_flags[nodes][kept].astype(np.float32))
        bits = np.zeros(len(order), dtype=np.int64)
        np.bitwise_or.at(bits, local, store.node_mechs[nodes][kept].astype(np.int64))
        other = 1 << self.mechanisms.index("other")
        bits = np.where(bits & ~other, bits & ~other, other)
        pair_node, pair_mech = np.nonzero((bits[:, None] >> np.arange(len(self.mechanisms))) & 1)
        start, end = store.pos_offsets[nodes.start + kept], store.pos_offsets[nodes.start + kept + 1]
        owner = np.repeat(local, end - start)
        where = store.positions[np.concatenate([np.arange(s, e) for s, e in zip(start, end, strict=True)])
                                if len(kept) else np.zeros(0, dtype=np.int64)]
        return {"token": order.astype(np.int64), "state": state, "flags": flags,
                "pair_node": pair_node.astype(np.int64), "pair_mech": pair_mech.astype(np.int64),
                "edge_index": _colocalisation(where, owner, k, shuffle_seed, store.isolates[row])}

    def column_names(self, view: str) -> list[str]:
        if view not in ("genestate", "presence"):
            raise ValueError(f"Unknown view: {view}")
        allowed = ("amr", "fam") if view == "presence" else ("amr", "res", "mge", "fam")
        names = [b for b, g in zip(self.blocks, self.block_group, strict=True) if g in allowed]
        if view == "genestate":
            names += [f"{b}={s}" for b, g in zip(self.blocks, self.block_group, strict=True) if g == "target"
                      for s in STATES]
        return names

    def matrix(self, store: ProfileStore, rows: Index, view: str = "genestate") -> sparse.csr_matrix:
        columns = {name: j for j, name in enumerate(self.column_names(view))}
        r_idx: list[int] = []
        c_idx: list[int] = []
        for r, row in enumerate(rows):
            nodes, block = self._node_blocks(store, int(row))
            names = {self.blocks[b] if self.block_group[b] != "target" else f"{self.blocks[b]}={NODE_STATES[s]}"
                     for b, s in zip(block[block >= 0], store.node_state[nodes][block >= 0], strict=True)}
            hits = [columns[name] for name in names if name in columns]
            r_idx += [r] * len(hits)
            c_idx += hits
        return sparse.csr_matrix((np.ones(len(r_idx), dtype=np.float32), (r_idx, c_idx)),
                                 shape=(len(rows), len(columns)))


def _presence(store: ProfileStore, rows: Index) -> sparse.csr_matrix:
    r_idx, c_idx = [], []
    for r, row in enumerate(rows):
        nodes = store.nodes(int(row))
        tokens = np.unique(np.concatenate([store.node_token[nodes], store.node_family[nodes]]))
        tokens = tokens[tokens >= 0]
        r_idx.append(np.full(len(tokens), r))
        c_idx.append(tokens)
    data = np.concatenate(r_idx), np.concatenate(c_idx)
    return sparse.csr_matrix((np.ones(len(data[0]), dtype=np.float32), data), shape=(len(rows), len(store.tokens)))


def _collinear_blocks(keep: NDArray[np.int64], presence: sparse.csr_matrix, group: NDArray[Any],
                      threshold: float) -> list[list[int]]:
    """Greedy blocks of tokens with r >= threshold in training presence; ``keep`` is priority-ordered so each
    block's first member is its representative. Blocks are encoded as "any member present", so only positively
    correlated tokens may merge (mutually exclusive alleles would become a constant). Targets are never merged."""
    dense = presence[:, keep].toarray()
    centred = dense - dense.mean(axis=0)
    norms = np.linalg.norm(centred, axis=0)
    norms[norms == 0] = 1.0
    corr = (centred / norms).T @ (centred / norms)
    is_target = group[keep] == "target"
    assigned = np.zeros(len(keep), dtype=bool)
    blocks = []
    for j in range(len(keep)):
        if assigned[j]:
            continue
        similar = [] if is_target[j] else [int(i) for i in np.flatnonzero(corr[j] >= threshold)
                                           if not assigned[i] and not is_target[i]]
        mates = sorted(set(similar) | {j}, key=lambda i: i != j)
        assigned[mates] = True
        blocks.append([int(keep[i]) for i in mates])
    return blocks


def _colocalisation(where: NDArray[np.int32], owner: NDArray[np.int64], k: int, shuffle_seed: int | None,
                    isolate: str) -> NDArray[np.int64]:
    """Undirected edges between nodes whose CDS lie within ±k positions on the same contig."""
    contig, cds = where[:, 0].astype(np.int64), where[:, 1].astype(np.int64)
    if shuffle_seed is not None and len(cds):  # gene-order control: permute CDS order within each contig
        digest = hashlib.sha256(f"{shuffle_seed}:{isolate}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        for c in np.unique(contig):
            on = contig == c
            values = np.unique(cds[on])
            shuffled = dict(zip(values.tolist(), rng.permutation(values).tolist(), strict=True))
            cds[on] = [shuffled[x] for x in cds[on].tolist()]
    order = np.lexsort((cds, contig))
    contig, cds, owner = contig[order], cds[order], owner[order]
    edges = set()
    for a in range(len(order)):
        b = a + 1
        while b < len(order) and contig[b] == contig[a] and cds[b] - cds[a] <= k:
            if owner[a] != owner[b]:
                edges.add((min(owner[a], owner[b]), max(owner[a], owner[b])))
            b += 1
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    one_way = np.array(sorted(edges), dtype=np.int64).T
    return np.concatenate([one_way, one_way[::-1]], axis=1)
