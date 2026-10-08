"""Compact columnar store of all isolate profiles (built once from the per-isolate JSON files)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from genocontext.features.profile import FLAGS, load_profile
from genocontext.features.targets import STATES

GROUPS = ("amr", "target", "res", "mge", "fam")
NODE_STATES = ("present", *STATES)


@dataclass
class ProfileStore:
    isolates: list[str]
    tokens: list[str]
    token_group: NDArray[np.int8]
    mechanisms: tuple[str, ...]
    node_offsets: NDArray[np.int64]  # (n_isolates + 1,) node range per isolate
    node_token: NDArray[np.int32]
    node_family: NDArray[np.int32]  # amr family token id or -1
    node_state: NDArray[np.int8]
    node_flags: NDArray[np.uint8]  # (n_nodes, len(FLAGS))
    node_mechs: NDArray[np.int32]  # bitmask over ``mechanisms``
    pos_offsets: NDArray[np.int64]  # (n_nodes + 1,) range into ``positions``
    positions: NDArray[np.int32]  # (n_positions, 2): contig index, CDS index
    qc: pd.DataFrame

    def nodes(self, row: int) -> slice:
        return slice(int(self.node_offsets[row]), int(self.node_offsets[row + 1]))

    @classmethod
    def build(cls, profile_dir: Path, isolates: list[str], mechanisms: tuple[str, ...], threads: int = 32
              ) -> ProfileStore:
        with ThreadPoolExecutor(threads) as pool:
            profiles = list(pool.map(lambda i: load_profile(profile_dir / f"{i}.json.gz"), isolates))
        vocab: dict[str, int] = {}
        groups: list[int] = []
        mech_bit = {m: 1 << i for i, m in enumerate(mechanisms)}

        def token_id(token: str, group: str) -> int:
            if token not in vocab:
                vocab[token] = len(vocab)
                groups.append(GROUPS.index(group))
            return vocab[token]

        cols: dict[str, list[Any]] = {k: [] for k in ("token", "family", "state", "flags", "mechs", "npos")}
        positions: list[list[int]] = []
        offsets = [0]
        for profile in profiles:
            for node in profile.nodes:
                cols["token"].append(token_id(node.token, node.group))
                cols["family"].append(token_id(node.family, "amr") if node.family else -1)
                cols["state"].append(NODE_STATES.index(node.state))
                cols["flags"].append(node.flags)
                cols["mechs"].append(sum(mech_bit[m] for m in set(node.mechanisms)))
                cols["npos"].append(len(node.positions))
                positions.extend(node.positions)
            offsets.append(len(cols["token"]))
        return cls(
            isolates=list(isolates), tokens=list(vocab), token_group=np.array(groups, dtype=np.int8),
            mechanisms=mechanisms, node_offsets=np.array(offsets, dtype=np.int64),
            node_token=np.array(cols["token"], dtype=np.int32), node_family=np.array(cols["family"], dtype=np.int32),
            node_state=np.array(cols["state"], dtype=np.int8),
            node_flags=np.array(cols["flags"], dtype=np.uint8).reshape(-1, len(FLAGS)),
            node_mechs=np.array(cols["mechs"], dtype=np.int32),
            pos_offsets=np.concatenate([[0], np.cumsum(cols["npos"])]).astype(np.int64),
            positions=np.array(positions, dtype=np.int32).reshape(-1, 2),
            qc=pd.DataFrame([p.qc for p in profiles], index=pd.Index(isolates, name="isolate_id")),
        )

    def save(self, path: Path) -> None:
        arrays = {k: getattr(self, k) for k in ("token_group", "node_offsets", "node_token", "node_family",
                                                 "node_state", "node_flags", "node_mechs", "pos_offsets", "positions")}
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps({"isolates": self.isolates, "tokens": self.tokens,
                                                         "mechanisms": list(self.mechanisms)}))
        self.qc.to_csv(path.with_suffix(".qc.tsv"), sep="\t")

    @classmethod
    def load(cls, path: Path) -> ProfileStore:
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        qc = pd.read_csv(path.with_suffix(".qc.tsv"), sep="\t", dtype={"isolate_id": str}).set_index("isolate_id")
        return cls(meta["isolates"], meta["tokens"], mechanisms=tuple(meta["mechanisms"]), qc=qc, **data)
