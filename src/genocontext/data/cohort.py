"""Labels, isolate metadata and split partitions.

Split *generation* lives in ``experiments/kp/make_splits.py``; this module only reads its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PARTITION_COLUMNS = ("isolate_id", "group", "outer", "inner_fold")


@dataclass(frozen=True)
class Cohort:
    labels: pd.DataFrame  # index isolate_id; one float column per drug with 1=R, 0=S, NaN=untested
    biosample: pd.Series
    assemblies: pd.Series  # isolate_id -> assembly FASTA path

    @property
    def drugs(self) -> tuple[str, ...]:
        return tuple(self.labels.columns)


def read_labels(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the wide AMR-GNN table ``id,Biosample,<drug>...`` with 0/1 phenotypes."""
    frame = pd.read_csv(path, dtype={"id": str, "Biosample": str}).set_index("id")
    frame.index.name = "isolate_id"
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate isolate IDs in {path}")
    labels = frame.drop(columns="Biosample").astype(float)
    if not labels.isin([0.0, 1.0]).where(labels.notna(), True).all().all():
        raise ValueError(f"Phenotypes must be 0/1 in {path}")
    return labels, frame["Biosample"]


def read_manifest(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, dtype=str)
    if not {"id", "assembly_path"} <= set(frame.columns):
        raise ValueError(f"Manifest needs id,assembly_path columns: {path}")
    return frame.set_index("id")["assembly_path"].map(Path).rename_axis("isolate_id")


def load_cohort(labels_path: str | Path, manifest_path: str | Path) -> Cohort:
    labels, biosample = read_labels(labels_path)
    assemblies = read_manifest(manifest_path)
    missing = labels.index.difference(assemblies.index)
    if len(missing):
        raise ValueError(f"{len(missing)} labelled isolates missing from manifest, e.g. {list(missing[:3])}")
    return Cohort(labels, biosample, assemblies.loc[labels.index])


def annotation_paths(assembly: Path) -> tuple[Path, Path]:
    """Bakta writes ``<stem>.gff3`` and ``<stem>.faa`` next to the assembly FASTA."""
    return assembly.with_suffix(".gff3"), assembly.with_suffix(".faa")


def partition_path(split_root: str | Path, scheme: str, seed: int, drug: str | None = None) -> Path:
    """Tier-1 schemes store one partition per target drug; isolate-level schemes store one per seed."""
    seed_dir = Path(split_root) / scheme / f"seed{seed}"
    per_drug = seed_dir / str(drug) / "partition.tsv"
    return per_drug if drug and per_drug.is_file() else seed_dir / "partition.tsv"


def read_partition(split_root: str | Path, scheme: str, seed: int, drug: str | None = None) -> pd.DataFrame:
    """Read an isolate-level partition (``outer`` in train/test/excluded, ``inner_fold`` 0-4 for train)."""
    path = partition_path(split_root, scheme, seed, drug)
    frame = pd.read_csv(path, sep="\t", dtype={"isolate_id": str, "group": str})
    if tuple(frame.columns) != PARTITION_COLUMNS or frame["isolate_id"].duplicated().any():
        raise ValueError(f"Malformed partition file: {path}")
    if not set(frame["outer"]) <= {"train", "test", "excluded"}:
        raise ValueError(f"Unknown outer partition values in {path}")
    return frame.set_index("isolate_id")
