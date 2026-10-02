from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from src.annotation.bakta import ensure_bakta_annotation, validate_gff3
from src.parsing.gff import GeneRecord, parse_gff3

LOG = logging.getLogger(__name__)


def parsed_cache_dir(config: dict) -> Path:
    return Path(config["paths"].get("parsed_cache", Path(config["paths"]["artifacts"]) / "features/parsed_gff"))


@dataclass(frozen=True)
class Isolate:
    isolate_id: str
    gff3_path: Path
    genome_path: Path | None = None


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def read_manifest(path: str | Path) -> dict[str, Isolate]:
    """Accept generic isolate_id/gff3_path or the Kp AMR-GNN manifest."""
    path = Path(path)
    result: dict[str, Isolate] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not ({"isolate_id", "gff3_path"} <= set(reader.fieldnames) or
                                         {"id", "assembly_path"} <= set(reader.fieldnames)):
            raise ValueError(f"Unsupported manifest columns in {path}")
        for row in reader:
            sample = (row.get("isolate_id") or row.get("id") or "").strip()
            if not sample or sample in result:
                raise ValueError(f"Empty or duplicate isolate ID in {path}: {sample!r}")
            assembly = row.get("genome_path") or row.get("assembly_path")
            gff = row.get("gff3_path")
            if not gff:
                if not assembly:
                    raise ValueError(f"No GFF3 or genome path for {sample}")
                gff = str(Path(assembly).with_suffix(".gff3"))
            result[sample] = Isolate(sample, Path(gff), Path(assembly) if assembly else None)
    return result


def _binary(value: str, source: Path) -> int:
    if value in {"1", "R"}:
        return 1
    if value in {"0", "S"}:
        return 0
    raise ValueError(f"Unsupported phenotype {value!r} in {source}; expected R/S or 0/1")


def read_phenotypes(path: str | Path) -> dict[tuple[str, str], int]:
    """Accept generic long R/S labels and the Kp AMR-GNN wide 0/1 table."""
    path = Path(path)
    labels: dict[tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        long_format = {"isolate_id", "antibiotic", "label"} <= columns
        if not long_format and "id" not in columns:
            raise ValueError(f"Unsupported phenotype columns in {path}")
        for row in reader:
            sample = (row["isolate_id"] if long_format else row["id"]).strip()
            pairs = [(row["antibiotic"].strip(), row["label"].strip())] if long_format else [
                (drug, value.strip()) for drug, value in row.items()
                if drug not in {"id", "Biosample"} and value is not None and value.strip()
            ]
            for drug, value in pairs:
                key = (sample, drug)
                label = _binary(value, path)
                if key in labels and labels[key] != label:
                    raise ValueError(f"Conflicting labels for {sample}, {drug} in {path}")
                labels[key] = label
    return labels


def read_split(split_root: str | Path, antibiotic: str, labels: dict[tuple[str, str], int]) -> Split:
    directory = Path(split_root) / antibiotic
    partitions: dict[str, tuple[str, ...]] = {}
    for name in ("train", "val", "test"):
        path = directory / f"{name}.ids"
        ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate IDs in {path}")
        missing = [sample for sample in ids if (sample, antibiotic) not in labels]
        if missing:
            raise ValueError(f"{len(missing)} split IDs have no {antibiotic} label: {missing[:5]}")
        partitions[name] = ids
    train, val, test = (set(partitions[name]) for name in ("train", "val", "test"))
    if train & val or train & test or val & test:
        raise ValueError(f"Overlapping train/val/test IDs for {antibiotic} in {split_root}")
    for name, ids in partitions.items():
        if {labels[(sample, antibiotic)] for sample in ids} != {0, 1}:
            raise ValueError(f"{name} must contain both classes for {antibiotic}")
    return Split(**partitions)


class RecordStore:
    """Read annotations and cache compact parsed CDS records on disk."""

    def __init__(
        self, manifest: dict[str, Isolate], cache_dir: Path,
        bakta_db: Path, bakta_threads: int = 8,
    ) -> None:
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.bakta_db = bakta_db
        self.bakta_threads = bakta_threads
        self._resolved: dict[str, Path] = {}
        cache_dir.mkdir(parents=True, exist_ok=True)

    def gff_path(self, sample: str) -> Path:
        if sample in self._resolved:
            return self._resolved[sample]
        if sample not in self.manifest:
            raise ValueError(f"Isolate missing from manifest: {sample}")
        item = self.manifest[sample]
        if item.gff3_path.exists():
            self._resolved[sample] = item.gff3_path
            return item.gff3_path
        if item.genome_path is None:
            raise FileNotFoundError(f"No GFF3 or genome FASTA for {sample}")
        path = ensure_bakta_annotation(
            item.genome_path, self.cache_dir.parent / "bakta" / sample,
            self.bakta_db, self.bakta_threads,
        )
        self._resolved[sample] = path
        return path

    def get(self, sample: str) -> list[GeneRecord]:
        source = self.gff_path(sample)
        stat = source.stat()
        cache = self.cache_dir / f"{sample}.json.gz"
        if cache.exists():
            try:
                with gzip.open(cache, "rt", encoding="utf-8") as handle:
                    saved = json.load(handle)
                if saved["source"] == str(source) and saved["size"] == stat.st_size and saved["mtime_ns"] == stat.st_mtime_ns:
                    return [GeneRecord(**{**row, "db_xrefs": tuple(row["db_xrefs"]),
                                           "ec_numbers": tuple(row["ec_numbers"])}) for row in saved["records"]]
            except (OSError, ValueError, KeyError, TypeError):
                LOG.warning("Rebuilding invalid parsed-GFF cache: %s", cache)
        validate_gff3(source)
        records = parse_gff3(source, sample)
        payload = {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                   "records": [asdict(record) for record in records]}
        temporary = cache.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        temporary.replace(cache)
        return records
