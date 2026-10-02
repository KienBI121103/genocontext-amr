"""Validate a generic manifest, phenotype table, and supplied split."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.inputs import RecordStore, parsed_cache_dir, read_manifest, read_phenotypes, read_split
from src.training.pipeline import preflight_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and cache GFF3 records")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--antibiotic", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    labels = read_phenotypes(config["paths"]["phenotypes"])
    split = read_split(args.split_root, args.antibiotic, labels)
    store = RecordStore(read_manifest(config["paths"]["manifest"]),
                        parsed_cache_dir(config),
                        Path(config["bakta"]["db"]), config["bakta"]["threads"])
    preflight_records(store, split)
    print(f"Validated {len(set(split.train) | set(split.val) | set(split.test))} isolates")


if __name__ == "__main__":
    main()
