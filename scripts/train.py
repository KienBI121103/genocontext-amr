"""Run the reusable five-model core on one caller-supplied exact split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.inputs import RecordStore, read_manifest, read_phenotypes, read_split
from src.training.pipeline import run_group
from src.training.train import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on one external train/val/test split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--antibiotic", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-mode", choices=("raw", "masked"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    labels = read_phenotypes(config["paths"]["phenotypes"])
    split = read_split(args.split_root, args.antibiotic, labels)
    store = RecordStore(read_manifest(config["paths"]["manifest"]),
                        Path(config["paths"]["artifacts"]) / "features/parsed_gff",
                        Path(config["bakta"]["db"]), config["bakta"]["threads"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    run_group(config, store, labels, split, args.antibiotic, args.seed,
              args.feature_mode, select_device())


if __name__ == "__main__":
    main()
