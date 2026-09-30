"""Aggregate core model metrics; dataset comparisons live in experiments/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.pipeline import collect_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed core model metrics")
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    metrics = collect_metrics(args.artifacts)
    print(f"Aggregated {len(metrics)} metric rows")


if __name__ == "__main__":
    main()
