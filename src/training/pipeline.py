"""Reusable one-antibiotic, one-seed training workflow.

Dataset paths and split selection belong to callers in scripts/ or experiments/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.inputs import RecordStore, Split
from src.evaluation.metrics import metric_row, prediction_rows, select_f1_threshold
from src.explain.attribution import attribute_nodes
from src.features.gene_features import GeneFeatureEncoder
from src.model.baselines import fit_baselines
from src.model.deepsets import DeepSets
from src.model.graphsage import GenoContextGNN
from src.training.parallel import FeatureBuilder
from src.training.train import fit_neural, predict, set_seed

LOG = logging.getLogger(__name__)

MODELS = (
    "logistic_regression", "random_forest", "deepsets",
    "genocontext_gnn_real", "genocontext_gnn_shuffled",
)


def write_csv(rows: list[dict] | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def preflight_records(store: RecordStore, split: Split) -> None:
    """Validate and parse every isolate before fitting any model."""
    for sample in sorted(set(split.train) | set(split.val) | set(split.test)):
        store.get(sample)


def _completed(output: Path, model_name: str) -> bool:
    files = [output / f"{model_name}_predictions.csv", output / f"{model_name}_metrics.csv"]
    if model_name == "genocontext_gnn_real":
        files.append(output / "genocontext_gnn_attribution.csv")
    return all(path.is_file() for path in files)


def _save_model_results(
    output: Path, sample_sets: dict[str, tuple[str, ...]], labels: dict,
    probabilities: dict[str, np.ndarray], antibiotic: str, seed: int,
    model_name: str, feature_mode: str, edge_mode: str,
) -> None:
    y_val = np.asarray([labels[(sample, antibiotic)] for sample in sample_sets["val"]])
    threshold = select_f1_threshold(y_val, probabilities["val"])
    metrics: list[dict] = []
    predictions: list[dict] = []
    for partition in ("val", "test"):
        ids = sample_sets[partition]
        truth = np.asarray([labels[(sample, antibiotic)] for sample in ids])
        predictions.extend(prediction_rows(
            ids, truth, probabilities[partition], threshold,
            antibiotic, model_name, partition, seed, feature_mode, edge_mode,
        ))
        metrics.append(metric_row(
            truth, probabilities[partition], threshold,
            antibiotic, model_name, partition, seed, feature_mode, edge_mode,
        ))
    write_csv(predictions, output / f"{model_name}_predictions.csv")
    write_csv(metrics, output / f"{model_name}_metrics.csv")


def run_group(
    config: dict, store: RecordStore, labels: dict, split: Split,
    antibiotic: str, seed: int, feature_mode: str, device: torch.device,
) -> None:
    """Run all five models on one externally supplied train/val/test split."""
    if config.get("fusion", {}).get("enabled", False):
        from src.training.fusion import run_fusion_group
        run_fusion_group(config, store, labels, split, antibiotic, seed, feature_mode, device)
        return
    if feature_mode not in {"raw", "masked"}:
        raise ValueError("feature_mode must be raw or masked")
    output = Path(config["paths"]["artifacts"]) / "results" / f"seed{seed}" / antibiotic / feature_mode
    samples = {"train": split.train, "val": split.val, "test": split.test}
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = output / "vocabularies.json"
    manifest_path = output / "run_manifest.json"
    run_details = {
        "antibiotic": antibiotic, "seed": seed, "feature_mode": feature_mode,
        "splits": {name: list(ids) for name, ids in samples.items()},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in run_details.items()):
            raise ValueError(f"Existing run uses different split IDs or settings: {manifest_path}")
    if all(_completed(output, name) for name in MODELS):
        return
    encoder = GeneFeatureEncoder(
        mask_amr_terms=feature_mode == "masked",
        min_category_frequency=config["features"]["min_category_frequency"],
    )
    if vocab_path.exists() and manifest_path.exists():
        encoder.vocabularies = json.loads(vocab_path.read_text(encoding="utf-8"))
        LOG.info("Reusing fitted vocabulary from %s", vocab_path)
    else:
        LOG.info("Fitting %s vocabulary from %d training isolates", feature_mode, len(split.train))
        encoder.fit(store.get(sample) for sample in split.train)
        vocab_path.write_text(json.dumps(encoder.vocabularies, sort_keys=True), encoding="utf-8")
        manifest_path.write_text(
            json.dumps({**run_details, "device": str(device)}, sort_keys=True),
            encoding="utf-8",
        )

    training = config["training"]
    workers = training.get("preprocess_workers", 1)
    with FeatureBuilder(
        store, encoder, labels, antibiotic, config["graph"]["neighbor_k"], workers,
    ) as builder:
        unfinished = [name for name in MODELS[:2] if not _completed(output, name)]
        if unfinished:
            LOG.info("Building genome vectors with %d worker(s)", workers)
            matrices = {
                part: builder.genome_matrix(ids) for part, ids in samples.items()
            }
            train_y = np.asarray([labels[(sample, antibiotic)] for sample in split.train])
            baselines = fit_baselines(
                matrices["train"], train_y, seed, training["cpu_threads"],
            )
            for name in unfinished:
                probabilities = {
                    part: baselines[name].predict_proba(matrices[part])[:, 1]
                    for part in ("val", "test")
                }
                _save_model_results(
                    output, samples, labels, probabilities, antibiotic, seed,
                    name, feature_mode, "none",
                )
                LOG.info("Finished %s", name)
            del matrices, baselines

        graphs = None
        for name in MODELS[2:]:
            if _completed(output, name):
                continue
            set_seed(seed)
            if graphs is None:
                LOG.info("Building graphs for %d isolates with %d worker(s)",
                         sum(map(len, samples.values())), workers)
                graphs = {part: builder.graphs(ids) for part, ids in samples.items()}
            shuffled = name == "genocontext_gnn_shuffled"
            if shuffled:
                LOG.info("Shuffling within-contig edges")
                for part, ids in samples.items():
                    builder.shuffle_edges(ids, graphs[part], seed)
            edge_mode = "shuffled" if shuffled else "real" if name == "genocontext_gnn_real" else "none"
            model_type = DeepSets if name == "deepsets" else GenoContextGNN
            model = model_type(
                encoder.category_sizes, config["model"]["hidden_dim"],
                config["model"]["dropout"],
                **({"pooling": config["model"].get("pooling", "mean")} if model_type is GenoContextGNN else {}),
            )
            checkpoint = (
                Path(config["paths"]["artifacts"]) / "models" / f"seed{seed}"
                / antibiotic / feature_mode / f"{name}.pt"
            )
            LOG.info("Training %s", name)
            model = fit_neural(
                model, graphs["train"], graphs["val"], device, seed, checkpoint,
                epochs=training["epochs"], patience=training["patience"],
                batch_size=training["batch_size"], learning_rate=training["learning_rate"],
                weight_decay=training["weight_decay"],
                checkpoint_metric=training.get("checkpoint_metric", "f1"),
            )
            probabilities = {
                part: predict(model, graphs[part], device, training["batch_size"])
                for part in ("val", "test")
            }
            _save_model_results(
                output, samples, labels, probabilities, antibiotic, seed,
                name, feature_mode, edge_mode,
            )
            if name == "genocontext_gnn_real":
                important = np.argsort(-probabilities["test"])[
                    :config["explain"]["test_isolates_per_run"]
                ]
                rows = []
                for position in important:
                    sample = split.test[int(position)]
                    rows.extend(attribute_nodes(
                        model, graphs["test"][int(position)], store.get(sample),
                        device, top_n=config["explain"]["top_genes"],
                        neighborhood_k=config["explain"]["neighborhood_k"],
                    ))
                write_csv(rows, output / "genocontext_gnn_attribution.csv")
            LOG.info("Finished %s", name)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()



def collect_metrics(artifacts: Path) -> pd.DataFrame:
    root = artifacts / "results"
    files = sorted(root.glob("seed*/*/*/*_metrics.csv"))
    if not files:
        raise ValueError(f"No completed model metrics under {root}")
    result = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    write_csv(result, root / "metrics.csv")
    return result
