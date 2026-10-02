"""Reusable RF / graph-branch training and validation-only fusion workflow."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import metric_row, prediction_rows, select_f1_threshold
from src.explain.attribution import attribute_nodes
from src.features.gene_features import GeneFeatureEncoder
from src.model.baselines import random_forest
from src.model.fusion import align_probabilities, select_fusion
from src.model.graphsage import GenoContextGNN
from src.training.parallel import FeatureBuilder
from src.training.pipeline import write_csv
from src.training.train import fit_neural, predict, set_seed

LOG = logging.getLogger(__name__)
FUSION_VERSION = "rf_gnn_fusion_v1"
FUSION_MODELS = (
    "random_forest", "genocontext_gnn_real", "genocontext_gnn_shuffled",
    "genocontext_fusion_real", "genocontext_fusion_shuffled",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write_json(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_details(config, store, labels, samples, antibiotic, seed, feature_mode):
    all_ids = sorted({sample for ids in samples.values() for sample in ids})
    for name, ids in samples.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate {name} IDs")
        if any((sample, antibiotic) not in labels for sample in ids):
            raise ValueError(f"Missing {antibiotic} labels in {name}")
        if {labels[(sample, antibiotic)] for sample in ids} != {0, 1}:
            raise ValueError(f"{name} requires both R and S isolates")
    if sum(map(len, samples.values())) != len(all_ids):
        raise ValueError("Overlapping train/val/test IDs")
    annotation_signatures = {}
    for sample in all_ids:
        if hasattr(store, "gff_path"):
            path = store.gff_path(sample)
            stat = path.stat()
            annotation_signatures[sample] = [str(path), stat.st_size, stat.st_mtime_ns]
        else:
            annotation_signatures[sample] = _digest([asdict(row) for row in store.get(sample)])
    source_root = Path(__file__).resolve().parents[1]
    source_signatures = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    return {
        "version": FUSION_VERSION, "source": _digest(source_signatures),
        "antibiotic": antibiotic, "seed": seed, "feature_mode": feature_mode,
        "splits": {name: list(ids) for name, ids in samples.items()},
        "labels": _digest({sample: labels[(sample, antibiotic)] for sample in all_ids}),
        "annotations": _digest(annotation_signatures),
        "settings": {key: config.get(key, {}) for key in
                     ("features", "graph", "model", "training", "explain", "fusion")},
    }


def _export(output, samples, labels, probabilities, antibiotic, seed, feature_mode,
            model_name, edge_mode, threshold, branches=None, alpha=None):
    metrics, predictions = [], []
    for part in ("val", "test"):
        ids = samples[part]
        truth = np.asarray([labels[(sample, antibiotic)] for sample in ids])
        rows = prediction_rows(ids, truth, probabilities[part], threshold,
                               antibiotic, model_name, part, seed, feature_mode, edge_mode)
        if branches is not None:
            for index, row in enumerate(rows):
                row.update(probability_rf=float(branches["rf"][part][index]),
                           probability_gnn=float(branches["gnn"][part][index]),
                           probability_fused=float(probabilities[part][index]), alpha=alpha)
        predictions.extend(rows)
        row = metric_row(truth, probabilities[part], threshold,
                         antibiotic, model_name, part, seed, feature_mode, edge_mode)
        row["probability_std"] = float(np.std(probabilities[part]))
        if alpha is not None:
            row["alpha"] = alpha
        metrics.append(row)
    write_csv(predictions, output / f"{model_name}_predictions.csv")
    write_csv(metrics, output / f"{model_name}_metrics.csv")


def _component_signatures(output, model_root, edge):
    files = [model_root / "random_forest.joblib", model_root / f"genocontext_gnn_{edge}.pt",
             output / f"genocontext_gnn_{edge}_history.csv"]
    files += [output / f"{name}_{suffix}.csv" for name in
              (f"genocontext_gnn_{edge}", f"genocontext_fusion_{edge}")
              for suffix in ("metrics", "predictions")]
    files += [output / f"genocontext_fusion_{edge}_{suffix}" for suffix in
              ("selection.json", "search.csv", "attribution.csv")]
    if not all(path.is_file() for path in files):
        return None
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _component_complete(output, model_root, edge):
    stamp = output / f"genocontext_fusion_{edge}_component.json"
    if not stamp.exists():
        return False
    signatures = _component_signatures(output, model_root, edge)
    return signatures is not None and json.loads(stamp.read_text(encoding="utf-8")) == signatures


def _complete(output, model_root) -> bool:
    return (all((output / f"random_forest_{suffix}.csv").is_file() for suffix in ("predictions", "metrics"))
            and all(_component_complete(output, model_root, edge) for edge in ("real", "shuffled")))


def _load_probabilities(output, name, samples, labels, antibiotic):
    path = output / f"{name}_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"isolate_id": str})
    if set(frame["split"]) != {"val", "test"}:
        raise ValueError(f"Invalid prediction partitions: {path}")
    result = {}
    for part in ("val", "test"):
        rows = frame[frame["split"] == part]
        ids = tuple(rows["isolate_id"])
        result[part] = align_probabilities(samples[part], ids, rows["probability_R"].to_numpy())
        expected_labels = ["R" if labels[(sample, antibiotic)] else "S" for sample in ids]
        if rows["true_label"].tolist() != expected_labels:
            raise ValueError(f"Prediction labels differ from input labels: {path}")
    return result


def validate_fusion_run(config, store, labels, split, antibiotic, seed, feature_mode):
    """Check a completed pilot against current inputs/settings before full expansion."""
    root = Path(config["paths"]["artifacts"])
    relative = Path(f"seed{seed}") / antibiotic / feature_mode
    output, model_root = root / "results" / relative, root / "models" / relative
    samples = {"train": split.train, "val": split.val, "test": split.test}
    details = _run_details(config, store, labels, samples, antibiotic, seed, feature_mode)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    vocab = json.loads((output / "vocabularies.json").read_text(encoding="utf-8"))
    if manifest.get("details") != details or manifest.get("vocabulary_sha256") != _digest(vocab):
        raise ValueError(f"Incompatible pilot artifacts: {output}")
    if not _complete(output, model_root):
        raise ValueError(f"Pilot components are incomplete: {output}")
    for name in FUSION_MODELS:
        _load_probabilities(output, name, samples, labels, antibiotic)


def run_fusion_group(config, store, labels, split, antibiotic, seed, feature_mode, device):
    if feature_mode not in {"raw", "masked"}:
        raise ValueError("feature_mode must be raw or masked")
    root = Path(config["paths"]["artifacts"])
    relative = Path(f"seed{seed}") / antibiotic / feature_mode
    output, model_root = root / "results" / relative, root / "models" / relative
    samples = {"train": split.train, "val": split.val, "test": split.test}
    details = _run_details(config, store, labels, samples, antibiotic, seed, feature_mode)
    manifest_path, vocab_path = output / "run_manifest.json", output / "vocabularies.json"
    encoder = GeneFeatureEncoder(mask_amr_terms=feature_mode == "masked",
                                 min_category_frequency=config["features"]["min_category_frequency"])
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("details") != details:
            raise ValueError(f"Incompatible fusion run; use a new artifacts directory: {manifest_path}")
        encoder.vocabularies = json.loads(vocab_path.read_text(encoding="utf-8"))
        if previous.get("vocabulary_sha256") != _digest(encoder.vocabularies):
            raise ValueError(f"Fitted vocabulary signature differs: {vocab_path}")
    else:
        if ((output.exists() and any(output.iterdir())) or
                (model_root.exists() and any(model_root.iterdir()))):
            raise ValueError("Fusion artifacts exist without a compatible run manifest")
        output.mkdir(parents=True, exist_ok=True)
        LOG.info("Fitting %s vocabulary from %d training isolates", feature_mode, len(split.train))
        encoder.fit(store.get(sample) for sample in split.train)
        _write_json(encoder.vocabularies, vocab_path)
        _write_json({"details": details, "vocabulary_sha256": _digest(encoder.vocabularies)}, manifest_path)
    for name in FUSION_MODELS:
        _load_probabilities(output, name, samples, labels, antibiotic)
    if _complete(output, model_root):
        LOG.info("Fusion group already complete: %s", output)
        return
    model_root.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    y_train = np.asarray([labels[(sample, antibiotic)] for sample in split.train])
    y_val = np.asarray([labels[(sample, antibiotic)] for sample in split.val])
    with FeatureBuilder(store, encoder, labels, antibiotic, config["graph"]["neighbor_k"],
                        training.get("preprocess_workers", 1)) as builder:
        rf_path = model_root / "random_forest.joblib"
        matrices = None
        rf_probabilities = _load_probabilities(output, "random_forest", samples, labels, antibiotic)
        if rf_probabilities is None:
            matrices = {part: builder.genome_matrix(ids) for part, ids in samples.items()}
            if rf_path.exists():
                rf = joblib.load(rf_path)
            else:
                rf_threads = training.get("rf_threads", training["cpu_threads"])
                LOG.info("Fitting random forest with %d threads", rf_threads)
                rf = random_forest(seed, rf_threads).fit(matrices["train"], y_train)
                temporary = rf_path.with_suffix(".tmp")
                joblib.dump(rf, temporary)
                temporary.replace(rf_path)
            rf_probabilities = {"val": rf.predict_proba(matrices["val"])[:, 1]}
        elif not rf_path.exists():
            raise ValueError("RF predictions exist without the persisted random forest")
        rf_threshold = select_f1_threshold(y_val, rf_probabilities["val"])
        graphs = None
        for edge in ("real", "shuffled"):
            gnn_name, fusion_name = f"genocontext_gnn_{edge}", f"genocontext_fusion_{edge}"
            checkpoint = model_root / f"{gnn_name}.pt"
            probabilities = _load_probabilities(output, gnn_name, samples, labels, antibiotic)
            component_complete = _component_complete(output, model_root, edge)
            if ("test" in rf_probabilities and component_complete and
                    (output / "random_forest_metrics.csv").exists()):
                continue
            if not component_complete:
                # Old scores may belong to a checkpoint replaced during an interrupted fit.
                probabilities = None
            (output / f"{fusion_name}_component.json").unlink(missing_ok=True)
            # Build independent shuffled copies; never mutate real graphs or cached topology.
            if graphs is None:
                LOG.info("Building fusion graphs with %d workers", training.get("preprocess_workers", 1))
                graphs = {part: builder.graphs(ids) for part, ids in samples.items()}
            branch_graphs = graphs
            if edge == "shuffled":
                branch_graphs = {part: [graph.clone() for graph in values] for part, values in graphs.items()}
                for part, ids in samples.items():
                    builder.shuffle_edges(ids, branch_graphs[part], seed)
            set_seed(seed)
            model = GenoContextGNN(encoder.category_sizes, config["model"]["hidden_dim"],
                                   config["model"]["dropout"], config["model"].get("pooling", "mean_max_count"))
            history = output / f"{gnn_name}_history.csv"
            if checkpoint.exists() and history.exists():
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
                model = model.to(device)
            else:
                LOG.info("Training %s", gnn_name)
                model = fit_neural(
                    model, branch_graphs["train"], branch_graphs["val"], device, seed, checkpoint,
                    epochs=training["epochs"], patience=training["patience"],
                    batch_size=training["batch_size"], learning_rate=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                    checkpoint_metric=training.get("checkpoint_metric", "auroc"), history_path=history,
                )
                probabilities = None
            if probabilities is None:
                probabilities = {"val": predict(model, branch_graphs["val"], device, training["batch_size"])}
            selection, search = select_fusion(y_val, rf_probabilities["val"], probabilities["val"])
            gnn_threshold = select_f1_threshold(y_val, probabilities["val"])
            # Persist all selected parameters before generating/evaluating test predictions.
            _write_json(asdict(selection), output / f"{fusion_name}_selection.json")
            write_csv(search, output / f"{fusion_name}_search.csv")
            if "test" not in rf_probabilities:
                rf_probabilities["test"] = rf.predict_proba(matrices["test"])[:, 1]
                del rf, matrices
                matrices = None
            if "test" not in probabilities:
                probabilities["test"] = predict(model, branch_graphs["test"], device, training["batch_size"])
            fused = {part: selection.predict(rf_probabilities[part], probabilities[part]) for part in ("val", "test")}
            common = (output, samples, labels)
            _export(*common, rf_probabilities, antibiotic, seed, feature_mode,
                    "random_forest", "none", rf_threshold)
            _export(*common, probabilities, antibiotic, seed, feature_mode,
                    gnn_name, edge, gnn_threshold)
            _export(*common, fused, antibiotic, seed, feature_mode,
                    fusion_name, edge, selection.threshold,
                    branches={"rf": rf_probabilities, "gnn": probabilities}, alpha=selection.alpha)
            rows = []
            positions = np.argsort(-fused["test"])[:config["explain"]["test_isolates_per_run"]]
            for position in positions:
                sample = split.test[int(position)]
                explanation = attribute_nodes(model, branch_graphs["test"][int(position)], store.get(sample), device,
                                              top_n=config["explain"]["top_genes"],
                                              neighborhood_k=config["explain"]["neighborhood_k"])
                for row in explanation:
                    row.update(explanation_scope="graph_branch", alpha=selection.alpha,
                               graph_contributes=selection.alpha > 0, model=fusion_name,
                               edge_mode=edge, seed=seed, antibiotic=antibiotic, feature_mode=feature_mode,
                               probability_fused=float(fused["test"][position]))
                rows.extend(explanation)
            columns = ["isolate_id", "contig", "feature_id", "explanation_scope", "alpha", "graph_contributes"]
            write_csv(pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns),
                      output / f"{fusion_name}_attribution.csv")
            _write_json(_component_signatures(output, model_root, edge),
                        output / f"{fusion_name}_component.json")
            LOG.info("Finished %s: alpha=%.2f validation_f1=%.4f", fusion_name, selection.alpha, selection.validation_f1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
