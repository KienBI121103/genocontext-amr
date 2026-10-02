from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from src.data.inputs import Split
from src.evaluation.fusion import summarize_fusion
from src.evaluation.metrics import metric_row
from src.explain.attribution import attribute_nodes
from src.features.gene_features import GeneFeatureEncoder
from src.graph.builder import build_graph
from src.model.fusion import align_probabilities, blend_probabilities, select_fusion
from src.model.graphsage import GenoContextGNN
from src.parsing.gff import GeneRecord
from src.training.fusion import FUSION_MODELS
from src.training.pipeline import collect_metrics, run_group


class SignalStore:
    def get(self, sample):
        signal = int(sample[1:]) % 2 == 0
        return [GeneRecord(sample, "contig", f"{sample}-{index}", 1 + 100 * index,
                           90 + 100 * index, "+", "CDS", "resistant" if signal else "susceptible",
                           "protein", (), (), 1000) for index in range(3 if signal else 2)]


def config_for(root):
    return {
        "paths": {"artifacts": str(root)}, "fusion": {"enabled": True},
        "features": {"min_category_frequency": 1}, "graph": {"neighbor_k": 1},
        "model": {"hidden_dim": 8, "dropout": 0., "pooling": "mean_max_count"},
        "training": {"epochs": 2, "patience": 2, "batch_size": 2, "learning_rate": .01,
                     "weight_decay": 0., "cpu_threads": 1, "preprocess_workers": 1,
                     "checkpoint_metric": "auroc"},
        "explain": {"test_isolates_per_run": 1, "top_genes": 1, "neighborhood_k": 1},
    }


def test_alignment_and_validation():
    np.testing.assert_array_equal(align_probabilities(("a", "b"), ("b", "a"), [.9, .1]), [.1, .9])
    for ids, probabilities in [(("a", "a"), [.1, .2]), (("a", "c"), [.1, .2]),
                               (("a", "b"), [.1]), (("a", "b"), [np.nan, .1]),
                               (("a", "b"), [-.1, .2])]:
        with pytest.raises(ValueError):
            align_probabilities(("a", "b"), ids, probabilities)
    with pytest.raises(ValueError):
        blend_probabilities([.1], [.2, .3], .5)
    with pytest.raises(ValueError):
        blend_probabilities([.1], [.2], 1.1)
    with pytest.raises(ValueError):
        select_fusion([0, 1.5], [.1, .2], [.2, .3])
    with pytest.raises(ValueError):
        select_fusion([1, 1], [.1, .2], [.2, .3])


def test_fusion_selection_and_exact_rf_fallback():
    rf = np.array([.05, .2, .8, .95])
    gnn = np.array([.9, .8, .2, .1])
    selected, search = select_fusion(np.array([0, 0, 1, 1]), rf, gnn)
    assert selected.alpha == 0
    assert selected.validation_f1 == 1
    assert len(search) == 21
    assert search[0]["alpha"] == 0 and search[-1]["alpha"] == 1
    np.testing.assert_array_equal(selected.predict(rf, gnn), rf)
    assert select_fusion(np.array([0, 0, 1, 1]), rf, gnn) == (selected, search)
    # Two branches each rank one susceptible isolate incorrectly, but blending fixes both.
    selected, search = select_fusion([0, 0, 1, 1], [.1, .9, .8, .8], [.9, .1, .8, .8])
    assert selected.alpha > 0
    assert selected.validation_f1 > search[0]["validation_f1"]


def test_readout_counts_gradients_and_attribution():
    torch.set_num_threads(1)
    model = GenoContextGNN((2, 2, 2, 2), hidden_dim=8, dropout=0, pooling="mean_max_count")
    hidden = torch.tensor([[1., 3.], [5., 1.], [2., 4.]], requires_grad=True)
    result = model.readout(hidden, torch.tensor([0, 0, 1]))
    expected = torch.tensor([[3., 2., 5., 3., np.log1p(2) / 10],
                             [2., 4., 2., 4., np.log1p(1) / 10]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)
    result[:, :-1].sum().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    store = SignalStore()
    encoder = GeneFeatureEncoder(min_category_frequency=1).fit([store.get("s0"), store.get("s1")])
    model = GenoContextGNN(encoder.category_sizes, hidden_dim=8, dropout=0, pooling="mean_max_count")
    graphs = [build_graph(store.get(sample), encoder, label) for sample, label in (("s0", 1), ("s1", 0))]
    assert model(Batch.from_data_list(graphs)).shape == (2,)
    rows = attribute_nodes(model, graphs[0], store.get("s0"), torch.device("cpu"), top_n=1)
    assert {row["feature_id"] for row in rows} <= {row.feature_id for row in store.get("s0")}
    assert np.isfinite([row["importance_score"] for row in rows]).all()
    assert GenoContextGNN(encoder.category_sizes, hidden_dim=8).classifier[0].in_features == 8


def test_end_to_end_resume_and_test_independence(tmp_path: Path):
    torch.set_num_threads(1)
    samples = tuple(f"s{index}" for index in range(10))
    labels = {(sample, "drug"): int(index % 2 == 0) for index, sample in enumerate(samples)}
    split = Split(samples[:6], samples[6:8], samples[8:])
    config = config_for(tmp_path / "original")
    run_group(config, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    output = Path(config["paths"]["artifacts"]) / "results/seed0/drug/raw"
    for model in FUSION_MODELS:
        assert (output / f"{model}_metrics.csv").exists()
        assert (output / f"{model}_predictions.csv").exists()
    assert not (output / "deepsets_metrics.csv").exists()
    assert (tmp_path / "original/models/seed0/drug/raw/random_forest.joblib").exists()
    history = pd.read_csv(output / "genocontext_gnn_real_history.csv")
    assert history["selected_checkpoint"].sum() == 1
    assert history.loc[history["selected_checkpoint"], "epoch"].iloc[0] == history["validation_auroc"].idxmax() + 1
    selection = json.loads((output / "genocontext_fusion_real_selection.json").read_text())
    frame = pd.read_csv(output / "genocontext_fusion_real_predictions.csv")
    assert {"probability_rf", "probability_gnn", "probability_fused", "alpha"} <= set(frame)
    rf_metrics = pd.read_csv(output / "random_forest_metrics.csv")
    assert selection["validation_f1"] >= rf_metrics.query("split == 'val'")["f1"].iloc[0]
    explanation = pd.read_csv(output / "genocontext_fusion_real_attribution.csv")
    assert set(explanation["explanation_scope"]) == {"graph_branch"}
    assert set(explanation["alpha"]) == {selection["alpha"]}

    # Partial resume rebuilds outputs using saved RF/GNN; no feature refit or neural training.
    (output / "random_forest_predictions.csv").unlink()
    (output / "genocontext_fusion_real_metrics.csv").unlink()
    with patch("src.training.fusion.GeneFeatureEncoder.fit", side_effect=AssertionError("refit")), \
            patch("src.training.fusion.fit_neural", side_effect=AssertionError("retrain")):
        run_group(config, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    changed = deepcopy(config)
    changed["model"]["dropout"] = .1
    with pytest.raises(ValueError, match="Incompatible"):
        run_group(changed, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    vocab_path = output / "vocabularies.json"
    original_vocab = vocab_path.read_text()
    vocab = json.loads(original_vocab)
    vocab["gene"]["unexpected"] = 100
    vocab_path.write_text(json.dumps(vocab))
    with pytest.raises(ValueError, match="vocabulary signature"):
        run_group(config, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    vocab_path.write_text(original_vocab)
    # Even a completed run must reject a changed output cohort.
    prediction_path = output / "genocontext_fusion_real_predictions.csv"
    original_predictions = prediction_path.read_text()
    invalid = pd.read_csv(prediction_path)
    invalid.loc[0, "isolate_id"] = "unmatched"
    invalid.to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="cohort"):
        run_group(config, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    prediction_path.write_text(original_predictions)

    # Changing test labels in a fresh run changes metrics, never selected parameters or checkpoints.
    altered_labels = {key: 1 - value if key[0] in split.test else value for key, value in labels.items()}
    with pytest.raises(ValueError, match="Incompatible"):
        run_group(config, SignalStore(), altered_labels, split, "drug", 0, "raw", torch.device("cpu"))
    changed = deepcopy(config)
    changed["paths"]["artifacts"] = str(tmp_path / "altered_test")
    run_group(changed, SignalStore(), altered_labels, split, "drug", 0, "raw", torch.device("cpu"))
    other = tmp_path / "altered_test/results/seed0/drug/raw"
    for edge in ("real", "shuffled"):
        name = f"genocontext_fusion_{edge}_selection.json"
        assert json.loads((output / name).read_text()) == json.loads((other / name).read_text())
        filename = f"genocontext_gnn_{edge}.pt"
        first = torch.load(tmp_path / "original/models/seed0/drug/raw" / filename, weights_only=True)
        second = torch.load(tmp_path / "altered_test/models/seed0/drug/raw" / filename, weights_only=True)
        assert first["best_epoch"] == second["best_epoch"]
        for key in first["state_dict"]:
            torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
    # Valid-looking stale probabilities must be recomputed from the saved checkpoint.
    gnn_path = output / "genocontext_gnn_real_predictions.csv"
    original_gnn = pd.read_csv(gnn_path)
    stale = original_gnn.copy()
    stale["probability_R"] = .01
    stale.to_csv(gnn_path, index=False)
    with patch("src.training.fusion.fit_neural", side_effect=AssertionError("retrain")):
        run_group(config, SignalStore(), labels, split, "drug", 0, "raw", torch.device("cpu"))
    np.testing.assert_allclose(pd.read_csv(gnn_path)["probability_R"], original_gnn["probability_R"], rtol=0, atol=1e-12)
    metrics = collect_metrics(Path(config["paths"]["artifacts"]))
    gate = summarize_fusion(Path(config["paths"]["artifacts"]), metrics, "drug")
    assert gate["status"] == "pilot_incomplete" and not gate["allow_full_run"]
    summary = pd.read_csv(tmp_path / "original/results/fusion_vs_rf.csv")
    assert {"delta_f1", "rf_errors_corrected", "rf_errors_introduced", "gnn_probability_std"} <= set(summary)


@pytest.mark.parametrize("improving_seeds,eligible", [((0, 1), True), ((0,), False), ((), False)])
def test_validation_gate_ignores_test_improvements(tmp_path, improving_seeds, eligible):
    metrics = []
    truth = np.array([0, 0, 1, 1])
    rf = np.array([.1, .7, .8, .9])
    perfect = np.array([.1, .2, .8, .9])
    for seed in (0, 1, 2):
        for mode in ("raw", "masked"):
            output = tmp_path / f"results/seed{seed}/ciprofloxacin/{mode}"
            output.mkdir(parents=True)
            rf_rows, fusion_rows = [], []
            for part in ("val", "test"):
                # Test improves for every seed, regardless of validation eligibility.
                fused = perfect if part == "test" or seed in improving_seeds else rf
                ids = tuple(f"{part}-{index}" for index in range(4))
                for model, scores in (("random_forest", rf), ("genocontext_fusion_real", fused)):
                    row = metric_row(truth, scores, .5, "ciprofloxacin", model, part, seed, mode,
                                     "none" if model == "random_forest" else "real")
                    row["alpha"] = .5
                    metrics.append(row)
                for index, sample in enumerate(ids):
                    base = {"isolate_id": sample, "split": part, "true_label": "R" if truth[index] else "S"}
                    rf_rows.append({**base, "probability_R": rf[index]})
                    fusion_rows.append({**base, "probability_R": fused[index],
                                        "probability_rf": rf[index], "probability_gnn": fused[index]})
            pd.DataFrame(rf_rows).to_csv(output / "random_forest_predictions.csv", index=False)
            pd.DataFrame(fusion_rows).to_csv(output / "genocontext_fusion_real_predictions.csv", index=False)
    gate = summarize_fusion(tmp_path, pd.DataFrame(metrics))
    assert gate["allow_full_run"] is eligible
    assert all(result["complete"] for result in gate["feature_modes"].values())
    if not eligible:
        assert gate["status"] == "no_demonstrated_fusion_benefit_retain_rf"
