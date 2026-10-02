from __future__ import annotations

import copy
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.evaluation.metrics import select_f1_threshold
from torch import nn
from torch_geometric.loader import DataLoader

LOG = logging.getLogger(__name__)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, graphs: list, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        scores = torch.sigmoid(model(batch.to(device))).cpu().numpy()
        values.append(scores)
    return np.concatenate(values)


def fit_neural(
    model: nn.Module, train_graphs: list, val_graphs: list,
    device: torch.device, seed: int, checkpoint: Path,
    epochs: int = 100, patience: int = 15, batch_size: int = 16,
    learning_rate: float = 0.001, weight_decay: float = 0.0001,
    checkpoint_metric: str = "f1", history_path: Path | None = None,
) -> nn.Module:
    if checkpoint_metric not in {"f1", "auroc"}:
        raise ValueError("checkpoint_metric must be f1 or auroc")
    set_seed(seed)
    model = model.to(device)
    labels = np.asarray([int(graph.y.item()) for graph in train_graphs])
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Training requires both R and S isolates")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    val_labels = np.asarray([int(graph.y.item()) for graph in val_graphs])
    if set(val_labels) != {0, 1}:
        raise ValueError("Validation requires both R and S isolates")
    best_score = float("-inf")
    best_state = None
    stale = 0
    best_epoch = 0
    history = []
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch.y.float().flatten())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.num_graphs
        probabilities = predict(model, val_graphs, device, batch_size)
        threshold = select_f1_threshold(val_labels, probabilities)
        row = {
            "epoch": epoch, "train_loss": total_loss / len(train_graphs),
            "validation_f1": float(f1_score(val_labels, probabilities >= threshold)),
            "validation_auroc": float(roc_auc_score(val_labels, probabilities)),
            "validation_auprc": float(average_precision_score(val_labels, probabilities)),
            "probability_std": float(np.std(probabilities)),
            "probability_min": float(np.min(probabilities)),
            "probability_max": float(np.max(probabilities)),
        }
        history.append(row)
        score = row[f"validation_{checkpoint_metric}"]
        if epoch == 1 or epoch % 5 == 0:
            LOG.info("epoch=%d train_loss=%.4f val_f1=%.4f val_auroc=%.4f val_auprc=%.4f "
                     "prob_std=%.6g best_%s=%.4f", epoch, row["train_loss"],
                     row["validation_f1"], row["validation_auroc"], row["validation_auprc"],
                     row["probability_std"], checkpoint_metric, max(best_score, score))
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No model checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save({"state_dict": best_state, "checkpoint_metric": checkpoint_metric,
                "best_score": best_score, "best_epoch": best_epoch,
                "validation_f1": history[best_epoch - 1]["validation_f1"],
                "validation_auroc": history[best_epoch - 1]["validation_auroc"]}, temporary)
    temporary.replace(checkpoint)
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = history_path.with_suffix(history_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*history[0], "selected_checkpoint"])
            writer.writeheader()
            writer.writerows({**row, "selected_checkpoint": row["epoch"] == best_epoch} for row in history)
        temporary.replace(history_path)
    LOG.info("Selected epoch=%d by validation_%s=%.4f", best_epoch, checkpoint_metric, best_score)
    return model
