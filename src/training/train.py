from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

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
) -> nn.Module:
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
    best_score = float("-inf")
    best_state = None
    stale = 0
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch.y.float().flatten())
            loss.backward()
            optimizer.step()
        probabilities = predict(model, val_graphs, device, batch_size)
        threshold = select_f1_threshold(val_labels, probabilities)
        score = float(f1_score(val_labels, probabilities >= threshold))
        if epoch == 1 or epoch % 5 == 0:
            LOG.info("epoch=%d validation_f1=%.4f best_f1=%.4f", epoch, score, max(best_score, score))
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No model checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "validation_f1": best_score}, checkpoint)
    return model
