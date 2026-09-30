from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class NodeEncoder(nn.Module):
    def __init__(self, category_sizes: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, 8) for size in category_sizes)
        self.projection = nn.Sequential(nn.Linear(9 + 8 * len(category_sizes), hidden_dim), nn.ReLU())

    def forward(self, data) -> torch.Tensor:
        columns = [embedding(data.x_cat[:, index]) for index, embedding in enumerate(self.embeddings)]
        return self.projection(torch.cat([data.x_num, *columns], dim=1))


def batch_index(data) -> torch.Tensor:
    return data.batch if getattr(data, "batch", None) is not None else torch.zeros(
        data.num_nodes, dtype=torch.long, device=data.x_num.device
    )


class GenoContextGNN(nn.Module):
    """Two-layer gene-neighborhood GraphSAGE; one logit per isolate."""

    def __init__(self, category_sizes: tuple[int, ...], hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.node_encoder = NodeEncoder(category_sizes, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, data, return_node_state: bool = False):
        initial = self.node_encoder(data)
        hidden = torch.relu(self.conv1(initial, data.edge_index))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.conv2(hidden, data.edge_index))
        pooled = global_mean_pool(hidden, batch_index(data))
        logits = self.classifier(pooled).flatten()
        return (logits, initial) if return_node_state else logits
