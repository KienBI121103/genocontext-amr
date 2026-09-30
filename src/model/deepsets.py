from __future__ import annotations

from torch import nn
from torch_geometric.nn import global_mean_pool

from src.model.graphsage import NodeEncoder, batch_index


class DeepSets(nn.Module):
    """The same node fields as GraphSAGE, with no message passing."""

    def __init__(self, category_sizes: tuple[int, ...], hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.node_encoder = NodeEncoder(category_sizes, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, data):
        initial = self.node_encoder(data)
        pooled = global_mean_pool(self.node_mlp(initial), batch_index(data))
        return self.classifier(pooled).flatten()
