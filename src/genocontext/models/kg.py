"""GenoContext-KG: gene-state tokens -> knowledge-masked mechanism nodes -> drug-class-conditioned logits.

For isolate b and drug d:
    logit[b, d] = bias[d] + sum_k c[b, d, k] + sum_{k<l} gamma[d, k, l] * s[b, k] * s[b, l]
    c[b, d, k]  = w[k, d] * (V_k m[b, k]) . q[d]
where m[b, k] is a gated sum of token embeddings assigned to mechanism k (prior assignment from the
knowledge file, or a learned soft assignment when ``use_prior`` is off), q[d] is a query built from the
drug's class features, and w[k, d] is 1 on prior mechanism->drug edges and a learnable, L1-penalised weight
elsewhere. The decomposition into c and the pairwise terms is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv

from genocontext.config import ModelConfig


@dataclass
class Batch:
    token: Tensor  # (N,) block ids
    state: Tensor  # (N,)
    flags: Tensor  # (N, F)
    node_graph: Tensor  # (N,) isolate index within the batch
    pair_node: Tensor  # (P,) node index of each (node, mechanism) membership
    pair_mech: Tensor  # (P,)
    edge_index: Tensor  # (2, E)
    size: int  # number of isolates


class AMRModel(nn.Module):
    """Base class: maps a Batch to (B, n_drugs) logits and exposes a regularisation penalty."""

    def penalty(self) -> Tensor:
        return torch.zeros(())


class TokenEncoder(nn.Module):
    def __init__(self, n_tokens: int, n_states: int, n_flags: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Embedding(n_tokens, hidden)
        self.state = nn.Embedding(n_states, hidden)
        self.flags = nn.Linear(n_flags, hidden)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, batch: Batch) -> Tensor:
        out: Tensor = self.mlp(self.token(batch.token) + self.state(batch.state) + self.flags(batch.flags))
        return out


class GenoContextKG(AMRModel):
    drug_features: Tensor
    prior: Tensor
    drug_embedding_mask: Tensor

    def __init__(self, n_tokens: int, n_states: int, n_flags: int, drug_features: Tensor, prior: Tensor,
                 config: ModelConfig) -> None:
        super().__init__()
        n_mech, n_drugs = prior.shape
        hidden, query = config.hidden, config.query
        self.config = config
        self.encoder = TokenEncoder(n_tokens, n_states, n_flags, hidden, config.dropout)
        self.context = GATv2Conv(hidden, hidden, add_self_loops=False) if config.context_layer else None
        self.gate = nn.Embedding(n_mech, hidden)
        self.assign = None if config.use_prior else nn.Linear(hidden, n_mech)
        self.project = nn.Parameter(torch.randn(n_mech, hidden, query) / hidden**0.5)
        self.activation = nn.Linear(hidden, 1)
        self.register_buffer("drug_features", drug_features)
        self.register_buffer("prior", prior if config.use_prior else torch.ones_like(prior))
        self.query_mlp = nn.Sequential(nn.Linear(drug_features.shape[1], query), nn.GELU(), nn.Linear(query, query))
        self.drug_embedding = nn.Embedding(n_drugs, query)
        nn.init.normal_(self.drug_embedding.weight, std=0.01)
        self.register_buffer("drug_embedding_mask", torch.ones(n_drugs, 1))
        self.edge_logit = nn.Parameter(torch.full((n_mech, n_drugs), -3.0))
        self.gamma = nn.Parameter(torch.zeros(n_drugs, n_mech, n_mech)) if config.interactions else None
        self.bias = nn.Parameter(torch.zeros(n_drugs))
        self.class_bias = nn.Linear(drug_features.shape[1], 1)

    # ---- building blocks ----------------------------------------------------------------------
    def queries(self) -> Tensor:
        own = self.drug_embedding.weight * self.drug_embedding_mask  # zero-shot drugs use class features only
        if not self.config.drug_features:
            return own
        shared: Tensor = self.query_mlp(self.drug_features)
        return shared + own

    def intercepts(self) -> Tensor:
        """Per-drug bias: a class-feature term plus a free term (masked for zero-shot drugs)."""
        shared: Tensor = self.class_bias(self.drug_features).squeeze(-1)
        return shared + self.bias * self.drug_embedding_mask.squeeze(-1)

    def edge_weights(self) -> Tensor:
        return self.prior + (1 - self.prior) * nn.functional.softplus(self.edge_logit)

    def mechanism_states(self, batch: Batch, node_scale: Tensor | None = None) -> Tensor:
        """(B, M, H) gated sums of token embeddings per mechanism, scaled by 1/sqrt(1 + count).

        ``node_scale`` (N,) multiplies each token embedding; integrated gradients interpolate it from 0 to 1.
        """
        h = self.encoder(batch)
        if node_scale is not None:
            h = h * node_scale.unsqueeze(-1)
        if self.context is not None and batch.edge_index.numel():
            h = h + torch.relu(self.context(h, batch.edge_index))
        n_mech = self.gate.num_embeddings
        if self.assign is None:
            node, mech = batch.pair_node, batch.pair_mech
            weight = torch.sigmoid((h[node] * self.gate(mech)).sum(-1, keepdim=True))
        else:  # dense control: every node to every mechanism with a learned soft assignment
            node = torch.arange(h.shape[0], device=h.device).repeat_interleave(n_mech)
            mech = torch.arange(n_mech, device=h.device).repeat(h.shape[0])
            weight = torch.softmax(self.assign(h), dim=-1).reshape(-1, 1)
        slot = batch.node_graph[node] * n_mech + mech
        total = torch.zeros(batch.size * n_mech, h.shape[1], device=h.device).index_add_(0, slot, weight * h[node])
        count = torch.zeros(batch.size * n_mech, device=h.device).index_add_(0, slot, weight.squeeze(-1))
        return (total / torch.sqrt(1 + count).unsqueeze(-1)).view(batch.size, n_mech, -1)

    def decompose(self, batch: Batch, node_scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return per-mechanism contributions (B, D, M) and pairwise interaction terms (B, D, M, M)."""
        m = self.mechanism_states(batch, node_scale)
        u = torch.einsum("bmh,mhq->bmq", m, self.project)
        c = torch.einsum("bmq,dq->bdm", u, self.queries()) * self.edge_weights().T.unsqueeze(0)
        if self.gamma is None:
            return c, torch.zeros(*c.shape, c.shape[-1], device=c.device)
        s = nn.functional.softplus(self.activation(m)).squeeze(-1)  # (B, M)
        upper = torch.triu(self.gamma, diagonal=1)
        pairs = upper.unsqueeze(0) * (s.unsqueeze(-1) * s.unsqueeze(-2)).unsqueeze(1)  # (B, D, M, M)
        return c, pairs

    def forward(self, batch: Batch, node_scale: Tensor | None = None) -> Tensor:
        c, pairs = self.decompose(batch, node_scale)
        return self.intercepts() + c.sum(-1) + pairs.sum((-1, -2))

    def penalty(self) -> Tensor:
        edges = (nn.functional.softplus(self.edge_logit) * (1 - self.prior)).sum()
        total = self.config.l1_edges * edges + self.config.l2_drug * self.drug_embedding.weight.pow(2).sum()
        if self.gamma is not None:
            total = total + self.config.l1_interactions * torch.triu(self.gamma, diagonal=1).abs().sum()
        return total
