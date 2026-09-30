from __future__ import annotations

import torch

from src.features.gene_features import ordered_genes
from src.parsing.gff import GeneRecord


def attribute_nodes(
    model: torch.nn.Module, graph, records: list[GeneRecord],
    device: torch.device, top_n: int = 5, neighborhood_k: int = 2,
) -> list[dict[str, object]]:
    """Gradient × initial node state, mapped to original GFF3 features."""
    records = ordered_genes(records)
    if len(records) != graph.num_nodes:
        raise ValueError("Graph node order does not match GFF3 records")
    model.eval()
    graph = graph.to(device)
    with torch.enable_grad():
        logit, node_state = model(graph, return_node_state=True)
        gradient = torch.autograd.grad(logit[0], node_state)[0]
        scores = (gradient * node_state).sum(dim=1).abs().detach().cpu()
    if scores.sum() > 0:
        scores /= scores.sum()
    focal = torch.topk(scores, min(top_n, len(records))).indices.tolist()
    selected: dict[int, tuple[int, int]] = {}
    for rank, index in enumerate(focal, 1):
        selected[index] = (rank, 0)
        for offset in range(1, neighborhood_k + 1):
            for other in (index - offset, index + offset):
                if 0 <= other < len(records) and records[other].contig_id == records[index].contig_id:
                    selected.setdefault(other, (rank, offset))
    rows = []
    for index in sorted(selected):
        record = records[index]
        rank, distance = selected[index]
        rows.append({
            "isolate_id": record.sample_id, "contig": record.contig_id,
            "gene": record.gene, "product": record.product,
            "start": record.start, "end": record.end,
            "feature_id": record.feature_id, "importance_score": float(scores[index]),
            "focal_rank": rank, "neighbor_distance": distance,
            "probability_R": float(torch.sigmoid(logit[0]).detach().cpu()),
        })
    return rows
