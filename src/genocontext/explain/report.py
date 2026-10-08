"""Interpret finished GenoContext-KG runs.

Per run (``explain``): exact mechanism decomposition, integrated-gradient token attributions mapped to GFF3
locus tags, learned mechanism->drug edges and drug-query similarity.
Across runs (``candidates``): the evidence ladder for putative determinants — stability across lineage-blocked
fits, differential to the shuffled-label control, association conditional on known determinants, AMRFinderPlus
status. Candidates are hypotheses requiring functional validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from torch import Tensor

from genocontext.config import Config
from genocontext.data.cohort import Cohort, read_partition
from genocontext.evaluation.stats import benjamini_hochberg
from genocontext.features.lexicon import Knowledge
from genocontext.features.profile import load_profile
from genocontext.features.space import FeatureSpace
from genocontext.features.store import GROUPS, NODE_STATES, ProfileStore
from genocontext.models.kg import AMRModel, Batch, GenoContextKG
from genocontext.pipeline import REGISTRY, Task, make_learner
from genocontext.training.fit import NeuralLearner, collate

LOG = logging.getLogger(__name__)
TOP_NODES = 25
LOSS_STATES = {NODE_STATES.index(s) for s in ("truncated", "is_disrupted")}


def integrated_gradients(model: AMRModel, batch: Batch, n_drugs: int, steps: int = 16) -> Tensor:
    """(N, D) attributions of each token to each drug logit; per isolate they sum to logit(1) - logit(0)."""
    model.eval()
    total = torch.zeros(len(batch.token), n_drugs)
    for alpha in (torch.arange(steps) + 0.5) / steps:  # midpoint rule from the all-zero baseline
        scale = torch.full((len(batch.token),), float(alpha), requires_grad=True)
        logits = model(batch, scale)
        for d in range(n_drugs):
            (grad,) = torch.autograd.grad(logits[:, d].sum(), scale, retain_graph=d < n_drugs - 1)
            total[:, d] += grad
    return total / steps


def _rebuild(config: Config, task: Task, cohort: Cohort, knowledge: Knowledge, store: ProfileStore
             ) -> tuple[FeatureSpace, NeuralLearner, np.ndarray, list[dict[str, Tensor]]]:
    run_dir = task.directory(config.paths.artifacts)
    spec = REGISTRY[task.model]
    if spec.kind != "kg":
        raise ValueError("Interpretation is defined for GenoContext-KG runs")
    part = read_partition(config.paths.splits, task.scheme, task.seed, task.drug).reindex(store.isolates)
    train = np.flatnonzero(part["outer"].to_numpy() == "train")
    test = np.flatnonzero(part["outer"].to_numpy() == "test")
    space = FeatureSpace.fit(store, train, config.features, spec.drop_groups)
    saved = pd.read_csv(run_dir / "feature_blocks.tsv.gz", sep="\t")["block"].tolist()
    if saved != space.blocks:
        raise ValueError(f"Refitted feature space differs from the saved one: {run_dir}")
    labels = cohort.labels.loc[store.isolates].to_numpy(dtype=float)
    learner = make_learner(spec, space, store, labels, knowledge, cohort.drugs, config, task.seed)
    assert isinstance(learner, NeuralLearner)
    states = torch.load(run_dir / "models.pt", weights_only=True)
    return space, learner, test, states


def explain_run(config: Config, task: Task, cohort: Cohort, knowledge: Knowledge, store: ProfileStore) -> Path:
    space, learner, test, states = _rebuild(config, task, cohort, knowledge, store)
    out = task.directory(config.paths.artifacts) / "explain"
    out.mkdir(exist_ok=True)
    drugs, mechs = cohort.drugs, knowledge.mechanism_names
    labels = cohort.labels.loc[store.isolates].to_numpy(dtype=float)
    models = [learner.model(s) for s in states]
    contribution_rows, node_rows, worst = [], [], 0.0
    for start in range(0, len(test), 32):
        rows = test[start:start + 32]
        batch = collate([learner.tensors[i] for i in rows])
        with torch.no_grad():
            parts = [m.decompose(batch) for m in models if isinstance(m, GenoContextKG)]
            c = torch.stack([p[0] for p in parts]).mean(0)  # (B, D, M)
            pairs = torch.stack([p[1].sum((-1, -2)) for p in parts]).mean(0)  # (B, D)
        attr = torch.stack([integrated_gradients(m, batch, len(drugs)) for m in models]).mean(0)
        with torch.no_grad():
            gap = torch.stack([m(batch) - m(batch, torch.zeros(len(batch.token))) for m in models]).mean(0)
        summed = torch.zeros_like(gap).index_add_(0, batch.node_graph, attr)
        worst = max(worst, float((summed - gap).abs().max()))
        contribution_rows += _contributions(rows, store, labels, c, pairs, drugs, mechs)
        node_rows += _top_nodes(rows, batch, attr, learner, space, store, labels, drugs, mechs)
    LOG.info("Integrated-gradient completeness: max |sum(attr) - (logit(1)-logit(0))| = %.2e", worst)
    pd.DataFrame(contribution_rows).to_csv(out / "mechanism_contributions.tsv.gz", sep="\t", index=False)
    _attach_loci(pd.DataFrame(node_rows), space, config).to_csv(out / "node_attributions.tsv.gz", sep="\t",
                                                                index=False)
    _global_tables(models, mechs, drugs, out, pd.DataFrame(contribution_rows))
    return out


def _contributions(rows: np.ndarray, store: ProfileStore, labels: np.ndarray, c: Tensor, pairs: Tensor,
                   drugs: tuple[str, ...], mechs: tuple[str, ...]) -> list[dict[str, object]]:
    out = []
    for b, row in enumerate(rows):
        isolate = store.isolates[row]
        for d, drug in enumerate(drugs):
            label = float(labels[row, d])
            if np.isnan(label):
                continue
            base = {"isolate_id": isolate, "drug": drug, "label": int(label)}
            out += [base | {"mechanism": m, "contribution": float(c[b, d, k])} for k, m in enumerate(mechs)]
            out.append(base | {"mechanism": "interactions", "contribution": float(pairs[b, d])})
    return out


def _top_nodes(rows: np.ndarray, batch: Batch, attr: Tensor, learner: NeuralLearner, space: FeatureSpace,
               store: ProfileStore, labels: np.ndarray, drugs: tuple[str, ...], mechs: tuple[str, ...]
               ) -> list[dict[str, object]]:
    out = []
    offsets = np.cumsum([0] + [len(learner.tensors[r]["token"]) for r in rows])
    for b, row in enumerate(rows):
        graph = learner.tensors[row]
        mech_of = dict(zip(graph["pair_node"].tolist(), graph["pair_mech"].tolist(), strict=False))
        for d, drug in enumerate(drugs):
            label = float(labels[row, d])
            if np.isnan(label):
                continue
            scores = attr[offsets[b]:offsets[b + 1], d]
            for i in torch.argsort(scores.abs(), descending=True)[:TOP_NODES].tolist():
                out.append({"isolate_id": store.isolates[row], "drug": drug, "label": int(label),
                            "token": space.blocks[int(graph["token"][i])],
                            "group": space.block_group[int(graph["token"][i])],
                            "state": NODE_STATES[int(graph["state"][i])], "mechanism": mechs[mech_of.get(i, 0)],
                            "attribution": float(scores[i])})
    return out


def _attach_loci(nodes: pd.DataFrame, space: FeatureSpace, config: Config) -> pd.DataFrame:
    """Map each attributed block back to the GFF3 locus tags of its member tokens in that isolate."""
    members = dict(zip(space.blocks, space.members, strict=True))
    loci = []
    for isolate, group in nodes.groupby("isolate_id", sort=False):
        profile = load_profile(config.paths.profiles / f"{isolate}.json.gz")
        by_token: dict[str, list[str]] = {}
        for node in profile.nodes:
            for token in (node.token, node.family):
                if token:
                    by_token.setdefault(token, []).extend(node.loci)
        for index, token in group["token"].items():
            tags = sorted({t for m in members[token] for t in by_token.get(m, [])})
            loci.append((index, ";".join(tags)))
    return nodes.join(pd.Series(dict(loci), name="loci"))


def _global_tables(models: list[AMRModel], mechs: tuple[str, ...], drugs: tuple[str, ...], out: Path,
                   contributions: pd.DataFrame) -> None:
    kg = [m for m in models if isinstance(m, GenoContextKG)]
    with torch.no_grad():
        weights = torch.stack([m.edge_weights() for m in kg]).mean(0).numpy()
        queries = torch.stack([torch.nn.functional.normalize(m.queries(), dim=-1) for m in kg]).mean(0)
    prior = kg[0].prior.numpy()
    pd.DataFrame([{"mechanism": m, "drug": d, "weight": float(weights[k, j]), "prior": int(prior[k, j])}
                  for k, m in enumerate(mechs) for j, d in enumerate(drugs)]).to_csv(
        out / "edge_weights.tsv", sep="\t", index=False)
    similarity = (queries @ queries.T).numpy()
    pd.DataFrame(similarity, index=list(drugs), columns=list(drugs)).to_csv(out / "drug_query_similarity.tsv",
                                                                           sep="\t")
    summary = contributions.groupby(["drug", "mechanism", "label"])["contribution"].mean().unstack("label")
    summary.columns = [f"mean_contribution_{'R' if str(c) == '1' else 'S'}" for c in summary.columns]
    summary.to_csv(out / "mechanism_summary.tsv", sep="\t")


def candidates(config: Config, scheme: str, protocol: str, knowledge: Knowledge, store: ProfileStore,
               cohort: Cohort, top_k: int = 20) -> pd.DataFrame:
    """Evidence ladder over all explained seeds of ``kg`` (and the ``kg@shuffled_labels`` control).

    The association test reuses labels that also trained the selecting models, so its p/q-values are
    exploratory (hypothesis-generating), not confirmatory.
    """
    runs = config.paths.artifacts / "runs" / scheme

    def stability(model: str) -> pd.Series:
        tops, n = [], 0
        for path in sorted(runs.glob(f"seed*/all/{model}/{protocol}/explain/node_attributions.tsv.gz")):
            nodes = pd.read_csv(path, sep="\t", dtype={"isolate_id": str})
            mean = nodes.groupby(["drug", "token"])["attribution"].apply(lambda s: s.abs().mean())
            tops.append(mean.groupby(level="drug", group_keys=False).nlargest(top_k).index)
            n += 1
        counts = pd.Series([key for idx in tops for key in idx]).value_counts()
        return (counts / max(n, 1)).rename(model)

    members: dict[str, list[str]] = {}
    for path in sorted(runs.glob(f"seed*/all/kg/{protocol}/feature_blocks.tsv.gz")):
        blocks = pd.read_csv(path, sep="\t")
        members.update({b: str(m).split(";") for b, m in zip(blocks["block"], blocks["members"], strict=True)})
    real, control = stability("kg"), stability("kg@shuffled_labels")
    table = pd.DataFrame({"stability": real}).join(control.rename("control_stability")).fillna(0.0)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["drug", "token"])
    table["differential"] = table["stability"] - table["control_stability"]
    table = table[table["stability"] >= 0.8].reset_index()
    tests = [_conditional_association(store, cohort, knowledge, str(drug), members.get(str(token), [str(token)]))
             for drug, token in zip(table["drug"], table["token"], strict=True)]
    table["odds_ratio"], table["exploratory_p"] = zip(*tests, strict=True) if tests else ([], [])
    table["exploratory_q"] = benjamini_hochberg(table["exploratory_p"].fillna(1.0)) if len(table) else []
    table["amrfinder_annotated"] = table["token"].str.startswith(("amr:", "amrfam:"))
    table["tier"] = 1 + (table["exploratory_q"] < 0.05).astype(int) + table["amrfinder_annotated"].astype(int)
    return table.sort_values(["drug", "tier", "differential"], ascending=[True, False, False])


def _conditional_association(store: ProfileStore, cohort: Cohort, knowledge: Knowledge, drug: str,
                             tokens: list[str]) -> tuple[float, float]:
    """Fisher test of block presence vs resistance among isolates lacking any known determinant for ``drug``.

    A block is present when any member token is; a target block is "present" when the locus is lost
    (truncated or IS-disrupted). Unknown tokens (e.g. ``unk:amr``) return NaN.
    """
    index = {t: i for i, t in enumerate(store.tokens)}
    ids = np.array([index[t] for t in tokens if t in index], dtype=np.int64)
    if not len(ids):
        return float("nan"), float("nan")
    is_target = all(t.startswith("target:") for t in tokens)
    prior = knowledge.prior_mask(cohort.drugs)[:, cohort.drugs.index(drug)]
    known_bits = sum(1 << k for k, flag in enumerate(prior) if flag)
    groups = np.array(GROUPS)[store.token_group]
    loss = list(LOSS_STATES)
    present, known = np.zeros(len(store.isolates), bool), np.zeros(len(store.isolates), bool)
    for row in range(len(store.isolates)):
        nodes = store.nodes(row)
        hit = np.isin(store.node_token[nodes], ids) | np.isin(store.node_family[nodes], ids)
        if is_target:
            hit &= np.isin(store.node_state[nodes], loss)
        present[row] = bool(hit.any())
        informative = (groups[store.node_token[nodes]] != "target") | np.isin(store.node_state[nodes], loss)
        known[row] = bool(((store.node_mechs[nodes] & known_bits) > 0)[informative & ~hit].any())
    labels = cohort.labels.loc[store.isolates, drug].to_numpy()
    use = ~known & ~np.isnan(labels)
    table = [[int((present & use & (labels == 1)).sum()), int((present & use & (labels == 0)).sum())],
             [int((~present & use & (labels == 1)).sum()), int((~present & use & (labels == 0)).sum())]]
    odds, p = fisher_exact(table)
    return float(odds), float(p)
