"""Experiment orchestration: model registry, resumable task runs and the dev-only E0 protocol check."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from genocontext.config import Config
from genocontext.data.cohort import Cohort, partition_path, read_partition
from genocontext.features.lexicon import Knowledge
from genocontext.features.profile import FLAGS
from genocontext.features.space import FeatureSpace
from genocontext.features.store import NODE_STATES, ProfileStore
from genocontext.models.baselines import DeepSets, QueryAttention
from genocontext.models.kg import AMRModel, GenoContextKG
from genocontext.training.fit import (
    KNNLearner,
    Learner,
    LightGBMLearner,
    NeuralLearner,
    Outcome,
    protocol_a,
    protocol_b,
)

LOG = logging.getLogger(__name__)
PIPELINE_VERSION = 2  # bump when feature/model semantics change so stale runs are never reused


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # kg | deepsets | attention | lgbm | knn
    overrides: dict[str, Any] = field(default_factory=dict)
    view: str = "genestate"
    drop_groups: tuple[str, ...] = ()
    shuffle_edges: bool = False
    rewire_prior: bool = False  # control: mechanism->drug prior rows randomly permuted


REGISTRY: dict[str, ModelSpec] = {
    "kg": ModelSpec("kg"),
    "kg_dense": ModelSpec("kg", {"use_prior": False}),
    "kg_no_context": ModelSpec("kg", {"context_layer": False}),
    "kg_shuffled_context": ModelSpec("kg", shuffle_edges=True),
    "kg_no_interactions": ModelSpec("kg", {"interactions": False}),
    "kg_free_drug": ModelSpec("kg", {"drug_features": False}),
    "kg_no_target_res": ModelSpec("kg", drop_groups=("target", "res")),
    "kg_rewired": ModelSpec("kg", rewire_prior=True),
    "deepsets": ModelSpec("deepsets"),
    "query_attention": ModelSpec("attention"),
    "lgbm_genestate": ModelSpec("lgbm"),
    "lgbm_presence": ModelSpec("lgbm", view="presence"),
    "knn_placement": ModelSpec("knn", view="presence"),
}


@dataclass(frozen=True)
class Task:
    scheme: str
    seed: int
    model: str
    protocol: str  # "A" (64/16/20) or "B" (80/20)
    drug: str | None = None  # tier-1 target drug
    variant: str = ""  # "", "rep<r>", "shuffled_labels", "zeroshot=<drug>", "n<count>-<drug>-r<rep>"

    def directory(self, artifacts: Path) -> Path:
        model = f"{self.model}@{self.variant}" if self.variant else self.model
        return artifacts / "runs" / self.scheme / f"seed{self.seed}" / (self.drug or "all") / model / self.protocol


@dataclass(frozen=True)
class Variant:
    labels: NDArray[np.float64]  # training labels after the variant's modification
    zero_shot: str | None
    train_seed: int


def apply_variant(variant: str, labels: NDArray[np.float64], train: NDArray[np.int64], drugs: tuple[str, ...],
                  seed: int) -> Variant:
    """Modify *training* labels for control and low-data experiments; test labels are never touched."""
    out, rng = labels.copy(), np.random.default_rng(seed + 7)
    if not variant:
        return Variant(out, None, seed)
    if match := re.fullmatch(r"rep(\d+)", variant):
        return Variant(out, None, seed + 1000 * int(match.group(1)))
    if variant == "shuffled_labels":
        for d in range(len(drugs)):
            rows = train[~np.isnan(out[train, d])]
            out[rows, d] = rng.permutation(out[rows, d])
        return Variant(out, None, seed)
    if match := re.fullmatch(r"zeroshot=(\w+)", variant):
        out[train, drugs.index(match.group(1))] = np.nan
        return Variant(out, match.group(1), seed)
    if match := re.fullmatch(r"n(\d+)-(\w+)-r(\d+)", variant):
        d = drugs.index(match.group(2))
        rows = train[~np.isnan(out[train, d])]
        keep = np.random.default_rng(int(match.group(3))).permutation(rows)[:int(match.group(1))]
        out[np.setdiff1d(rows, keep), d] = np.nan
        return Variant(out, None, seed)
    raise ValueError(f"Unknown variant: {variant}")


def load_store(config: Config, cohort: Cohort, knowledge: Knowledge) -> ProfileStore:
    path = config.paths.artifacts / "profile_store"
    if not path.with_suffix(".npz").exists():
        LOG.info("Building profile store for %d isolates", len(cohort.labels))
        path.parent.mkdir(parents=True, exist_ok=True)
        ProfileStore.build(config.paths.profiles, list(cohort.labels.index), knowledge.mechanism_names).save(path)
    store = ProfileStore.load(path)
    if store.isolates != list(cohort.labels.index) or store.mechanisms != knowledge.mechanism_names:
        raise ValueError(f"Profile store {path} does not match the cohort/knowledge; delete it to rebuild")
    return store


def make_learner(spec: ModelSpec, space: FeatureSpace, store: ProfileStore, labels: NDArray[np.float64],
                 knowledge: Knowledge, drugs: tuple[str, ...], config: Config, seed: int,
                 zero_shot: str | None = None) -> Learner:
    if spec.kind == "lgbm":
        return LightGBMLearner(space.matrix(store, np.arange(len(store.isolates)), spec.view), labels,
                               config.training.threads)
    if spec.kind == "knn":
        return KNNLearner(space.matrix(store, np.arange(len(store.isolates)), spec.view), labels)
    model_config = replace(config.model, **spec.overrides)
    shuffle = seed if spec.shuffle_edges else None
    graphs = [space.graph(store, r, config.features.colocalization_k, shuffle) for r in range(len(store.isolates))]
    sizes = (space.n_tokens, len(NODE_STATES), len(FLAGS))
    drug_features = torch.from_numpy(knowledge.drug_features(drugs))
    prior = torch.from_numpy(knowledge.prior_mask(drugs))
    if spec.rewire_prior:
        prior = prior[torch.from_numpy(np.random.default_rng(12345).permutation(prior.shape[0]))]

    def build_kg() -> AMRModel:
        model = GenoContextKG(*sizes, drug_features, prior, model_config)
        if zero_shot is not None:  # the held-out drug is predicted from its class features alone
            model.drug_embedding_mask[drugs.index(zero_shot)] = 0.0
        return model

    builders: dict[str, Callable[[], AMRModel]] = {
        "kg": build_kg,
        "deepsets": lambda: DeepSets(*sizes, len(drugs), model_config),
        "attention": lambda: QueryAttention(*sizes, len(drugs), model_config),
    }
    return NeuralLearner(builders[spec.kind], graphs, labels, config.training)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _long(isolates: list[str], drugs: tuple[str, ...], probs: NDArray[np.float64], labels: NDArray[np.float64],
          part: str) -> pd.DataFrame:
    def melt(values: NDArray[np.float64], name: str) -> pd.DataFrame:
        wide = pd.DataFrame(values, columns=list(drugs)).assign(isolate_id=isolates)
        return wide.melt(id_vars="isolate_id", var_name="drug", value_name=name)

    long = melt(probs, "probability").merge(melt(labels, "label"), on=["isolate_id", "drug"])
    return long.dropna(subset=["probability", "label"]).assign(part=part)


def run_task(config: Config, task: Task, cohort: Cohort, knowledge: Knowledge, store: ProfileStore) -> Path:
    out = task.directory(config.paths.artifacts)
    split_file = partition_path(config.paths.splits, task.scheme, task.seed, task.drug)
    stamp = {"task": asdict(task), "spec": asdict(REGISTRY[task.model]), "config": _config_dict(config),
             "partition_sha256": _sha256(split_file), "knowledge_sha256": knowledge.sha256,
             "pipeline_version": PIPELINE_VERSION}
    if (out / "task.json").exists():
        if json.loads((out / "task.json").read_text()) != stamp:
            raise ValueError(f"Existing results were produced differently: {out}")
        return out
    torch.set_num_threads(config.training.threads)
    spec, drugs = REGISTRY[task.model], cohort.drugs
    part = read_partition(config.paths.splits, task.scheme, task.seed, task.drug).reindex(store.isolates)
    labels = cohort.labels.loc[store.isolates].to_numpy(dtype=float)
    if task.drug and spec.kind in ("lgbm", "knn"):  # per-drug learners only need the target drug
        labels = np.where(np.array(drugs) == task.drug, labels, np.nan)
    train = np.flatnonzero(part["outer"].to_numpy() == "train")
    test = np.flatnonzero(part["outer"].to_numpy() == "test")
    inner = part["inner_fold"].to_numpy()[train].astype(int)
    variant = apply_variant(task.variant, labels, train, drugs, task.seed)
    space = FeatureSpace.fit(store, train, config.features, spec.drop_groups)
    learner = make_learner(spec, space, store, variant.labels, knowledge, drugs, config, task.seed, variant.zero_shot)
    LOG.info("%s: %d train / %d test isolates, %d feature blocks", out, len(train), len(test), space.n_tokens)
    seed = variant.train_seed
    outcome = (protocol_a(learner, variant.labels, train, inner, test, seed) if task.protocol == "A" else
               protocol_b(learner, variant.labels, train, inner, test, seed, config.training.ensemble_seeds))
    _save(out, task, outcome, store, train, test, labels, drugs, space)
    (out / "task.json").write_text(json.dumps(stamp, indent=1, sort_keys=True, default=str))
    return out


def _save(out: Path, task: Task, outcome: Outcome, store: ProfileStore, train: NDArray[np.int64],
          test: NDArray[np.int64], labels: NDArray[np.float64], drugs: tuple[str, ...], space: FeatureSpace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keep = [drugs.index(task.drug)] if task.drug else list(range(len(drugs)))
    kept = tuple(drugs[d] for d in keep)
    isolates = np.array(store.isolates)
    frames = [_long(list(isolates[test]), kept, outcome.test[:, keep], labels[test][:, keep], "test"),
              _long(list(isolates[train]), kept, outcome.selection[:, keep], labels[train][:, keep], "selection")]
    predictions = pd.concat(frames).dropna(subset=["label", "probability"])
    predictions["threshold"] = predictions["drug"].map(dict(zip(drugs, outcome.thresholds, strict=True)))
    predictions.to_csv(out / "predictions.tsv.gz", sep="\t", index=False)
    (out / "selection.json").write_text(json.dumps({"thresholds": dict(zip(drugs, outcome.thresholds.tolist(),
                                                                           strict=True)),
                                                    "budget": outcome.budget.tolist()}, indent=1))
    pd.DataFrame({"block": space.blocks, "group": space.block_group,
                  "members": [";".join(m) for m in space.members]}).to_csv(out / "feature_blocks.tsv.gz",
                                                                           sep="\t", index=False)
    if isinstance(outcome.fitted[0], dict):
        torch.save(outcome.fitted, out / "models.pt")


def _config_dict(config: Config) -> dict[str, Any]:
    plain: dict[str, Any] = json.loads(json.dumps(asdict(config), default=str))
    return plain


# ---- E0: does training on train ∪ val (80/20) beat 64/16/20? decided on development data only -------
def e0_inner(part: pd.DataFrame, held: int) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64],
                                                     NDArray[np.int64]]:
    """Within the outer-training set, hold out inner fold ``held``; return (train, A-folds, B-folds, test)."""
    folds = part["inner_fold"].to_numpy()
    train_mask = (part["outer"].to_numpy() == "train") & (folds != held) & (folds >= 0)
    test = np.flatnonzero((part["outer"].to_numpy() == "train") & (folds == held))
    train = np.flatnonzero(train_mask)
    remaining = sorted(set(folds[train].tolist()))
    a_folds = np.where(folds[train] == _e0_validation_fold(part, held, remaining), 0, 1)  # A: 1 fold validates
    b_folds = np.searchsorted(remaining, folds[train])  # B: 4-fold inner CV over all remaining folds
    return train, a_folds, b_folds, test


def _e0_validation_fold(part: pd.DataFrame, held: int, remaining: list[int], big: float = 0.05) -> int:
    """Protocol A validates on fold 0 as in the main runs; when fold 0 is held out, use the remaining fold whose
    largest group is smallest (main-run fold 0 never contains groups above ``big`` of the cohort)."""
    if held != 0:
        return 0
    share = part["group"].value_counts(normalize=True)
    worst = {f: float(share[part.loc[part["inner_fold"] == f, "group"]].max()) for f in remaining}
    return min(remaining, key=lambda f: (worst[f] > big, worst[f], f))


def run_e0(config: Config, scheme: str, model: str, cohort: Cohort, knowledge: Knowledge,
           store: ProfileStore) -> pd.DataFrame:
    out = config.paths.artifacts / "e0" / scheme / model
    if (out / "predictions.tsv.gz").exists():
        return pd.read_csv(out / "predictions.tsv.gz", sep="\t", dtype={"isolate_id": str})
    torch.set_num_threads(config.training.threads)
    spec, drugs = REGISTRY[model], cohort.drugs
    part = read_partition(config.paths.splits, scheme, 0).reindex(store.isolates)
    labels = cohort.labels.loc[store.isolates].to_numpy(dtype=float)
    frames = []
    for held in range(5):
        train, a_folds, b_folds, test = e0_inner(part, held)
        space = FeatureSpace.fit(store, train, config.features, spec.drop_groups)
        learner = make_learner(spec, space, store, labels, knowledge, drugs, config, held)
        for protocol, outcome in (("A", protocol_a(learner, labels, train, a_folds, test, held)),
                                  ("B", protocol_b(learner, labels, train, b_folds, test, held, 1))):
            frame = _long(list(np.array(store.isolates)[test]), drugs, outcome.test, labels[test], "test")
            frame["threshold"] = frame["drug"].map(dict(zip(drugs, outcome.thresholds, strict=True)))
            frames.append(frame.dropna(subset=["label"]).assign(protocol=protocol, held_fold=held))
        LOG.info("E0 %s/%s fold %d done", scheme, model, held)
    result = pd.concat(frames)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "predictions.tsv.gz", sep="\t", index=False)
    return result
