from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.gene_features import GeneFeatureEncoder
from src.parsing.gff import GeneRecord


def aggregate_genomes(
    sample_ids: tuple[str, ...], get_records: Callable[[str], list[GeneRecord]],
    encoder: GeneFeatureEncoder,
) -> sparse.csr_matrix:
    return sparse.vstack(
        [encoder.genome_vector(get_records(sample)) for sample in sample_ids],
        format="csr",
    )


def random_forest(seed: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs,
        random_state=seed, min_samples_leaf=2,
    )


def fit_baselines(
    train_x: sparse.csr_matrix, train_y: np.ndarray,
    seed: int, n_jobs: int,
) -> dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed,
        ),
        "random_forest": random_forest(seed, n_jobs),
    }
    for model in models.values():
        model.fit(train_x, train_y)
    return models
