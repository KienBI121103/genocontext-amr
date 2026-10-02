"""Post-hoc gene/neighborhood exports from completed RF–GNN fusion checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.inputs import RecordStore, Split, parsed_cache_dir, read_manifest, read_phenotypes
from src.explain.attribution import attribute_nodes
from src.features.gene_features import FIELDS, GeneFeatureEncoder
from src.graph.builder import build_graph
from src.model.fusion import align_probabilities
from src.model.graphsage import GenoContextGNN
from src.training.fusion import validate_fusion_run
from src.training.pipeline import write_csv
from src.training.train import select_device

LOG = logging.getLogger(__name__)
QUINOLONE_SCREEN = r"qnr|gyr[ab]|par[ce]|aac.*ib.*cr|oqx|qep|acr[ab]|tolc|efflux"
CONTEXT_SCREEN = r"transpos|integras|insertion sequence|plasmid|beta[- ]?lactamase|\bbla[a-z0-9_-]*"
NUMERIC_FEATURES = (
    "mean_log_gene_length", "mean_strand", "mean_relative_position", "mean_previous_gap",
    "mean_next_gap", "previous_overlap_fraction", "next_overlap_fraction",
    "previous_same_strand_fraction", "next_same_strand_fraction", "log_gene_count",
)


def read_predictions(path: Path, expected_ids: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"isolate_id": str}, float_precision="round_trip")
    frame = frame[frame["split"] == "test"].copy()
    align_probabilities(expected_ids, tuple(frame["isolate_id"]), frame["probability_R"].to_numpy())
    return frame.set_index("isolate_id").loc[list(expected_ids)]


def case_table(rf: pd.DataFrame, fusion: pd.DataFrame) -> pd.DataFrame:
    if not rf.index.equals(fusion.index) or not rf["true_label"].equals(fusion["true_label"]):
        raise ValueError("RF/fusion isolate IDs or labels differ")
    rf_correct = rf["predicted_label"] == rf["true_label"]
    fusion_correct = fusion["predicted_label"] == fusion["true_label"]
    frame = fusion.copy()
    frame["case_group"] = np.select(
        [~rf_correct & fusion_correct, rf_correct & ~fusion_correct, rf_correct & fusion_correct],
        ["rf_corrected", "error_introduced", "both_correct"], default="both_wrong",
    )
    frame["rf_prediction"] = rf["predicted_label"]
    frame["fusion_prediction"] = fusion["predicted_label"]
    return frame.reset_index()


def rf_feature_names(encoder: GeneFeatureEncoder) -> list[str]:
    names = ["numeric::" + name for name in NUMERIC_FEATURES]
    for field in FIELDS:
        vocabulary = encoder.vocabularies[field]
        names.extend(field + "::" + token for token, _index in sorted(vocabulary.items(), key=lambda item: item[1]))
    return names


def summarize_genes(attributions: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    columns = ["gene", "product", "n_isolates_top", "n_resistant_top", "n_susceptible_top",
               "mean_importance", "max_importance", "resistant_top_rate", "susceptible_top_rate"]
    if attributions.empty:
        return pd.DataFrame(columns=columns)
    focal = attributions[attributions["neighbor_distance"] == 0].copy()
    focal[["gene", "product"]] = focal[["gene", "product"]].fillna("")
    # Multiple copies of one annotation in an isolate count once in recurrence rates.
    per_isolate = focal.groupby(["gene", "product", "isolate_id", "true_label"], as_index=False).agg(
        importance_score=("importance_score", "max"))
    per_isolate["resistant"] = per_isolate["true_label"].eq("R").astype(int)
    per_isolate["susceptible"] = per_isolate["true_label"].eq("S").astype(int)
    summary = per_isolate.groupby(["gene", "product"], as_index=False).agg(
        n_isolates_top=("isolate_id", "nunique"), n_resistant_top=("resistant", "sum"),
        n_susceptible_top=("susceptible", "sum"), mean_importance=("importance_score", "mean"),
        max_importance=("importance_score", "max"))
    n_r, n_s = int(cases["true_label"].eq("R").sum()), int(cases["true_label"].eq("S").sum())
    summary["resistant_top_rate"] = summary["n_resistant_top"] / n_r if n_r else np.nan
    summary["susceptible_top_rate"] = summary["n_susceptible_top"] / n_s if n_s else np.nan
    return summary.sort_values(["n_isolates_top", "mean_importance"], ascending=False)


def explain_run(config: dict, antibiotic: str, seed: int, feature_mode: str, cohort: str,
                top_genes: int = 20, neighborhood_k: int = 5, limit: int | None = None,
                output_root: Path | None = None, threads: int = 8) -> Path:
    if top_genes < 1 or neighborhood_k < 0 or (limit is not None and limit < 1):
        raise ValueError("top_genes/limit must be positive and neighborhood_k nonnegative")
    if threads < 1 or threads > 64:
        raise ValueError("threads must be between 1 and 64")
    if cohort not in {"discordant", "errors", "all"}:
        raise ValueError("cohort must be discordant, errors, or all")
    torch.set_num_threads(threads)
    root = Path(config["paths"]["artifacts"])
    relative = Path(f"seed{seed}") / antibiotic / feature_mode
    run = root / "results" / relative
    details = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["details"]
    split = Split(**{part: tuple(ids) for part, ids in details["splits"].items()})
    labels = read_phenotypes(config["paths"]["phenotypes"])
    store = RecordStore(read_manifest(config["paths"]["manifest"]), parsed_cache_dir(config),
                        Path(config["bakta"]["db"]), config["bakta"]["threads"])
    LOG.info("Checking completed model signatures and exact cohorts")
    validate_fusion_run(config, store, labels, split, antibiotic, seed, feature_mode)
    rf_predictions = read_predictions(run / "random_forest_predictions.csv", split.test)
    fusion_predictions = read_predictions(run / "genocontext_fusion_real_predictions.csv", split.test)
    cases = case_table(rf_predictions, fusion_predictions)
    if cohort == "discordant":
        cases = cases[cases["case_group"].isin(["rf_corrected", "error_introduced"])]
    elif cohort == "errors":
        cases = cases[cases["true_label"] != cases["fusion_prediction"]]
    if limit is not None:
        cases = cases.head(limit)
    cases = cases.copy()
    output = (output_root if output_root is not None else root / "interpretation") / relative / cohort
    output.mkdir(parents=True, exist_ok=True)
    write_csv(cases, output / "isolate_cases.csv")

    encoder = GeneFeatureEncoder(mask_amr_terms=feature_mode == "masked",
                                 min_category_frequency=config["features"]["min_category_frequency"])
    encoder.vocabularies = json.loads((run / "vocabularies.json").read_text(encoding="utf-8"))
    device = select_device()
    model = GenoContextGNN(encoder.category_sizes, config["model"]["hidden_dim"],
                           config["model"]["dropout"], config["model"].get("pooling", "mean_max_count"))
    checkpoint = root / "models" / relative / "genocontext_gnn_real.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    model.to(device).eval()
    LOG.info("Explaining %d test isolates (%s, %s) on %s", len(cases), feature_mode, cohort, device)
    attribution_rows = []
    for index, case in enumerate(cases.to_dict("records"), 1):
        sample = case["isolate_id"]
        records = store.get(sample)
        graph = build_graph(records, encoder, labels[(sample, antibiotic)], config["graph"]["neighbor_k"])
        rows = attribute_nodes(model, graph, records, device, top_n=top_genes, neighborhood_k=neighborhood_k)
        if rows and not np.isclose(rows[0]["probability_R"], case["probability_gnn"], atol=1e-6, rtol=1e-4):
            raise ValueError(f"Checkpoint probability differs from saved prediction for {sample}")
        annotations = {(record.contig_id, record.feature_id): record for record in records}
        for row in rows:
            record = annotations[(row["contig"], row["feature_id"])]
            row.update(true_label=case["true_label"], rf_prediction=case["rf_prediction"],
                       fusion_prediction=case["fusion_prediction"], case_group=case["case_group"],
                       probability_rf=case["probability_rf"], probability_fused=case["probability_fused"],
                       alpha=case["alpha"], feature_mode=feature_mode, antibiotic=antibiotic, seed=seed,
                       explanation_scope="graph_branch", direction="unsigned_magnitude",
                       strand=record.strand, db_xrefs=";".join(record.db_xrefs),
                       ec_numbers=";".join(record.ec_numbers), gff3_path=str(store.gff_path(sample)))
        attribution_rows.extend(rows)
        if index % 10 == 0 or index == len(cases):
            LOG.info("Explained %d/%d isolates", index, len(cases))
    empty_columns = ["isolate_id", "contig", "feature_id", "gene", "product", "neighbor_distance", "importance_score"]
    attributions = pd.DataFrame(attribution_rows) if attribution_rows else pd.DataFrame(columns=empty_columns)
    write_csv(attributions, output / "gene_neighborhoods.csv")
    summary = summarize_genes(attributions, cases)
    write_csv(summary, output / "gene_summary.csv")
    annotation_text = summary["gene"].fillna("") + " " + summary["product"].fillna("")
    screened = summary.copy()
    screened["quinolone_keyword_match"] = annotation_text.str.contains(QUINOLONE_SCREEN, flags=re.I, regex=True)
    screened["mobile_or_other_amr_keyword_match"] = annotation_text.str.contains(CONTEXT_SCREEN, flags=re.I, regex=True)
    screened = screened[screened["quinolone_keyword_match"] | screened["mobile_or_other_amr_keyword_match"]]
    write_csv(screened, output / "candidate_annotations.csv")
    rf = joblib.load(root / "models" / relative / "random_forest.joblib")
    feature_names = rf_feature_names(encoder)
    if len(feature_names) != len(rf.feature_importances_):
        raise ValueError("RF feature names do not match fitted model dimensions")
    importances = pd.DataFrame({"feature": feature_names, "impurity_importance": rf.feature_importances_})
    write_csv(importances.sort_values("impurity_importance", ascending=False), output / "rf_global_features.csv")
    metadata = {"antibiotic": antibiotic, "seed": seed, "feature_mode": feature_mode,
                "cohort": cohort, "n_explained": len(cases), "limit": limit, "top_genes": top_genes,
                "neighborhood_k": neighborhood_k, "threads": threads,
                "allowed_cpu_count": len(os.sched_getaffinity(0)),
                "case_counts": cases["case_group"].value_counts().to_dict(),
                "checkpoint": str(checkpoint), "retrained": False,
                "salience": "normalized absolute gradient times initial node state",
                "interpretation": "candidate associations, not causal resistance mechanisms"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "README.md").write_text(
        "# Post-hoc gene-context interpretation\n\n"
        f"{antibiotic}, seed {seed}, {feature_mode}, cohort {cohort}: {len(cases)} isolates.\n\n"
        "- gene_neighborhoods.csv: salient genes and nearby genes within each contig, with original GFF3 IDs and coordinates. Neighborhood radius is for display, not a new model edge setting.\n"
        "- gene_summary.csv: recurrence of top-ranked annotations and mean/max salience. R/S rates refer to top-ranked annotations among selected isolates, not gene-presence enrichment or statistical significance.\n"
        "- candidate_annotations.csv: keyword screening for quinolone targets, efflux, mobile elements, and other AMR annotations; no sequence-based AMR calling is performed.\n"
        "- rf_global_features.csv: RF impurity feature importance; global, unsigned, potentially affected by correlated features. This is not a local explanation.\n"
        "- isolate_cases.csv: corrected/introduced/both-correct/both-wrong classifications and frozen probabilities.\n\n"
        "Graph salience is normalized absolute gradient × initial node state. It cannot determine whether a gene increases or decreases resistance, is not a causal effect, and does not explain the complete fused score or the graph's direct gene-count input.\n\n"
        "Masked-model outputs display original GFF3 names for mapping; the model received the masked feature representation, not those original AMR names.\n\n"
        "The discordant/error cohorts are selected by outcomes and are unsuitable for population enrichment claims. Use all test isolates for descriptive R/S comparisons and independent validation for biological conclusions. GFF3-only inputs cannot establish gyrA/parC resistance substitutions, gene expression, plasmid location, or causality. A missing keyword hit does not establish absence of a determinant. Beta-lactamase importance for a quinolone phenotype may reflect co-resistance or lineage.\n\n"
        "Biological references: [qnrB](https://pubmed.ncbi.nlm.nih.gov/16569827/), "
        "[AAC(6')-Ib-cr](https://pubmed.ncbi.nlm.nih.gov/16369542/), "
        "[Kp GyrA substitutions](https://pmc.ncbi.nlm.nih.gov/articles/PMC2493132/), "
        "[SHV-28](https://card.mcmaster.ca/ontology/37466).\n"
    )
    LOG.info("Saved interpretation exports to %s", output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Extract biological candidates from completed fusion models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--antibiotic", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-mode", choices=("raw", "masked"), required=True)
    parser.add_argument("--cohort", choices=("discordant", "errors", "all"), default="discordant")
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--neighborhood-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Output root; seed/antibiotic/feature/cohort appended")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_interop_threads(1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    explain_run(config, args.antibiotic, args.seed, args.feature_mode, args.cohort,
                args.top_genes, args.neighborhood_k, args.limit, args.output, args.threads)


if __name__ == "__main__":
    main()
