# Pre-registered evaluation protocol (GenoContext v2)

**Status:** draft. Freeze this file with `git commit`, record the hashes below, and only then run any step that scores a test set (run book step 4 onwards). After freezing, change nothing here without a dated amendment note.

| Item | Value |
|---|---|
| `experiments/kp/config.yaml` sha256 | _fill in after E0_ |
| `knowledge/mechanisms.yaml` sha256 | _fill in after E0_ |
| `knowledge/references/kp_targets.faa` sha256 | `82c9af84…` (13 targets after removing MgrB; re-hash) |
| `splits_v2/split_protocol.json` | created 2026-10-08, 385 clonal groups |
| Chosen protocol (E0) | _A or B_ |

## Data and splits

- **Data:** 3,949 *K. pneumoniae* genomes (Bakta GFF3 + FAA) and 17 antibiotics. Labels are the AMR-GNN `ast_labels.csv` (0 = S, 1 = R). Document the breakpoint version and how intermediate (I) results were handled when they are known.
- **Tier 1, `amrgnn_fixed` (descriptive only):**
  - AMR-GNN test sets kept verbatim; protocol B trains on train∪val.
  - The training pool for each target drug excludes BioSample twins of that drug's test isolates.
  - train∪val is identical across AMR-GNN seeds 0–9, so seeds measure optimisation variance only.
  - No endpoint uses tier 1.
- **Tier 2, `random_grouped`:**
  - BioSample-deduplicated (3,941 genomes).
  - Balanced 5-fold × 2 repeats, giving seeds 0–9.
- **Tier 3, `cg_blocked` (headline):**
  - The same builder, over clonal groups (single-locus-variant stars around large STs).
  - Train and test share no clonal group.
- **Inner folds** follow the same grouping.
  - Protocol A validates on inner fold 0.
  - Clonal groups larger than 5% of isolates never serve as the protocol-A validation fold.

## E0: protocol choice, decided without test data

- **Universe:** the seed-0 outer-training sets of tiers 2 and 3.
- **Procedure:** each inner fold is held out once.
  - Protocol A trains on 3 folds and validates on 1.
  - Protocol B trains on 4 folds, using a nested 4-fold CV for epochs, rounds and thresholds, then a single refit.
- **Models:** GenoContext-KG, LightGBM-gene-state, LightGBM-presence.
- **Rule:** adopt B for all main runs if, for every scheme × model:
  - mean ΔAUROC ≥ 0 and mean ΔAUPRC ≥ 0 (Δ = B − A, over held fold × drug pairs), and
  - the bootstrap 95% CI lower bounds of Δ balanced accuracy and Δ F1 are both ≥ −0.01.
- Expected effect from learning-curve literature: ΔAUROC ≈ 0.002–0.01.

## Models (fixed before testing)

- **GenoContext-KG:** `kg`, with the configuration in `config.yaml`.
- **Ablations:**
  - `kg_dense`, `kg_rewired`
  - `kg_no_target_res`, `kg_no_context`, `kg_shuffled_context`
  - `kg_no_interactions`, `kg_free_drug`
- **Baselines:**
  - `deepsets` (mean pooling)
  - `query_attention` (GL-HAT-style)
  - `lgbm_genestate`
  - `lgbm_presence` (PanKA-like)
  - `knn_placement` (lineage placement)
- **Stacked variant:** `kg+lgbm_genestate`, a logistic stacker on selection/out-of-fold predictions.
- **Hyperparameters:** a single fixed configuration. Selection only chooses epochs or boosting rounds (early stopping) and thresholds (maximum balanced accuracy).

## Endpoints (tier 3 unless stated; macro = equal weight per drug)

| ID | Comparison | Metric | Test |
|---|---|---|---|
| P1 | `kg` vs `lgbm_presence` | macro AUROC (also AUPRC, balanced accuracy) | corrected repeated 5-fold × 2 t-test (Bouckaert & Frank); clonal-group cluster bootstrap of pooled OOF predictions |
| P2 | `kg` vs `kg_dense`, `deepsets`, `kg_rewired` | macro AUROC | as P1, BH within family |
| P3 | low-data: subsampling curves (n = 100–1000) and zero-shot drugs (cefuroxime, levofloxacin, tobramycin) | AUROC, AUPRC (MCC secondary; zero-shot thresholds default to 0.5 and the intercept comes from class features) | mean ± SD over repeats; Δ vs `lgbm_genestate` at each n |
| P4 | `kg` vs `kg_no_target_res`; `lgbm_genestate` vs `lgbm_presence` | macro AUROC | as P1 |

Per-drug results are reported with BH-adjusted q-values. Tier 2 is reported alongside tier 3; the difference between them measures lineage confounding.

**Additional reporting:**
- VME and ME with Wilson 95% CIs on pooled OOF counts, judged against the FDA criteria: VME upper CL ≤ 7.5% and lower CL ≤ 1.5%; ME ≤ 3%.
- Brier score, calibration slope and calibration intercept.

**Decision gate:** if `kg` is inferior to `lgbm_genestate` on tier-3 macro AUROC, the paper says so. It then presents `kg+lgbm_genestate` as the performance model and `kg` as the interpretable model.

## Interpretation validity

- Explanations are averaged over refits and seeds.
- Every explanation is compared with the `shuffled_labels` control (differential stability) and the `kg_rewired` prior.
- **Evidence ladder:**
  1. In the top 20 attributions in at least 80% of tier-3 seeds.
  2. Fisher association with resistance among isolates lacking known determinants for that drug (BH q < 0.05).
     These p/q-values are **exploratory**: the same labels trained the selecting models.
  3. AMRFinderPlus annotation status.
- Candidates are reported as hypotheses requiring functional validation.

## Implementation notes (code review, 2026-10-08)

- **Collinear blocks:** only positively correlated tokens merge (r ≥ 0.98). Mutually exclusive alleles stay separate.
- **E0, protocol A:** validates on inner fold 0, as in the main runs. When fold 0 is held out, it uses the remaining fold with the smallest largest-group share.
- **Tier 1:** BioSample twins of validation isolates are kept inside fold 0.
- **Run stamps:** every run records `pipeline_version` (currently 2), so results from older semantics are never reused.
