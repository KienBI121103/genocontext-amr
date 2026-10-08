# K. pneumoniae run book (GenoContext v2)

Run every command from the repository root, one at a time, in your own terminal. Use tmux/screen yourself if you want to detach; nothing here starts background jobs.
- Threads: 8 per process, BLAS at 1. Each process stays within 8 cores; `taskset` pins it.
- Outputs go to `/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/v2/` (the `artifacts` path in `config.yaml`).
- Tasks are resumable. A finished task, marked by its `task.json`, is skipped. A task that was produced with different settings raises an error instead of being overwritten.

```bash
cd /home/kiennm/AMR/AMR_GNN/GenoContext-AMR
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv sync                                   # once
uv run pytest -q && uv run ruff check src tests experiments && uv run mypy   # sanity: 30 tests pass
```

## Steps 0–2: inputs (already done on 2026-10-08)

| Step | Command | Output |
|---|---|---|
| 0 Splits (standalone, not core) | `uv run python experiments/kp/make_splits.py --out /mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/splits_v2` | `splits_v2/{amrgnn_fixed,random_grouped,cg_blocked}/seed0..9/`, `manifest.tsv`, `clonal_groups.tsv`, `split_protocol.json` |
| 1 Check inputs | `uv run genocontext preflight` | log only |
| 2 Gene-state profiles (about 5 min with 64 workers) | `taskset -c 0-63 uv run genocontext profiles` | `v2/profiles/*.json.gz`, `v2/assembly_qc.tsv` |

The profile store (`v2/profile_store.npz`) is built automatically on first use. This one-time build takes about 11 minutes. To rebuild the splits, first delete `splits_v2/`; the script refuses to overwrite it.

## Step 3: E0, does 80/20 (protocol B) beat 64/16/20 (protocol A)?

E0 uses only the seed-0 outer-training set and never touches any test set. Run one command per line, in any order. Each takes 8 cores. Approximate times: LightGBM 10–30 min, GenoContext-KG 2–6 h per scheme.

```bash
taskset -c 0-7   uv run genocontext e0 --scheme cg_blocked     --model lgbm_genestate lgbm_presence
taskset -c 8-15  uv run genocontext e0 --scheme random_grouped --model lgbm_genestate lgbm_presence
taskset -c 16-23 uv run genocontext e0 --scheme cg_blocked     --model kg
taskset -c 24-31 uv run genocontext e0 --scheme random_grouped --model kg
uv run genocontext compare                      # writes v2/e0/e0_decision.json and e0_summary.tsv
cat /mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/v2/e0/e0_decision.json | head -5
```

The decision rule is fixed in `docs/protocol.md`. If `adopt_protocol_B` is `true`, use `--protocol B` everywhere below; otherwise use `--protocol A`. **Then freeze the protocol:** record the config hash, and `git commit` `docs/protocol.md` before running any step below.

```bash
sha256sum experiments/kp/config.yaml knowledge/mechanisms.yaml knowledge/references/kp_targets.faa
```

## Step 4: main runs (each test set is scored once)

`P=B` below; set `P=A` if E0 chose protocol A. Seeds can run in parallel, each with its own 8-core range.
- GenoContext-KG takes about 50 min per split under B (5 inner fits + 5 refits).
- LightGBM takes a few minutes per split.

```bash
P=B
# Tier 3 (headline, clonal-group blocked) and tier 2 (random, BioSample-grouped): 10 seeds each
taskset -c 0-7   uv run genocontext run --scheme cg_blocked random_grouped --seed 0 1 2 3 4 --protocol $P \
                 --model kg lgbm_genestate lgbm_presence knn_placement
taskset -c 8-15  uv run genocontext run --scheme cg_blocked random_grouped --seed 5 6 7 8 9 --protocol $P \
                 --model kg lgbm_genestate lgbm_presence knn_placement
# Tier 1 (AMR-GNN split, descriptive; per target drug). train∪val is identical across AMR-GNN seeds,
# so seed 0 suffices for protocol B; run both protocols for the six-method table.
taskset -c 16-23 uv run genocontext run --scheme amrgnn_fixed --seed 0 --protocol B --model kg lgbm_genestate
taskset -c 24-31 uv run genocontext run --scheme amrgnn_fixed --seed 0 --protocol A --model kg lgbm_genestate
# Peak-performance variant: stack KG + LightGBM (fitted on out-of-fold/selection predictions only)
uv run genocontext stack --scheme cg_blocked random_grouped --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P \
       --model kg lgbm_genestate
uv run genocontext stack --scheme amrgnn_fixed --seed 0 --protocol B --model kg lgbm_genestate
```

## Step 5: ablations and controls (tier 3; add `random_grouped` if time allows)

```bash
taskset -c 0-7   uv run genocontext run --scheme cg_blocked --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P \
                 --model kg_dense deepsets query_attention kg_rewired
taskset -c 8-15  uv run genocontext run --scheme cg_blocked --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P \
                 --model kg_no_target_res kg_no_context kg_shuffled_context kg_no_interactions kg_free_drug
# Shuffled-label control (for differential attributions)
taskset -c 16-23 uv run genocontext run --scheme cg_blocked --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P \
                 --model kg --variant shuffled_labels
```

## Step 6: low-data experiments (P3)

```bash
# Zero-shot: the drug's training labels are hidden; KG predicts it from drug-class features
taskset -c 0-7 uv run genocontext run --scheme cg_blocked --seed 0 1 2 3 4 --protocol $P --model kg \
               --variant zeroshot=cefuroxime zeroshot=levofloxacin zeroshot=tobramycin
# Label subsampling: 3 drugs × n ∈ {100, 250, 500, 1000} × 5 repeats ("all" = the main runs)
for d in gentamicin ciprofloxacin ceftazidime; do
  V=""; for n in 100 250 500 1000; do for r in 0 1 2 3 4; do V="$V n$n-$d-r$r"; done; done
  taskset -c 8-15 uv run genocontext run --scheme cg_blocked --seed 0 --protocol $P \
                  --model kg lgbm_genestate --variant $V
done
```

## Step 7: interpretation (GenoContext-KG runs only; inference only)

```bash
taskset -c 0-7 uv run genocontext explain --scheme cg_blocked --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P --model kg
taskset -c 0-7 uv run genocontext explain --scheme cg_blocked --seed 0 1 2 3 4 5 6 7 8 9 --protocol $P --model kg \
               --variant shuffled_labels
uv run genocontext candidates --scheme cg_blocked --protocol $P
```

Each run's `explain/` directory contains:
- `mechanism_contributions.tsv.gz`: the exact logit decomposition
- `node_attributions.tsv.gz`: integrated gradients mapped to GFF3 locus tags
- `edge_weights.tsv`: learned and prior mechanism→drug edges
- `drug_query_similarity.tsv`
- `mechanism_summary.tsv`

`results/candidates_cg_blocked_<P>.tsv` is the evidence ladder.

## Step 8: tables

```bash
uv run genocontext compare
```

`v2/results/` will contain:
- `metrics.tsv`
- `seed_summary.tsv`
- `endpoints_protocol{A,B}.tsv`: P1, P2 and P4, with the corrected repeated K-fold t-test and BH correction
- `cluster_bootstrap.tsv`: clonal-group bootstrap of pooled out-of-fold predictions
- `six_methods_tier1.tsv`
- `low_data.tsv`
