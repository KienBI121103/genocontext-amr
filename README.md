# GenoContext-AMR

**GenoContext-GNN** predicts binary antimicrobial resistance from gene-neighborhood graphs built from Bakta GFF3. The reusable implementation lives in `src/`; generic commands live in `scripts/`; the *Klebsiella pneumoniae* study is launched from `experiments/kp/run.py`.

## Layout

```text
configs/kp.yaml       Kp paths and model settings
experiments/kp/run.py  Kp preflight, training, and metric aggregation
scripts/              Generic preprocessing, training, and evaluation commands
src/                  Annotation, data, parsing, features, graph, model, training, evaluation, explanation
tests/                Synthetic checks and a small end-to-end run
NAS Kp_collected/    Annotation cache, checkpoints, predictions, metrics, explanations
```

## Environment

From /home/kiennm/AMR/AMR_GNN/GenoContext-AMR:

```bash
uv sync --python 3.11.7
uv run pytest -q
```

This creates this project's own `.venv`. The current host has no working NVIDIA driver, so `pyproject.toml` installs CPU PyTorch. The training code also supports CUDA when used in an environment with a CUDA build of PyTorch.

## Kp experiment: run in order

All Kp artifacts are written directly to `/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected` on the NAS; the project .venv remains in the project directory. The commands below use eight preprocessing workers and eight PyTorch intra-op threads, while keeping BLAS at one thread per process. Bakta remains capped at four threads. Run one command at a time. The complete study covers 17 antibiotics and seeds 0–9. The Kp split files are inputs; this project does not import or compare model results from other runs.

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
uv run python -m experiments.kp.run preflight
uv run python -m experiments.kp.run run --seed 0 --antibiotic ciprofloxacin
uv run python -m experiments.kp.run run
uv run python -m experiments.kp.run summarize
```

`preflight` verifies labels, split IDs, the shared fixed test cohort, and all required annotations before full training. Existing valid GFF3 files are reused; Bakta runs only when a GFF3 is missing. An invalid existing GFF3 stops the run. The focused seed-0 command is an operational check; the full command resumes completed model outputs and matching fitted vocabularies.

The model selects a checkpoint and one binary threshold by validation F1. `/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/results/` contains validation and test predictions, F1, AUROC, AUPRC and other metrics, plus GFF3-mapped gene attributions. `/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/results/metrics.csv` aggregates completed model metrics. Raw and AMR-term-masked features, real and shuffled within-contig edges, logistic regression, random forest, and DeepSets are part of this study.

## Generic commands

These accept a compatible manifest, binary phenotype table, and train/val/test split directory:

```bash
uv run python -m scripts.preprocess --config configs/kp.yaml --split-root /path/to/splits --antibiotic ciprofloxacin
uv run python -m scripts.train --config configs/kp.yaml --split-root /path/to/splits --antibiotic ciprofloxacin --seed 0 --feature-mode masked
uv run python -m scripts.evaluate --artifacts /mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected
```

The manifest may use `isolate_id,gff3_path,genome_path` or `id,assembly_path`. Phenotypes may use long `isolate_id,antibiotic,label` rows with `R`/`S`, or the Kp wide 0/1 table. FFN, FAA, and FNA files are not model inputs.

## RF–GNN fusion pilot: run in order

**GenoContext-GNN-Fusion** combines the existing random forest with an independently trained GraphSAGE branch. The graph readout concatenates mean pooling, max pooling, and `log1p(gene_count)/10`. Neural checkpoints maximize validation AUROC; a 21-point graph-weight search and resistant-class F1 threshold selection use validation only. Ties prefer zero/smaller graph weight. The final test receives the frozen checkpoint, weight, and threshold. Zero weight reproduces RF and is a valid outcome; improved test performance is not guaranteed.

Run these commands from the project directory, sequentially:

```bash
cd /home/kiennm/AMR/AMR_GNN/GenoContext-AMR
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c 0-63 uv run python -m experiments.kp.run preflight --config configs/kp_fusion.yaml
taskset -c 0-63 uv run python -m experiments.kp.run run --config configs/kp_fusion.yaml --seed 0
taskset -c 0-63 uv run python -m experiments.kp.run run --config configs/kp_fusion.yaml --seed 1 --seed 2
uv run python -m experiments.kp.run summarize --config configs/kp_fusion.yaml
```

The configuration runs ciprofloxacin only, seeds 0–2, both raw and masked features. Its five outputs are RF, real/shuffled improved GNN, and real/shuffled fusion. It shares the existing NAS parsed-GFF cache. All new predictions, checkpoints, and summaries go to `/mnt/nas/earth/KienNM/genecontext-gnn/Kp_collected/fusion_v1`; existing results remain intact. Jobs run sequentially, with eight neural training threads, 32 RF threads, 16 annotation/feature preprocessing workers limited to one PyTorch thread each, and automatic CPU/CUDA selection. `taskset -c 0-63` restricts the complete job and inherited workers to 64 logical CPUs on this host; OS thread counts may include additional idle helper threads. A representative 16-isolate batch on this host took 1.07 seconds at eight training threads, 1.16 at 16, and 1.12 at 32; these timings informed the neural thread setting, but do not measure a full-run speedup. No new environment or dependency installation is required if `uv sync` has already completed.

Within the new output directory:

- `results/metrics.csv`: validation/test F1, AUROC, AUPRC, specificity, precision, recall, balanced accuracy, MCC, log loss, confusion counts, and probability spread.
- `results/fusion_vs_rf.csv`: per-seed F1 and other metric differences, branch score spread, and RF errors corrected/introduced.
- `results/fusion_summary.csv`: means and standard deviations across seeds. These seeds share a test cohort; they are not independent test datasets.
- `results/fusion_gate.json`: full-run eligibility, using only real-edge validation results for ciprofloxacin seeds 0–2.
- Per-run fusion predictions: RF/GNN/fused probabilities, graph weight, and threshold; selection JSON and all 21 validation search rows.
- Per-run GNN histories: training loss, validation metrics, score spread, and selected epoch. Models include persisted RF and GNN checkpoints.
- Per-run attributions: GFF3-mapped **graph-branch** explanations, marked with the graph weight and whether that branch contributes. They do not explain the complete RF–GNN prediction.

Rerunning a pilot command resumes compatible components and reuses fitted vocabularies/models. Resume checks exact split IDs, labels, annotation path/size/modification time, settings, source version, and vocabulary signature. Component signatures tie predictions to saved checkpoints; interrupted exports are regenerated. Changed inputs/settings require a new output directory. An interrupted individual neural fit restarts; completed branch checkpoints are reused.

The full experiment is a separate command:

```bash
taskset -c 0-63 uv run python -m experiments.kp.run run --config configs/kp_fusion.yaml --full
```

It requires complete pilot artifacts and a feature mode with strictly better validation F1 in at least two of three seeds and positive mean improvement. The gate is recomputed and pilot compatibility checked before training. This launches both feature modes across all 17 antibiotics and seeds 0–9, resuming the pilot groups. If the gate fails, the command stops; retain RF and inspect the graph branch. Previously inspected ciprofloxacin test results make this an exploratory pilot.

## Biological interpretation from saved fusion models

The post-hoc command loads the completed RF/GNN models and exports original GFF3 gene IDs, coordinates, nearby genes, graph salience, RF feature rankings, and candidate annotation keywords. It does not fit a new model or select a new threshold. Explanations are written to `fusion_v1/interpretation/seed0/ciprofloxacin/<feature-mode>/<cohort>/` on the NAS, outside the completed training artifacts.

Start with RF–fusion disagreements, then extend to every test isolate:

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
taskset -c 0-63 uv run python -m scripts.explain --config configs/kp_fusion.yaml --antibiotic ciprofloxacin --seed 0 --feature-mode raw --cohort discordant --top-genes 20 --neighborhood-k 5
taskset -c 0-63 uv run python -m scripts.explain --config configs/kp_fusion.yaml --antibiotic ciprofloxacin --seed 0 --feature-mode masked --cohort discordant --top-genes 20 --neighborhood-k 5
taskset -c 0-63 uv run python -m scripts.explain --config configs/kp_fusion.yaml --antibiotic ciprofloxacin --seed 0 --feature-mode raw --cohort all --top-genes 20 --neighborhood-k 5
taskset -c 0-63 uv run python -m scripts.explain --config configs/kp_fusion.yaml --antibiotic ciprofloxacin --seed 0 --feature-mode masked --cohort all --top-genes 20 --neighborhood-k 5
```

`--cohort discordant` selects RF errors corrected by fusion and new errors introduced by fusion; `errors` selects all fusion mistakes; `all` explains all test isolates. `--limit N` provides a small smoke run and `--threads` controls neural explanation threads (default eight). Run jobs sequentially within the 64-CPU cap.

Read `isolate_cases.csv`, `gene_summary.csv`, `candidate_annotations.csv`, `gene_neighborhoods.csv`, and `rf_global_features.csv` together. Neighborhood radius controls exported context, not model edges. Salience is unsigned gradient × initial node state, not a causal effect or complete fusion explanation. RF impurity rankings are global, not isolate-specific. Top-gene recurrence rates are descriptive, not gene-presence enrichment; the discordant/error subsets are selected by outcomes. GFF3-only inputs do not identify quinolone target substitutions or establish expression, plasmid location, or causality. Masked explanations display original annotations for mapping even though the model received masked features.
