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
