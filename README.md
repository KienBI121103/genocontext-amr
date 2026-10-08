# GenoContext-AMR v2

**GenoContext-KG** is a knowledge-guided graph network for genome-based antimicrobial resistance prediction in *Klebsiella pneumoniae*. The pipeline:

1. Reads Bakta GFF3 and protein FASTA files.
2. Represents each isolate as **gene-state** nodes:
   - AMR alleles
   - porin and regulator integrity (intact / truncated / IS-disrupted / absent)
   - QRDR residues and OmpK36 insertions
   - mobile elements
   - accessory gene families
3. Routes them through 14 curated resistance-mechanism nodes to drug-class-conditioned outputs for 17 antibiotics, in one model.

Every prediction decomposes exactly into mechanism contributions, which trace back to GFF3 locus tags.

- **Methods:** `docs/methods.md`
- **Pre-registered evaluation:** `docs/protocol.md`
- **Run book with all commands:** `experiments/kp/README.md`

## Layout

```text
src/genocontext/          installable package (src layout, typed, mypy --strict)
  config.py               frozen dataclass config; unknown YAML keys are rejected
  data/                   cohort labels/partitions, GFF3 and FAA readers
  features/               knowledge lexicon, target caller, profiles, profile store, training-only feature space
  models/                 GenoContextKG (kg.py); DeepSets, query attention, LightGBM, kNN placement (baselines.py)
  training/fit.py         learners and protocols A (64/16/20) and B (80/20)
  evaluation/             metrics (VME/ME, calibration), statistics, stacking, result tables
  explain/report.py       decomposition, integrated gradients, evidence ladder
  pipeline.py, cli.py     model registry, resumable tasks, `genocontext` command
knowledge/                mechanisms.yaml (curated prior, cited) and pinned reference proteins
experiments/kp/           config.yaml, make_splits.py (standalone split generator), README.md (run book)
tests/                    unit tests plus an end-to-end CLI test on a synthetic cohort
reports/, research_notes/ literature review behind the design (2026-10-08)
```

## Quick start

```bash
uv sync
uv run pytest -q && uv run ruff check src tests experiments && uv run mypy
uv run genocontext --help
```

## GenoContext v1 (RF + GraphSAGE fusion)

v1 is preserved at git tag `v1-fusion-seed0` and runs from its own worktree:

```bash
git worktree add ../GenoContext-AMR-v1 v1-fusion-seed0
cd ../GenoContext-AMR-v1 && uv sync && uv run pytest -q
```

**Known v1 issues** (audit of 8 Oct 2026; details in `reports/`):
- The "masked" mode was not AMR-blind.
- Real vs shuffled gene order differed significantly in only 2 of 17 drugs, so the fusion gain was mostly ensembling.
- The fusion alpha was selected on a small validation set that was also used for the checkpoint and the threshold.
- The upstream AMR-GNN splits put 8 BioSample twin pairs across train and test. 84–92% of test isolates shared an ST with training isolates.
