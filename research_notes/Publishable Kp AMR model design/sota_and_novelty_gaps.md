# State of the art and novelty gaps: genome-based AMR prediction in K. pneumoniae (2019–2026)

These notes condense the research output of 8 Oct 2026. Preprints are labelled. Every DOI should be re-verified before citing.

## PanKA (the LightGBM reference method)
- Do VH et al., *iScience* 27(9), 2024, DOI 10.1016/j.isci.2024.110623 (PMC11369404).
- **Pan-genome:** PanTA on Prokka annotations.
- **Features:**
  - gene presence/absence
  - amino-acid variants of pan-genes (label-encoded alignment columns)
  - protein 10-mers from AMR gene clusters
  - chi² selection keeps the top 2,000 features per block per antibiotic
- **Model:** LightGBM, tuned with GridSearchCV on mean 5-fold F1.
- **Data and evaluation:**
  - *K. pneumoniae* from PATRIC: 17 antibiotics, 1,612–2,239 isolates per drug
  - random 80/20 train/test split
  - ST-stratified evaluation only for *E. coli*
  - Kp results appear only in figures, and **no AUROC is reported**
- **Headline result (*E. coli*):** mean F1 0.893 vs 0.856 for PanPred.
- **Interpretability:** limited to the top-10 features by gain or split count.
- **Cost:** 25 min and 3.6 GB on CPU for Kp.

## Methods published 2022–2026

| Method | Input / architecture | K. pneumoniae evidence | Split | Notes |
|---|---|---|---|---|
| AMR-GNN (*Nat Commun* 17:3555, 2026, 10.1038/s41467-026-69934-8) | Unitig node features; isolate-similarity graphs (SNP Hamming distance + FCGR); dual GCN | BV-BRC Kp n=7,072; ceftriaxone AUROC 0.985 (0.981–0.990); VME 0.008–0.133, ME 0.054–0.336 | Stratified random 80/20 × 10 | MLST decoupling tested only in *P. aeruginosa*; requires a GPU |
| GL-HAT (bioRxiv 2025, preprint) | ESM-2 650M protein embeddings in genome order; hierarchical attention; per-antibiotic query | Multi-taxon, 53 antibiotics: AUROC 0.953 (max-pool) vs 0.931 (attention), XGBoost 0.950 | Jaccard ≥0.9 cluster split | Attention readout lost to mean pooling; requires a GPU |
| Bacformer (bioRxiv 2025, preprint) | Protein-sequence genome transformer | AMR claimed in the abstract, but no AMR numbers could be found | — | 4× A100 for 2 weeks |
| DeepMDC (Front Cell Infect Microbiol 2026 / bioRxiv 2025) | Attention multiple-instance learning (MIL) over an unordered bag of ORFs (Hopfield attention) | Kp meropenem AUROC 0.93, ceftazidime 0.88, gentamicin 0.89, cefepime 0.83; attention highlights OmpK36, Tn4401 and ISKpn11 | 10:1 + CV, not lineage-blocked | Gene order is not needed |
| GeneBac (bioRxiv 2024) | Gene CNN + STRING-edge GAT, multi-task, MIC | MTB and *P. aeruginosa* only | Lineage holdout (MTB) | — |
| Kervancı TabTransformer+CatBoost (*Bioinform Adv* 2026, vbag193) | 195 AMR markers | Kp meropenem, 554 unique genomes: clade split AUROC 0.867, external 0.81 | Clade-aware | Random-split model collapsed externally (recall 22%) |
| Cheon et al. RF (*Brief Bioinform* 2026, bbag451) | 111 engineered features (ARG, expression proxies, mutations) | Kp 1,824 genomes, F1 0.873 | Stratified vs ANI-blocked: F1 0.876 → 0.753 | Kp has the highest "silent gene" share of the species studied (23.9%) |
| Xie et al. (*mBio* 2025, 10.1128/mbio.02852-24) | Gene presence + IS presence + IS–ARG pairs | *A. baumannii* | Random 70/30 | No ablation with and without IS features |
| Kover / SCM (Drouin 2016, 2019) | k-mer rules | 12 species, 56 antibiotics | — | Highly interpretable, but misses rare determinants |
| Nguyen 2018 (*Sci Rep* 8:421) | 10-mer XGBoost, MIC | Kp 1,668 isolates, 20 antibiotics; 92% within ±1 dilution | 10-fold CV | — |

## Lineage confounding
- Yu, Wheeler & Barquist, *PLOS Biol* 23(12):e3003539 (2025), 10.1371/journal.pbio.3003539:
  - Based on more than 24,000 genomes from 5 species, including Kp.
  - Clade-biased training sharply lowers AUC.
  - Top features overlap little between clades.
  - The authors recommend phylogeny-aware CV, lineage-aware ML and a phylogenetic-placement baseline.
- Random-split Kp AUROCs of 0.93–0.99 are essentially all from random splits.
- **No lineage-blocked, 17-drug, per-antibiotic Kp benchmark exists.**

## Ranked novelty angles (CPU-feasible, GFF3 + FAA)
1. **Gene-integrity / IS-disruption / residue features** (porin loss, ramR/acrR IS, QRDR). No one has used these as ML inputs for Kp.
2. **A lineage-blocked Kp benchmark** with a placement baseline. This answers the PLOS Biology call.
3. **Gene×drug sharing across drugs with low-data rescue.** Multi-task models for Kp exist only for TB, not Kp.
4. **A quantitative attribution-faithfulness audit** with lineage controls.
5. **Allele-resolved protein language model embeddings** of pan-genome clusters. GL-HAT partly pre-empts this.
6. **Conformal or calibrated prediction under lineage shift.**

## Local evidence (GenoContext v1, seed 0, 17 antibiotics)
- **Mean test AUROC:**
  - RF 0.925
  - GNN (real edges) 0.926
  - GNN (shuffled edges, masked mode) 0.910
  - fusion (real) 0.938
  - fusion (shuffled) 0.935
- **Paired bootstrap:**
  - fusion > RF in 10 of 17 antibiotics
  - real > shuffled edges in 2 of 17 (aztreonam, TMP-SMX)
- **Split issues:**
  - 8 BioSamples have two genome IDs each, and up to 5 such pairs cross the split per drug.
  - 84–92% of test isolates share an ST with training isolates.
  - train∪val and test are identical across AMR-GNN seeds 0–9.
- **The "masked" mode is not AMR-blind:** the product vocabulary is the same size in both modes.
