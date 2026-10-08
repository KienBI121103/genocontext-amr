I read the researcher instructions and ran 17 search/fetch calls. Per plan mode and the task brief, nothing was written to disk — complete notes follow inline.

---

# Multi-task / drug-aware AMR models and publication-grade evaluation for K. pneumoniae

## (A1) Published multi-label / multi-task / drug-aware AMR models: inputs, architectures, gains, code

### Takeaway
Drug-aware sharing is **not new in concept** — "antibiotic as an input feature" has been standard in genome→MIC models since Nguyen et al. 2018, and an explicit *molecular-structure* drug representation for K. pneumoniae was already published in Scientific Reports in 2024. What is genuinely under-explored is a **factorised gene×drug / learned drug-embedding model benchmarked specifically for low-data drug rescue**, with an ablation against per-drug models — because the two best existing head-to-head tests of naive multi-output sharing both found **no significant gain** (p = 0.374 and p = 0.48).

### Cited Findings

**Drug-as-feature shared models (closest prior art, K. pneumoniae)**
- Nguyen et al. (2018, *Scientific Reports*) trained **one XGBoost regression model over 1,668 K. pneumoniae clinical genomes for 20 antibiotics**, using overlapping 10-mer counts merged with the antibiotic identity as features — i.e. a single shared model with the drug as an input, not 20 per-drug models. Accuracy 92% within ±1 two-fold dilution; ≥90% for 15/20 antibiotics; **average raw (exact-match) accuracy only 69%** under 10-fold CV — [Scientific Reports 41598-017-18972-w](https://www.nature.com/articles/s41598-017-18972-w); [PMC5765115](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5765115/); [preprint PDF](https://www.biorxiv.org/content/10.1101/193797.full.pdf)
- **Direct precedent for a drug-embedding model on K. pneumoniae:** "Integrating genomic and molecular data to predict antimicrobial minimum inhibitory concentration in Klebsiella pneumoniae" (2024, *Scientific Reports*) uses a **CNN over 10-mer genomic features plus molecular-structure data for the 20 drugs**, trained on 1,667 genomes / **32,312 genome–antibiotic pairs**, reporting a **~20% increase in raw accuracy** over the Nguyen 2018 state of the art while matching its 1-tier (±1 dilution) accuracy — [Scientific Reports 41598-024-75973-2](https://www.nature.com/articles/s41598-024-75973-2) (full text was behind an IDP redirect; figures above are from the indexed abstract and [ResearchGate record](https://www.researchgate.net/publication/385354808_Integrating_genomic_and_molecular_data_to_predict_antimicrobial_minimum_inhibitory_concentration_in_Klebsiella_pneumoniae))
- MIC censoring matters for any shared-head regression model: a separate study on 4,367 genomes notes MICs are "semi-quantitative, with varying resolution, and typically also left- and right-censored within varying ranges" — [PMC10995625 / "Optimising machine learning prediction of minimum inhibitory concentrations in Klebsiella pneumoniae"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10995625/)

**Multi-output / multi-species shared encoders (genomic)**
- Aytan-Aktug et al. (2020, *mSystems*) trained neural networks on ResFinder/PointFinder marker features across **3,528 M. tuberculosis + 1,694 E. coli + 658 S. enterica + 1,236 S. aureus** isolates. They explicitly compared **per-antibiotic models vs. one aggregated multi-output model** and found **no significant difference: P = 0.374 for both random forests and neural networks** — [mSystems 10.1128/msystems.00774-19](https://journals.asm.org/doi/10.1128/msystems.00774-19); [PMC6977075](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6977075/)
- The independent Briefings in Bioinformatics benchmark re-ran this: **multi-antibiotic vs. single-antibiotic models, P = 0.48 (not significant)**; whereas **leave-one-species-out (cross-species) models were dramatically worse than single-species models, P between 1e-13 and 1e-19** — [Brief Bioinform 25(3):bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)

**Transfer learning from a data-rich drug to data-poor drugs (the low-data-rescue evidence)**
- Ren et al. (2022, *Antibiotics*): 12-layer 1D CNN on one-hot SNP matrices (E. coli, 809 in-house strains; 1,509 public strains as external validation). The **ciprofloxacin model's weights were transferred** to cefotaxime, ceftazidime and gentamicin with two normalisation layers and one conv layer frozen. Internal test MCC gains: **CTX 0.47→0.56, CTZ 0.46→0.55, GEN 0.33→0.53**. On the imbalanced external public set (resistance 18/8/5/7%): **MCC CTX 0.06→0.41, CTZ 0.08→0.29, GEN 0.11→0.26**; **AUROC CTX 0.74→0.87 and CTZ 0.79→0.86 (significant), GEN 0.69→0.72 (not significant)**. Note the honest negative: for the *source* drug CIP, AUROC fell 0.93→0.89. Code: [github.com/YunxiaoRen/deep_transfer_learning_AMR](https://github.com/YunxiaoRen/deep_transfer_learning_AMR) — [PMC9686617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9686617/)
- The same paper **did not test a genuinely novel/unseen antibiotic**; gains were demonstrated only for existing antibiotics with imbalanced labels — [PMC9686617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9686617/)

**Multi-label from non-genomic inputs (useful as framing, not as competition)**
- Astudillo et al., multi-label AMR from MALDI-TOF spectra (DRIAMS-A training, external validation on DRIAMS-B/C), covering *K. pneumoniae*–ciprofloxacin and *K. pneumoniae*–ceftriaxone among others; the tabulated result is **multi-label ≈ single-label**, with external generalisation confirmed — cited in a comparison table in [Bioinformatics Advances vbaf303](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbaf303/8341552) (secondary citation; original not fetched)
- Multi-task EHR model across **nine antibiotic classes**, 59,551 patients, three Korean tertiary hospitals: hard-parameter-sharing ranked first for 5/9 classes on external validation (mean AUC ≈ 79.6, mean AUPRC ≈ 80.3); soft sharing won when prior culture results were missing. Motivation explicitly framed as **handling partial/incomplete label matrices** — [PMC13066429](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066429/); [PubMed 41775804](https://pubmed.ncbi.nlm.nih.gov/41775804/)
- "Personalized Antibiogram" multitask framework predicts resistance to several antimicrobials simultaneously, reporting reduced false-negative rates for lower-prevalence carbapenem resistance — [Clin Infect Dis 10.1093/cid/ciag027](https://doi.org/10.1093/cid/ciag027)
- A K. pneumoniae antibiogram-based RF study states the mechanistic rationale for label sharing: resistance phenotypes are "highly correlated due to shared mechanisms (e.g. ESBL or carbapenemase activity) and plasmid-mediated co-transmission of genes" — [Diagnostics 10.3390/diagnostics16040555](https://doi.org/10.3390/diagnostics16040555); [PMC12939074](https://pmc.ncbi.nlm.nih.gov/articles/PMC12939074/)

**Zero-shot / unseen-drug generalisation**
- ApexOracle (2025 preprint) combines **genomic embeddings of the target strain with a molecule representation** and, without fine-tuning on strain-specific small-molecule data, "matched or outperformed two of the four fine-tuned baseline models"; ablation showed **removing genomic embeddings caused the sharpest performance drop** — [arXiv 2507.07862](https://arxiv.org/html/2507.07862). Its task is antibacterial activity, not clinical AST phenotype.
- BaCNet encodes compounds as **Morgan fingerprints + Chemical Checker signatures + ChemBERTa-2 vectors** and proteins with ESM-2, trained on STITCH records restricted to 23 drug-resistant species — but the output is binding affinity, not an AST phenotype, and no zero-shot unseen-antibiotic evaluation is shown — [bioRxiv 2025.02.26.640477](https://www.biorxiv.org/content/10.1101/2025.02.26.640477v1.full)

**Comparator landscape for your paper**
- PanKA extracts a concise feature set from the population pangenome and trains **LightGBM**, benchmarked on E. coli and K. pneumoniae, claiming higher accuracy plus faster training than conventional and state-of-the-art tools — [PMC11369404](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11369404/)
- Independent benchmark comparator set is Kover, PhenotypeSeeker, Seq2Geno2Pheno, Aytan-Aktug, a majority-class baseline, and rule-based ResFinder 4.0 — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- A 2026 K. pneumoniae WGS study built **multi-level** models (R/S, R/I/S, and high/low-level resistance) for **11 antibiotics across 5,239 strains in three cohorts**, reporting binary R/S AUC > 0.9 for all 11 and mean categorical agreement 0.96 — [PubMed 41997433](https://pubmed.ncbi.nlm.nih.gov/41997433/)
- Condorelli et al., K. pneumoniae AMR from gene content and genome composition — [PLOS ONE 10.1371/journal.pone.0309333](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0309333)

### Inferences
- **Novelty positioning:** "A single model over 17 antibiotics" alone is *not* novel (Nguyen 2018; Aytan-Aktug 2020), and "drug represented by chemistry" is *not* novel for K. pneumoniae (Sci Rep 2024). The defensible novelty claims are (i) an explicitly **factorised gene×drug bilinear/low-rank interaction** rather than drug-as-concatenated-feature, (ii) **systematic low-data-drug rescue as the primary endpoint** with per-drug gain plotted against per-drug label count, and (iii) **evaluation under lineage-blocked splits**, which no multi-task AMR paper found here has done.
- The two existing head-to-head tests (p = 0.374, p = 0.48) are the strongest reason a reviewer will be sceptical. You should **pre-empt them by citing both and explaining why your factorisation differs** (shared low-rank gene-effect basis with per-drug loadings vs. a shared trunk with independent heads), and by showing the gain is concentrated where label counts are lowest (nitrofurantoin 1,296) rather than averaged across all 17.
- Because the Ren et al. transfer-learning gains were largest **on the external, imbalanced validation set** and small or negative internally, the right headline for a low-data-rescue claim is **external-set MCC/AUPRC**, not internal AUROC.
- Classifier chains that consume other drugs' *observed phenotypes* at test time are not comparable to a genotype-only model, since at deployment an isolate has no AST results. If you include a chain baseline, run it twice: once with oracle sibling labels (upper bound) and once with *predicted* sibling labels (deployable).

### Gaps
- I found **no published AMR model using ATC codes or CARD ARO drug-class ontology terms as drug embeddings**. If that is your representation, it appears unclaimed — but I could not confirm absence, only that targeted searches returned nothing.
- I found **no genome-based AMR paper that evaluates true zero-shot transfer to a held-out antibiotic**. ApexOracle is the nearest, and its task is activity prediction, not AST phenotype.
- Could not retrieve the full text of the 2024 Sci Rep drug-structure CNN (publisher auth redirect), so its exact drug encoding (SMILES vs. descriptors), its split design, and whether it has code are unverified. **This is the single most important paper for you to read in full** before writing a novelty claim.
- The Astudillo multi-label MALDI result is a secondary citation from a comparison table; the primary paper was not retrieved.

---

## (A2) Does multi-task learning actually improve AUROC/F1 over per-drug models in bacteria?

### Takeaway
On the published evidence, **naive multi-task sharing gives no significant average improvement in bacterial genomic AMR** (two independent tests, p = 0.374 and p = 0.48). The improvements that *are* documented come from **transfer/initialisation from a data-rich drug** and show up mainly as MCC/AUPRC gains on imbalanced external data, not as average AUROC gains.

### Cited Findings
- Aytan-Aktug et al.: aggregated multi-output model vs. per-antimicrobial models, **P = 0.374** for both RF and NN — [mSystems 10.1128/msystems.00774-19](https://journals.asm.org/doi/10.1128/msystems.00774-19)
- Independent benchmark: **multi-antibiotic vs. single-antibiotic, P = 0.48**; cross-species (LOSO) sharing was significantly *harmful*, P = 1e-13 to 1e-19 — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- Transfer learning did help under imbalance: external-set MCC rose from 0.06→0.41 (CTX), 0.08→0.29 (CTZ), 0.11→0.26 (GEN); AUROC 0.74→0.87 and 0.79→0.86 significant for CTX/CTZ, 0.69→0.72 not significant for GEN — [PMC9686617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9686617/)
- Multi-label vs. single-label on MALDI-TOF spectra: **approximately equal** — [Bioinformatics Advances vbaf303](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbaf303/8341552)
- Multi-task in the EHR domain *did* win, but only partially: hard sharing best in **5 of 9** antibiotic classes on external validation — [PMC13066429](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066429/)
- A meta-model across **6,740 trained models** found species and drug class strongly affect achievable AUC, and **Gram-negative training data gave lower AUC** overall — [PLOS Biol 23(12):e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)

### Inferences
- The honest expected effect size is **small on average and heterogeneous by drug**. Design your statistics accordingly: power for a *per-drug, low-data-subset* comparison, not an all-17 average, and report per-drug deltas with CIs rather than a single mean.
- Reviewers who know bbae206 will ask "is this just Aytan-Aktug's multi-output model again?" Having the **p = 0.48 result reproduced on your own data as a baseline** (shared-trunk/independent-heads MTL) and then beaten by your factorised variant is the most convincing possible framing.
- Because multi-task benefit should scale inversely with label count, the strongest single figure in your paper is **Δmetric vs. n_labelled per drug, with a fitted trend**, plus a label-subsampling experiment (artificially down-sample gentamicin from 3,864 to 1,296 and show the gap opens).

### Gaps
- No source quantifies MTL gain **as a function of per-drug sample size** in bacterial AMR. This is an open question your paper can own.

---

## (B1) Split design: lineage blocking, deduplication, seeds, fixed vs. resampled test sets

### Takeaway
The 2025 PLOS Biology paper has made **phylogeny-aware/clade-held-out cross-validation the new expected standard** for genomic AMR ML, and the Briefings in Bioinformatics benchmark quantifies exactly how much performance collapses when you use it. A random 64/16/20 split with a fixed test set and reshuffled train/val is now, on its own, below the publication bar.

### Cited Findings
- Yu, Wheeler & Barquist (2025) analysed **>24,000 genomes, 5 WHO priority pathogens (incl. K. pneumoniae), 27 antibiotics, 6,740 models** and concluded that models trained on structured, biased surveillance data "conflate lineage markers with genuine indicators of AMR", and that **increasing training sample size does not fix this** — [PLOS Biol 10.1371/journal.pbio.3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)
- Their explicit recommendations: **phylogeny-aware CV testing on held-out clades; balanced test sets; report precision and recall alongside AUC; include a simple phylogenetic-placement baseline to show the model adds information beyond lineage; develop lineage-aware methods borrowing mixed-effect approaches from bacterial GWAS; build globally representative benchmarks** — [PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)
- Their clade-holdout design: all resistant *or* all susceptible samples from one clade are excluded, while other clades retain both phenotypes — [PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)
- K. pneumoniae-specific robustness check in that paper: **1,429 genomes from three clades (of 3,205), ciprofloxacin, unitig features** — unitig results matched SNP/accessory-gene results, showing **feature engineering alone does not fix non-independence** — [PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)
- Quantified collapse under harder splits, across 78 species–antibiotic datasets (11 species, 44 antimicrobials, 31,195 PATRIC genomes, 200–13,500 genomes per dataset): **fraction of datasets with ML F1-macro ≥ 0.9 fell from 64% (random folds) → 33% (phylogeny-aware) → 25% (homology-aware)**; ML beat ResFinder in **66% → 47% → 40%** of cases respectively; under phylogeny-aware and homology-aware folds, **ResFinder was the best method in 44% and 50%** of combinations vs. Kover's 28% and 34% — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- That benchmark's protocol: **10-fold nested CV with inner 9-fold hyperparameter selection**, minimum **100 genomes per class**, phylogeny-aware folds built from core-gene trees (Seq2Geno/Prokka/modified Roary), homology-aware folds, and LOSO for cross-species. Metrics: **F1-macro primary (mean ± SD over 10 folds)**, accuracy, per-class precision/recall/F1, with **negative-class (susceptible) precision and F1 emphasised for clinical relevance**. Statistics: one-sided paired t-tests — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- Its K. pneumoniae results are directly comparable to yours: **median F1-macro 0.79–0.89 under random splits**, declining further under harder splits; K. pneumoniae was **not** among the robustly predictable species (C. jejuni and E. faecium were); the **worst dataset in the entire study was aztreonam–K. pneumoniae at F1-macro 0.59** — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- Reusable artefacts: [github.com/hzi-bifo/AMR_benchmarking](https://github.com/hzi-bifo/AMR_benchmarking), [github.com/hzi-bifo/AMR_prediction_pipeline](https://github.com/hzi-bifo/AMR_prediction_pipeline), and **sample partitions on Mendeley Data, doi 10.17632/6vc2msmsxj.2** — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- General methodological backing: block cross-validation should be used wherever dependence structures exist, since random CV — even with models that nominally correct for dependence — yields **error estimates that are too low** — [Roberts et al., Ecography 2017, 10.1111/ecog.02881](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881); [PDF](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)
- Known counter-caveat to blocking: blocking by phylogenetic distance can **unintentionally induce extrapolation** by restricting the predictor combinations available in training, which can *overestimate* interpolation error — [Roberts et al. 2017](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
- Variance correction for repeated resampling: Nadeau & Bengio's corrected resampled t-test replaces the naive 1/n variance factor with **1/n + n_test/n_train**, referred to a t distribution with n−1 df; the plain paired t-test **underestimates the variance of CV mean estimators**, inflating false positives. Bouckaert & Frank extend this to T repeats of K-fold with F = 1 + TK/(K−1) — [Nadeau & Bengio, NeurIPS, "Inference for the Generalization Error"](http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf); [replicability study, UGR](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf)
- 10×10-fold CV with the correction, or 100 random-resampling runs, were the configurations with the best replicability — [UGR replicability study](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf)

### Inferences
- **Your `fixed_test_multiseed` design (seeds 1–9, test set frozen) measures only train/val reshuffling variance.** It cannot estimate test-set sampling variance, so the resulting error bars are systematically too narrow and are *not* a valid basis for claiming your model beats PanKA/Kover. Keep it as an *optimisation-stability* ablation (legitimate and worth reporting as such), but add repeated **resampled** splits (e.g. 10×5-fold or 10 repeats of 64/16/20 with the test set redrawn) for the headline comparison, analysed with the corrected resampled t-test.
- Given K. pneumoniae's median F1-macro of 0.79–0.89 under *random* folds in bbae206, a random-split result above ~0.95 in your paper is more likely a sign of lineage leakage than of a better model — reviewers will read it that way. Report the random-split and the lineage-blocked number side by side and treat the drop as a finding, not an embarrassment.
- Deduplication: the task context does not say whether identical/near-identical BioSamples were removed. Outbreak clusters in NCBI/BV-BRC can place near-clonal isolates on both sides of a random split. At minimum, dedupe exact assemblies and collapse isolates below a cgMLST/SNP threshold to one representative, and report counts before and after. I found no AMR-specific paper prescribing a numeric threshold, so state your choice and show sensitivity to it.
- An MLST-blocked split is a cheap, defensible approximation to clade-blocked CV for K. pneumoniae (ST is already in your AMR-GNN+MLST pipeline) and is CPU-cheap, which matters given your constraint. A full core-gene tree is also CPU-feasible at n ≈ 3,949 but Roary/Panaroo plus IQ-TREE is the expensive step; note that bbae206 **dropped 11 of 78 datasets because phylogeny-aware fold construction exceeded runtime limits**.

### Gaps
- Neither key source prescribes a specific **number of clades/blocks** or a tree-cut depth. You will have to justify your own (e.g. clades defined at a fixed patristic distance, or Kleborate sublineages) and show robustness to the choice.
- I found **no source giving a recommended deduplication threshold (SNP or cgMLST allele distance) for AMR ML datasets**.
- I found no published guidance specifically on temporal or geographic holdout for genomic AMR, though PLOS Biology's "globally representative benchmark / expand into under-represented regions" recommendation implies geographic holdout is favourably viewed — [PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)

---

## (B2) External validation cohorts for K. pneumoniae (public genomes + AST)

### Takeaway
The strongest and most current external resource is the **KlebNET-GSP AMR Genotype-Phenotype collection (12,167 matched isolates + 7,030 external)**, followed by EuSCAPE (ENA PRJEB10018, ~1,500–1,717 isolates with AST for 8 agents), NCBI Pathogen Detection's AST Browser, and BV-BRC. Note that BV-BRC phenotype coverage is far thinner than its genome count suggests.

### Cited Findings
- **KlebNET-GSP AMR Genotype-Phenotype Group**: **12,167 K. pneumoniae species complex isolates with matched genotype and phenotype, 27 countries, 2001–2021** (discovery), plus **7,030 externally contributed isolates** for validation, plus **31,319 KpSC genomes from Pathogenwatch, 109 countries, 2000–2023** for global analysis. Corresponding authors Kara K. Tsang and Kathryn E. Holt. Code/data at [github.com/klebgenomics/cipropaper](https://github.com/klebgenomics/cipropaper) and [klebgenomics/Kleborate](https://github.com/klebgenomics/Kleborate) — [bioRxiv 2025.09.24.678318](https://www.biorxiv.org/content/10.1101/2025.09.24.678318v1)
- That paper's **performance bar is the one to beat on ciprofloxacin**: a rules-based classifier (QRDR mutation counts in gyrA/parC, PMQR gene count, aac(6')-Ib-cr presence) achieved **CA, sensitivity and specificity all >96% with ME and VME <4% on discovery**, and on external validation **CA 93.12% (95% CI 92.50–93.74), ME 8.65% (7.34–9.97), VME 6.20% (5.51–6.90)** — and the authors conclude accuracy is **sufficient for surveillance but not for clinical use** — [bioRxiv 2025.09.24.678318](https://www.biorxiv.org/content/10.1101/2025.09.24.678318v1)
- A separate KlebNET-GSP output notes the matched collection spans 24 countries in its SHV subset and contributed **ten novel protein variants to Kleborate v2.4.1** — [Microbial Genomics 10.1099/mgen.0.001294](https://www.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.001294)
- **EuSCAPE**: >1,700 K. pneumoniae from **244 hospitals in 32 countries**; raw and assembled Illumina data under **ENA PRJEB10018 / ERP011196**; per-run accessions in Supplementary Table 4; Microreact project "EuSCAPE_Kp" — [Nat Microbiol 10.1038/s41564-019-0492-8](https://www.nature.com/articles/s41564-019-0492-8)
- Kleborate used **1,490 EuSCAPE K. pneumoniae genomes** from RefSeq (of 1,649 reported) and plotted **mean meropenem MIC** for them, so MIC data exist for that set — [Nat Commun 10.1038/s41467-021-24448-3](https://www.nature.com/articles/s41467-021-24448-3)
- The RASE preprint compiled **genomes, ST and AST for 8 antimicrobial agents for 1,511 K. pneumoniae sensu lato strains from EuSCAPE** — [bioRxiv 2025.09.03.673989](https://www.biorxiv.org/content/10.1101/2025.09.03.673989.full.pdf)
- **COMBACTE-EURECA**: 687 carbapenem-resistant strains from 41 hospitals in nine Southern European countries, 2016–2018; reads under **ENA PRJEB63349** — [Nat Commun 10.1038/s41467-024-49349-z](https://www.nature.com/articles/s41467-024-49349-z); [PMC11178878](https://pmc.ncbi.nlm.nih.gov/articles/PMC11178878/)
- **NCBI Pathogen Detection AST Browser** (announced May 2024): searchable, downloadable table filterable by organism group, antibiotic and MIC, with a cross-browser link to the Isolates Browser and MicroBIGG-E. **Critical caveat: "the phenotype data displayed in this interface is supplied by submitters"; "Beyond basic quality control (e.g. negative MIC values), NCBI does not vet the methods used or values supplied for AST data"**; submitter AST may conflict with in-silico genotype — [NCBI Insights, 1 May 2024](https://ncbiinsights.ncbi.nlm.nih.gov/2024/05/01/pathogen-detection-ast-browser/); [Pathogen Detection How-To](https://www.ncbi.nlm.nih.gov/pathogens/docs/HowTo/)
- **BV-BRC phenotype sparsity**: of 18,645 K. pneumoniae WGS filtered for assembly quality, **only 4,976 had antimicrobial resistance data, covering 76 antibiotics across 15 classes**; where MICs were missing, labels were taken as provided by BV-BRC; most samples are short-read, leaving fragmented assemblies with potentially collapsed tandem repeats — [Sci Rep 10.1038/s41598-025-24333-9](https://www.nature.com/articles/s41598-025-24333-9)
- **Pathogenwatch** runs MLST and cgMLST, resistance-gene identification, virulence loci, capsule and O-antigen typing, and replicon typing on every Klebsiella assembly — [bioRxiv 2021.06.22.448967](https://www.biorxiv.org/content/10.1101/2021.06.22.448967.full.pdf); 16,537 high-quality geolocated Klebsiella genomes at that time
- Independent benchmark genome source for a like-for-like comparison: **PATRIC, accessed December 2020**, with partitions archived at Mendeley doi 10.17632/6vc2msmsxj.2 — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)

### Inferences
- The **highest-value, lowest-cost external validation for your paper** is the KlebNET-GSP/Pathogenwatch route: it gives you a genuinely independent, multi-country cohort and a published rules-based benchmark (CA 93.12%, ME 8.65%, VME 6.20% on ciprofloxacin external data) you can beat or match. The downside is it is currently a 2025 preprint and coverage is ciprofloxacin-only in that publication.
- For multi-drug external validation across your 17 antibiotics, the practical combination is **NCBI Pathogen Detection AST Browser + BV-BRC**, filtered to isolates **not present in your training set by BioSample accession**, with submitter-method heterogeneity disclosed as a limitation (NCBI explicitly does not vet methods).
- Because EuSCAPE AST sits in supplements rather than ENA, budget time to merge phenotypes manually; the RASE preprint's 1,511-strain, 8-agent compilation is the shortest path to a ready-made EuSCAPE AST table.
- A geographic-holdout experiment is nearly free given these resources and would directly answer the PLOS Biology critique: train on your 3,949 and test on a European cohort (EuSCAPE/EURECA), or vice versa.

### Gaps
- **CRACKLE / CRACKLE-2** — I did not search for it and have **no verified figures**; do not cite it without checking.
- **Southeast-Asia / Vietnam K. pneumoniae collections** — not researched here; no reliable source found in this pass. Given your location, a local collection would be a strong differentiator, but I cannot cite one.
- EuSCAPE isolate counts are **inconsistent across sources** (>1,700 in the 2019 Nat Microbiol report; 1,649 in the Kleborate reference; 1,717 in a 2024 comparison; 1,490 analysed; 1,511 in RASE). State which version and supplement you used.
- I could **not confirm whether KlebNET-GSP's full 12,167-isolate phenotype table is publicly downloadable**; the GitHub repos are the place to check.

---

## (B3) Metrics, error rates, calibration and statistical comparison

### Takeaway
Reviewers in 2025–2026 expect discrimination metrics **plus** clinical-microbiology error rates (VME/ME) with confidence intervals, **plus** calibration, **plus** a variance-correct statistical comparison. Note that ISO 20776-2:2021 **removed** categorical agreement, so cite FDA (which retains CA/VME/ME) and ISO separately and correctly.

### Cited Findings

**Clinical-microbiology acceptance thresholds**
- FDA Class II special controls guidance for AST systems: passing requires the **upper 95% confidence limit for the true very-major-discrepancy rate to be under 7.5% and the lower 95% confidence limit under 1.5%**; **major error rate <3%** computed over susceptible isolates; **essential and categorical agreement above 89.9% is acceptable** (CA <90% can be acceptable where EA is very good and most discrepancies are minor); **overall reproducibility across sites must be ≥95% within ±1 dilution** — [FDA guidance PDF](https://www.fda.gov/media/88069/download); [CDRH overview](https://www.fda.gov/files/drugs/published/Overview-of-AST-Device-Clearance-Process--CDRH-Perspective.pdf); [Federal Register notice](https://www.federalregister.gov/documents/2003/02/05/03-2657/medical-devices-class-ii-special-controls-guidance-document-antimicrobial-susceptibility-test)
- Conflicting secondary figures exist in the literature (some papers quote a flat CLSI VME ≤1.5%, others a "FDA maximum of 3%"), so quote the guidance's confidence-bound criteria directly rather than a secondary summary — [FDA guidance](https://www.fda.gov/media/88069/download); conflict noted in [arXiv 2005.11454](https://arxiv.org/pdf/2005.11454)
- **ISO 20776-2:2021 removed interpretive categories (SIR), categorical agreement and associated terminology**, because those reflect breakpoint performance as well as assay performance and would force re-assessment whenever only a breakpoint changed; it **added a bias requirement** and **sensitivity/specificity acceptance provisions for qualitative AST devices (subclause 5.1.3)** — [ISO 20776-2:2021 preview](https://webstore.ansi.org/preview-pages/ISO/preview_ISO+20776-2-2021.pdf); [PMC10281161](https://pmc.ncbi.nlm.nih.gov/articles/PMC10281161/)
- CA is still required for US submissions even where ISO no longer asks for it; studies now report EA + bias for worldwide registration and EA + CA + trend + VME + ME for US registration — [Open Forum Infect Dis ofae631.2273](https://academic.oup.com/ofid/article/12/Supplement_1/ofae631.2273/7988273?login=false)
- Worked example of this dual reporting in a 2026 clinical evaluation citing ISO 20776-2:2021 — [J Clin Microbiol 10.1128/jcm.00558-26](https://journals.asm.org/doi/full/10.1128/jcm.00558-26)

**Discrimination / class-imbalance metrics as used in the field**
- Benchmark primary metric: **F1-macro averaged over 10 folds with standard deviation**, plus accuracy and per-class precision/recall/F1, with **susceptible-class (negative-class) precision and F1 explicitly emphasised for clinical relevance** — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- PLOS Biology recommends **balanced test sets and reporting precision and recall alongside AUC**, because AUC alone hides large precision/recall biases that persist even as AUC improves with more data — [PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)
- MCC was the metric that revealed transfer-learning benefit where AUROC did not (GEN: MCC 0.11→0.26 but AUROC 0.69→0.72 n.s.) — [PMC9686617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9686617/)
- The KlebNET-GSP paper reports **CA, ME and VME with 95% CIs** as its headline external-validation metrics — [bioRxiv 2025.09.24.678318](https://www.biorxiv.org/content/10.1101/2025.09.24.678318v1)

**Statistics**
- Use the **corrected resampled t-test** (variance factor 1/n + n_test/n_train; Bouckaert–Frank extension F = 1 + TK/(K−1) for T repeats of K folds) because the naive paired t-test underestimates CV variance and inflates false positives — [Nadeau & Bengio](http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf); [UGR replicability study](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf); implemented in Weka as "corrected resampled t-test"
- 10×10-fold CV with the correction (or 100 resampling runs) had the best replicability — [UGR replicability study](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf)
- Multiple-comparison correction is applied in practice as **Holm–Bonferroni across metric-level p-values within each comparison** — [UGR replicability study](https://sci2s.ugr.es/keel/pdf/specific/congreso/evaluating-the-replicability-of.pdf); see also the Bayesian alternative for comparing cross-validated algorithms across datasets — [Mach Learn 10.1007/s10994-015-5486-z](https://link.springer.com/article/10.1007/s10994-015-5486-z)
- The AMR benchmark itself used **one-sided paired t-tests**, e.g. Kover vs. PhenotypeSeeker P = 0.01, Kover vs. ResFinder P = 3.27e-7 — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)
- DOME explicitly asks authors to **evaluate on a final independent hold-out set and report confidence/error intervals** to gauge prediction robustness — [Nat Methods 10.1038/s41592-021-01205-4](https://www.nature.com/articles/s41592-021-01205-4)

### Inferences
- With **15 susceptible isolates in the cefuroxime test set, a single major error is 6.67%**, already more than double FDA's 3% ME ceiling, and the 95% CI on an ME rate from n=15 spans roughly 0–20%. **Cefuroxime ME is not evaluable** at that sample size. State this explicitly rather than reporting a point estimate — reviewers will otherwise read 0% ME as a claim. The same logic applies to any drug where the minority class in the test set is below roughly 30–50 isolates.
- bbae206's minimum of **100 genomes per class** is a useful, citable inclusion rule. Drugs in your panel that fall below it under lineage-blocked folds should be reported in a clearly separated "underpowered" tier.
- Report both **AUROC and AUPRC**: with susceptible classes as small as 15, AUROC is dominated by the majority class and AUPRC is the metric that will move.
- **Calibration** (reliability diagram, Brier score, expected calibration error) is explicitly called for by TRIPOD+AI's clinical-prediction framing and is cheap to add; it is also the natural place to show a multi-task model's advantage on low-data drugs, where per-drug models are typically badly over-confident.
- For paired AUROC comparisons on a single fixed test set, DeLong's test is the conventional choice, and paired bootstrap over test isolates (≥2,000 resamples, BCa intervals) generalises to F1/MCC/VME/ME where no closed form exists. I did not retrieve a citation for DeLong in this pass — look it up before citing.
- Harmonise phenotypes to a **single, stated breakpoint version** (EUCAST clinical breakpoints vXX or CLSI MYY) and state how Intermediate was handled (dropped vs. merged into R), since this single choice can move VME/ME by several points. The EUCAST "susceptible, increased exposure" redefinition materially affects AST evaluation — [Clin Microbiol Infect S1198-743X(24)00339-2](https://www.clinicalmicrobiologyandinfection.org/article/S1198-743X(24)00339-2/fulltext)

### Gaps
- The FDA guidance version I located traces to the 2003 Federal Register notice and is cited elsewhere as 2009; **verify the current revision on fda.gov** before quoting thresholds in a submission-adjacent context.
- No DeLong-test primary citation retrieved in this pass.
- I found no AMR-ML paper that reports calibration metrics, so there is no in-field precedent to cite — but also no in-field competition on that axis.

---

## (B4) Reporting guidelines, what recent papers actually report, and target journals

### Takeaway
Two guidelines are directly applicable and should both be cited and supplied as completed checklists: **TRIPOD+AI** (if you frame the model as clinically predictive) and **DOME** (supervised ML in biology, the better fit for a genomics method paper). The **EUCAST WGS subcommittee's 2026 update** is the domain-authority document and now explicitly covers ML/AI.

### Cited Findings
- **TRIPOD+AI** (BMJ, 2024): **27-item checklist, an expanded explanation-and-elaboration checklist, and a 13-item TRIPOD+AI for Abstracts checklist**; it **supersedes the TRIPOD 2015 checklist, which "should no longer be used"**; it provides harmonised guidance "irrespective of whether regression modelling or machine learning methods have been used"; checklists are in Supplementary Tables 1 and 2 — [PMC11019967](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11019967/); [EQUATOR Network listing](https://www.equator-network.org/reporting-guidelines/tripod-statement/)
- **DOME** (Nature Methods, July 2021): community recommendations structured as questions across **Data, Optimization, Model, Evaluation**, written for supervised ML in biology "in the absence of direct experimental validation"; the authors describe it as a non-exhaustive consensus first iteration — [Nat Methods 10.1038/s41592-021-01205-4](https://www.nature.com/articles/s41592-021-01205-4); [arXiv 2006.16189](https://arxiv.org/abs/2006.16189)
- **DOME Registry** (GigaScience, 2024) lets authors deposit and reviewers inspect DOME annotations for published ML studies — [GigaScience 10.1093/gigascience/giae094](https://doi.org/10.1093/gigascience/giae094); [PMC11633452](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11633452/); [arXiv 2408.07721](https://arxiv.org/abs/2408.07721)
- **EUCAST WGS-AST subcommittee, 2017 report**: *Clinical Microbiology and Infection* 23(1):2–22, doi **10.1016/j.cmi.2016.11.012**; its verdict was that "the published evidence for using WGS to infer antimicrobial susceptibility accurately is currently either poor or non-existent" — [ORA record](https://ora.ox.ac.uk/objects/uuid:c7779e4e-9df2-470b-83d4-f8ad51837c6c)
- **EUCAST 2025 update** ("The role of whole genome sequencing in antimicrobial susceptibility **prediction** of bacteria: 2025 update"), *Clinical Microbiology and Infection*, doi **10.1016/j.cmi.2026.05.012**; subcommittee re-established 2023/24 under Antonio Oliver (chair) and Ørjan Samuelsen (co-chair); the new review **adds sections on metagenomics, other omics, and ML/AI**, and extends organism coverage to enterococci, H. influenzae and B. fragilis; public consultation November 2025; **published 12 June 2026** — [EUCAST subcommittee page](https://www.eucast.org/about-eucast/subcommittees/subcommittee-wgs/); [ScienceDirect S1198743X2600248X](https://www.sciencedirect.com/science/article/abs/pii/S1198743X2600248X)
- Field-specific critical review of practice and limitations, useful for framing the gap: [Clin Microbiol Rev 10.1128/cmr.00179-21](https://journals.asm.org/doi/10.1128/cmr.00179-21)
- Evidence that reporting compliance is actively audited post-publication: a systematic review scored ML prognostic models against TRIPOD/TRIPOD+AI and found they "lag behind in reporting quality" — [PMC12866346](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12866346/)
- Recent venue precedents for this exact kind of paper: *Briefings in Bioinformatics* (the 78-dataset benchmark) — [bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136); *PLOS Biology* (the population-structure critique) — [e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539); *Nature Communications* (Kleborate) — [41467-021-24448-3](https://www.nature.com/articles/s41467-021-24448-3); *Scientific Reports* (both K. pneumoniae MIC-panel papers) — [41598-017-18972-w](https://www.nature.com/articles/s41598-017-18972-w), [41598-024-75973-2](https://www.nature.com/articles/s41598-024-75973-2); *Bioinformatics* (PanKA) — [PMC11369404](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11369404/); *Microbial Genomics* (KlebNET-GSP SHV) — [mgen.0.001294](https://www.microbiologyresearch.org/content/journal/mgen/10.1099/mgen.0.001294); a 2026 scalable interpretable MIC-prediction paper in the Nature Partner Journals family — [npj 10.1038/s44259-026-00217-4](https://www.nature.com/articles/s44259-026-00217-4)
- Precedent for open artefacts at the level reviewers now expect: benchmark code, adapted-software repo, prediction pipeline, **and archived sample partitions on Mendeley (doi 10.17632/6vc2msmsxj.2)** — [Brief Bioinform bbae206](https://academic.oup.com/bib/article/25/3/bbae206/7665136)

### Inferences
- **DOME is the better primary checklist** for a genomics method paper (no patient-level clinical deployment claim); add TRIPOD+AI if you make any clinical-utility or decision-support claim, which the VME/ME framing implicitly does. Supplying both as supplementary tables is cheap insurance and is now common.
- The **EUCAST 2026 update is the single most important citation for your Discussion**: it is the domain authority, it is brand new, and it explicitly now covers ML/AI. You must read it in full — I could only retrieve its metadata and scope, not its recommendations or any thresholds it sets. **This is the top follow-up action.**
- Realistic venue ladder given a CPU-only, single-species, 3,949-genome study with a methods contribution: *Microbial Genomics* / *Bioinformatics* / *Briefings in Bioinformatics* are well matched; *Genome Medicine* or *Nature Communications* become plausible only with a strong external validation cohort plus lineage-blocked results plus the low-data-rescue effect demonstrated prospectively; *Lancet Microbe* and *Clinical Microbiology and Infection* would require clinical-grade VME/ME performance, which the KlebNET-GSP authors explicitly judged not yet achieved even for a well-understood drug like ciprofloxacin. This ladder is my inference, not a sourced claim.
- Archive your **exact fold assignments** (not just code) alongside the paper. bbae206 set that precedent and it is the single cheapest way to make your comparison reproducible and to defuse "you tuned on the test set" concerns.

### Gaps
- I could **not read the EUCAST 2026 update's actual recommendations, thresholds, or its verdict on ML/AI** (ScienceDirect abstract-only; the mirror returned HTTP 403). Everything above about it is scope and metadata only.
- I did not verify individual journals' stated submission requirements (e.g. whether a given journal mandates a DOME or TRIPOD+AI checklist). Treat the venue guidance as inference.

---

## Recommended evaluation protocol checklist

**1. Dataset construction and disclosure**
- [ ] Report, per antibiotic: n labelled, n resistant, n susceptible, n intermediate, and how I was handled (dropped vs. merged into R) — for all 17 drugs, in the main text.
- [ ] State the exact breakpoint standard and version (EUCAST vXX or CLSI MYY) used to derive S/R; justify, since this moves VME/ME by several points ([CMI 2024](https://www.clinicalmicrobiologyandinfection.org/article/S1198-743X(24)00339-2/fulltext)).
- [ ] Deduplicate: remove identical assemblies, collapse near-clonal isolates (cgMLST/SNP threshold of your choosing) to one representative; report n before/after and show sensitivity to the threshold.
- [ ] Report ST/sublineage composition (Kleborate) and per-drug resistance prevalence by sublineage, so lineage confounding is visible ([PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)).
- [ ] Apply and state an inclusion floor (bbae206 used ≥100 genomes per class); put drugs below it in a clearly flagged "underpowered" tier.

**2. Splits — three tiers, all reported**
- [ ] **Tier 1 (comparability):** random repeated splits, test set **resampled** each repeat, 10 repeats (or 10×5-fold nested CV with inner-loop hyperparameter selection). This replaces `fixed_test_multiseed` for the headline comparison.
- [ ] **Tier 2 (the standard reviewers now expect):** lineage/clade-blocked CV — MLST/sublineage-blocked as the cheap version, core-gene-tree clade holdout as the full version; follow the PLOS Biology construction (hold out a clade's phenotypes while other clades retain both) ([PLOS Biol e3003539](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3003539)).
- [ ] **Tier 3 (external):** at least one fully held-out cohort by accession — KlebNET-GSP/Pathogenwatch, EuSCAPE (ENA PRJEB10018, or the RASE 1,511-strain 8-agent table), NCBI AST Browser, or BV-BRC — with explicit confirmation of **no BioSample overlap** with training.
- [ ] Keep `fixed_test_multiseed` (seeds 1–9) but **relabel it as an optimisation-stability ablation**, stating that it does not estimate test-set sampling variance.
- [ ] Deposit the exact fold assignments (Zenodo/Mendeley), as bbae206 did.

**3. Baselines**
- [ ] Majority-class per drug.
- [ ] **Phylogenetic-placement / nearest-neighbour-by-ST baseline** — explicitly demanded by PLOS Biology to show the model adds information beyond lineage.
- [ ] Rule-based: ResFinder (and Kleborate for K. pneumoniae).
- [ ] Per-drug ML: PanKA (LightGBM), Kover, AMR-GNN(+MLST), Panex.
- [ ] **Naive MTL ablation:** shared trunk + 17 independent heads — i.e. reproduce the configuration that gave p = 0.374 and p = 0.48 in the literature, then show your factorisation beats it.
- [ ] Drug-representation ablation: one-hot drug ID vs. ATC/CARD-ARO class vs. SMILES-derived embedding.
- [ ] If you run a classifier chain, run it twice: oracle sibling phenotypes (upper bound, clearly labelled non-deployable) and predicted sibling phenotypes (deployable).

**4. Metrics — per drug, never only averaged**
- [ ] AUROC **and** AUPRC (AUPRC is the one that moves with 15 susceptible isolates).
- [ ] F1-macro (primary, as in bbae206), balanced accuracy, MCC, and **susceptible-class precision/recall/F1 reported separately**.
- [ ] **VME (over resistant isolates) and ME (over susceptible isolates) with 95% CIs**, benchmarked against FDA's criteria: VME upper 95% CL <7.5% and lower 95% CL <1.5%; ME <3%; CA ≥90% ([FDA guidance](https://www.fda.gov/media/88069/download)).
- [ ] Explicitly mark drugs where VME/ME is **not evaluable** for lack of minority-class isolates — cefuroxime at 15 susceptible means one ME = 6.67%, already above the 3% ceiling.
- [ ] Cite ISO 20776-2:2021 correctly: it **removed** CA/SIR and now requires EA + bias, with sensitivity/specificity for qualitative devices ([ISO preview](https://webstore.ansi.org/preview-pages/ISO/preview_ISO+20776-2-2021.pdf)).
- [ ] Calibration: reliability diagram, Brier score, ECE — per drug, with a focus on low-data drugs.

**5. Statistics**
- [ ] **Corrected resampled t-test** (Nadeau–Bengio variance factor 1/n + n_test/n_train; Bouckaert–Frank F = 1 + TK/(K−1)) for all cross-validated model comparisons ([NeurIPS PDF](http://papers.neurips.cc/paper/1661-inference-for-the-generalization-error.pdf)).
- [ ] Paired bootstrap over test isolates (≥2,000 resamples, BCa CIs) for F1/MCC/VME/ME on fixed test sets; DeLong for paired AUROC (verify the citation).
- [ ] **Holm–Bonferroni or Benjamini–Hochberg across the 17 drugs** — state which, and report both raw and adjusted p-values.
- [ ] Report effect sizes with CIs, not only p-values.

**6. The low-data-drug claim (your core contribution)**
- [ ] Plot Δmetric (MTL − per-drug) against n_labelled per drug, with a fitted trend and CIs.
- [ ] **Label-subsampling experiment:** down-sample a data-rich drug (gentamicin, 3,864) to 1,296 and to intermediate sizes, and show the MTL gap opens as labels shrink. This is the cleanest causal evidence and is cheap on CPU.
- [ ] Leave-one-drug-out zero-shot test using the drug embedding — currently unclaimed in the genomic AMR literature.
- [ ] Report gains on **external** data in MCC/AUPRC, following the Ren et al. pattern where external imbalanced data showed the effect that internal AUROC hid.

**7. Reporting and artefacts**
- [ ] Completed **DOME** checklist as supplementary (and deposit in the DOME Registry); add **TRIPOD+AI** (27 items + 13-item abstract checklist) if any clinical-utility claim is made.
- [ ] Cite and engage with the **EUCAST 2026 WGS-AST update** (doi 10.1016/j.cmi.2026.05.012) — read it in full first.
- [ ] Engage directly with the PLOS Biology population-structure critique rather than ignoring it; report the random-vs-lineage-blocked drop as a finding.
- [ ] Release code, trained models, fold assignments, accession lists and per-drug prediction tables; report CPU runtime and memory (a genuine selling point given your CPU-only constraint and PanKA's speed claim).

**Top three follow-ups before writing:** (1) read the 2024 *Scientific Reports* K. pneumoniae drug-structure CNN in full — it is the closest prior art and determines your novelty framing; (2) read the EUCAST 2026 update in full; (3) confirm whether KlebNET-GSP's 12,167-isolate genotype-phenotype table is downloadable, since that single dataset would carry your external validation.
