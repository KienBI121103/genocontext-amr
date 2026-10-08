# Prior art for a knowledge-guided GNN, P-NET lessons, and evidence on the 80/20 split

Condensed research output from 8 Oct 2026. Preprints are labelled; verify every DOI before citing.

## Closest prior work for a gene → mechanism → drug graph model

| Design element | Closest work | Key facts |
|---|---|---|
| Knowledge graph (KG) linking genes, mechanisms and drugs for isolate phenotype | **KG-TRACE** (arXiv 2606.26179, preprint, 2026) | *M. tuberculosis*, CRyPTIC, 37,761 isolates. A mutation vector feeds an MLP; RotatE KG embeddings come from WHO catalogue triples (gene / mutation / drug / mechanism); the genome vector acts as a query attending over the KG; fusion uses a trust gate. Isoniazid AUROC 0.9760, vs XGBoost 0.9760 and RF 0.9806, so the **KG adds no accuracy**; the stated value is symbolic grounding (BGR@10 = 0.20). Code: github.com/semintelligence/KG-TRACE |
| KG for Kp / *E. coli* | **BRIDGE** (bioRxiv 2026, preprint) | Link prediction on a CARD + STRING + DrugBank KG. It does not predict isolate phenotypes. |
| Drug-conditioned attention over a genome's gene set | **GL-HAT** (bioRxiv 2025, preprint) | Learned query vector per antibiotic over ESM-2 gene embeddings; no drug features. Attention readout: F1 0.798 / AUROC 0.931; mean pooling: 0.845 / 0.953; XGBoost: 0.835 / 0.950. |
| Drug structure × sample | De Waele et al., *eLife* 2024;13:93242 (MALDI-TOF drug recommender); Compound RNN (ACM BCB 2023, SMILES + contigs → MIC) | Not genome-to-AST with a mechanism layer |
| Heterogeneous GNN | **HGAT-AMR**, Yang et al., *Brief Bioinform* 22(6):bbab299 (2021) | *M. tuberculosis*, 13,402 isolates, isolate–SNP heterogeneous attention, multi-label with missing labels. INH AUROC 98.5%, RIF 99.1%; baselines sometimes win. No code. |
| Biologically informed (P-NET-style) networks for bacterial AMR | **None found** | ONN4ARG is an ontology network for ARG annotation, not phenotype prediction. KEGG-orthology features with XGBoost (*A. baumannii*). |

**Novelty verdict.** The combination appears to be unpublished. The strongest claims are:
- a mechanism-layer, knowledge-primed network for isolate AMR in Gram-negatives
- gene → mechanism → drug path attribution
- held-out-drug transfer driven by drug-class features

The weakest claims are drug-conditioned attention alone and a heterogeneous GNN alone.

## Lessons from P-NET and biologically informed neural networks

**P-NET** (Elmarakeby, *Nature* 598:348, 2021):
- Test AUC 0.93 versus a dense network.
- It won significantly only when there were fewer than about 500 samples. The prior helps most when data are small.

**Esser-Skala & Fortelny**, *npj Syst Biol Appl* 9:50 (2023):
- Retraining with different seeds gives divergent node importances. Replicate correlations in DTox ranged from R = 0.98 to −0.98.
- Importance tracks graph centrality.
- **Fix:** average over several seeds and subtract a shuffled-label control to get differential scores.

**Related findings:**
- **Pedersen et al.** (reusability report, arXiv 2309.16645, preprint): P-NET beat randomly sparsified networks, so the biology does add something.
- **Miller et al.** (bioRxiv 2025, preprint): biologically informed networks mainly exploit linear signal and help interpretability more than accuracy.
- **Selby et al.** (review, *Front Artif Intell* 2025): these networks are often comparable to dense networks rather than better. Reviewers will expect random-sparse controls, repeated training and released code.
- **DrugCell** (*Cancer Cell* 2020): per-drug Spearman ρ = 0.37 vs 0.35 for elastic net.

**Implications for GenoContext-KG:**
- Train ≥ 20–30 seeds.
- Run a shuffled-label control and report differential path attributions.
- Run a randomly rewired mechanism mask.
- Compare against a parameter-matched dense network and a mean-pooling model.
- Expect accuracy gains mainly at small n and on held-out drugs.

## Evidence on training size and protocol for an 80/20 split

**Size effect:**
- Lüftinger et al. (*Front Cell Infect Microbiol* 2021, 10.3389/fcimb.2021.610348) found a positive correlation between training size and performance, but did not quantify the effect.
- Learning curves plateau in several studies:
  - Benkwitz-Bedford, *mSystems* 2021: monotonic improvement in only 6 of 19 conditions.
  - Liu, *Microbiol Spectr* 2025: plateau well before the full training set is used.
  - Moradigaravand, *PLoS Comput Biol* 2018: no loss down to 130 strains when causal genes have high penetrance.
- Diversity matters more than count: Nguyen, *J Clin Microbiol* 2019, built accurate models from fewer than 500 diverse genomes.
- Yu et al. (PLOS Biol 2025): more data does not fix lineage confounding.

**Expected effect of going from 64% to 80% training data:**
- Empirical learning-curve exponents are −0.07 to −0.35 (Hestness, arXiv 1712.00409).
- A 25% increase in training data therefore gives roughly ΔAUROC ≈ 0.002–0.01, within cross-validation noise.

**Protocol when train and val are merged:**
- Lüftinger: 5-fold outer CV with 10-fold inner CV, grouped by distance; tune only a few hyperparameters; redo feature selection inside each fold.
- Varoquaux (*NeuroImage* 2017): tune inside nested CV; averaging models gives the most stable weights.
- Stacking (Lüftinger): sensitivity +1.77%, specificity +3.20%, and the fewest failures.
- Thresholds: pick them from pooled out-of-fold predictions, as scikit-learn's `TunedThresholdClassifierCV` does (balanced accuracy by default).
- Refitting a neural network on all data: reuse the median early-stopping epoch across folds (Goodfellow et al., *Deep Learning* §7.8, Algorithm 7.2), or use a fold or seed ensemble. No AMR-specific precedent was found.
