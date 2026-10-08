# Methods (GenoContext v2)

## Gene-state representation (label-free, per isolate)

The representation is built from each Bakta GFF3 and the matching protein FASTA (`src/genocontext/features/`). Every isolate becomes a set of **nodes**:

| Group | Definition |
|---|---|
| `amr` | AMRFinderPlus-tagged CDS (`NCBIProtein:` cross-reference), named at allele level (e.g. `blaKPC-3`). The family name (`blaKPC`) is a fallback for alleles that are rare in training. The `-cr` variant of `aac(6')-Ib` is taken from the product text. |
| `target` | 13 chromosomal loci: GyrA, ParC, OmpK35/36/37, PhoE, RamR, AcrR, MarR, SoxR, NfsA, NfsB, RibE. They are found by a 5-mer prefilter over the whole proteome, then local alignment (BLOSUM62) to pinned references (`knowledge/references/kp_targets.faa`), with identity ≥ 0.8. The best reference wins, so the 381-aa porin paralogs are not called OmpK36. Each target gets one state: *intact*, *truncated* (pseudogene, missing N-terminus, or < 90% reference coverage), *IS-disrupted* (a mobile-element CDS within 200 bp), *edge-unknown* (fragment at a contig end), or *absent*. |
| `res` | Residue calls against the reference coordinates: GyrA S83/D87 and ParC S80/E84, versus explicitly pinned wild-type residues. OmpK36 insertions and deletions are also called (e.g. the L3 insertion `OmpK36_ins134_DT`). |
| `mge` | IS elements (`IS:` cross-reference), integrases, transposases, recombinases. |
| `fam` | Accessory UniRef90 families (falling back to UniRef50, then the gene name). |

**Node flags:**
- ≥ 2 assembled copies
- near a contig end
- within ±3 CDS of a mobile element

These flags are deliberately binary. Assemblies carry no coverage information, so copy number and contig size are not trusted.

**Graph edges:** connect nodes within ±5 CDS on the same contig.

**Feature space (fitted on training isolates only):**
- Prevalence ≥ 10.
- Accessory families capped at the 2,000 with the highest p(1−p).
- Tokens with |r| ≥ 0.98 are merged into blocks. Targets are never merged.

## GenoContext-KG

Each node embedding combines token, state and flags. An optional GATv2 layer mixes neighbouring nodes. Nodes are then pooled into **14 resistance-mechanism nodes**, using membership from `knowledge/mechanisms.yaml`, by a gated, count-normalised sum.

Each drug has a query q_d. It is an MLP of the drug's class features, plus a small L2-penalised drug embedding. The logit is:

`logit[b,d] = bias_d + Σ_k w_kd · (V_k m_bk)·q_d + Σ_{k<l} γ_dkl s_bk s_bl`

- w_kd = 1 on prior mechanism→drug-class edges. Elsewhere it is a learnable softplus weight with an L1 penalty, so non-prior links can be discovered.
- The γ terms are L1-penalised pairwise mechanism interactions, for example ESBL × porin loss.
- The per-mechanism contributions and interaction terms sum exactly to the logit.
- **Training:** masked binary cross-entropy averaged per drug, over all observed (isolate, drug) labels, with AdamW. Other drugs' phenotypes are never used as inputs.
- **Zero-shot:** the held-out drug's labels are removed from training and its embedding is zeroed, so it is predicted from class features alone.

## Training protocols

- **A (64/16/20):** fit on inner folds 1–4, then use inner fold 0 for early stopping and the threshold.
- **B (80/20):** a 5-fold inner CV provides the training budget (median best epoch or boosting rounds) and out-of-fold probabilities, which set the thresholds. The final model is refit on all training isolates, averaged over 5 seeds.
- **Thresholds:** maximise balanced accuracy, with ties going to the threshold closest to 0.5.

## Interpretation

- **Exact contributions:** mechanism-level contributions c_bdk, plus interaction terms.
- **Integrated gradients:** node scales are interpolated from 0 to 1, giving node-level attributions. Per isolate and drug these sum to logit(1) − logit(0); the completeness error is logged. Attributions are mapped back to GFF3 locus tags.
- **Global tables:**
  - learned mechanism→drug weights against the prior
  - drug-query cosine similarity
  - mean contribution per mechanism in R and S isolates
- **Evidence ladder:** see `docs/protocol.md`.
