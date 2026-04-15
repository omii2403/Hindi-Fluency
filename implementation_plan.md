# Hindi VFT + SpAM Complete Analysis — Implementation Plan

## Overview

The mid-project report (`Report_Mid.md`) already covers the core VFT Phase 1 analysis (RQ1, RQ2, EH1–EH4). The goal now is to extend `VFT_Final_Analysis.ipynb` into a **complete, submission-ready notebook** that:

1. **Adds multilingual transformer embeddings** (LaBSE / MuRIL / IndicBERT) for semantic clustering
2. **Answers all three confirmatory RQs and exploratory hypotheses** with proper statistics
3. **Completes the SpAM analysis** (Phase 2) using the coordinate data in `responses.json`
4. **Adds Hindi fluency proficiency as a predictor** of retrieval efficiency
5. **Updates / extends the final report** (`Report_Final.md`)

---

## Repository Scan Summary

| File | Role |
|---|---|
| `responses.json` | Raw session data (35 sessions). Each session has `data[]` array with trial objects (trial_type: `html-keyboard-response`, `survey-multi-choice`, etc.). SpAM coordinate data is likely in a canvas/drag trial type. |
| `vft_responses.csv` | Processed per-word VFT data (1045 rows): subject_id, session_id, domain, word, rt_ms, position, language_type |
| `vft_responses_enriched.csv` | Same + demographics (Hi_Read, Hi_Write, En_Read, En_Write, first_language, hi_confidence, state_ut, age, gender, education, etc.) |
| `VFT_Final_Analysis.ipynb` | Current analysis notebook (partial) |
| `eda.ipynb` | Exploratory data analysis |
| `images/` | 34 existing figures (VFT figs 01–17, SpAM figs 01–08, demo figs) |

---

## Proposed Changes

### Section A — New Notebook: `VFT_Complete_Analysis.ipynb`

A new notebook replacing/extending `VFT_Final_Analysis.ipynb` with all sections below.

---

#### § 0 – Setup & Data Loading
- Install/import: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `sklearn`, `pingouin`, `statsmodels`, `transformers`, `torch`
- Load `vft_responses_enriched.csv`
- Parse `responses.json` to extract SpAM coordinates
- Define consistent colour palette and style

---

#### § 1 – Preprocessing & EDA (extend existing)
- Already done: IRT computation, language tagging, cluster scoring
- **Add:** Normalization check, outlier flagging (IRT > 60 s → likely page-away events)
- **Add:** Confirmation of 30 vs 35 participants (report says 30, data has 35)

> [!WARNING]
> The Mid-Report abstract says "Thirty-five IIIT Hyderabad students" but the requirements say 30. Need to confirm and reconcile.

---

#### § 2 – Demographics Analysis (already in `gen_demographics.py`)
- Reproduce demographic figures using enriched CSV
- Add: Fluency proficiency distribution (hi_confidence, en_confidence)
- Add: State distribution map

---

#### § 3 – VFT Core Analysis (extend existing RQ1 & RQ2)

**RQ1: Within-Cluster vs Between-Cluster IRTs**
- Welch t-test (already done, t(34)=−8.91, p<.001, d=1.51) — verify with `pingouin`
- Add: Wilcoxon signed-rank test (non-parametric backup given skewness)
- Add: Mixed-effects ANOVA: `IRT ~ cluster_type * domain + (1|subject_id)` using `statsmodels`
- Add: Effect size reporting (Cohen's d, r)
- Plots: violin + box + raw points per domain

**RQ2: Cluster Size → Fluency Score**
- Pearson r (already done, r=.55, p<.001) — verify with `pingouin`
- Add: Spearman ρ backup
- Add: OLS regression with bootstrapped CIs
- Add: Multiple regression: `fluency ~ mean_cluster_size + language_count + hi_confidence`

**EH1: Domain IRT Differences**
- Kruskal-Wallis H test across 4 domains
- Dunn post-hoc tests with Bonferroni/FDR correction

**EH2: Serial Position Effect**
- Mixed-effects regression: `log(IRT) ~ position * domain + (1+position|subject_id)` using `statsmodels MixedLM`

---

#### § 4 – Transformer-Based Semantic Embeddings (NEW)

Model choice: **LaBSE** (Language-agnostic BERT Sentence Embeddings) — handles Hindi, Hinglish, English equally well. Fallback: `ai4bharat/indic-bert` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Steps:
1. Normalise all words (strip whitespace, lowercase transliteration)
2. Encode all unique words per domain using LaBSE (`sentence-transformers` library)
3. Build per-domain cosine similarity matrices
4. Compute **neighbourhood density** = mean cosine similarity to k=3 nearest neighbours
5. Hierarchical clustering (Ward) on 1−similarity distance matrix

**Analysis using embeddings:**
- Do temporal clusters (adaptive IRT threshold) align with embedding-derived clusters?
  - Adjusted Rand Index (ARI) between temporal and embedding clusters
- Does neighbourhood density predict IRT?
  - `IRT ~ neighbourhood_density + position + (1|subject_id)` — Mixed LM
- t-SNE / UMAP visualisation of word embeddings coloured by domain and temporal cluster

**Outputs:**
- `vft_fig18_embedding_clusters.png` — clustered embedding heatmaps per domain
- `vft_fig19_tsne_domain.png` — 2-D t-SNE per domain
- `vft_fig20_density_irt_scatter.png` — neighbourhood density vs IRT

---

#### § 5 – SpAM Analysis (Phase 2 — complete)

**5.1 Extract SpAM data from `responses.json`**
- Iterate sessions, find trial objects with `trial_type` containing canvas/drag
- Extract per-word (x, y) normalised coordinates per domain per participant
- Build pairwise Euclidean distance matrices

**5.2 Consensus Distance Matrices**
- For each domain, average pairwise distances across participants
- Clamp to words appearing in ≥ 5 participants

**5.3 MDS Visualisation** (existing `spam_fig02_mds_maps.png` — regenerate with better labels)

**5.4 Hierarchical Clustering on SpAM**
- Ward linkage; determine optimal k via silhouette and gap statistics
- Label clusters with dominant words (existing `spam_fig03_dendrograms.png`)

**5.5 RQ3: SpAM Distance → VFT IRT (Confirmatory)**

> **H0:** ρ(SpAM distance, VFT IRT) = 0  
> **H1:** ρ > 0 (one-tailed)

Method:
- For each same-domain word-pair, get SpAM consensus distance + mean VFT IRT gap between consecutive appearances
- Spearman ρ per domain, then meta-analytic summary
- BH FDR correction across 4 domains
- Multilevel model: `IRT_gap ~ SpAM_distance + domain + (1|subject_id)` 

**5.6 Neighbourhood Density Cross-Task**
- Compute SpAM neighbourhood density per word
- Correlate with mean VFT IRT per word
- Compare density across domains (Foods > Animals > Body-parts > Colours?)

**Plots to add/update:**
- `spam_fig09_rq3_correlation.png` — scatter of SpAM distance vs VFT IRT
- `spam_fig10_cross_task_summary.png` — forest plot of domain-level correlations

---

#### § 6 – Hindi Fluency as Predictor (RQ from prompt)

**"Does Hindi fluency predict lexical retrieval efficiency?"**

Variables from enriched CSV:
- `hi_confidence` (1–5 self-rated)
- `Hi_Read`, `Hi_Write` (1–5)
- Composite `hi_fluency_score = mean(hi_confidence, Hi_Read, Hi_Write)`

Outcomes:
- Total Hindi words per participant
- Mean IRT
- Mean cluster size

Methods:
- Pearson/Spearman correlations
- Multiple regression: `total_hindi_words ~ hi_fluency + language_count + age + education`
- Group comparison: high-fluency (hi_confidence ≥ 4) vs low-fluency t-test

Plots:
- `vft_fig21_fluency_regression.png` — scatter matrix of fluency predictors vs outcomes

---

#### § 7 – Summary Statistics Tables

Publication-quality tables:
- Table 1: Participant demographics
- Table 2: IRT descriptives by domain
- Table 3: Cluster metrics by domain
- Table 4: Hypothesis test summary (RQ1, RQ2, RQ3 + all EHs)
- Table 5: Embedding cluster alignment ARI per domain

---

### Section B — Updated Report: `Report_Final.md`

Extend `Report_Mid.md` with:
- New section: **Embeddings-Based Semantic Clustering**
- New section: **SpAM Phase 2 Results** (complete with RQ3)
- New section: **Hindi Fluency as Predictor**
- Updated **Abstract** with Phase 2 results
- Updated **References** (add Hills et al. 2012 optimal foraging; Dautriche et al. 2016; psyarxiv/2bazx)

---

## Open Questions

> [!IMPORTANT]
> **Q1: Participant count** — The mid-report says N=35 but the requirements say "Number of participants: 30". Should we filter to the 30 Hindi-comfortable participants, or keep all 35?

> [!IMPORTANT]
> **Q2: SpAM data availability** — The `responses.json` contains trial data but we couldn't confirm the exact trial type name for SpAM coordinates (`html-keyboard-response`, `survey-multi-choice`?). The existing `spam_fig*` images suggest SpAM was already extracted. Do you want us to re-extract from JSON or use the existing SpAM figures as-is?

> [!NOTE]
> **Q3: Transformer model preference** — LaBSE is highly recommended for multilingual Hindi/English text. However it requires ~1.8 GB download. An alternative is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~120 MB). Which is acceptable given your compute?

> [!NOTE]
> **Q4: Final deliverable format** — Should the final report remain as Markdown (compiled via Pandoc to PDF) or do you need a separate `.ipynb` notebook as the primary submission artifact?

---

## Verification Plan

### Automated
- Run all notebook cells end-to-end without errors
- Assert key statistics match mid-report values (RQ1: t≈-8.91; RQ2: r≈.55)
- Check all figure files are generated in `images/`

### Manual
- Inspect all plots for correct labels, axes, and colour encoding
- Verify RQ3 SpAM-VFT correlation direction and significance
- Review Report_Final.md for completeness and academic tone
