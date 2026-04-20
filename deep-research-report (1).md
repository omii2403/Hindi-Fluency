# Analysis of Hindi Verbal Fluency Study

**Executive Summary:** We reviewed an IPython notebook analyzing a Hindi verbal fluency experiment with 35 bilingual participants. The study involved two tasks: a 60-second word-generation task (VFT) across four semantic categories (animals, foods, colours, body-parts) and a spatial-arrangement task (SpAM) where participants placed their words by similarity in 2D. The notebook posed seven research questions (RQ1-RQ7) about domain differences, participant factors, spatial organization, and phonological effects. We reconstruct the analysis steps, clarify each statistical test, report all results, and interpret them in simple terms. Key findings include: **no significant domain differences** in word count or retrieval speed (RQ1); **no support** for the hypothesis that higher self-rated Hindi confidence predicts better fluency (RQ2); **no clear differences** in spatial compactness by domain (RQ3); **mixed semantic alignment evidence** in RQ4 (sign-test positivity but permutation-based non-significance), with phonetic alignment not supported; **no reliable improvement** from composite scoring over VFT-only (RQ5); and **no phonological facilitation effect** (RQ6-RQ7) in this sample.  

**Update note (MuRIL rerun):** The notebook was re-run after switching embeddings to `google/muril-base-cased` and keeping Devanagari words in native script (no Devanagari-to-Latin conversion). The observations below reflect this rerun.

## Study Context and Data

- **Design:** 35 Hindi-English bilinguals completed (1) a **Verbal Fluency Task (VFT)**: naming as many words as possible in 60 seconds within four semantic categories (animals, foods, colours, body-parts); and (2) a **Spatial Arrangement Method (SpAM)** task: placing their produced words on a screen by perceived similarity. An exit survey collected demographics and self-rated Hindi confidence (3–5 scale).  
- **Data:** The analysis focused on **723 Hindi/Hinglish word responses** (out of 1040 total), spanning all categories (subjects could skip categories). Each response has timestamps (to compute inter-response times, IRT) and normalized (x,y) coordinates from SpAM. Participant count: 35. We treat each “subject_id” as independent.

- **Variables (key):**  
  - *Total words* (per participant): count of distinct Hindi words produced.  
  - *IRT* (inter-word response time, ms): time between consecutive word entries. Longer IRT means slower retrieval. The notebook often reports both individual IRTs and participant-level mean IRT.  
  - *Cluster metrics:* Words were algorithmically grouped into semantic clusters based on retrieval sequence. Relevant measures include *within-cluster IRT* (time between words in same cluster) vs *between-cluster IRT*, and *mean cluster size*.  
  - *SpAM compactness:* For each participant-category, the mean nearest-neighbor distance among their placed words. Smaller = tighter clusters.  
  - *Alignment metrics:* Clustering (e.g. with UMAP on word embeddings) of words by meaning or phonetic form was compared to participant-based clusters, quantified by Adjusted Rand Index (ARI).  
  - *Hindi confidence:* Self-rating (3–5, higher = more confident).  

- **Preprocessing:** The notebook filtered data to Hindi/Hinglish rows (723). It removed extremely short IRTs (<200 ms) as likely artifacts. We note no explicit missing-data imputation is needed, as analysis is complete-case. Importantly, **“Colours” category had very low coverage** (only 4 of 35 participants provided words), so inferential tests treat it as descriptive-only (excluded from omnibus tests) per the analysis plan.

## Phase 2: Descriptive Stats and Distributions

**Goal:** Summarize IRT and word counts by domain (RQ1 background).  

- **Process:** Compute by-domain means, medians, SD, skewness (for IRT) and participant-level word-count summaries. Test for normality using Shapiro–Wilk.  

- **Results (Table):**

  | Domain      | N (words) | Mean IRT (ms) | Median IRT (ms) | SD (ms) | Skewness | Word-count (mean ± SD) |
  |-------------|-----------|---------------|-----------------|---------|----------|------------------------|
  | Animals     | 233       | 6547.07       | 5595.10         | 4695.36 | 2.97     | 8.32 ± 2.93 (n=28)     |
  | Foods       | 258       | 6289.96       | 5016.55         | 5289.14 | 2.29     | 7.37 ± 3.46 (n=35)     |
  | Colours     | 41        | 4938.68       | 3483.50         | 3499.81 | 0.73     | 10.25 ± 3.40 (n=4)     |
  | Body-parts  | 191       | 6580.88       | 5234.10         | 4928.08 | 2.55     | 8.30 ± 2.88 (n=23)     |

  We see all domains have **positively skewed IRT distributions** (Skewness > 1 for 3 of 4 domains). Shapiro–Wilk tests confirm: IRT is significantly non-normal in all domains (p < 0.001 for animals, foods, body-parts; p = 0.0027 for colours). For word counts, “foods” is non-normal (p < 0.001) while others are roughly normal. Median IRT suggests *colours* yields the fastest retrieval (3483 ms) and *animals* the slowest (5595 ms), but formal tests (below) account for these.

- **Visuals:** A histogram (Graph 2.1) and boxplot (Graph 2.2) of IRTs confirm long right tails and large spread in animals/foods/body-parts, with colours showing less extreme values (but only 4 data points).

- **Interpretation:** Because IRTs are highly skewed and non-normal, we should avoid parametric mean-based tests on raw IRT (the notebook notes this as a methodological precaution). Also, due to the very small sample in colours (n=4 participants), any inference involving it is unreliable. The data profile simply shows large individual differences in fluency (range of word counts from 2 to 50 per participant) and IRTs ranging from ~1895 to ~15169 ms.

*Implication:* We will use non-parametric or robust methods for testing differences across domains (the notebook follows this, e.g. Kruskal-Wallis for RQ1). 

## Phase 3: Participant Profiles and RQ2 (Confidence Effects)

**Goal (RQ2):** Test whether self-rated Hindi confidence correlates with VFT performance (productivity or speed). Hypotheses (from pre-registration):  
- H5: Confidence positively correlates with total words (higher confidence → more words).  
- H6: Confidence negatively correlates with mean IRT (higher confidence → faster retrieval).  
- H7: Confidence positively correlates with mean cluster size.  
All are directional (one-tailed) expectations.

- **Data Prep (Cell 38-39):** Built a participant table with columns: total_words, mean_irt, median_irt, number of domains attempted, `hi_confidence`, plus clustering metrics (n_clusters, mean_cluster_size, etc.). Confidence column (`hi_confidence`) was found for all 35 participants (mean ≈ 4.31, SD ≈ 0.72, range 3–5). Notably, many reported maximum confidence (4.0 or 5.0), causing ceiling restriction.

- **Diagnostic:** Before hypothesis tests, the notebook checked the “confidence” distribution (Graph C1B) – it is heavily skewed to the high end. They also checked that a **range restriction** existed (16/35 at ceiling). Indeed, an initial Spearman correlation of confidence vs productivity was -0.395 (p=0.0188), suggesting the unexpected *higher confidence, lower word count* trend. After trimming the top/bottom 25% of confidence values, the correlation weakened (ρ = -0.209, p=0.268). This suggests that ceiling effects could be distorting any true relationship.

- **Hypothesis Tests (Cell 20 outputs):** The notebook used **Spearman’s rank correlation** (one-tailed) for H5–H7. Results:

  | Hypothesis | Variables (Spearman ρ) | Observed ρ | p(two-tailed) | p(one-tailed) | Supported? |
  |------------|------------------------|------------|---------------|---------------|------------|
  | H5         | Total words vs hi_confidence, expected ρ>0 | -0.395 (negative) | 0.0188 | 0.9906 | No (opp. sign) |
  | H6         | Mean IRT vs hi_confidence, expected ρ<0   | +0.276 (positive) | 0.1092 | 0.9454 | No (opp. sign) |
  | H7         | Mean cluster size vs hi_confidence, expected ρ>0 | -0.226 (negative) | 0.2062 | 0.8969 | No (opp. sign) |

  All p(one-tailed) are near 1, because the observed correlations are in the opposite direction of the hypotheses (e.g., negative when positive was expected). No result is significant or even in the predicted direction. Cohen’s rule of thumb (though not needed here) suggests these effects are modest.

- **Interpretation:** Contrary to expectations, *higher self-rated confidence was not associated with better fluency*. If anything, we observed an inverse trend (though not significant). The likely reason is ceiling effects on the confidence measure; the participants’ confidence scores didn’t vary much, undermining any clear correlation (supported by the diagnostic check). Graph C2 (panel scatterplots) shows no clear positive trend, and Graph 3.2 (efficiency bubble chart) shows no systematic confidence gradient.  

**Conclusion for RQ2 (H5–H7):** No evidence that personal Hindi confidence predicts word count, speed, or clustering. Thus RQ2 answered negatively: *Hindi confidence does not reliably explain fluency performance in this sample.*

## Module A: Semantic Clustering (H1) and Lexical Exhaustion (H2)

This “Module A” implements foundational VFT analyses (related to RQ1). The pre-registered hypotheses:

- **H1 (Semantic clustering):** Within-cluster IRT vs between-cluster IRT. *H₀:* Mean within-cluster IRT = mean between-cluster IRT. *H₁:* Between-cluster IRT > Within-cluster IRT (i.e., retrieving words that jump to a new semantic cluster takes longer). Test: Welch’s t-test (independent two-sample, one-tailed) plus Cohen’s d for effect size.

- **H2 (Lexical exhaustion):** Does retrieval slow down over time? Use a mixed-effects model predicting log-RT by serial position. *H₀:* slope = 0 (no change), *H₁:* slope > 0 (slowing down; later words take longer). Tested domain-wise with a linear mixed-effects model (`IRT ~ position + (1|subject)`), one-tailed on the position coefficient.

### Module A – Data Preparation (Cell 6–7)

The notebook combined VFT and SpAM data (merged dataset). It reports: **Hindi/Hinglish rows: 723** (from ~1040 total), Participants: 35. All four domains present (with very low “colours” usage noted). It built “cluster_records” by grouping each subject’s sequence of words into semantic clusters using an adaptive threshold (algorithmic detail not fully shown). This yielded 172 cluster-transition records (i.e. cases of within vs between cluster transitions). A robustness check is mentioned but outputs not fully given.

### H1 Test (Cell 8)

- **Procedure:** They compared all “within-cluster” IRTs (when a participant stays in the same semantic cluster) vs “between-cluster” IRTs (transition to a new cluster). Since these two samples have unequal variance, they used **Welch’s t-test** (a version of the two-sample t-test not requiring equal variances【1†L41-L43】) and conducted it one-tailed (expecting between > within). They also computed Cohen’s d for effect size.

- **Results (from output):**  
  - Pooled across domains: *Within-cluster mean IRT* = **5841.9 ms** (n = 646), *Between-cluster mean IRT* = **10960.3 ms** (n = 69).  
  - Welch’s t = **9.30**, one-tailed p < 0.0001. Cohen’s d ≈ **1.12** (a large effect by conventional benchmarks, e.g. d>0.8 is large【10†L1-L3】). The ratio of means (between/within) is ~1.88×. 
  - All four domains separately also showed significant longer between-cluster times (e.g. animals: t=6.31, p<0.0001; foods: t=4.41, p=0.0001; colours: t=4.10, p=0.0027; body-parts: t=4.58, p=0.0001). (Though “colours” has tiny N, so treat that cautiously.)

- **Interpretation:** The time to produce a word that starts a new cluster is **much longer** on average than time to produce consecutive words within a semantic cluster. In plain language: *participants tend to “search” longer after they leave a sub-category*. This strongly supports a semantic clustering process (as theorized by Troyer et al., 1997). All effects are highly significant and the effect size is large, indicating a robust difference【1†L41-L43】. The figures (Graph H1A violin + H1B scatter) visually confirm that between-cluster latencies are longer.

- **Assumptions:** Welch’s t-test assumes each group’s sample is roughly normal. Given the large sample of IRTs (646, 69) and known non-normality, strictly speaking this is a mild assumption violation, but with such strong effects the conclusion is still meaningful. (Nonparametric alternatives exist, but the huge effect size makes it clear.)

- **H1 Conclusion:** *Supported.* Participants retrieve words in identifiable semantic clusters (within-cluster transitions are faster).

### H2 Test (Cell 10)

- **Procedure:** For each domain, they ran a linear mixed-effects model predicting log-RT by word position (i.e. serial order 1st, 2nd, ...), with subject as a random intercept. This tests if there’s an overall “slowing down” across retrieval. (Taking log of IRT helps with skew.) They then inspected the slope β: under H₀ it should be ≤0 (no increase), under H₁ we expect β>0 (IRTs get longer). A one-tailed inference was planned. They report each domain’s β and p-value.

- **Results:** None of the domains showed a significant positive slope. In fact, all estimated slopes were **negative** (faster times on average as position increases) or not significantly positive. E.g. animals: β = –376 ms/position (p_one-tailed = 0.9705); foods: β = –653 (p=1.0000); colours: β = +364 (p≈0.055, a non-significant upward trend only for n=4); body-parts: β = –718 (p=0.9949). All one-tailed p-values > 0.05. (The output says 0/4 domains met the directional criterion.)

- **Interpretation:** Contrary to the “lexical exhaustion” idea, later responses were not slower; if anything, there was a (non-significant) trend toward faster responses over time (except in “colours”, which was very small N). Graph H2 (regression lines) shows mostly flat or downward slopes. We must conclude H2 is **not supported** in this sample. In simple terms: *we do not find consistent evidence that participants slow down as they exhaust easier words*【1†L41-L43】. It might be due to practice effects, strategic chunking, or just noise.

- **Assumptions:** Linear mixed-effects models assume normal residuals, homoskedasticity, and linearity. The notebook did not report formal residual checks, but given non-significance, assumption violations are less critical. With small domain samples, one-tailed testing here may be underpowered.

- **H2 Conclusion:** *Not supported.* No reliable increase in IRT with word order; thus no clear lexical exhaustion effect was found.

## Module B: Domain Differences (RQ1, H3 & H4)

**Goal (RQ1):** Are there differences **across semantic categories** in VFT performance?  
- **H3:** Productivity (total words). *H₀:* All domains have the same median word count. *H₁:* At least one differs. Test plan: Kruskal–Wallis (non-parametric one-way ANOVA on ranks) on participant word counts (using participants as units, N differs by domain).  
- **H4:** Retrieval speed (IRT). *H₀:* All domains have the same distribution of IRTs. *H₁:* At least one differs. Test: Kruskal–Wallis on all response-level IRTs by domain. (“Colours” excluded from inferential tests; only animals, foods, body-parts used.)

- **Normality Check:** Confirming Phase 2, word-counts: foods non-normal; IRT non-normal in all.

- **Tests and Results:**

  - **H3 (word count):** Kruskal–Wallis on 3 domains (animals n=28, foods n=35, body-parts n=23). The test statistic H = **3.264**, df=2, p = **0.1956**. Effect size (ε² ≈ 0.015) is tiny. Conclusion: no significant differences (p > 0.05).  
  - **H4 (IRT):** Kruskal–Wallis on IRT (over all responses) for 3 domains. H = **3.903**, df=2, p = **0.1421** (ε² ≈ 0.0028). Also non-significant.

- **Visuals:** Graph B1A/B show boxplots and points of word counts: medians are roughly similar (animals ≈8.5, foods ≈7, body ≈8). Graph B2A/B for IRT densities/boxes: the distributions largely overlap, with similar right-tailed shapes.

- **Interpretation:** No domain difference in either number of words or retrieval speed was statistically supported. In plain terms: *Hindi speakers performed similarly across animals, foods, and body-parts categories.* (“Colours” was not tested due to tiny n.) Any observed medians differences (e.g. animals slightly higher productivity) could be random. This suggests RQ1 (domain differences) is **not confirmed**: the analysis finds *no significant domain effect* on productivity or speed.

- **Assumptions:** Kruskal–Wallis assumes independent samples and similar-shaped distributions. We have independent participants, and given severe skew (non-normal), KW was the correct non-parametric choice【5†L162-L166】. (We did not pursue post-hoc pairwise tests since omnibus was non-significant.)

- **H3/H4 Conclusion:** *Not supported.* No evidence of domain effects under required tests.

## Module D: SpAM Structure (RQ3–RQ4, H8–H10)

**Goal:** Analyze the spatial placement (SpAM) of words.  
- **RQ3:** Compare **spatial compactness** of word clouds across domains. Hypothesis H8.  
- **RQ4:** Check alignment of participant spatial clusters with pre-defined cluster models: H9 for semantic embeddings, H10 for phonetic embeddings. (The notebook uses ARI to quantify alignment.)  

Hypotheses summarized (from plan):
- **H8:** Mean nearest-neighbor distance (compactness) differs by domain. Test: Kruskal–Wallis on each participant’s domain-specific compactness.  
- **H9:** ARI of SpAM clusters vs semantic model exceeds chance. *H₀:* ARI = 0 (null level). *H₁:* ARI > 0. Tested via a sign test on domain-level ARIs and permutation.  
- **H10:** ARI of SpAM vs phonetic model > chance. Tested similarly.

- **Data:** They merged all (x,y) coordinates of Hindi words by participant and category (85+ valid participant-category records depending on filtering stage). In the rerun, the semantic model used **MuRIL** (`google/muril-base-cased`) and retained native Devanagari text. A phonological representation was also tested. Clustering (KMeans/Agglomerative/HDBSCAN) and ARI-based alignment were then compared against participant SpAM structure.

- **H8 – Compactness (Cell 25):**  
  - **Descriptives:** Mean nearest-neighbor (NN) distances: animals 0.0578, foods 0.0596, colours 0.0643, body-parts 0.0794 (lower = tighter clusters). (n_records equals n_subjects because 1 record per participant-domain.)  
  - **Kruskal–Wallis:** H = **3.445**, df=3, p = **0.3280** (all four domains). Excluding “colours” (low N): H = **3.315**, df=2, p = **0.1906**. Both p > 0.05.  
  - **Conclusion:** No significant domain effect on SpAM compactness. Body-parts appear descriptively tightest and colours loosest, but not beyond chance.

- **H9 – Semantic alignment:**  
  - In the rerun Module-D output: **68/86** participant-category units had semantic ARI > 0, mean ARI = **0.1765**, sign-test p = **0.0000**, but permutation p = **0.5022**.  
  - In the extended permutation analysis (Phase 9 deep-dive): **17/44** positive ARIs, sign-test p = **0.9519**, mean observed ARI = **-0.0062**, mean null ARI = **0.0007**, and 0/44 units significant by permutation.  
  - **Interpretation:** Evidence is **mixed** and not robust under permutation-based inference. The sign-test-only signal is not corroborated by permutation tests.
  - **Decision:** Treat H9 as **weak/inconclusive** rather than robustly supported.

- **H10 – Phonetic alignment:**  
  - In rerun Module-D output: **52/86** participant-category units had ARI > 0, mean ARI = **0.0745**, sign-test p = **0.0662**, permutation p = **0.5034**.  
  - **Conclusion:** No reliable above-chance phonetic alignment. H10 remains **not supported**.

- **Graphs:** Graph D2 shows boxplots of ARIs by domain and a bar of means. Semantic ARIs are generally higher than phonetic, but all are low (<0.3). 

- **Interpretation (RQ3–RQ4):**  
  - **SpAM Compactness (RQ3/H8):** No statistical domain differences. We cannot conclude some categories had tighter spatial maps than others.  
  - **SpAM-Semantic Alignment (RQ4/H9):** The rerun does not show stable above-chance alignment once permutation-based criteria are enforced. In plain terms: there may be weak positive overlap in some summaries, but it is **not robust** under stronger null testing.  
  - **SpAM-Phonetic Alignment (H10):** No evidence that sound similarity drives spatial grouping beyond chance.

- **Assumptions:** Kruskal–Wallis and sign tests have minimal assumptions (independence, ordinal data)【5†L162-L166】. Here sample sizes for ARI analyses are moderate (85 records). The sign test is non-parametric and appropriate for testing above-chance ARI.

- **Conclusions:** *H8 not supported.* *H9 inconclusive/weak.* *H10 not supported.*

## Phase 12: Phonological Similarity (RQ6–RQ7, H11–H12)

**Goal:** Replicate findings of Kumar et al. (2022) about phonological overlap. (The notebook title says RQ6–RQ7 but content calls these H11/H12.)  
- **H11:** Does phonological similarity of consecutive word pairs **increase** over retrieval order? (They mention also semantic similarity decreasing, but focus is phonological vs position.)  
- **H12:** Does higher mean phonological similarity predict more words? (Spearman ρ > 0 with total words.)  

- **Procedure:** For each pair of consecutive responses, they measured phonological similarity (1 – [Levenshtein distance]/max_length) and semantic similarity (cosine of embeddings). Then they fit a mixed-effects model of (phonological – semantic) similarity vs position. For H12 they correlated mean phon_similarity with total_words.

- **Results (rerun):**  
  - H11 interaction (type × position): beta = **+0.003127**, p = **0.0905** (not significant).  
  - Phonological slope: beta = **+0.000077**, p = **0.9699**.  
  - Semantic slope: beta = **-0.000141**, p = **0.0002**.  
  - H12 Spearman (mean phonological similarity vs total words): rho = **+0.114**, one-tailed p = **0.2570**.  
  - Both H11 and H12 are marked **not supported**.

- **Interpretation:** The analysis found *no strong evidence* that participants increasingly use sound patterns in later responses, nor that those with more phonological overlap produced more words. Essentially, RQ6 and RQ7 are not supported in this dataset. In simple terms: *no clear phonological priming effect was detected*. (Given the complexity, we trust the notebook: p=0.4689 for the test of Rho differences also suggests no effect in H12.)

- **Assumptions:** The mixed model assumes linearity and normally-distributed residuals for similarity differences; Spearman ρ assumes monotonic relation. The notebook does not detail assumption checks, but sample size is moderate (∼640 transitions) so normality is less critical.

## Phase 13: Composite Fluency Score (RQ5, H13)

**Goal (RQ5):** Evaluate whether a combined VFT+SpAM score better captures fluency vs VFT alone. They constructed an “integrated” score combining total words, mean IRT, cluster info, and spatial compactness. They tested if confidence (or presumably “language fluency score”) correlates more strongly with the integrated score than with the VFT-only score. Hypothesis: H13 – The dependent correlation coefficient with confidence is different between the two scores.

- **Procedure:** They compute Spearman ρ(confidence, VFT-only score) and ρ(confidence, integrated score), then use **Steiger’s Z-test** for comparing two dependent correlations (same confidence variable)【3†L174-L176】.

- **Results (rerun Cells 118):**  
  - VFT-only vs confidence: rho = **-0.461**, p = **0.0053**.  
  - Integrated vs confidence: rho = **-0.461**, p = **0.0053**.  
  - Steiger test for difference: z = **0.000**, p = **1.0000** (non-significant). 

  In the rerun, both scores have identical correlation magnitude and significance; there is no measurable difference between them.

- **Interpretation:** Higher confidence again correlates negatively with performance score (rho < 0). However, after rerun, integrated and VFT-only scores perform identically against confidence (same rho and p), and Steiger confirms no difference.

  In plain terms: *adding SpAM/spatial factors did not change the confidence correlation at all in this rerun*.

- **Conclusion (H13):** *Not supported.* No statistically reliable difference between integrated and VFT-only score correlations (p = 1.0000).

## Summary of Findings and Recommendations

- **RQ1 (domain differences):** No significant effect. Hypotheses H3/H4 not supported. Conclusion: *Hindi word retrieval did not show clear differences across animals, foods, body-parts categories*.

- **RQ2 (participant variables):** Self-rated Hindi confidence did **not** positively predict fluency. H5–H7 all failed (observed trends were opposite). Likely cause: restricted range on confidence. *Thus, personal confidence was not a useful predictor* here.

- **RQ3 (SpAM compactness):** No reliable domain effect. H8 not supported. Descriptively, “body-parts” words were more tightly clustered, but p>0.05.  
- **RQ4 (SpAM alignment):** With MuRIL rerun and stricter permutation interpretation, semantic alignment is **mixed/inconclusive** (not robust above chance), and phonetic alignment (H10) remains unsupported.

- **RQ5 (composite score):** Integrated VFT+SpAM score did not outperform VFT alone; in rerun they are numerically identical against confidence (rho = -0.461 for both). H13 not supported.

- **RQ6/H11 & RQ7/H12 (phonology):** No evidence found for phonological facilitation. Neither increasing sound-similarity over sequence nor its relation to productivity was significant.

**Limitations:** The sample is relatively small (n=35) and uneven by domain. The “colours” category has only 4 participants, so all inferences about colours are tentative or descriptive. Many variables (IRT, word counts) are skewed or have ceiling effects, so non-parametric tests and caution in interpretation are appropriate. The language confidence measure had restricted variability, limiting its usefulness.

**Recommendations:** Future studies could collect larger samples per category and use more sensitive measures of language skill. Alternative approaches (e.g. log-transformed IRT, Bayesian models, or power analyses for one-tailed tests) might clarify ambiguous trends. Also, collecting more nuanced confidence or proficiency metrics could help test RQ2 more decisively.

**Statistical References:** We used standard tests: Welch’s t-test for unequal variances【1†L41-L43】, Shapiro–Wilk to assess normality【3†L174-L176】, Kruskal–Wallis for non-parametric group comparisons【5†L162-L166】, Spearman rank correlation for monotonic association, and Steiger’s test for comparing two correlated correlations. (Spearman’s ρ is a nonparametric rank correlation measure【7†L173-L178】.) Effect sizes (Cohen’s d, epsilon-squared) and confidence intervals (where not shown) should also accompany these tests in reporting.

