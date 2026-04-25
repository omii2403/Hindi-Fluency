# Hindi Fluency Notebook: Beginner-Friendly Hypothesis Guide

This document explains the hypothesis tests used in `Hindi_fluency_final.ipynb` in simple language. It is written for someone who is new to statistics and wants to understand:

- what data was used,
- what each hypothesis is asking,
- which test was applied,
- why that test was chosen,
- what the result means.

## 1) What this notebook is about

The notebook studies Hindi / Hinglish verbal fluency and SpAM data. In a verbal fluency task, people are asked to quickly say words from a category, like animals or foods. In the SpAM task, they place those words in a 2D space based on how similar they feel the words are.

The notebook tries to answer questions such as:

- Do people slow down when they move from one semantic cluster to another?
- Do some semantic domains produce more words than others?
- Does the SpAM map match semantic or phonological structure?
- Can phonological similarity predict how productive a participant is?
- Does combining VFT and SpAM information improve prediction compared with VFT alone?

## 2) Data used in the notebook

The main dataset is `merged_vft_spam_responses_enriched.csv`. The notebook first filters the data and then uses subsets of it for different tests.

### Main filters and subsets

- Reaction times (`rt_ms`) are filtered to keep values less than or equal to 60000 ms.
- The main inferential analyses focus on Hindi / Hinglish rows.
- The domains in the notebook are `animals`, `foods`, `colours`, and `body-parts`.
- The notebook treats `colours` as a low-sample-size domain, so it is mostly descriptive rather than used in some inferential comparisons.

### Important variables

- `rt_ms`: response time in milliseconds. Lower values mean faster responses.
- `subject_id`: the participant identifier.
- `domain`: the semantic category of the task, such as animals or foods.
- `word`: the spoken or produced word.
- `position`: the order of the word in a participant's response list.
- `language_type`: whether the response is Hindi, Hinglish, or another language-related label used in the dataset.
- `x`, `y`: the 2D SpAM coordinates assigned to a word or response.
- `cluster` or cluster labels: grouping labels used when comparing within-cluster and between-cluster behavior.
- `ARI`: Adjusted Rand Index, a score that measures how similar two clusterings are. A score near 0 means the agreement is about chance; larger positive values mean better agreement.

## 3) Short statistics primer

A few words appear repeatedly in the notebook, so here is the basic idea.

### p-value

A p-value tells you how surprising the observed result would be if the null hypothesis were true.

- Small p-value, usually below 0.05: evidence against the null hypothesis.
- Large p-value: not enough evidence to reject the null hypothesis.

### One-tailed vs two-tailed

- One-tailed test: used when the hypothesis predicts a direction, such as "greater than zero".
- Two-tailed test: used when the hypothesis only predicts a difference, not a direction.

### Effect size

A p-value tells you whether something is statistically significant, but not how large the effect is.

- Cohen's d: common effect size for mean differences.
- Correlation coefficient `rho` or `r`: tells you how strongly two variables move together.
- Larger absolute values mean stronger relationships.

### Mixed-effects model

A mixed-effects model is useful when data are repeated within participants. It can handle:

- fixed effects: the overall effect you want to study,
- random effects: participant-specific differences.

This is important because the same participant contributes many responses, so the observations are not independent.

## 4) Hypotheses and tests

### H1 - Semantic clustering should show faster retrieval within clusters

**Question:** Do people respond faster within a semantic cluster than when they move between clusters?

**Null hypothesis:** There is no difference in response time between within-cluster and between-cluster transitions.

**Alternative hypothesis:** Within-cluster responses are faster.

**Test used:** Welch's t-test, plus Cohen's d.

**Why this test:**

- Welch's t-test compares the mean of two groups.
- It is a good choice when the two groups may have unequal variance.
- Cohen's d helps show how large the difference is, not just whether it is statistically significant.

**Result:** Supported.

- Welch t = 9.2995
- Cohen's d = 1.1206
- p < 0.001

**Interpretation:**

This is a strong result. It means the data show a clear clustering effect: when participants stay inside a semantic cluster, they tend to retrieve words faster than when they shift to a new cluster. The effect size is large, so this is not just a tiny difference that happened to be statistically significant.

---

### H2 - Retrieval time should increase with retrieval position

**Question:** Do participants slow down as they produce more words, which would suggest lexical exhaustion or increased difficulty over time?

**Null hypothesis:** Retrieval time does not change systematically with position.

**Alternative hypothesis:** Retrieval time increases with position.

**Test used:** Linear mixed-effects model: `rt_ms ~ position + (1 + position | subject)`.

**Why this test:**

- The response time is measured repeatedly for each participant.
- A mixed-effects model can estimate a general trend across all participants while allowing individual participants to have their own baseline speed and their own slope over position.
- This is much better than using a plain regression, because plain regression would ignore the repeated-measures structure.

**Result:** Not supported.

**Interpretation:**

The notebook does not find a stable, reliable pattern showing that response times consistently increase as participants move later in the list. In simple words, the data do not strongly support the idea of a general lexical-exhaustion slowdown across position.

---

### H3 - Word counts should differ across semantic domains

**Question:** Do some semantic domains produce more words than others?

**Null hypothesis:** Mean word count is the same across domains.

**Alternative hypothesis:** At least one domain has a different mean word count.

**Test used:** Kruskal-Wallis test, with Shapiro-Wilk diagnostics first.

**Why this test:**

- Shapiro-Wilk is used to check whether the data look normally distributed.
- The notebook finds that the word-count distributions are not well behaved enough for standard ANOVA assumptions.
- Kruskal-Wallis is a non-parametric alternative to ANOVA and is safer when normality is doubtful.

**Result:** Not supported.

- Kruskal-Wallis H = 3.2639
- p = 0.1956

**Interpretation:**

The domains do not show a statistically reliable difference in word count in this analysis. So even if the average counts look a little different, the differences are not strong enough to conclude that domain alone changes productivity.

---

### H4 - Mean response time should differ across semantic domains

**Question:** Do some semantic domains lead to faster or slower retrieval overall?

**Null hypothesis:** Mean IRT is the same across domains.

**Alternative hypothesis:** At least one domain differs in mean IRT.

**Test used:** Kruskal-Wallis test, with Shapiro-Wilk diagnostics first.

**Why this test:**

- The notebook again treats the data as non-normal / not ideal for ANOVA.
- Kruskal-Wallis compares groups without assuming normality.

**Result:** Not supported.

- Kruskal-Wallis H = 3.9028
- p = 0.1421

**Interpretation:**

There is no strong evidence that one domain is reliably faster or slower than another in the available sample.

---

### H5 - SpAM clusters should align with semantic embeddings above chance

**Question:** Do the clusters made by participants in SpAM match the semantic structure of the words better than random chance would?

**Null hypothesis:** ARI between SpAM clusters and semantic embeddings is at chance.

**Alternative hypothesis:** ARI is greater than chance.

**Test used:** Binomial sign test plus an ARI permutation test.

**Why this test:**

- ARI gives a similarity score between two clusterings.
- The sign test checks whether the observed ARI is consistently above a chance baseline.
- A permutation test is useful because it builds a chance distribution by randomizing the labels or assignments, which helps answer: "Is this score better than we would expect by luck?"

**Result:** Partly supported.

- Mean ARI = 0.1765
- Sign test was positive, p < 0.001
- Permutation estimate = 0.5022

**Interpretation:**

The SpAM clusters do show some real semantic structure, but the effect is modest. The mean ARI is above zero, so the map is not random, yet it is not extremely strong. The important beginner-level takeaway is: participants were doing better than chance, but the semantic signal is only partial, not perfect.

A nuance in the notebook: one code section includes simulated or random ARI generation text. That should not be read as the main empirical result itself. The real interpretation comes from the observed ARI summaries and the sign / permutation checks.

---

### H6 - SpAM clusters should align with phonetic embeddings above chance

**Question:** Do SpAM clusters also match phonetic structure?

**Null hypothesis:** ARI between SpAM clusters and phonetic embeddings is at chance.

**Alternative hypothesis:** ARI is greater than chance.

**Test used:** Binomial sign test.

**Why this test:**

- Same logic as H5, but now the comparison is with phonetic structure rather than semantic structure.
- The goal is to see whether the participant maps reflect sound-based organization.

**Result:** Not supported.

- Mean ARI = 0.0745
- p = 0.0662

**Interpretation:**

There is some weak positive signal, but it is not strong enough to be called statistically significant at the usual 0.05 threshold. So the notebook does not have strong evidence that the SpAM clusters align with phonetic embeddings.

---

### H7 - Phonological similarity should change over retrieval order

**Question:** As participants continue producing words, does phonological similarity increase, while semantic similarity decreases?

**Null hypothesis:** There is no interaction between similarity type and retrieval position.

**Alternative hypothesis:** Phonological similarity increases with position while semantic similarity decreases.

**Test used:** Linear mixed-effects model: `similarity ~ type * position + (1 + position | subject)`.

**Why this test:**

- The same participant contributes many observations.
- The model needs to account for that repeated structure.
- The interaction term `type * position` checks whether the slope over position differs for semantic vs phonological similarity.

**Result:** Not supported.

- beta = -0.004144
- p = 0.1028

**Interpretation:**

The notebook does not find convincing evidence that similarity changes over retrieval order in the predicted way. The direction is not strongly supported statistically.

---

### H8 - Phonological similarity should predict higher word count

**Question:** Do people who show higher mean phonological similarity also produce more words?

**Null hypothesis:** Spearman rho = 0 between mean phonological similarity and total words.

**Alternative hypothesis:** Spearman rho > 0.

**Test used:** One-tailed Spearman correlation.

**Why this test:**

- Spearman correlation is used when you want to measure whether two variables move together in a monotonic way.
- It does not require the relationship to be perfectly linear.
- The notebook uses a one-tailed test because the prediction is specifically positive.

**Result:** Not supported.

- rho = 0.114
- p = 0.2570

**Interpretation:**

There is a small positive relationship, but it is weak and not statistically significant. In beginner language: people with more phonologically similar response patterns were not clearly more productive in a reliable way.

---

### H9 - Integrated VFT+SpAM score should outperform VFT-only score

**Question:** Does combining verbal fluency and SpAM information produce a better score than using VFT alone?

**Null hypothesis:** The two dependent correlations are equal in strength.

**Alternative hypothesis:** The correlations differ.

**Test used:** Steiger Z test for dependent correlations.

**Why this test:**

- The same participants contribute both scores.
- The correlations are therefore dependent, not independent.
- Steiger's test is designed specifically to compare two correlations that share a variable.

**Result:** Not supported.

- Integrated score vs confidence: rho = -0.461, p = 0.0053, n = 35
- VFT-only score vs confidence: rho = -0.461, p = 0.0053, n = 35
- Steiger Z: z = -0.000, p = 1.0000

**Interpretation:**

The integrated score does not outperform the VFT-only score here. They are essentially identical in their correlation with confidence, so there is no evidence that adding the SpAM component improves the prediction.

## 5) Overall takeaway

The notebook tells a fairly coherent story:

- H1 is the strongest positive result: semantic clustering clearly matters for retrieval speed.
- H5 shows that SpAM captures some real semantic structure, but only partially.
- H2, H3, H4, H6, H7, H8, and H9 are not strongly supported in this phase.

So the main scientific message is not "everything worked," but rather:

- semantic clustering is real,
- SpAM captures some structure,
- many other expected links are weak or absent in the current data.

## 6) How to read the notebook like a beginner

If you want the simplest mental model, use this order:

1. The notebook cleans the data and keeps only usable response times.
2. It checks whether the data are normal enough for standard tests.
3. If normality looks doubtful, it uses non-parametric tests.
4. For repeated measurements from the same people, it uses mixed-effects models.
5. For relationships between two numeric variables, it uses correlations.
6. For comparing two related correlations, it uses Steiger's test.

That is the core logic of the analysis.

## 7) Note on later labels in the notebook

The notebook also contains later planning or status notes around H10-H12. The only explicit later status statement I found clearly says that H12 is not formally tested in this phase because it requires phonological similarity data from completed SpAM spatial mappings.

So, for the fully executed hypothesis test story, H1-H9 are the main tested hypotheses in this phase.
