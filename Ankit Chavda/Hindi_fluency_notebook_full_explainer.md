# Hindi Fluency Notebook: Full Beginner-Friendly Explanation

## What this document does
This is a full walkthrough of the notebook in plain language, for someone who does not feel comfortable with statistics.

It explains:
- how the research questions were chosen,
- how each hypothesis comes from each question,
- why each statistical test was used,
- what each major graph is showing,
- and what each result means in practical terms.

I use the notebook's own values and decisions.

---

## 1) Big picture of the study

### Study design in simple words
The notebook studies Hindi/Hinglish verbal fluency using three sources:
- VFT task: say as many words as possible in 60 seconds for domains like animals, foods, colours, and body-parts.
- SpAM task: place words in 2D space based on perceived similarity.
- Exit poll: self-ratings like Hindi confidence.

Main analysis subset:
- Hindi/Hinglish responses only.
- About 723 response rows.
- 35 participants.

### Core idea behind the research questions
The questions are built from a cognitive pipeline:
1. Performance basics: Are some domains faster/easier than others? (RQ1)
2. Individual differences: Does self-confidence relate to performance? (RQ2)
3. Spatial structure: Do semantic maps differ by domain? (RQ3)
4. Human vs model structure: Do SpAM clusters match embedding clusters? (RQ4)
5. Scoring framework: Is combined scoring better than VFT-only? (RQ5)
6. Retrieval dynamics: Does phonological similarity increase over retrieval order? (RQ6)
7. Productivity link: Does phonological similarity predict more words? (RQ7)

So the notebook is not random tests. It moves from basic behavior -> participant factors -> map structure -> model alignment -> scoring comparison.

---

## 2) How hypotheses were identified from RQs

The notebook pre-defines hypotheses H1-H13 before reporting results.

### Mapping from RQs to hypotheses
- RQ1 (domain differences): H3, H4
- RQ2 (participant variables): H5, H6, H7
- RQ3 (SpAM compactness): H8
- RQ4 (alignment with embeddings): H9, H10
- RQ5 (combined vs VFT-only score): H13
- RQ6 (phonological shift over order): H11
- RQ7 (phonological similarity and productivity): H12
- Foundational VFT mechanisms: H1, H2 (used as base cognitive checks)

### Why one-tailed tests appear in many places
Many hypotheses are directional, not just "different".
Examples:
- "greater than" (rho > 0)
- "less than" (rho < 0)
- "increase over position" (beta > 0)

Because direction is pre-registered, one-tailed p-values are used in those cases.

---

## 3) Stats primer (quick, non-technical)

If you are new to stats, these are the only rules you need here:
- p-value: smaller means stronger evidence against H0.
- Common threshold: p < 0.05 is often treated as "statistically supported".
- Effect size: how big the pattern is (not just whether it is detectable).
- Correlation rho:
  - positive rho: as X goes up, Y tends to go up.
  - negative rho: as X goes up, Y tends to go down.
- Kruskal-Wallis: compares groups when normality is not safe.
- Mixed-effects model: useful for repeated data per participant.
- ARI (Adjusted Rand Index): overlap between two clusterings.
  - near 0 means chance-level overlap.
  - higher positive values mean stronger overlap.

---

## 4) Data preparation and quality checks

The notebook does the following setup:
- Reads merged enriched CSV.
- Keeps RT <= 60000 ms.
- Focuses on Hindi/Hinglish rows.
- Uses domains: animals, foods, colours, body-parts.
- Uses participant-level and response-level tables depending on the hypothesis.

Important data caveat repeatedly handled in notebook:
- colours has very low participant coverage (n=4 in several contexts).
- Because of this, colours is often treated as descriptive-only in inferential comparisons.

---

## 5) Detailed hypothesis-by-hypothesis explanation

## Module A: Foundational VFT structure

### H1: Between-cluster IRT > within-cluster IRT (semantic clustering)
Question in plain language:
- Do people pause longer when they jump between semantic clusters than when staying inside the same cluster?

Why this test:
- Two groups are compared (within vs between transitions).
- Welch t-test is used because variance/sample balance may differ.
- Direction is pre-registered (between should be larger), so one-tailed test is used.
- Cohen d is added to show effect size.

Result:
- Within mean IRT: 5841.89 ms (n=646)
- Between mean IRT: 10960.34 ms (n=69)
- Welch t = 9.2995
- One-tailed p < 0.0001
- Cohen d = 1.1206 (large)
- Ratio between/within = 1.8762
- Decision: H1 supported strongly.

Per-domain pattern:
- Animals: t=6.31, p(one)<0.0001, d=1.19 (supported)
- Foods: t=4.41, p(one)=0.0001, d=1.22 (supported)
- Colours: t=4.10, p(one)=0.0027, d=1.93 (supported; low N caution)
- Body-parts: t=4.58, p(one)=0.0001, d=0.86 (supported)

Graph interpretation:
- Graph H1 split violin: between-cluster values are shifted upward.
- Participant mean scatter vs y=x line: most points above diagonal.
- Visual and test agree fully.

Meaning:
- Retrieval is not random; it has semantic run structure with boundary costs.

---

### H2: Lexical exhaustion (IRT should increase with serial position)
Question in plain language:
- As people continue speaking, do they slow down because easy words are used up?

Why this test:
- Serial position is a slope question.
- Repeated observations per participant require mixed-effects model.
- Pre-registered direction is beta > 0, so one-tailed decision is used.

Result by domain:
- Animals: beta = -376.05 ms/position, p(one)=0.9705 (not supported)
- Foods: beta = -653.49, p(one)=1.0000 (not supported)
- Colours: beta = +364.37, p(one)=0.0549 (not supported)
- Body-parts: beta = -717.96, p(one)=0.9949 (not supported)
- Decision: H2 not supported (0/4 domains).

Graph interpretation:
- Most fitted slopes are negative, opposite to predicted slowdown.
- Colours is positive but just misses one-tailed threshold and has low N.

Meaning:
- This sample does not show reliable serial slowdown in the predicted way.

---

## Module B: Domain differences (RQ1)

### Why non-parametric tests were chosen here
Notebook runs Shapiro checks and sees non-normal RT distributions (and some non-normal word-count distributions).
So Kruskal-Wallis is correctly used for robust group comparison.

### H3: Domain differences in productivity (word count)
- Test: Kruskal-Wallis on inferential domains (animals, foods, body-parts)
- Result: H=3.2639, df=2, p=0.1956, epsilon^2=0.0152
- Decision: H3 not supported.

Graph B1 interpretation:
- Box/point overlap is strong across inferential domains.
- colours may look high descriptively, but low N prevents inferential claim.

### H4: Domain differences in retrieval speed
- Test: Kruskal-Wallis on response-level IRT for inferential domains
- Result: H=3.9028, df=2, p=0.1421, epsilon^2=0.0028
- Decision: H4 not supported.

Graph B2 interpretation:
- KDE curves overlap heavily.
- Boxplots show close medians and wide within-domain spread.

Meaning for RQ1:
- No strong inferential evidence that domain drives productivity or speed in this subset.

---

## Module C: Participant-level effects (RQ2)

### Confidence variable diagnostics
Notebook detects confidence column as hi_confidence.
- N=35 participants.
- Mean confidence=4.314 (range 3 to 5).
- Ceiling at max score: 16/35 (45.7%).

Why this matters:
- Heavy ceiling compresses ranking and weakens correlation reliability.

### H5: Confidence -> total words (expected positive)
- Observed rho = -0.3951
- p(two)=0.0188
- one-tailed p for expected positive direction = 0.9906
- Decision: not supported (direction opposite expectation).

### H6: Confidence -> mean IRT (expected negative)
- Observed rho = +0.2755
- p(two)=0.1092
- one-tailed p for expected negative direction = 0.9454
- Decision: not supported.

### H7: Confidence -> mean cluster size (expected positive)
- Observed rho = -0.2259
- p(two)=0.2062
- one-tailed p = 0.8969
- Decision: not supported.

Graph C1/C2 interpretation:
- Trends are in unexpected directions (H5 negative, H6 positive, H7 negative).
- Histogram confirms confidence ceiling effect.

Meaning for RQ2:
- In this sample, self-rated Hindi confidence does not behave as expected predictor.

---

## Module D: SpAM structure and alignment (RQ3, RQ4)

### H8: SpAM compactness differs by domain
Question:
- Are some semantic maps tighter (more compact) than others?

Test choice:
- Uses mean nearest-neighbor distance from coordinates.
- Kruskal-Wallis compares domains.

Result:
- All 4 domains: H=3.4451, df=3, p=0.3280, epsilon^2=0.0054
- Sensitivity excluding low-N colours: H=3.3148, df=2, p=0.1906
- Decision: H8 not supported.

Descriptive means (lower is tighter):
- body-parts 0.0794
- foods 0.1034
- animals 0.1111
- colours 0.1154 (low N)

Graph D1 interpretation:
- Heavy overlap in distributions.
- Descriptive ordering exists but inferential support is absent.

### H9: SpAM-semantic alignment above chance
Question:
- Do participant clusters match semantic embedding clusters more than chance?

Planned decision framework in notebook:
- Sign test on ARI > 0
- Plus permutation test on mean ARI magnitude

Result:
- Sign test: 48/85 positive, p=0.1390 (not significant)
- Mean ARI: 0.1199
- Permutation test: p < 0.001 (magnitude above shuffled null)
- Final decision used: H9 not supported (because sign criterion not met).

Interpretation:
- There is some average semantic signal, but not enough directional consistency under the notebook's strict rule.

### H10: SpAM-phonetic alignment above chance
- Sign test positives: 39/85
- Sign-test p=0.8072
- Mean ARI (descriptive)=0.0486
- Decision: H10 not supported.

Graph D2 interpretation:
- ARI values cluster near zero with wide spread.
- Semantic alignment mean is above phonetic alignment mean, but both are modest.

Meaning for RQ3/RQ4:
- Clear strong alignment claims are not supported by the inferential framework used.

---

## Phase 9 extended embedding analysis (important context)

This part builds semantic space using multilingual transformer embeddings and compares clustering methods.

Key findings:
- K-Means best k examples: animals=2, foods=2, colours=8, body-parts=2.
- K-Means vs agglomerative ARI is high in several domains:
  - animals 0.905
  - foods 0.861
  - colours 0.714
  - body-parts 0.461
- HDBSCAN labeled 100% as noise in reported run, indicating parameter/data mismatch for dense-cluster detection in this setup.

Extended permutation alignment summary:
- Units tested: 62
- Mean observed ARI: 0.0887
- Mean null ARI: 0.0001
- Sign test ARI>0: 27/62, p=0.8736

Interpretation for beginners:
- Representation has structure (methods can recover clusters),
- but behavior-link consistency is weaker and depends on criterion.

---

## Phase 10 extended behavior-link analysis (RQ2 extension)

This phase asks whether neighborhood structure predicts retrieval timing.

### Step 10.1: SpAM vs embedding geometry alignment
Spearman r by domain:
- animals: -0.070, p<0.001
- foods: -0.036, p=0.0007
- colours: +0.139, p=0.0093
- body-parts: -0.033, p=0.0764

Interpretation:
- Effect sizes are tiny and mostly not in desired positive direction.

### Step 10.2/10.3: isolation vs mean IRT (word-level)
- Retained words with participant coverage >=5: 31 words.
- Domain-level FastText rho values:
  - animals +0.174 (p=0.5528)
  - foods +0.383 (p=0.3085)
  - body-parts -0.571 (p=0.1390)
- No domain supports H8-extended criterion.

### Step 10.4: IRT-run clusters vs FastText clusters
Mean ARI by domain:
- animals -0.014
- foods 0.112
- colours 0.025
- body-parts 0.150
- overall mean ARI=0.068

Interpretation:
- Practical overlap is low.

---

## Module E: phonological facilitation (RQ6, RQ7)

Important note:
- The notebook has an earlier section where label H11 is reused for serial-position IRT reporting.
- The dedicated Phase 12 Module E section is the relevant one for phonological H11/H12 definitions.

### H11 (official Module E form): interaction over retrieval order
Model summary from Phase 12:
- Consecutive pairs: 633
- Interaction type x position: beta=-0.004144, p=0.1028
- Phonological slope: beta=+0.000077, p=0.9699
- Semantic slope: beta=+0.002948, p=0.1414
- Decision: H11 not supported.

Graph 12.1 interpretation:
- Curves do not show a reliable monotonic handoff to phonological dominance.
- Uncertainty overlap supports non-significant interaction.

### H12: mean phonological similarity predicts total words
- Spearman rho=+0.114
- one-tailed p=0.2570
- Decision: H12 not supported.

Meaning for RQ6/RQ7:
- No inferential evidence of strong phonological transition/productivity mechanism in this run.

---

## Module F: composite score comparison (RQ5)

### H13: integrated score vs VFT-only score for confidence association
Question:
- Is integrated score significantly better (or at least significantly different) than VFT-only for confidence relation?

Test choice:
- Two dependent correlations share the same outcome (confidence), so Steiger Z is the right comparison.

Results:
- Integrated score vs confidence: rho=-0.323, p=0.0586, n=35
- VFT-only score vs confidence: rho=-0.461, p=0.0053, n=35
- Steiger test: z=0.724, p=0.4689
- Decision: H13 not supported (no significant difference between correlations).

Graph 13.1 interpretation:
- Panel A (VFT-only): clearer and statistically significant negative trend.
- Panel B (integrated): weaker, misses significance.
- But because Steiger test is non-significant, do not claim confirmed superiority.

Meaning for RQ5:
- Integrated score is conceptually richer, but this sample does not show statistically stronger external validity than VFT-only.

---

## 6) What each major plot family tells you (high-level)

- H1 plots: strong visual confirmation of semantic boundary cost.
- H2 plots: mostly opposite-direction slopes; no lexical exhaustion support.
- Phase 2 distribution plots: strong right-skew and heterogeneity; justify non-parametric testing.
- Module B plots: domain overlaps dominate; no reliable group separation.
- Module C plots: confidence has ceiling compression and unexpected trend directions.
- Module D plots: moderate descriptive semantic signal but weak directional inferential support.
- Phase 9 maps/heatmaps/dendrograms: representational structure exists.
- Phase 10 linking plots: structure-to-behavior coupling is weak.
- Graph 12.1: no reliable semantic-to-phonological handoff.
- Graph 13.1: VFT-only looks stronger descriptively, but difference is not significant by Steiger.

---

## 7) Final integrated interpretation (plain language)

### What is clearly supported
- H1 is strongly supported.
- Semantic clustering in retrieval is a robust finding in this dataset.

### What is mostly not supported
- H2 through H13 are mostly not supported under pre-registered directional tests and decision rules in this run.
- Several effects exist descriptively, but they are weak, directionally inconsistent, low-N sensitive, or not significant.

### Why this still matters scientifically
Null or non-supported findings are still important because they:
- show which mechanisms are stable (H1) versus fragile or context-dependent,
- prevent overclaiming,
- and guide better next-study design (more balanced domain coverage, less confidence ceiling, stronger phonological instrumentation).

---

## 8) Key limitations explicitly visible in this notebook

- Uneven domain participation (especially colours low N in many places).
- Confidence variable has strong ceiling effect.
- Some sections include exploratory extensions with different decision criteria than primary hypothesis blocks.
- H11 label reuse appears in two contexts; Module E/Phase 12 is the proper phonological H11 implementation.

---

## 9) Reporting-ready summary paragraph

In this Hindi/Hinglish verbal fluency notebook, the most stable finding is strong semantic clustering: between-cluster transitions are much slower than within-cluster transitions, with a large effect size. In contrast, expected serial lexical exhaustion, confidence-based performance advantages, cross-domain compactness differences, and phonological facilitation hypotheses were not supported under planned directional inferential criteria. Extended embedding analyses show that semantic structure can be computationally recovered, but links from structure to participant behavior are modest and criterion-sensitive. Composite integrated scoring does not significantly outperform the simpler VFT-only score in confidence association under dependent-correlation testing.

---

## 10) If you are weak at stats: how to read results fast

Use this checklist:
1. What was the expected direction? (for example rho > 0)
2. Did observed direction match expectation?
3. Is p-value below 0.05 for the correct directional test?
4. Is effect size practically meaningful?
5. Does the graph show the same story as the table?

If any of these fail, treat the claim as weak or not supported.

That is exactly how this notebook reached its decisions.