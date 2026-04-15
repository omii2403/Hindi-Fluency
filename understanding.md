# Understanding File

This file is for simple understanding of what we did.
It explains RQ and EH in easy words.
It also explains what happened in final results.

Important data rule used in final rerun:

- Core inferential analysis used only Hindi/Hinglish responses.
- Hindi-only rows used: 723.
- We kept all responses (1040) only for EH3 Hindi vs English domain comparison.
- Final run used merged_vft_spam_responses.csv as the single final data source.

## One line meaning of RQ and EH

- RQ means research question.
- EH means exploratory hypothesis.
- RQ is main target question.
- EH is extra pattern check.

## Simple meaning of each RQ and EH

## RQ1 simple meaning

Question asks this.
Is switching between clusters slower than staying inside same cluster.

What happened.
Difference was significant in final rerun.
So switching between clusters is reliably slower at 0.05.

## RQ2 simple meaning

Question asks this.
If student makes bigger clusters then does student produce more total words.

What happened.
Positive trend was present but not significant.
So we cannot claim reliable effect at 0.05.

## RQ3 simple meaning

Question asks this.
If two words are far in SpAM space then are they also slower in VFT retrieval.

What happened.
Overall relation was not significant in Hindi-only rerun.
Domain-wise correlations were also weak and not significant.

## RQ4 simple meaning

Question asks this.
Is semantic compactness same in all domains in SpAM.

What happened.
Significant domain difference was found.
Colours domain was clearly different from others.

## RQ5 simple meaning

Question asks this.
If a person has more spread SpAM map then does person show higher VFT switching cost.

What happened.
No significant relation.
So spread in SpAM map does not explain switch cost in this sample.

## EH1 simple meaning

Question asks this.
Are domain level response times different.

What happened.
Yes significant difference found.
So retrieval speed changes by domain.

## EH2 simple meaning

Question asks this.
Does response position affect response time.

What happened.
Global mixed model was not significant.
Some domains showed significant slope and some did not.
So effect is domain dependent.

## EH3 simple meaning

Question asks this.
Is Hindi and English use pattern same in all domains.

What happened.
No it is not same.
Domain wise Hindi vs English shares were:

- Animals: Hindi 63.8%, English 36.2%
- Foods: Hindi 80.1%, English 19.9%
- Colours: Hindi 27.9%, English 72.1%
- Body-parts: Hindi 92.7%, English 7.3%

Colours had lowest Hindi share and highest English share.
So code switching depends on domain.

## EH4 simple meaning

Question asks this.
Which cluster metric is most linked with total fluency.

What happened.
Switch count and number of clusters gave strongest and significant relation.
Mean cluster size showed only positive trend and was not significant.

## Transformer usage

Transformer model used:

- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Simple meaning:

- Model converts each word into numeric vector.
- Similar vectors mean semantically close words.
- We used this to compute cosine similarity and cosine distance.
- Then we compared semantic closeness with behavior timing.

Where it is used in code:

1. analysis_part1.py model load and encoding
2. analysis_part1.py cosine similarity and distance matrix
3. analysis_part2.py neighbourhood density creation
4. analysis_part2.py mixed model with neighbourhood density and rt_ms

## Test selection rule

We used normality first rule.

1. Shapiro Wilk test was used.
2. If p < 0.05 then non normal.
3. If non normal then non parametric test.
4. If normal then parametric test can be used.

In this project most variables were non normal.
So non parametric tests are primary.

## What test was used and why

- RQ1: Wilcoxon signed rank because paired difference was non normal.
- RQ2: Spearman because predictor and residual were non normal.
- RQ3: Spearman because pairwise distance and pairwise IRT were non normal.
- RQ4: Kruskal Wallis plus BH corrected Mann Whitney because domain comparison was non normal.
- RQ5: Spearman plus MixedLM because both variables were non normal and data was repeated by participant.
- EH1: Kruskal Wallis because domain rt values were non normal.
- EH2: Mixed model because repeated data structure.
- EH3: proportion check because it is count pattern.
- EH4: Spearman because cluster metrics were mostly non normal.

## Final simple takeaway

1. RQ1 and RQ4 are supported.
2. RQ2 RQ3 and RQ5 are not supported.
3. Domain effect in timing and compactness is significant.
4. Code switching pattern is strongly domain dependent.
5. Colours domain is most different in compactness and language pattern.
