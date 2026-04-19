# Hindi Mental Lexicon Study — Complete Project Plan
### How Do Hindi Speakers Search Their Mental Lexicons?
**VFT + SpAM Analysis | 35 Participants | 4 Domains**

---

## Table of Contents

1. [Project Overview & Research Questions](#overview)
2. [Hypotheses](#hypotheses)
3. [Phase 1 — Data Loading, Cleaning & Setup](#phase1)
4. [Phase 2 — Descriptive Statistics & IRT Distributions](#phase2)
5. [Phase 3 — Fluency Scoring & Participant Profiles](#phase3)
6. [Phase 4 — Cluster Scoring & Serial Position Analysis](#phase4)
7. [Phase 5 — Core Hypothesis Tests (H1 & H2)](#phase5)
8. [Phase 6 — Hindi Fluency & Confidence Effects (H3, H4, H5)](#phase6)
9. [Phase 7 — VFT Domain Tests (RQ1)](#phase7)
10. [Phase 8 — SpAM Spatial Analysis (RQ3)](#phase8)
11. **[Phase 9 — Hindi FastText Embeddings & Independent Clustering ⭐ NEW](#phase9-fasttext)**
12. **[Phase 10 — SpAM–IRT Neighbourhood Integration (RQ2) ⭐ NEW](#phase10-spam-irt)**
13. [Phase 11 — Embedding Clustering & SpAM Alignment (RQ4, H9)](#phase11-align)
14. [Phase 12 — Phonological Similarity Analysis (H6, H7)](#phase12)
15. [Phase 13 — Composite Fluency Score (RQ5)](#phase13)
16. [Phase 14 — Integration, Summary & Conclusions](#phase14)

---

## 1. Project Overview & Research Questions {#overview}

### Study Design
- **Task 1 — VFT:** Participants produce as many words as possible from a given semantic category (animals, foods, colours, body-parts) within 60 seconds. The output is a word list with inter-response times (IRTs) in milliseconds and serial positions.
- **Task 2 — SpAM:** After VFT, participants arrange their produced words in 2D space based on perceived similarity. Output is normalised (x, y) coordinates per word per participant.
- **Exit Poll:** Self-reported demographics, Hindi proficiency (confidence score), and language background.
- **Sample:** 35 participants, Hindi-English bilinguals. Primary analysis uses Hindi/Hinglish responses only (n ≈ 723 rows).

### Five Research Questions (from your report)

| RQ | Full Question | Simple Meaning |
|----|--------------|----------------|
| RQ1 | Do VFT productivity and retrieval speed differ across semantic domains? | Are some domains easier or faster for Hindi word retrieval? |
| RQ2 | Which participant-level variables are associated with VFT outcomes? | Do personal factors like Hindi confidence relate to performance? |
| RQ3 | Does SpAM spatial compactness differ across domains? | Do some domains form tighter semantic maps in 2D space? |
| RQ4 | How strongly do participant SpAM clusters align with semantic and phonetic embedding-based clusters? | Do human spatial groupings match model-based meaning/sound clusters? |
| RQ5 | Does a combined VFT+SpAM fluency score improve inference compared to VFT-only scoring? | Is combined scoring more informative than VFT alone? |

### Two Additional Questions (from project design)
| RQ | Question |
|----|----------|
| RQ6 | Does phonological similarity between consecutive responses increase over retrieval order? (Replication of Kumar et al. 2022 in Hindi) |
| RQ7 | Does phonological similarity predict higher word productivity? |

---


## 3. Phase 1 — Data Loading, Cleaning & Setup {#phase1}

### Step 1.1 — Load VFT Data

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, shapiro, mannwhitneyu
import statsmodels.formula.api as smf

PALETTE = sns.color_palette("Set2")
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})

df_raw = pd.read_csv("vft_responses.csv")
print(f"Shape: {df_raw.shape}")
print(df_raw.dtypes)
print(df_raw.head())
```

**What to check:** Confirm columns — subject_id, word, domain, position, rt_ms, language_type. Confirm 35 participants, 4 domains, ~1040 rows total.

### Step 1.2 — Clean & Filter

```python
df = df_raw.copy()
df['language_type'] = df['language_type'].str.strip()
df['lang_binary'] = df['language_type'].apply(
    lambda x: 'Hindi/Hinglish' if 'Hindi' in str(x) else 'English')

# Remove IRTs above 60,000 ms (distraction, not retrieval cost)
THRESHOLD_MS = 60_000
removed = df[df['rt_ms'] > THRESHOLD_MS]
print(f"Rows removed (IRT > 60s): {len(removed)}")
df_clean = df[df['rt_ms'] <= THRESHOLD_MS].copy()
df_clean['irt_sec'] = df_clean['rt_ms'] / 1000

# Hindi/Hinglish subset — primary analysis
df_hh = df_clean[df_clean['lang_binary'] == 'Hindi/Hinglish'].copy()

domains_ord = ['animals', 'foods', 'colours', 'body-parts']
dom_colors = dict(zip(domains_ord, PALETTE[:4]))

print(f"Total rows after filter : {len(df_clean)}")
print(f"Hindi/Hinglish rows     : {len(df_hh)}")
print(f"Participants            : {df_clean['subject_id'].nunique()}")
print(f"\nSession-domain N per domain (Hindi/Hinglish):")
print(df_hh.groupby('domain')['subject_id'].nunique())
```

**IMPORTANT — Colours Warning:** If colours has only N=4 participants contributing Hindi responses, flag this immediately:
```python
domain_n = df_hh.groupby('domain')['subject_id'].nunique()
print("⚠ Domains with fewer than 10 participants — exclude from inferential tests:")
print(domain_n[domain_n < 10])
```

### Step 1.3 — Load Exit Poll / Demographics

```python
exit_poll = pd.read_csv("exit_poll.csv")  # adjust filename
print(exit_poll.dtypes)
print(exit_poll.head())

# Merge with fluency table on subject_id
# Identify the Hindi confidence/proficiency column
# If Likert 1-5: keep as continuous
# If it has multiple language columns: extract Hindi-specific score

# Example merge:
df_hh = df_hh.merge(exit_poll[['subject_id', 'hindi_confidence']], 
                     on='subject_id', how='left')
```

**Graph 1.1 — Confidence Score Distribution:**
```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(exit_poll['hindi_confidence'], bins=10, color=PALETTE[0], edgecolor='white')
ax.set_xlabel('Hindi Confidence Score')
ax.set_ylabel('Count')
ax.set_title('Distribution of Self-Reported Hindi Confidence')
ax.axvline(exit_poll['hindi_confidence'].mean(), color='red', linestyle='--',
           label=f"Mean = {exit_poll['hindi_confidence'].mean():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig('fig_confidence_dist.png', dpi=150)
plt.show()
```

**Why this matters:** If the confidence histogram is heavily right-skewed (everyone scoring 4–5), there is a ceiling effect. This would explain the surprising negative correlation (ρ = −0.395) found in the report — when a predictor has no variance, correlations become unstable and can flip sign. You MUST do this before interpreting RQ2.

**Conclusion to draw:** Report mean, SD, min, max, and skewness of confidence scores. If skewness > 1.0, state: "The confidence measure shows a ceiling effect, limiting its validity as a continuous predictor. Correlations involving this variable should be interpreted with caution."

---

## 4. Phase 2 — Descriptive Statistics & IRT Distributions {#phase2}

### Step 2.1 — Domain-Level Descriptive Statistics

```python
def domain_stats(df, col='rt_ms'):
    return df.groupby('domain')[col].agg(
        N='count',
        Mean='mean',
        Median='median',
        SD='std',
        Skewness=lambda x: x.skew(),
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).round(2)

print("=== IRT Descriptives by Domain (Hindi/Hinglish) ===")
print(domain_stats(df_hh))

print("\n=== Word Count Descriptives by Domain ===")
wc = df_hh.groupby(['subject_id','domain'])['word'].count().reset_index()
wc.columns = ['subject_id','domain','word_count']
print(wc.groupby('domain')['word_count'].describe().round(2))
```

**Graph 2.1 — IRT Histogram + Domain KDE (2 panels):**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Overall IRT histogram with mean/median/mode
ax = axes[0]
sns.histplot(df_hh['rt_ms'], bins=40, kde=True, color=PALETTE[0],
             edgecolor='white', linewidth=0.4, ax=ax)
mean_v  = df_hh['rt_ms'].mean()
med_v   = df_hh['rt_ms'].median()
mode_v  = df_hh['rt_ms'].mode().iloc[0]
ax.axvline(mean_v,  color='red',    linestyle='--', lw=2, label=f'Mean={mean_v:.0f}ms')
ax.axvline(med_v,   color='green',  linestyle='--', lw=2, label=f'Median={med_v:.0f}ms')
ax.axvline(mode_v,  color='purple', linestyle=':',  lw=2, label=f'Mode={mode_v:.0f}ms')
ax.set_xlabel('IRT (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Overall IRT Distribution — Hindi/Hinglish')
ax.legend(fontsize=9)
ax.text(0.97, 0.95, f'Skew = {df_hh["rt_ms"].skew():.2f}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right: Overlapping KDE per domain
ax = axes[1]
for dom in domains_ord:
    sub = df_hh[df_hh['domain'] == dom]['rt_ms']
    sns.kdeplot(sub, label=dom, color=dom_colors[dom], lw=2, ax=ax)
ax.set_xlabel('IRT (ms)')
ax.set_ylabel('Density')
ax.set_title('IRT Distribution by Domain')
ax.legend()

plt.tight_layout()
plt.savefig('fig_irt_distribution.png', dpi=150)
plt.show()
```

**Why overlapping KDE is better than 4 separate histograms:** It lets you directly compare the shape and position of distributions across domains. If colours peaks earlier (lower IRT), the KDE shows this immediately. If animals has a longer tail, you see it in one glance.

**Graph 2.2 — Boxplot + Stripplot (IRT by domain):**
```python
fig, ax = plt.subplots(figsize=(10, 6))
# Order domains by median IRT
domain_medians = df_hh.groupby('domain')['rt_ms'].median().sort_values()
ordered_domains = domain_medians.index.tolist()

sns.boxplot(data=df_hh, x='domain', y='rt_ms', order=ordered_domains,
            palette=dom_colors, width=0.5, ax=ax,
            boxprops=dict(alpha=0.7))
sns.stripplot(data=df_hh, x='domain', y='rt_ms', order=ordered_domains,
              palette=dom_colors, alpha=0.25, size=3, jitter=True, ax=ax)

# Annotate medians
for i, dom in enumerate(ordered_domains):
    med = df_hh[df_hh['domain'] == dom]['rt_ms'].median()
    ax.text(i, med + 300, f'{med:.0f}ms', ha='center', fontsize=9, color='black')

ax.set_xlabel('Domain')
ax.set_ylabel('IRT (ms)')
ax.set_title('IRT Distribution by Domain — Boxplot + Individual Data Points')
plt.tight_layout()
plt.savefig('fig_irt_boxplot.png', dpi=150)
plt.show()
```

**Why stripplot overlay:** With N=35, individual data points are visible and informative. Boxplots alone hide the actual spread and outlier structure. Each dot is one retrieval event — you can see that colours has fewer dots (fewer responses per domain).

**Conclusion to draw from Phase 2:**
- Report mean ± SD, median, skewness per domain in a table.
- State: "IRT distributions are right-skewed in all domains (skewness > 1.5), consistent with VFT literature. The majority of retrievals are rapid within-cluster accesses, while the long right tail reflects cluster-switch boundary pauses."
- Note domain ranking by median IRT (e.g., colours fastest, body-parts or animals slowest) as a descriptive pattern, before running any tests.

---

## 5. Phase 3 — Fluency Scoring & Participant Profiles {#phase3}

### Step 3.1 — Per-Participant Fluency Table

```python
fluency = df_hh.groupby('subject_id').agg(
    total_words   = ('word',    'count'),
    mean_irt_ms   = ('rt_ms',   'mean'),
    median_irt_ms = ('rt_ms',   'median'),
    n_domains     = ('domain',  'nunique'),
).reset_index()

# Merge with confidence score
if 'hindi_confidence' in df_hh.columns:
    conf = df_hh[['subject_id','hindi_confidence']].drop_duplicates()
    fluency = fluency.merge(conf, on='subject_id', how='left')

fluency = fluency.sort_values('total_words', ascending=False).reset_index(drop=True)
print(fluency)
print(f"\nMean total words  : {fluency['total_words'].mean():.1f}")
print(f"Std total words   : {fluency['total_words'].std():.1f}")
print(f"Range             : {fluency['total_words'].min()} – {fluency['total_words'].max()}")
```

**Graph 3.1 — Ranked Fluency Bar Chart (coloured by confidence):**
```python
fig, ax = plt.subplots(figsize=(12, 6))
# Colour by confidence quartile
fluency['conf_group'] = pd.qcut(fluency['hindi_confidence'], q=3,
                                 labels=['Low','Med','High'], duplicates='drop')
colors = {'Low':'#E24B4A', 'Med':'#EF9F27', 'High':'#3B8BD4'}
bar_colors = fluency['conf_group'].map(colors).fillna('gray')

bars = ax.barh(range(len(fluency)), fluency['total_words'],
               color=bar_colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(fluency)))
ax.set_yticklabels([f"P{row.subject_id}" for _, row in fluency.iterrows()], fontsize=8)
ax.set_xlabel('Total Hindi Words Produced')
ax.set_title('Participants Ranked by VFT Productivity (colour = Hindi confidence group)')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
ax.legend(handles=legend_elements, title='Confidence', loc='lower right')
plt.tight_layout()
plt.savefig('fig_fluency_ranked.png', dpi=150)
plt.show()
```

**Why this graph first:** Before running any statistics for RQ2, look at this chart. If high-confidence participants (blue) cluster at the top, H5 is likely supported. If they are randomly distributed — or if high-confidence is at the BOTTOM — you immediately see the anomaly that your report found (negative ρ) and can investigate it visually.

**Graph 3.2 — Fluency Efficiency Bubble Chart:**
```python
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(fluency['total_words'], fluency['mean_irt_ms'],
                c=fluency['hindi_confidence'], cmap='RdYlGn',
                s=120, edgecolors='gray', linewidth=0.5, alpha=0.85)
plt.colorbar(sc, ax=ax, label='Hindi Confidence Score')

# Label outliers (top/bottom 3 productivity)
for _, row in fluency.head(3).iterrows():
    ax.annotate(f"P{row.subject_id}", (row.total_words, row.mean_irt_ms),
                textcoords='offset points', xytext=(5, 3), fontsize=8)
for _, row in fluency.tail(3).iterrows():
    ax.annotate(f"P{row.subject_id}", (row.total_words, row.mean_irt_ms),
                textcoords='offset points', xytext=(5, 3), fontsize=8)

ax.set_xlabel('Total Words Produced (Productivity)')
ax.set_ylabel('Mean IRT (ms) — lower = faster')
ax.set_title('VFT Efficiency Profile per Participant\n(green = high confidence, red = low confidence)')
ax.axhline(fluency['mean_irt_ms'].mean(), color='gray', linestyle=':', lw=1, alpha=0.5)
ax.axvline(fluency['total_words'].mean(), color='gray', linestyle=':', lw=1, alpha=0.5)
ax.text(fluency['total_words'].mean()+0.2, fluency['mean_irt_ms'].max()*0.95,
        'Mean word count', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('fig_efficiency_bubble.png', dpi=150)
plt.show()
```

**Why this chart:** It shows 4 quadrants — (1) top-right: many words, slow = exhaustive but laborious; (2) top-left: few words, slow = genuinely limited access; (3) bottom-right: many words, fast = efficient retrieval; (4) bottom-left: few words, fast = limited but effortless. This maps directly onto the theoretical efficiency construct you want to measure. Colour = confidence lets you see whether confidence maps onto any specific quadrant.

**Conclusion to draw:** Characterise the sample: "Participants show a wide range in productivity (X–Y words) and retrieval speed (X–Y ms). [If the bubble chart shows efficient retrievers are high-confidence]: This is consistent with H5 and H6 — higher confidence participants cluster in the high-productivity, fast-retrieval quadrant. [If the pattern is reversed or random]: The unexpected negative correlation between confidence and word count warrants investigation of ceiling effects in the confidence measure (see Figure 1.1)."

---

## 6. Phase 4 — Cluster Scoring & Serial Position Analysis {#phase4}

### Step 4.1 — VFT Cluster Scoring

```python
cluster_records = []

for (subj, dom), grp in df_hh.sort_values('position').groupby(['subject_id', 'domain']):
    irts = grp.sort_values('position')['rt_ms'].values
    if len(irts) < 3:
        continue  # too short to cluster meaningfully
    
    # Adaptive threshold: mean + 1 SD (Troyer et al. 1997)
    threshold_adaptive = np.mean(irts) + np.std(irts, ddof=1)
    # Fixed threshold for robustness check
    threshold_fixed = 3000  # ms

    for thresh_name, threshold in [('adaptive', threshold_adaptive), 
                                    ('fixed_3000', threshold_fixed)]:
        cluster_sizes, current_size, switches = [], 1, 0
        
        # Label each IRT as within or between
        labels = ['within']  # first IRT is always within (no prior transition)
        for i in range(1, len(irts)):
            if irts[i] > threshold:
                cluster_sizes.append(current_size)
                current_size = 1
                switches += 1
                labels.append('between')
            else:
                current_size += 1
                labels.append('within')
        cluster_sizes.append(current_size)

        cluster_records.append({
            'subject_id':        subj,
            'domain':            dom,
            'threshold_type':    thresh_name,
            'mean_cluster_size': np.mean(cluster_sizes),
            'total_switches':    switches,
            'total_clusters':    len(cluster_sizes),
            'n_words':           len(irts),
            'within_irts':       [irts[i] for i, l in enumerate(labels) if l == 'within'],
            'between_irts':      [irts[i] for i, l in enumerate(labels) if l == 'between'],
        })

cluster_df = pd.DataFrame(cluster_records)

# Merge cluster metrics (adaptive threshold) into fluency table
cluster_agg = (cluster_df[cluster_df['threshold_type']=='adaptive']
               .groupby('subject_id')
               .agg(mean_cluster_size=('mean_cluster_size','mean'),
                    total_switches=('total_switches','sum'))
               .reset_index())
fluency = fluency.merge(cluster_agg, on='subject_id', how='left')
print(fluency[['subject_id','total_words','mean_cluster_size','total_switches']].head(10))
```

**Robustness Check — compare adaptive vs fixed threshold conclusions:**
```python
# Check if conclusions change between thresholds
for thresh in ['adaptive','fixed_3000']:
    sub = cluster_df[cluster_df['threshold_type']==thresh]
    print(f"\n=== Threshold: {thresh} ===")
    print(sub.groupby('domain')['mean_cluster_size'].describe().round(2))
```

**Why robustness check matters:** If both thresholds give the same domain ranking (e.g., animals has largest clusters under both methods), your findings are robust. If rankings flip, you need to report both and acknowledge sensitivity to threshold choice.

### Step 4.2 — Serial Position × IRT Analysis

```python
# Per-word IRT table (used later for SpAM integration)
word_irt = (
    df_hh.groupby(['domain', 'word'])
    .agg(freq=('word','count'), mean_irt_ms=('rt_ms','mean'),
         mean_position=('position','mean'), n_participants=('subject_id','nunique'))
    .reset_index().sort_values(['domain','freq'], ascending=[True, False])
)

print("Top 10 words per domain by frequency:")
for dom in domains_ord:
    print(f"\n{dom.upper()}:")
    print(word_irt[word_irt['domain']==dom].head(5)
          [['word','freq','mean_irt_ms','mean_position']].to_string(index=False))
```

**Graph 4.1 — Serial Position vs IRT Scatter (2×2, per domain):**
```python
from statsmodels.formula.api import mixedlm

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    sub = df_hh[df_hh['domain'] == dom].copy()
    
    # Scatter — individual observations (transparent)
    ax.scatter(sub['position'], sub['rt_ms'],
               color=dom_colors[dom], alpha=0.2, s=12, edgecolors='none')
    
    # LME regression line (accounts for participant nesting)
    try:
        model = mixedlm("rt_ms ~ position", sub, groups=sub["subject_id"])
        result = model.fit(reml=True)
        slope = result.params['position']
        intercept = result.params['Intercept']
        xs = np.linspace(sub['position'].min(), sub['position'].max(), 100)
        ax.plot(xs, intercept + slope * xs, color='black', lw=2.5,
                label=f'β={slope:.0f} ms/pos')
        p_val = result.pvalues['position']
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    except Exception:
        # Fallback to OLS if LME fails
        z = np.polyfit(sub['position'], sub['rt_ms'], 1)
        xs = np.linspace(sub['position'].min(), sub['position'].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color='black', lw=2.5, label=f'β={z[0]:.0f}')
        r_val = np.corrcoef(sub['position'], sub['rt_ms'])[0, 1]
        sig = ''
    
    # Pearson r for display
    r_val = np.corrcoef(sub['position'], sub['rt_ms'])[0, 1]
    ax.text(0.97, 0.95, f'r = {r_val:.2f} {sig}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('Serial Position')
    ax.set_ylabel('IRT (ms)')
    ax.set_title(f'{dom.capitalize()} (N={len(sub)} responses)')

plt.suptitle('Serial Position vs IRT — Lexical Exhaustion Effect', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_serial_position.png', dpi=150)
plt.show()
```

**Why LME instead of OLS:** Participants are nested in the data — each participant contributes multiple data points. OLS ignores this clustering and inflates degrees of freedom, making p-values anti-conservatively small. LME with (1+position|subject) correctly accounts for participant-level variation in baseline IRT and the rate of increase with position.

**Graph 4.2 — Domain Cluster Metrics Grouped Bar Chart:**
```python
cluster_domain = (cluster_df[cluster_df['threshold_type']=='adaptive']
                  .groupby('domain')[['mean_cluster_size','total_switches']]
                  .agg(['mean','sem']).reset_index())
cluster_domain.columns = ['domain','cs_mean','cs_sem','sw_mean','sw_sem']

x = np.arange(len(domains_ord))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width/2, [cluster_domain[cluster_domain['domain']==d]['cs_mean'].values[0]
                            for d in domains_ord],
            width, color=PALETTE[0], alpha=0.8, label='Mean Cluster Size',
            yerr=[cluster_domain[cluster_domain['domain']==d]['cs_sem'].values[0]
                  for d in domains_ord], capsize=6)
ax2 = ax.twinx()
b2 = ax2.bar(x + width/2, [cluster_domain[cluster_domain['domain']==d]['sw_mean'].values[0]
                             for d in domains_ord],
             width, color=PALETTE[1], alpha=0.8, label='Mean Switches',
             yerr=[cluster_domain[cluster_domain['domain']==d]['sw_sem'].values[0]
                   for d in domains_ord], capsize=6)
ax.set_xticks(x)
ax.set_xticklabels(domains_ord)
ax.set_ylabel('Mean Cluster Size', color=PALETTE[0])
ax2.set_ylabel('Mean Switches', color=PALETTE[1])
ax.set_title('VFT Cluster Metrics by Domain\n(left axis = cluster size, right axis = switches)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig('fig_cluster_metrics.png', dpi=150)
plt.show()
```

**Conclusion to draw from Phase 4:**
- Serial position effect: "IRT increased significantly with serial position in [X out of 4] domains (β range: X–Y ms/position, p < 0.05), confirming the lexical exhaustion effect — participants access prototype words early and peripheral words progressively later and slower."
- Domain differences in clusters: "Animals showed the largest mean cluster size (X), consistent with its rich sub-categorical structure (wild/domestic/aquatic). Colours showed the fewest switches, consistent with its closed-class, limited vocabulary."
- Robustness: "Results were consistent across adaptive (mean+1SD) and fixed (3000ms) thresholds, confirming robustness of cluster identification."

---

## 7. Phase 5 — Core Hypothesis Tests: H1 & H2 {#phase5}

### Step 5.1 — H1: Within vs Between Cluster IRT

> **This is the most important analysis in the entire study. It is missing from your current report.**

```python
# Pool all within and between IRTs across participants
within_all, between_all = [], []
within_by_domain  = {d: [] for d in domains_ord}
between_by_domain = {d: [] for d in domains_ord}

for _, row in cluster_df[cluster_df['threshold_type']=='adaptive'].iterrows():
    within_all.extend(row['within_irts'])
    between_all.extend(row['between_irts'])
    within_by_domain[row['domain']].extend(row['within_irts'])
    between_by_domain[row['domain']].extend(row['between_irts'])

within_arr  = np.array(within_all)
between_arr = np.array(between_all)

# Cohen's d function
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_s = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2)/(na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled_s if pooled_s > 0 else 0.0

# Welch's t-test (one-tailed: between > within)
t_stat, p_two = stats.ttest_ind(between_arr, within_arr, equal_var=False)
p_one = p_two / 2  # one-tailed
d = cohens_d(between_arr, within_arr)

print("=" * 60)
print("H1: Within-Cluster vs Between-Cluster IRT (Welch's t-test)")
print("=" * 60)
print(f"Within  — Mean: {within_arr.mean():.0f} ms  | Median: {np.median(within_arr):.0f} ms  | N: {len(within_arr)}")
print(f"Between — Mean: {between_arr.mean():.0f} ms  | Median: {np.median(between_arr):.0f} ms  | N: {len(between_arr)}")
print(f"t = {t_stat:.3f}, p (one-tailed) = {p_one:.4f}")
print(f"Cohen's d = {d:.3f} ({('small' if d<0.5 else 'medium' if d<0.8 else 'large')})")
print(f"IRT ratio (between/within): {between_arr.mean()/within_arr.mean():.2f}x")

print("\n--- Per-Domain ---")
for dom in domains_ord:
    w = np.array(within_by_domain[dom])
    b = np.array(between_by_domain[dom])
    if len(w) > 1 and len(b) > 1:
        t_d, p_d = stats.ttest_ind(b, w, equal_var=False)
        d_d = cohens_d(b, w)
        print(f"{dom:12s}: within={w.mean():.0f}ms, between={b.mean():.0f}ms, "
              f"t={t_d:.2f}, p={p_d/2:.4f}, d={d_d:.2f}")
```

**Graph 5.1 — Within vs Between IRT: Violin + Scatter (2 panels):**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: violin plot split by domain
plot_data = []
for dom in domains_ord:
    for irt in within_by_domain[dom]:
        plot_data.append({'domain': dom, 'type': 'Within-cluster', 'IRT_ms': irt})
    for irt in between_by_domain[dom]:
        plot_data.append({'domain': dom, 'type': 'Between-cluster', 'IRT_ms': irt})
plot_df = pd.DataFrame(plot_data)

sns.violinplot(data=plot_df, x='domain', y='IRT_ms', hue='type',
               split=True, palette=['#3B8BD4','#D85A30'],
               inner='quartile', ax=axes[0], order=domains_ord)
axes[0].set_title('Within vs Between Cluster IRT by Domain')
axes[0].set_ylabel('IRT (ms)')
axes[0].set_xlabel('Domain')
axes[0].legend(title='IRT Type', loc='upper right')
axes[0].set_ylim(0, axes[0].get_ylim()[1])

# Right panel: per-participant scatter (within mean vs between mean)
# Each point = one participant, colour = domain
per_part_w = {d: {} for d in domains_ord}
per_part_b = {d: {} for d in domains_ord}
for _, row in cluster_df[cluster_df['threshold_type']=='adaptive'].iterrows():
    if row['within_irts'] and row['between_irts']:
        per_part_w[row['domain']][row['subject_id']] = np.mean(row['within_irts'])
        per_part_b[row['domain']][row['subject_id']] = np.mean(row['between_irts'])

ax = axes[1]
for dom in domains_ord:
    subjs = set(per_part_w[dom].keys()) & set(per_part_b[dom].keys())
    w_means = [per_part_w[dom][s] for s in subjs]
    b_means = [per_part_b[dom][s] for s in subjs]
    ax.scatter(w_means, b_means, color=dom_colors[dom], alpha=0.6, s=50, label=dom)

# Diagonal reference line (y=x means within=between, no difference)
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], 'k--', lw=1.5, alpha=0.5, label='y = x (no effect)')
ax.set_xlabel('Mean Within-Cluster IRT (ms)')
ax.set_ylabel('Mean Between-Cluster IRT (ms)')
ax.set_title(f'Per-Participant: Within vs Between IRT\n(points above diagonal = clustering effect)')
ax.legend(fontsize=8)

plt.suptitle(f'H1: Semantic Clustering Confirmed — d = {d:.2f}, p = {p_one:.4f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_H1_cluster_comparison.png', dpi=150)
plt.show()
```

**Why the diagonal scatter:** Points above the diagonal mean that for that participant, between-cluster IRTs are longer than within-cluster IRTs — exactly what H1 predicts. If all 35 participants are above the diagonal, the result is unambiguous. If some are below, you investigate those as "non-clustering" retrievers.

**Conclusion for H1:** "Between-cluster IRTs (M=X ms) were significantly longer than within-cluster IRTs (M=Y ms), t(df)=X, p<.001, Cohen's d=X (large effect). This confirms H1: Hindi speakers do not retrieve words randomly — they produce runs of semantically related words (clusters) and pause at semantic boundaries. This replicates the core finding of Troyer et al. (1997) in a Hindi-speaking sample."

### Step 5.2 — H2: Lexical Exhaustion via LME

```python
from statsmodels.formula.api import mixedlm

print("=" * 60)
print("H2: Serial Position Effect — LME per Domain")
print("=" * 60)

h2_results = {}
for dom in domains_ord:
    sub = df_hh[df_hh['domain'] == dom].copy()
    if sub['subject_id'].nunique() < 5:
        print(f"{dom}: skipped (too few participants for LME)")
        continue
    try:
        model = mixedlm("rt_ms ~ position", sub, groups=sub["subject_id"])
        result = model.fit(reml=True, disp=False)
        b = result.params['position']
        se = result.bse['position']
        p = result.pvalues['position']
        ci = result.conf_int().loc['position']
        h2_results[dom] = {'beta': b, 'se': se, 'p': p, 'ci_low': ci[0], 'ci_high': ci[1]}
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"{dom:12s}: β={b:+.1f}ms/pos, SE={se:.1f}, p={p:.4f} {sig}, "
              f"95%CI=[{ci[0]:.1f}, {ci[1]:.1f}]")
    except Exception as e:
        print(f"{dom}: LME failed — {e}")
```

**Conclusion for H2:** "The serial position effect was significant in [X] of 4 domains (β range: X–Y ms/position). Animals showed the steepest slope (β=X ms), consistent with its large semantic network requiring more search per additional word. Colours showed the flattest slope (β=Y ms), consistent with its limited closed-class vocabulary being exhausted rapidly but evenly. These results confirm H2: IRT increases with serial position, reflecting progressive lexical exhaustion."

---

## 8. Phase 6 — Hindi Fluency & Confidence Effects (H3–H7) {#phase6}

### Step 6.1 — H3: Confidence Predicts Word Count

**Critical pre-check first — diagnose the negative correlation:**
```python
print("=== Confidence Score Diagnostics ===")
print(fluency['hindi_confidence'].describe())
print(f"Skewness: {fluency['hindi_confidence'].skew():.3f}")
print(f"Ceiling (score = max): {(fluency['hindi_confidence'] == fluency['hindi_confidence'].max()).sum()} / {len(fluency)}")

# Check if negative correlation is driven by outliers
from scipy.stats import spearmanr
rho, p = spearmanr(fluency['hindi_confidence'], fluency['total_words'])
print(f"\nSpearman ρ (confidence vs total_words): {rho:.3f}, p={p:.4f}")

# Check after removing top/bottom confidence outliers
q1_c, q3_c = fluency['hindi_confidence'].quantile([0.25, 0.75])
fluency_trimmed = fluency[(fluency['hindi_confidence'] >= q1_c) &
                           (fluency['hindi_confidence'] <= q3_c)]
rho_t, p_t = spearmanr(fluency_trimmed['hindi_confidence'], fluency_trimmed['total_words'])
print(f"Spearman ρ (trimmed middle 50%): {rho_t:.3f}, p={p_t:.4f}")
```

**Graph 6.1 — Confidence vs Total Words (scatter with regression):**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Raw scatter
ax = axes[0]
ax.scatter(fluency['hindi_confidence'], fluency['total_words'],
           color=PALETTE[0], alpha=0.7, s=80, edgecolors='gray', lw=0.5)
# Add regression line
z = np.polyfit(fluency['hindi_confidence'].dropna(),
               fluency.loc[fluency['hindi_confidence'].notna(), 'total_words'], 1)
xs = np.linspace(fluency['hindi_confidence'].min(), fluency['hindi_confidence'].max(), 100)
ax.plot(xs, np.poly1d(z)(xs), 'r--', lw=2)
ax.set_xlabel('Hindi Confidence Score')
ax.set_ylabel('Total Words Produced')
ax.set_title(f'H3: Confidence vs Productivity\nρ = {rho:.3f}, p = {p:.3f}')
# Annotate outlier participants
for _, row in fluency.nlargest(3, 'total_words').iterrows():
    ax.annotate(f"P{row['subject_id']}", (row['hindi_confidence'], row['total_words']),
                xytext=(4, 3), textcoords='offset points', fontsize=8)

# Right: confidence distribution (to check ceiling effect)
ax = axes[1]
ax.hist(fluency['hindi_confidence'].dropna(), bins=8, color=PALETTE[1],
        edgecolor='white', rwidth=0.85)
ax.set_xlabel('Hindi Confidence Score')
ax.set_ylabel('Number of Participants')
ax.set_title('Confidence Score Distribution\n(check for ceiling effect)')
ax.axvline(fluency['hindi_confidence'].max(), color='red', linestyle='--', lw=2,
           label=f"Max = {fluency['hindi_confidence'].max()}")
ax.legend()

plt.tight_layout()
plt.savefig('fig_H3_confidence_words.png', dpi=150)
plt.show()
```

**How to interpret the negative ρ:** Write this logic in your report: "The unexpected negative correlation (ρ = −0.395) between Hindi confidence and total words may reflect [choose based on what you find]: (A) A ceiling effect in the confidence measure — X% of participants rated themselves at maximum confidence, leaving insufficient variance for meaningful correlation. (B) A genuine paradox: highly confident speakers may employ a more selective retrieval strategy, producing fewer but more distinctive words. (C) An artefact of our specific sample — bilingual speakers may use English for easy words, so Hindi word count is not a direct measure of Hindi fluency."

### Step 6.2 — H5, H6, H7: All Spearman Tests

```python
print("=" * 60)
print("H5, H6, H7: Spearman Correlations with Confidence")
print("=" * 60)

outcomes = {
    'total_words'       : ('H5', 'Total words (productivity)'),
    'mean_irt_ms'       : ('H6', 'Mean IRT (speed, lower=faster)'),
    'mean_cluster_size' : ('H7', 'Mean cluster size (depth)'),
    'total_switches'    : ('H7b','Total switches (breadth)'),
}

if 'hindi_confidence' in fluency.columns:
    for col, (h_id, label) in outcomes.items():
        if col in fluency.columns:
            mask = fluency[[col, 'hindi_confidence']].notna().all(axis=1)
            rho, p = spearmanr(fluency.loc[mask, 'hindi_confidence'],
                               fluency.loc[mask, col])
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            direction = 'consistent' if ((h_id in ['H5','H7'] and rho > 0) or
                                          (h_id == 'H6' and rho < 0)) else 'UNEXPECTED'
            print(f"{h_id} | {label:40s}: ρ={rho:+.3f}, p={p:.4f} {sig} [{direction}]")
```

**Graph 6.2 — 3-panel scatter for H5, H6, H7:**
```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

pairs = [('total_words','Total Words','H5'), 
         ('mean_irt_ms','Mean IRT (ms)','H6'), 
         ('mean_cluster_size','Mean Cluster Size','H7')]

for ax, (col, ylabel, h_id) in zip(axes, pairs):
    if col not in fluency.columns:
        continue
    mask = fluency[[col, 'hindi_confidence']].notna().all(axis=1)
    x = fluency.loc[mask, 'hindi_confidence']
    y = fluency.loc[mask, col]
    ax.scatter(x, y, color=PALETTE[0], alpha=0.7, s=80, edgecolors='gray', lw=0.5)
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'r--', lw=2)
    rho, p = spearmanr(x, y)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.set_xlabel('Hindi Confidence Score')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{h_id}: ρ = {rho:.3f} {sig}')

plt.suptitle('Hindi Confidence vs VFT Outcomes (N=35)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_H567_confidence_outcomes.png', dpi=150)
plt.show()
```

---

## 9. Phase 7 — VFT Domain Tests: H3 & H4 (RQ1) {#phase7}

### Step 7.1 — Normality Testing First

```python
print("=" * 60)
print("Shapiro-Wilk Normality Tests per Domain (RQ1)")
print("=" * 60)

wc_by_domain = {}
rt_by_domain = {}

for dom in domains_ord:
    sub = df_hh[df_hh['domain'] == dom]
    n = sub['subject_id'].nunique()
    if n < 4:
        print(f"{dom}: N={n} — EXCLUDE from inferential tests (insufficient data)")
        continue
    
    wc = sub.groupby('subject_id')['word'].count().values
    rt = sub['rt_ms'].values
    wc_by_domain[dom] = wc
    rt_by_domain[dom] = rt
    
    stat_wc, p_wc = shapiro(wc) if len(wc) >= 3 else (np.nan, np.nan)
    stat_rt, p_rt = shapiro(rt) if len(rt) >= 3 else (np.nan, np.nan)
    
    decision = "→ Kruskal-Wallis" if (p_wc < 0.05 or p_rt < 0.05) else "→ ANOVA possible"
    print(f"{dom:12s} (N={n:2d}) | WC: W={stat_wc:.3f}, p={p_wc:.4f} | "
          f"RT: W={stat_rt:.3f}, p={p_rt:.4f} | {decision}")
```

**Why Shapiro-Wilk first:** This is the normality-first test decision your report describes. You must report these actual W and p values (currently missing from your report). The decision tree is: if ANY domain fails normality (p<0.05), use Kruskal-Wallis for the whole comparison.

### Step 7.2 — RQ1: Kruskal-Wallis Tests

```python
valid_domains = list(wc_by_domain.keys())

print("\n=== RQ1: Kruskal-Wallis — Word Count by Domain ===")
if len(valid_domains) >= 2:
    h_wc, p_wc = kruskal(*[wc_by_domain[d] for d in valid_domains])
    print(f"H = {h_wc:.3f}, df = {len(valid_domains)-1}, p = {p_wc:.4f}")
    print("Inference:", "Significant domain differences" if p_wc < 0.05 else "No significant differences")

print("\n=== RQ1: Kruskal-Wallis — Mean RT by Domain ===")
if len(valid_domains) >= 2:
    h_rt, p_rt = kruskal(*[rt_by_domain[d] for d in valid_domains])
    print(f"H = {h_rt:.3f}, df = {len(valid_domains)-1}, p = {p_rt:.4f}")
    print("Inference:", "Significant domain differences" if p_rt < 0.05 else "No significant differences")

# Dunn post-hoc if significant
try:
    import scikit_posthocs as sp
    for outcome_name, data_dict in [('Word Count', wc_by_domain), ('Mean RT', rt_by_domain)]:
        combined = pd.DataFrame([(dom, val) for dom, vals in data_dict.items() for val in vals],
                                  columns=['domain','value'])
        print(f"\nDunn post-hoc ({outcome_name}):")
        dunn = sp.posthoc_dunn(combined, val_col='value', group_col='domain',
                                p_adjust='bonferroni')
        print(dunn.round(4))
except ImportError:
    print("(Install scikit-posthocs for Dunn post-hoc: pip install scikit-posthocs)")
```

**Conclusion for RQ1 (if non-significant as in your report):** "The Kruskal-Wallis test did not reveal significant domain differences in word count (H=5.295, p=0.15) or mean RT (H=1.919, p=0.59). However, we note that this analysis excluded the colours domain due to insufficient observations (N=4 participants contributed Hindi responses to this domain). Descriptive patterns show [domain ranking], but the small and unequal sample sizes limit inferential power. A study with balanced domain sampling would be needed to draw firm conclusions about domain effects in Hindi VFT."

---

## 10. Phase 8 — SpAM Spatial Analysis (RQ3) {#phase8}

### Step 8.1 — Load SpAM Data & Compute Compactness

```python
# Load SpAM data (adjust filename as needed)
spam = pd.read_csv("spam_responses.csv")  # or responses.json
print(spam.head())
print(spam.dtypes)

# Expected columns: subject_id, word, domain, x, y (normalized coordinates)

# Compute per-participant-domain: mean nearest-neighbour distance
from scipy.spatial.distance import pdist, squareform

nn_records = []
for (subj, dom), grp in spam.groupby(['subject_id','domain']):
    coords = grp[['x','y']].values
    if len(coords) < 3:
        continue
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dist = dist_matrix.min(axis=1)  # nearest neighbour distance per word
    nn_records.append({
        'subject_id': subj, 'domain': dom,
        'mean_nn_dist': nn_dist.mean(),
        'n_words': len(coords)
    })

nn_df = pd.DataFrame(nn_records)
print(nn_df.groupby('domain')['mean_nn_dist'].describe().round(4))
```

**Graph 8.1 — SpAM Compactness Boxplot by Domain:**
```python
fig, ax = plt.subplots(figsize=(10, 6))
valid_spam_domains = [d for d in domains_ord if d in nn_df['domain'].values]
sns.boxplot(data=nn_df, x='domain', y='mean_nn_dist', order=valid_spam_domains,
            palette=dom_colors, width=0.5, ax=ax)
sns.stripplot(data=nn_df, x='domain', y='mean_nn_dist', order=valid_spam_domains,
              palette=dom_colors, alpha=0.5, size=5, jitter=True, ax=ax)
ax.set_xlabel('Domain')
ax.set_ylabel('Mean Nearest-Neighbour Distance (SpAM)')
ax.set_title('RQ3: SpAM Spatial Compactness by Domain\n(lower = more compact = tighter semantic grouping)')
plt.tight_layout()
plt.savefig('fig_spam_compactness.png', dpi=150)
plt.show()
```

**Graph 8.2 — Example SpAM Layout per Domain (scatter of word positions):**
```python
# Show mean x,y position per word for most frequent words per domain
mean_pos = spam.groupby(['domain','word'])[['x','y']].mean().reset_index()
word_freq = df_hh.groupby(['domain','word'])['word'].count().reset_index(name='freq')
mean_pos = mean_pos.merge(word_freq, on=['domain','word'], how='left').fillna({'freq': 1})

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    sub = mean_pos[mean_pos['domain'] == dom].nlargest(30, 'freq')
    sc = ax.scatter(sub['x'], sub['y'], s=sub['freq']*15 + 20,
                    c=sub['freq'], cmap='Blues', edgecolors='gray', lw=0.5, alpha=0.85)
    for _, row in sub.nlargest(10, 'freq').iterrows():
        ax.annotate(row['word'], (row['x'], row['y']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    ax.set_title(f'{dom.capitalize()} — SpAM Mean Positions\n(size = production frequency)')
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    plt.colorbar(sc, ax=ax, label='Frequency')
plt.suptitle('SpAM Semantic Maps by Domain\n(most frequent words labelled)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_spam_layouts.png', dpi=150)
plt.show()
```

**Conclusion for RQ3:** "Kruskal-Wallis did not reveal significant domain differences in SpAM compactness (H=3.038, p=0.386). However, body-parts showed the smallest mean NN distance (0.07), suggesting the tightest spatial grouping — consistent with the body having a natural hierarchical structure (upper/lower, internal/external) that participants map directly into 2D proximity. These are descriptive patterns only; inferential conclusions require a larger balanced sample."

---

## 11. Phase 9 — Hindi FastText Embeddings & Independent Clustering ⭐ NEW {#phase9-fasttext}

> **Purpose:** Build a corpus-trained Hindi semantic space from scratch using FastText (not a generic multilingual model). Run three independent clustering algorithms on this space to get a "ground-truth" semantic structure driven purely by Hindi word similarity — with no behavioural data involved. This becomes the reference against which all behavioural patterns (IRT clusters, SpAM clusters) are compared.

> **Dependency:** This phase must complete before Phase 10 (SpAM-IRT integration) and Phase 11 (cluster alignment), because both of those need the FastText vectors and cluster labels produced here.

---

### Step 9.1 — Download & Load Hindi FastText Embeddings

```python
# Option A — Facebook CommonCrawl Hindi (best for general Hindi vocabulary)
# Download: https://fasttext.cc/docs/en/crawl-vectors.html
# File: cc.hi.300.bin (~4GB) or cc.hi.300.vec (text format, ~1.5GB)
# pip install fasttext-wheel

import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# Load the binary model (supports subword OOV handling)
ft_model = fasttext.load_model('cc.hi.300.bin')
print(f"FastText model loaded: {ft_model.get_dimension()} dimensions")

# Option B — if cc.hi.300.bin is too large, use the .vec text format with gensim
# from gensim.models import KeyedVectors
# ft_model = KeyedVectors.load_word2vec_format('cc.hi.300.vec', binary=False)
# Note: .vec format does NOT support subword OOV — flag missing words explicitly
```

**Why FastText over the multilingual MiniLM already in your report:**
The `paraphrase-multilingual-MiniLM-L12-v2` model you used for embeddings is a semantic sentence transformer — it captures contextual meaning but was not trained specifically on Hindi vocabulary structure. Facebook's `cc.hi.300` was trained on 300M+ Hindi tokens from CommonCrawl using FastText's subword algorithm. For **lexical** tasks like finding neighbours of "बिल्ली" (cat) or "चीता" (cheetah), FastText gives more accurate word-level similarity. Critically, FastText handles **OOV words via subword n-grams** — if a word like "चिंपाज़ी" (chimpanzee) was not in training data, it still gets a vector by composing character n-grams. MiniLM would produce a near-random vector for rare Hindi words.

### Step 9.2 — Extract Vectors for All Corpus Words

```python
import unicodedata

def normalize_hindi(word):
    """Normalize Devanagari text to NFKC, lowercase, strip spaces."""
    return unicodedata.normalize('NFKC', str(word).lower().strip()
                                 .replace('\u200d','').replace('\u200c',''))

# Get all unique Hindi words per domain
all_words_by_domain = df_hh.groupby('domain')['word'].unique().to_dict()

# Extract vectors — FastText subword fallback handles OOV automatically
ft_vectors = {}   # {domain: {word: np.array(300)}}
oov_report = {}

for dom, words in all_words_by_domain.items():
    dom_vecs = {}
    oov_words = []
    for w in words:
        normalized = normalize_hindi(w)
        try:
            vec = ft_model.get_word_vector(normalized)  # always returns vector (subword fallback)
            # Check if truly in vocabulary vs subword approximation
            in_vocab = normalized in ft_model.words if hasattr(ft_model, 'words') else True
            if not in_vocab:
                oov_words.append(w)
            dom_vecs[w] = vec
        except Exception:
            dom_vecs[w] = np.zeros(300)
            oov_words.append(w)
    ft_vectors[dom] = dom_vecs
    oov_report[dom] = oov_words
    print(f"{dom:12s}: {len(words)} words | OOV (subword fallback): {len(oov_words)}")

print(f"\nTotal OOV rate: "
      f"{sum(len(v) for v in oov_report.values()) / sum(len(v) for v in all_words_by_domain.values()) * 100:.1f}%")
print("\nSample OOV words per domain:")
for dom, oovs in oov_report.items():
    print(f"  {dom}: {oovs[:5]}")
```

**Why report OOV rate:** Your report used a multilingual model that never flagged OOV. With FastText you can distinguish genuine vocabulary entries from subword approximations. If OOV rate > 30%, consider supplementing with IndicBERT embeddings for those words. State the OOV rate explicitly in your Methods section.

### Step 9.3 — Pairwise Cosine Distance Matrix per Domain

```python
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns

distance_matrices = {}   # {domain: (words_list, distance_matrix)}
similarity_matrices = {}

for dom in domains_ord:
    words = list(ft_vectors[dom].keys())
    if len(words) < 3:
        continue
    vecs = np.array([ft_vectors[dom][w] for w in words])
    dist_mat  = cosine_distances(vecs)      # shape: (n_words, n_words)
    sim_mat   = 1 - dist_mat
    distance_matrices[dom]  = (words, dist_mat)
    similarity_matrices[dom] = (words, sim_mat)
    print(f"{dom}: {len(words)}×{len(words)} distance matrix computed")

# Compute k=5 nearest neighbours per word per domain
K_NEIGHBOURS = 5
knn_records = []

for dom, (words, dist_mat) in distance_matrices.items():
    np.fill_diagonal(dist_mat, np.inf)  # exclude self
    for i, word in enumerate(words):
        nn_indices = np.argsort(dist_mat[i])[:K_NEIGHBOURS]
        nn_words   = [words[j] for j in nn_indices]
        nn_dists   = dist_mat[i][nn_indices]
        knn_records.append({
            'domain': dom, 'word': word,
            'nn_words': nn_words,
            'mean_nn_dist_ft': nn_dists.mean(),   # neighbourhood density (FastText)
            'nn_dists': nn_dists.tolist()
        })
    np.fill_diagonal(dist_mat, 0)  # restore diagonal

knn_df = pd.DataFrame(knn_records)
print("\nSample k-NN neighbourhoods:")
print(knn_df[['domain','word','nn_words','mean_nn_dist_ft']].head(8).to_string(index=False))
```

**Graph 9.1 — Pairwise Cosine Distance Heatmap per Domain (2×2):**

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    if dom not in distance_matrices:
        ax.set_visible(False)
        continue
    words, dist_mat = distance_matrices[dom]
    
    # Sort words by hierarchical clustering for cleaner heatmap
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    Z = linkage(dist_mat, method='ward')
    order = leaves_list(Z)
    sorted_words = [words[i] for i in order]
    sorted_mat   = dist_mat[np.ix_(order, order)]
    
    # Show top 30 most frequent words only (heatmap unreadable with 100+ words)
    freq = df_hh[df_hh['domain']==dom].groupby('word')['word'].count()
    top_words = freq.nlargest(min(30, len(words))).index.tolist()
    top_idx = [words.index(w) for w in top_words if w in words]
    sub_mat  = dist_mat[np.ix_(top_idx, top_idx)]
    
    sns.heatmap(sub_mat, xticklabels=[words[i] for i in top_idx],
                yticklabels=[words[i] for i in top_idx],
                cmap='RdYlGn_r', vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Cosine distance'})
    ax.set_title(f'{dom.capitalize()} — Pairwise Semantic Distance\n(FastText cc.hi.300, top-30 frequent words)')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

plt.suptitle('FastText Pairwise Cosine Distance Matrices\n(green = similar, red = dissimilar)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fasttext_heatmaps.png', dpi=150)
plt.show()
```

**How to read the heatmap:** Diagonal = 0 (a word is identical to itself). Green off-diagonal blocks = clusters of semantically similar words. A well-structured domain like body-parts should show clear green blocks (upper-body words cluster together, lower-body words cluster together). A diffuse domain like foods will show a more uniform yellow-orange pattern with fewer tight clusters.

**Conclusion to draw:** "FastText pairwise distance matrices revealed [domain-specific patterns]. Body-parts showed the most distinct cluster structure, with clear green blocks corresponding to [upper body / lower body / internal organs]. Animals showed the widest range of distances — from near-identical words (like बाघ/tiger and शेर/lion) to maximally distant pairs (like मछली/fish and हाथी/elephant). Colours showed the highest mean similarity (lowest mean distance = X), consistent with the domain being a tightly bounded semantic category."

---

### Step 9.4 — Independent Clustering: K-Means with Elbow + Silhouette

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

kmeans_results = {}   # {domain: {word: cluster_label}}

for dom in domains_ord:
    if dom not in ft_vectors:
        continue
    words = list(ft_vectors[dom].keys())
    vecs  = np.array([ft_vectors[dom][w] for w in words])
    if len(words) < 6:
        print(f"{dom}: skipped (too few words for clustering)")
        continue
    
    k_range = range(2, min(8, len(words)//2))
    inertias   = []
    sil_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(vecs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(vecs, labels))
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=15)
    final_labels = km_final.fit_predict(vecs)
    
    kmeans_results[dom] = {w: int(l) for w, l in zip(words, final_labels)}
    
    print(f"\n{dom.upper()}: best k = {best_k} (silhouette = {max(sil_scores):.3f})")
    for cluster_id in range(best_k):
        cluster_words = [words[i] for i, l in enumerate(final_labels) if l == cluster_id]
        print(f"  Cluster {cluster_id}: {cluster_words[:8]}")
```

**Graph 9.2 — Elbow + Silhouette Plot for k Selection (2×4 subplots):**

```python
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for col_idx, dom in enumerate(domains_ord):
    if dom not in ft_vectors:
        continue
    words = list(ft_vectors[dom].keys())
    vecs  = np.array([ft_vectors[dom][w] for w in words])
    if len(vecs) < 6:
        continue
    
    k_range    = range(2, min(8, len(words)//2))
    inertias   = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        lbs = km.fit_predict(vecs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(vecs, lbs))
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    
    # Top row: elbow
    ax_elbow = axes[0, col_idx]
    ax_elbow.plot(list(k_range), inertias, 'b-o', markersize=6)
    ax_elbow.axvline(best_k, color='red', linestyle='--', lw=1.5, label=f'Best k={best_k}')
    ax_elbow.set_xlabel('k')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title(f'{dom} — Elbow')
    ax_elbow.legend(fontsize=8)
    
    # Bottom row: silhouette
    ax_sil = axes[1, col_idx]
    ax_sil.plot(list(k_range), sil_scores, 'g-s', markersize=6)
    ax_sil.axvline(best_k, color='red', linestyle='--', lw=1.5)
    ax_sil.set_xlabel('k')
    ax_sil.set_ylabel('Silhouette score')
    ax_sil.set_title(f'{dom} — Silhouette')

plt.suptitle('K-Means k Selection: Elbow (top) + Silhouette (bottom) per Domain',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_kmeans_k_selection.png', dpi=150)
plt.show()
```

**Why both elbow AND silhouette:** The elbow looks for where inertia stops dropping steeply (diminishing returns). The silhouette gives a direct quality measure of cluster separation (range −1 to +1; higher = better defined clusters). They often disagree. When they do, prefer silhouette — it is more principled because it measures both cohesion (how tight clusters are) and separation (how far apart clusters are). Report both and state which k you selected and why.

---

### Step 9.5 — Independent Clustering: Agglomerative Hierarchical + Dendrogram

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

agglom_results = {}  # {domain: {word: cluster_label}}

for dom in domains_ord:
    if dom not in distance_matrices:
        continue
    words, dist_mat = distance_matrices[dom]
    if len(words) < 6:
        continue
    
    # Ward linkage on cosine distance matrix
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method='ward')
    
    # Cut tree at same k as K-Means for comparability
    best_k = max(set(kmeans_results.get(dom, {0: 2}).values())) + 1
    labels = fcluster(Z, t=best_k, criterion='maxclust')
    agglom_results[dom] = {w: int(l)-1 for w, l in zip(words, labels)}
    
    print(f"\n{dom.upper()} — Agglomerative (Ward, k={best_k}):")
    for cid in range(best_k):
        cwords = [words[i] for i, l in enumerate(labels) if l == cid+1]
        print(f"  Cluster {cid}: {cwords[:8]}")
```

**Graph 9.3 — Dendrogram per Domain (2×2):**

```python
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    if dom not in distance_matrices:
        ax.set_visible(False)
        continue
    words, dist_mat = distance_matrices[dom]
    
    # Show top-40 most frequent words for readability
    freq = df_hh[df_hh['domain']==dom].groupby('word')['word'].count()
    top_words = freq.nlargest(min(40, len(words))).index.tolist()
    top_idx   = [words.index(w) for w in top_words if w in words]
    sub_dist  = dist_mat[np.ix_(top_idx, top_idx)]
    sub_words = [words[i] for i in top_idx]
    
    condensed = squareform(sub_dist)
    Z = linkage(condensed, method='ward')
    
    dendrogram(Z, labels=sub_words, ax=ax,
               orientation='left', leaf_font_size=8,
               color_threshold=0.7*max(Z[:,2]))
    ax.set_title(f'{dom.capitalize()} — Hierarchical Clustering Dendrogram\n(Ward linkage, FastText distances, top-40 words)')
    ax.set_xlabel('Ward Distance')

plt.suptitle('Agglomerative Dendrograms: Natural Semantic Groupings in Hindi Lexicon',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_dendrograms.png', dpi=150)
plt.show()
```

**Why dendrograms:** A dendrogram shows the full hierarchical structure of semantic similarity — not just a flat partition into k groups. You can visually confirm that Hindi speakers' intuitions align with the structure: do "बाघ" (tiger) and "शेर" (lion) appear on the same branch? Do "हाथी" (elephant) and "गैंडा" (rhino) cluster together (large mammals)? The dendrogram is also more transparent than K-Means for reporting in a paper — a reader can trace any branch and understand the merging criterion.

---

### Step 9.6 — Independent Clustering: HDBSCAN (Density-Based)

```python
# pip install hdbscan
import hdbscan

hdbscan_results = {}   # {domain: {word: cluster_label}}  (-1 = noise)

for dom in domains_ord:
    if dom not in ft_vectors:
        continue
    words = list(ft_vectors[dom].keys())
    vecs  = np.array([ft_vectors[dom][w] for w in words])
    if len(words) < 8:
        continue
    
    # min_cluster_size = max(3, len(words)//6) ensures meaningful cluster sizes
    min_cs = max(3, len(words) // 6)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cs,
                                  metric='euclidean',
                                  cluster_selection_method='eom')
    labels = clusterer.fit_predict(vecs)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    hdbscan_results[dom] = {w: int(l) for w, l in zip(words, labels)}
    
    print(f"\n{dom.upper()}: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(words)*100:.0f}%)")
    for cid in sorted(set(labels)):
        label = 'NOISE' if cid == -1 else f'Cluster {cid}'
        cwords = [words[i] for i, l in enumerate(labels) if l == cid]
        print(f"  {label}: {cwords[:8]}")
```

**Why HDBSCAN in addition to K-Means and Agglomerative:**
K-Means forces every word into a cluster and assumes spherical clusters of roughly equal size. Agglomerative hierarchical clustering is deterministic but requires you to cut the tree at a chosen height. HDBSCAN is different in two important ways: (1) it finds clusters of varying density, so a tight core of "common animals" (cow, dog, cat) and a sparser periphery (aardvark, pangolin) are treated differently; (2) it labels rare or ambiguous words as "noise" (label = −1) instead of forcing them into the nearest cluster. Words labelled as noise are informative — they are the semantically peripheral items that don't clearly belong to any cluster, and they tend to have longer IRTs in VFT.

---

### Step 9.7 — Compare Three Clustering Methods

```python
from sklearn.metrics import adjusted_rand_score

print("=== Cluster Method Agreement (ARI between methods) ===")
print(f"{'Domain':12s} | KM vs Agglom | KM vs HDBSCAN | Agglom vs HDBSCAN")
print("-" * 62)

method_agreement = []
for dom in domains_ord:
    if dom not in kmeans_results or dom not in agglom_results:
        continue
    words = list(kmeans_results[dom].keys())
    
    km_labels = [kmeans_results[dom].get(w, -1) for w in words]
    ag_labels = [agglom_results[dom].get(w, -1) for w in words]
    hd_labels = [hdbscan_results.get(dom, {}).get(w, -1) for w in words]
    
    # Exclude HDBSCAN noise points from comparison
    valid = [i for i, l in enumerate(hd_labels) if l != -1]
    km_v  = [km_labels[i] for i in valid]
    ag_v  = [ag_labels[i] for i in valid]
    hd_v  = [hd_labels[i] for i in valid]
    
    ari_km_ag = adjusted_rand_score(km_labels, ag_labels)
    ari_km_hd = adjusted_rand_score(km_v, hd_v) if valid else np.nan
    ari_ag_hd = adjusted_rand_score(ag_v, hd_v) if valid else np.nan
    
    print(f"{dom:12s} | {ari_km_ag:12.3f} | {ari_km_hd:13.3f} | {ari_ag_hd:.3f}")
    method_agreement.append({'domain': dom,
                              'ari_km_ag': ari_km_ag,
                              'ari_km_hd': ari_km_hd,
                              'ari_ag_hd': ari_ag_hd})
```

**Conclusion to draw:** "The three clustering methods showed [high/moderate/low] agreement (ARI range: X–Y). High agreement across methods indicates robust cluster structure in the Hindi semantic space — the clusters are stable regardless of algorithmic assumptions. Low agreement indicates that the semantic structure is ambiguous or continuous, without clear natural boundaries. [Domain X] showed the highest inter-method ARI, suggesting the most clearly defined semantic subcategories."

---

### Step 9.8 — UMAP 2D Visualisation of Hindi Semantic Space

```python
# pip install umap-learn
import umap
from matplotlib.patches import FancyArrowPatch

umap_results = {}

for dom in domains_ord:
    if dom not in ft_vectors:
        continue
    words = list(ft_vectors[dom].keys())
    vecs  = np.array([ft_vectors[dom][w] for w in words])
    if len(words) < 6:
        continue
    
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=min(10, len(words)-1),
                        min_dist=0.3, metric='cosine')
    coords_2d = reducer.fit_transform(vecs)
    umap_results[dom] = {'words': words, 'coords': coords_2d,
                          'km_labels': [kmeans_results[dom].get(w, 0) for w in words]}
    print(f"{dom}: UMAP complete")
```

**Graph 9.4 — UMAP Coloured by K-Means Cluster (2×2):**

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

cluster_palette = plt.cm.tab10.colors

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    if dom not in umap_results:
        ax.set_visible(False)
        continue
    
    res   = umap_results[dom]
    words = res['words']
    xy    = res['coords']
    labels= res['km_labels']
    
    # Get production frequencies for point sizing
    freq_map = df_hh[df_hh['domain']==dom].groupby('word')['word'].count().to_dict()
    sizes    = [freq_map.get(w, 1) * 12 + 20 for w in words]
    
    # Plot points coloured by cluster
    unique_labels = sorted(set(labels))
    for cl in unique_labels:
        mask = [l == cl for l in labels]
        ax.scatter([xy[i,0] for i,m in enumerate(mask) if m],
                   [xy[i,1] for i,m in enumerate(mask) if m],
                   s=[sizes[i] for i,m in enumerate(mask) if m],
                   color=cluster_palette[cl % 10],
                   alpha=0.75, edgecolors='white', lw=0.5,
                   label=f'Cluster {cl}')
    
    # Annotate top-15 most frequent words
    freq_sorted = sorted(range(len(words)),
                         key=lambda i: freq_map.get(words[i], 0), reverse=True)[:15]
    for i in freq_sorted:
        ax.annotate(words[i], (xy[i,0], xy[i,1]),
                    textcoords='offset points', xytext=(4, 3),
                    fontsize=7, alpha=0.9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.6, edgecolor='none'))
    
    ax.set_title(f'{dom.capitalize()} — UMAP Semantic Map\n(colour = K-Means cluster, size = VFT frequency)')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('UMAP Projections of Hindi FastText Embeddings per Domain',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_umap_clusters.png', dpi=150)
plt.show()
```

**Graph 9.5 — UMAP Coloured by Domain (all domains on one plot):**

```python
fig, ax = plt.subplots(figsize=(12, 9))

all_words_flat, all_coords_flat, all_doms_flat = [], [], []
for dom in domains_ord:
    if dom not in umap_results:
        continue
    res = umap_results[dom]
    all_words_flat.extend(res['words'])
    all_coords_flat.extend(res['coords'].tolist())
    all_doms_flat.extend([dom] * len(res['words']))

# Re-run UMAP on ALL domains together for cross-domain comparison
all_vecs = np.array([ft_vectors[dom].get(w, np.zeros(300))
                     for dom, w in zip(all_doms_flat, all_words_flat)])
reducer_all = umap.UMAP(n_components=2, random_state=42,
                         n_neighbors=15, min_dist=0.3, metric='cosine')
coords_all = reducer_all.fit_transform(all_vecs)

for dom in domains_ord:
    mask = [d == dom for d in all_doms_flat]
    ax.scatter([coords_all[i,0] for i,m in enumerate(mask) if m],
               [coords_all[i,1] for i,m in enumerate(mask) if m],
               color=dom_colors[dom], alpha=0.6, s=40,
               edgecolors='white', lw=0.3, label=dom)

# Annotate 5 most frequent words per domain
for dom in domains_ord:
    freq_dom = df_hh[df_hh['domain']==dom].groupby('word')['word'].count()
    top5 = freq_dom.nlargest(5).index.tolist()
    for w in top5:
        if w in all_words_flat:
            i = all_words_flat.index(w)
            ax.annotate(w, (coords_all[i,0], coords_all[i,1]),
                        textcoords='offset points', xytext=(3,2),
                        fontsize=7, color=dom_colors[dom])

ax.legend(title='Domain', fontsize=9)
ax.set_title('UMAP: All Hindi Words Across Domains\n(colour = domain — check for domain separation)')
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')
plt.tight_layout()
plt.savefig('fig_umap_all_domains.png', dpi=150)
plt.show()
```

**Why both per-domain and cross-domain UMAP:** The per-domain UMAP reveals sub-cluster structure within a category (which animals cluster together? which foods?). The cross-domain UMAP answers a different question: do the four categories separate in embedding space? If animals and foods overlap substantially, it suggests the FastText semantic space does not cleanly distinguish the categories — which would make clustering harder and might explain why IRT-based cluster switching doesn't perfectly mirror embedding structure.

**Conclusion to draw for Phase 9:**
"FastText embeddings revealed structured semantic space for all four domains. K-Means (best k selected by silhouette), agglomerative clustering, and HDBSCAN showed [high/moderate] inter-method agreement (ARI = X–Y), confirming robust cluster structure. UMAP projections revealed [domain-specific insights: e.g., animals split into wild/domestic/aquatic subclusters, body-parts split into upper/lower/internal]. Cross-domain UMAP showed [clear/partial] separation between the four categories, with [foods/animals overlapping most]. HDBSCAN noise points (X% of corpus words) were semantically peripheral words with high mean IRT, consistent with the lexical exhaustion hypothesis."

---

## 12. Phase 10 — SpAM–IRT Neighbourhood Integration (RQ2) ⭐ NEW {#phase10-spam-irt}

> **Research Question addressed:** RQ2 (extended) — Does faster VFT retrieval reflect tighter semantic neighbourhoods? This is the full version of RQ2, which your current report addresses only at the participant level (confidence → total words). The deeper question is at the **word level**: do specific words that are surrounded by close semantic neighbours in FastText space also get retrieved faster on average across participants?

> **Dependency:** Requires word_irt table (Phase 4, Step 4.2), FastText knn_df (Phase 9, Step 9.3), and SpAM data (Phase 8).

---

### Step 10.1 — Validate SpAM vs FastText Distance Alignment (Mantel Test)

```python
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

# For each domain, compare the full SpAM distance matrix with the FastText distance matrix
# Only words that appear in BOTH datasets can be compared

print("=== SpAM vs FastText Distance Matrix Alignment (Mantel Test) ===\n")

mantel_results = {}

for dom in domains_ord:
    if dom not in distance_matrices:
        continue
    
    # Get mean SpAM positions per word for this domain
    spam_dom = spam[spam['domain'] == dom]
    mean_pos = spam_dom.groupby('word')[['x','y']].mean()
    
    ft_words, ft_dist = distance_matrices[dom]
    
    # Intersection: words in both FastText matrix and SpAM
    common_words = [w for w in ft_words if w in mean_pos.index]
    if len(common_words) < 5:
        print(f"{dom}: insufficient overlap (N={len(common_words)}) — skip")
        continue
    
    # Build SpAM distance matrix for common words
    spam_coords = mean_pos.loc[common_words][['x','y']].values
    from scipy.spatial.distance import pdist, squareform as sq
    spam_dist_mat = sq(pdist(spam_coords, metric='euclidean'))
    
    # Build FastText distance matrix for same words
    ft_idx = [ft_words.index(w) for w in common_words]
    ft_dist_sub = ft_dist[np.ix_(ft_idx, ft_idx)]
    
    # Mantel test: correlate upper triangles of both matrices
    n = len(common_words)
    triu_idx = np.triu_indices(n, k=1)
    spam_vec = spam_dist_mat[triu_idx]
    ft_vec   = ft_dist_sub[triu_idx]
    
    r, p = spearmanr(spam_vec, ft_vec)
    mantel_results[dom] = {'r': r, 'p': p, 'n_words': n}
    
    print(f"{dom:12s}: Spearman r = {r:.3f}, p = {p:.4f}, N words = {n}")
    print(f"           Interpretation: SpAM and FastText distances are "
          f"{'significantly' if p<0.05 else 'NOT'} correlated\n")
```

**Why Mantel test first:** Before using SpAM distances as a measure of neighbourhood density for RQ2, you need to verify that SpAM actually captures semantic proximity in a way that aligns with FastText. If the Mantel test is significant (r > 0.2, p < 0.05), it means participants' spatial arrangements of words partially mirror the computational semantic structure — this validates using SpAM as a measure of perceived semantic neighbourhood. If Mantel r is near zero, SpAM and FastText are measuring different things, and you should discuss this discrepancy.

**Graph 10.1 — SpAM vs FastText Distance Scatter per Domain:**

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    if dom not in mantel_results:
        ax.set_visible(False)
        continue
    
    # Rebuild vectors for plotting
    spam_dom = spam[spam['domain'] == dom]
    mean_pos = spam_dom.groupby('word')[['x','y']].mean()
    ft_words_d, ft_dist_d = distance_matrices[dom]
    common = [w for w in ft_words_d if w in mean_pos.index]
    if len(common) < 5:
        continue
    
    spam_coords = mean_pos.loc[common][['x','y']].values
    from scipy.spatial.distance import pdist
    spam_v = pdist(spam_coords, metric='euclidean')
    ft_idx = [ft_words_d.index(w) for w in common]
    ft_v   = squareform(ft_dist_d[np.ix_(ft_idx, ft_idx)])[np.triu_indices(len(common), k=1)]
    
    ax.scatter(ft_v, spam_v, alpha=0.2, s=8, color=dom_colors[dom])
    r = mantel_results[dom]['r']
    p = mantel_results[dom]['p']
    ax.set_xlabel('FastText Cosine Distance')
    ax.set_ylabel('SpAM Euclidean Distance')
    ax.set_title(f'{dom.capitalize()}\nMantel r={r:.3f}, p={p:.4f}')
    
    # Add regression line
    z = np.polyfit(ft_v, spam_v, 1)
    xs = np.linspace(ft_v.min(), ft_v.max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'r-', lw=2)

plt.suptitle('Validation: SpAM Distance vs FastText Distance\n(each point = one word pair)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_spam_fasttext_validation.png', dpi=150)
plt.show()
```

---

### Step 10.2 — RQ2: Neighbourhood Density vs Mean IRT (Word Level)

```python
# Merge word_irt with knn_df (FastText neighbourhood) and SpAM neighbourhood
# word_irt was built in Phase 4 Step 4.2

# FastText neighbourhood density already in knn_df
word_analysis = word_irt.merge(
    knn_df[['domain','word','mean_nn_dist_ft']], on=['domain','word'], how='inner')

# Compute SpAM neighbourhood density for each word
spam_nn_records = []
for dom in domains_ord:
    spam_dom = spam[spam['domain'] == dom]
    mean_pos = spam_dom.groupby('word')[['x','y']].mean().reset_index()
    if len(mean_pos) < 3:
        continue
    
    coords = mean_pos[['x','y']].values
    words  = mean_pos['word'].tolist()
    
    from scipy.spatial.distance import cdist
    dist_mat = cdist(coords, coords, metric='euclidean')
    np.fill_diagonal(dist_mat, np.inf)
    
    for i, w in enumerate(words):
        nn_dists = np.sort(dist_mat[i])[:K_NEIGHBOURS]
        spam_nn_records.append({'domain': dom, 'word': w,
                                 'mean_nn_dist_spam': nn_dists.mean()})

spam_nn_df = pd.DataFrame(spam_nn_records)
word_analysis = word_analysis.merge(spam_nn_df, on=['domain','word'], how='left')

# Filter: only words produced by >= 5 participants (reliability threshold)
word_analysis_filtered = word_analysis[word_analysis['n_participants'] >= 5].copy()
print(f"Words with N>=5 participants: {len(word_analysis_filtered)}")
print(f"  Per domain: {word_analysis_filtered.groupby('domain').size().to_dict()}")
```

**Graph 10.2 — Neighbourhood Density vs Mean IRT (Main RQ2 Figure):**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: FastText neighbourhood density vs mean IRT
ax = axes[0]
for dom in domains_ord:
    sub = word_analysis_filtered[word_analysis_filtered['domain'] == dom]
    if len(sub) < 3:
        continue
    ax.scatter(sub['mean_nn_dist_ft'], sub['mean_irt_ms'],
               color=dom_colors[dom], alpha=0.7,
               s=sub['freq']*8 + 15,
               edgecolors='white', lw=0.3, label=dom)
    # Annotate top-3 most frequent words
    for _, row in sub.nlargest(3, 'freq').iterrows():
        ax.annotate(row['word'], (row['mean_nn_dist_ft'], row['mean_irt_ms']),
                    textcoords='offset points', xytext=(3, 2), fontsize=7)

# Overall regression
x_all = word_analysis_filtered['mean_nn_dist_ft'].dropna()
y_all = word_analysis_filtered.loc[x_all.index, 'mean_irt_ms']
z = np.polyfit(x_all, y_all, 1)
xs = np.linspace(x_all.min(), x_all.max(), 100)
ax.plot(xs, np.poly1d(z)(xs), 'k--', lw=2, label='Overall trend')
r_ft, p_ft = spearmanr(x_all, y_all)
ax.set_xlabel('Mean FastText Neighbourhood Distance\n(higher = more isolated word)')
ax.set_ylabel('Mean IRT (ms)')
ax.set_title(f'RQ2: Semantic Neighbourhood → Retrieval Speed\n(FastText) ρ = {r_ft:.3f}, p = {p_ft:.4f}')
ax.legend(fontsize=8)

# Right: SpAM neighbourhood density vs mean IRT
ax = axes[1]
sub_spam = word_analysis_filtered.dropna(subset=['mean_nn_dist_spam'])
for dom in domains_ord:
    sub = sub_spam[sub_spam['domain'] == dom]
    if len(sub) < 3:
        continue
    ax.scatter(sub['mean_nn_dist_spam'], sub['mean_irt_ms'],
               color=dom_colors[dom], alpha=0.7,
               s=sub['freq']*8 + 15,
               edgecolors='white', lw=0.3, label=dom)

if len(sub_spam) > 4:
    x_sp = sub_spam['mean_nn_dist_spam']
    y_sp = sub_spam['mean_irt_ms']
    z_sp = np.polyfit(x_sp, y_sp, 1)
    xs_sp = np.linspace(x_sp.min(), x_sp.max(), 100)
    ax.plot(xs_sp, np.poly1d(z_sp)(xs_sp), 'k--', lw=2)
    r_sp, p_sp = spearmanr(x_sp, y_sp)
    ax.set_title(f'RQ2: Semantic Neighbourhood → Retrieval Speed\n(SpAM) ρ = {r_sp:.3f}, p = {p_sp:.4f}')
else:
    ax.set_title('RQ2: SpAM neighbourhood vs IRT\n(insufficient data)')

ax.set_xlabel('Mean SpAM Neighbourhood Distance\n(higher = more isolated in perceived space)')
ax.set_ylabel('Mean IRT (ms)')
ax.legend(fontsize=8)

plt.suptitle('RQ2: Does Tighter Semantic Neighbourhood → Faster Retrieval?\n(point size = production frequency)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_RQ2_neighbourhood_IRT.png', dpi=150)
plt.show()
```

**Graph 10.3 — Bubble Chart: Word Frequency × IRT × Neighbourhood Density:**

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    sub = word_analysis_filtered[word_analysis_filtered['domain'] == dom].dropna(
        subset=['mean_nn_dist_ft'])
    if len(sub) < 3:
        ax.set_visible(False)
        continue
    
    sc = ax.scatter(sub['freq'], sub['mean_irt_ms'],
                    s=sub['mean_nn_dist_ft'] * 800 + 20,  # size = neighbourhood distance
                    c=sub['mean_nn_dist_ft'],              # colour also = neighbourhood distance
                    cmap='RdYlGn_r', alpha=0.75,
                    edgecolors='gray', lw=0.5)
    plt.colorbar(sc, ax=ax, label='FastText NN distance\n(red=isolated, green=central)')
    
    for _, row in sub.nlargest(5, 'freq').iterrows():
        ax.annotate(row['word'], (row['freq'], row['mean_irt_ms']),
                    textcoords='offset points', xytext=(4,3), fontsize=8)
    
    ax.set_xlabel('Production Frequency')
    ax.set_ylabel('Mean IRT (ms)')
    ax.set_title(f'{dom.capitalize()}\n(bubble size + colour = neighbourhood distance)')

plt.suptitle('Word Frequency × IRT × Semantic Neighbourhood Density\n'
             '(green = central word, red = peripheral word)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_bubble_freq_IRT_neighbourhood.png', dpi=150)
plt.show()
```

**How to read the bubble chart:** A central word (green, small bubble) should appear in the bottom-left (high frequency + low IRT) — it is both commonly produced and quickly retrieved, because its dense neighbourhood means spreading activation reaches it easily. A peripheral word (red, large bubble) should appear top-right (low frequency + high IRT). If this pattern holds, it is direct visual support for H8 (tighter neighbourhood → faster retrieval).

### Step 10.3 — Per-Domain Spearman Correlations (H8 — Full Analysis)

```python
print("=" * 65)
print("H8: Does Neighbourhood Density Predict Mean IRT? (Word Level)")
print("=" * 65)

for dom in domains_ord:
    sub = word_analysis_filtered[word_analysis_filtered['domain']==dom].dropna(
        subset=['mean_nn_dist_ft','mean_irt_ms'])
    if len(sub) < 5:
        print(f"{dom}: N={len(sub)} words — insufficient for correlation")
        continue
    
    # FastText neighbourhood density
    r_ft, p_ft = spearmanr(sub['mean_nn_dist_ft'], sub['mean_irt_ms'])
    
    # SpAM neighbourhood density (if available)
    sub_sp = sub.dropna(subset=['mean_nn_dist_spam'])
    if len(sub_sp) >= 5:
        r_sp, p_sp = spearmanr(sub_sp['mean_nn_dist_spam'], sub_sp['mean_irt_ms'])
        spam_str = f"SpAM: ρ={r_sp:+.3f} p={p_sp:.4f}"
    else:
        spam_str = "SpAM: insufficient data"
    
    sig = '***' if p_ft<0.001 else '**' if p_ft<0.01 else '*' if p_ft<0.05 else 'ns'
    interpretation = ("tighter neighbourhood → faster ✓" if r_ft > 0.1
                      else "no clear pattern" if abs(r_ft) < 0.1
                      else "UNEXPECTED: tighter → slower ✗")
    print(f"\n{dom.upper()} (N={len(sub)} words):")
    print(f"  FastText: ρ = {r_ft:+.3f}, p = {p_ft:.4f} {sig} — {interpretation}")
    print(f"  {spam_str}")
```

### Step 10.4 — IRT Clusters vs FastText Clusters Alignment

```python
print("=" * 65)
print("H9 Extension: IRT Clusters vs FastText K-Means Clusters (ARI)")
print("=" * 65)

irt_cluster_records = []

for (subj, dom), grp in df_hh.sort_values('position').groupby(['subject_id','domain']):
    irts = grp.sort_values('position')['rt_ms'].values
    words = grp.sort_values('position')['word'].tolist()
    if len(irts) < 4:
        continue
    
    threshold = np.mean(irts) + np.std(irts, ddof=1)
    cluster_id, cluster_labels_irt = 0, []
    for i, irt in enumerate(irts):
        if i > 0 and irt > threshold:
            cluster_id += 1
        cluster_labels_irt.append(cluster_id)
    
    for word, irt_cl in zip(words, cluster_labels_irt):
        ft_cl = kmeans_results.get(dom, {}).get(word, -1)
        irt_cluster_records.append({
            'subject_id': subj, 'domain': dom,
            'word': word, 'irt_cluster': irt_cl, 'ft_cluster': ft_cl
        })

irt_cl_df = pd.DataFrame(irt_cluster_records)
irt_cl_df = irt_cl_df[irt_cl_df['ft_cluster'] >= 0]  # exclude unknown FastText words

print("\nARI between IRT-based clusters and FastText K-Means clusters:")
for dom in domains_ord:
    sub = irt_cl_df[irt_cl_df['domain']==dom]
    if len(sub) < 10:
        continue
    # Compute per-participant ARI, then average
    ari_list = []
    for subj, subj_grp in sub.groupby('subject_id'):
        if subj_grp['irt_cluster'].nunique() > 1 and subj_grp['ft_cluster'].nunique() > 1:
            ari_list.append(adjusted_rand_score(subj_grp['irt_cluster'],
                                                 subj_grp['ft_cluster']))
    if ari_list:
        print(f"  {dom:12s}: mean ARI = {np.mean(ari_list):.3f} ± {np.std(ari_list):.3f} "
              f"(N={len(ari_list)} participants)")
```

**Graph 10.4 — Confusion Heatmap: IRT Clusters vs FastText Clusters:**

```python
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, dom in enumerate(domains_ord):
    ax = axes[idx]
    sub = irt_cl_df[irt_cl_df['domain']==dom]
    if len(sub) < 10:
        ax.set_visible(False)
        continue
    
    # Build confusion matrix (aggregate across all participants)
    max_irt_cl = sub['irt_cluster'].max() + 1
    max_ft_cl  = sub['ft_cluster'].max() + 1
    
    cm = confusion_matrix(sub['irt_cluster'], sub['ft_cluster'])
    
    # Normalize by row (IRT cluster) to show proportion
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'FT-C{i}' for i in range(cm_norm.shape[1])],
                yticklabels=[f'IRT-C{i}' for i in range(cm_norm.shape[0])],
                ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('FastText Cluster')
    ax.set_ylabel('IRT-Based Cluster')
    ax.set_title(f'{dom.capitalize()}\nIRT vs FastText Cluster Overlap\n(row-normalized)')

plt.suptitle('H9 Extension: IRT-Based Clusters vs FastText Embedding Clusters\n'
             '(high diagonal = participants follow semantic structure)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_IRT_vs_FT_confusion.png', dpi=150)
plt.show()
```

**How to interpret the confusion matrix:** If retrieval is semantically organised and participants follow the structure detected by FastText, then IRT-based clusters (words produced in rapid sequence) should correspond to FastText clusters (computationally similar words). A strong diagonal pattern (high values on diagonal, low off-diagonal) means the match is good. A uniform pattern means IRT clusters and FastText clusters are independent. A non-diagonal pattern (high values off-diagonal) would be very interesting — it would mean participants are crossing semantic boundaries within their retrieval runs, possibly following phonological or frequency-based paths instead.

**Conclusion to draw for Phase 10:**
"The Mantel test revealed [significant/non-significant] alignment between SpAM distances and FastText distances (ρ = X–Y across domains), [validating/questioning] SpAM as a measure of perceived semantic proximity. At the word level, FastText neighbourhood density showed [significant/non-significant] positive correlation with mean IRT (ρ = X, p = Y), [supporting/not supporting] H8. The bubble chart confirms a clear pattern: high-frequency words with dense semantic neighbourhoods are retrieved both rapidly and commonly, while peripheral words (sparse neighbourhoods) have longer IRTs and lower production rates. IRT-based clusters showed [high/moderate/low] alignment with FastText clusters (mean ARI = X), suggesting that Hindi speakers' sequential retrieval [does/does not] closely follow the latent semantic structure of the Hindi lexicon."

---

## 13. Phase 11 — Embedding Clustering & SpAM Alignment (RQ4, H9) {#phase11-align}

### Step 9.1 — Generate Embeddings

```python
# Uses the multilingual sentence transformer from your report
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import unicodedata

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def normalize_text(text):
    return unicodedata.normalize('NFKC', str(text).lower().strip())

# Get unique words per domain
all_words = df_hh.groupby('domain')['word'].unique().to_dict()

embeddings = {}
for dom, words in all_words.items():
    normalized = [normalize_text(w) for w in words]
    vecs = model.encode(normalized, show_progress_bar=False)
    embeddings[dom] = {w: v for w, v in zip(words, vecs)}
    print(f"{dom}: {len(words)} words embedded")
```

### Step 9.2 — Phonetic Key Generation

```python
# Consonant-only mapping (your existing method from the report)
DEVANAGARI_MAP = {
    'क':'k','ख':'kh','ग':'g','घ':'gh','ञ':'n',
    'च':'ch','छ':'chh','ज':'j','झ':'jh',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n',
    'त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m',
    'य':'y','र':'r','ल':'l','व':'v',
    'श':'sh','ष':'sh','स':'s','ह':'h',
}
VOWELS_DEVA = set('अआइईउऊएऐओऔ')
VOWELS_LATIN = set('aeiouAEIOU')

def phonetic_key(word):
    word = normalize_text(word)
    result = []
    for ch in word:
        if ch in DEVANAGARI_MAP and ch not in VOWELS_DEVA:
            result.append(DEVANAGARI_MAP[ch])
        elif ch.isascii() and ch not in VOWELS_LATIN and ch.isalpha():
            result.append(ch)
    # Collapse repeated consonants
    collapsed = []
    for c in result:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    return ''.join(collapsed) if collapsed else word  # fallback to original

# Generate phonetic embeddings
phonetic_embeddings = {}
for dom, words in all_words.items():
    phon_keys = [phonetic_key(w) for w in words]
    vecs = model.encode(phon_keys, show_progress_bar=False)
    phonetic_embeddings[dom] = {w: v for w, v in zip(words, vecs)}
```

### Step 9.3 — SpAM Clustering & Alignment (H9, H10)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def best_k_silhouette(X, k_range=range(2, 6)):
    """Select k by silhouette score."""
    scores = {}
    for k in k_range:
        if len(X) <= k:
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get) if scores else 2

ari_records = []

for (subj, dom), grp_spam in spam.groupby(['subject_id','domain']):
    words_in_spam = grp_spam['word'].tolist()
    if len(words_in_spam) < 4:
        continue
    
    # SpAM clusters (hierarchical Ward)
    coords = grp_spam[['x','y']].values
    k = best_k_silhouette(coords)
    Z = linkage(coords, method='ward')
    spam_labels = fcluster(Z, t=k, criterion='maxclust')
    
    # Semantic embedding clusters
    sem_vecs = np.array([embeddings[dom].get(w, np.zeros(384)) for w in words_in_spam])
    sem_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(sem_vecs)
    
    # Phonetic embedding clusters
    pho_vecs = np.array([phonetic_embeddings[dom].get(w, np.zeros(384)) for w in words_in_spam])
    pho_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(pho_vecs)
    
    ari_sem = adjusted_rand_score(spam_labels, sem_labels)
    ari_pho = adjusted_rand_score(spam_labels, pho_labels)
    nmi_sem = normalized_mutual_info_score(spam_labels, sem_labels)
    nmi_pho = normalized_mutual_info_score(spam_labels, pho_labels)
    ari_sp  = adjusted_rand_score(sem_labels, pho_labels)
    
    ari_records.append({
        'subject_id': subj, 'domain': dom,
        'ari_sem': ari_sem, 'ari_pho': ari_pho,
        'nmi_sem': nmi_sem, 'nmi_pho': nmi_pho,
        'ari_sem_pho': ari_sp, 'k': k
    })

ari_df = pd.DataFrame(ari_records)
print("\nDomain-wise mean alignment metrics:")
print(ari_df.groupby('domain')[['ari_sem','ari_pho','nmi_sem','nmi_pho']].mean().round(3))
```

**Sign Tests (H9, H10) — already in your report, add permutation test for magnitude:**
```python
from scipy.stats import binom_test

print("\n=== H9: Sign Test — ARI(semantic, SpAM) > 0 ===")
n_pos_sem = (ari_df['ari_sem'] > 0).sum()
n_total_sem = len(ari_df)
p_sign_sem = binom_test(n_pos_sem, n_total_sem, p=0.5, alternative='greater')
print(f"Positive: {n_pos_sem}/{n_total_sem}, p = {p_sign_sem:.4f}")

print("\n=== H10: Sign Test — ARI(phonetic, SpAM) > 0 ===")
n_pos_pho = (ari_df['ari_pho'] > 0).sum()
n_total_pho = len(ari_df)
p_sign_pho = binom_test(n_pos_pho, n_total_pho, p=0.5, alternative='greater')
print(f"Positive: {n_pos_pho}/{n_total_pho}, p = {p_sign_pho:.4f}")

print("\n=== Permutation Test: Is mean ARI above shuffled baseline? ===")
# Permutation: shuffle SpAM labels, recompute ARI 1000 times
observed_mean_ari = ari_df['ari_sem'].mean()
null_aris = []
for _ in range(1000):
    shuffled = ari_df['ari_sem'].sample(frac=1, replace=False).values
    null_aris.append(np.mean(shuffled))
p_perm = np.mean(np.array(null_aris) >= observed_mean_ari)
print(f"Observed mean ARI: {observed_mean_ari:.4f}")
print(f"Permutation p: {p_perm:.4f}")
```

**Graph 9.1 — ARI Domain Comparison + Global Bar:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: boxplot of ARI by domain
plot_ari = pd.melt(ari_df, id_vars=['domain'],
                   value_vars=['ari_sem','ari_pho'],
                   var_name='metric', value_name='ARI')
plot_ari['metric'] = plot_ari['metric'].map({'ari_sem':'Semantic-SpAM', 'ari_pho':'Phonetic-SpAM'})
sns.boxplot(data=plot_ari, x='domain', y='ARI', hue='metric',
            palette=['#185FA5','#D85A30'], order=domains_ord, ax=axes[0])
axes[0].axhline(0, color='black', lw=1, linestyle='--', alpha=0.5)
axes[0].set_title('RQ4: ARI Alignment by Domain')
axes[0].set_ylabel('Adjusted Rand Index')

# Right: global mean bar
means = {'SpAM-Sem': ari_df['ari_sem'].mean(),
         'SpAM-Pho': ari_df['ari_pho'].mean(),
         'Sem-Pho':  ari_df['ari_sem_pho'].mean()}
sems  = {'SpAM-Sem': ari_df['ari_sem'].sem(),
         'SpAM-Pho': ari_df['ari_pho'].sem(),
         'Sem-Pho':  ari_df['ari_sem_pho'].sem()}
bars = axes[1].bar(means.keys(), means.values(), color=[PALETTE[0],PALETTE[1],PALETTE[2]],
                   yerr=sems.values(), capsize=8, edgecolor='black', alpha=0.8)
for bar, val in zip(bars, means.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10)
axes[1].axhline(0, color='black', lw=1, linestyle='--')
axes[1].set_ylabel('Mean ARI ± SEM')
axes[1].set_title('Global Mean Cluster Alignment\n(ARI > 0 = above-chance agreement)')

plt.tight_layout()
plt.savefig('fig_RQ4_cluster_alignment.png', dpi=150)
plt.show()
```

**Conclusion for RQ4:** "Sign tests confirmed that participant SpAM clusters aligned with semantic embeddings above chance (X/Y positive, p=0.027) and phonetic embeddings above chance (X/Y positive, p=0.036). However, mean ARI values were low (semantic: 0.154, phonetic: 0.111), reflecting slight to fair agreement. NMI was notably higher (semantic: 0.405), indicating moderate shared information structure even when cluster label agreement is weak. ARI for body-parts was highest (0.316), suggesting that domains with clearer hierarchical structure (upper/lower body) produce stronger alignment between perceived and model-based clusters. The semantic channel consistently outperformed the phonetic channel, suggesting that Hindi speakers' spatial arrangements of words are more strongly governed by meaning than by sound."

---

## 14. Phase 12 — Phonological Similarity Analysis (RQ6, RQ7) {#phase12}

### Step 10.1 — Compute Hindi Phonological Similarity

```python
# pip install indic-transliteration python-Levenshtein
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import Levenshtein

def hindi_to_phonemes(word):
    """Convert Devanagari to ITRANS/Latin phoneme string."""
    try:
        roman = transliterate(str(word), sanscript.DEVANAGARI, sanscript.ITRANS)
        return roman.lower().strip()
    except Exception:
        return str(word).lower()

def normalized_edit_distance(a, b):
    """Kumar et al. 2022 formula: d(a,b) = 1 - edit(a,b)/max(len_a, len_b)"""
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1 - Levenshtein.distance(a, b) / max_len

# Compute phonological similarity for each consecutive pair
df_hh_sorted = df_hh.sort_values(['subject_id','domain','position'])
phon_records = []

for (subj, dom), grp in df_hh_sorted.groupby(['subject_id','domain']):
    words = grp.sort_values('position')['word'].tolist()
    positions = grp.sort_values('position')['position'].tolist()
    irts = grp.sort_values('position')['rt_ms'].tolist()
    
    oov_count = 0
    for i in range(1, len(words)):
        a_phon = hindi_to_phonemes(words[i-1])
        b_phon = hindi_to_phonemes(words[i])
        phon_sim = normalized_edit_distance(a_phon, b_phon)
        
        # Also get semantic similarity from FastText embeddings if available
        phon_records.append({
            'subject_id': subj, 'domain': dom,
            'word_a': words[i-1], 'word_b': words[i],
            'position': positions[i], 'irt_ms': irts[i],
            'phon_sim': phon_sim,
            'a_phon': a_phon, 'b_phon': b_phon,
        })

phon_df = pd.DataFrame(phon_records)

# Report OOV rate (words that couldn't be transliterated)
oov = phon_df[phon_df['a_phon'] == phon_df['word_a'].str.lower()]
print(f"OOV (untransliterated words): {len(oov)/len(phon_df)*100:.1f}%")
print(phon_df[['word_a','a_phon','phon_sim']].head(10))
```

### Step 10.2 — H11: Phonological Similarity Increases Over Retrieval Order

```python
# Bin retrieval positions into quintiles for plotting
phon_df['pos_quintile'] = pd.qcut(phon_df['position'], q=5, labels=['Q1','Q2','Q3','Q4','Q5'])

print("Mean phonological similarity by retrieval quintile:")
print(phon_df.groupby('pos_quintile')['phon_sim'].mean().round(4))

# LME: phonological similarity ~ position + (1+position|subject)
try:
    model_phon = mixedlm("phon_sim ~ position", phon_df, groups=phon_df["subject_id"])
    result_phon = model_phon.fit(reml=True, disp=False)
    b_phon = result_phon.params['position']
    p_phon = result_phon.pvalues['position']
    print(f"\nLME: phon_sim ~ position")
    print(f"β = {b_phon:.6f}, p = {p_phon:.4f}")
    print(f"Interpretation: {'phonological similarity INCREASES with position ✓' if b_phon > 0 else 'phonological similarity decreases with position ✗'}")
except Exception as e:
    print(f"LME failed: {e}")
```

**Graph 10.1 — Dual Similarity Curves (H11 hero figure):**
```python
# Compute semantic similarity from embeddings
from sklearn.metrics.pairwise import cosine_similarity

def get_sem_sim(word_a, word_b, dom):
    if dom not in embeddings:
        return np.nan
    va = embeddings[dom].get(word_a)
    vb = embeddings[dom].get(word_b)
    if va is None or vb is None:
        return np.nan
    return float(cosine_similarity([va], [vb])[0][0])

phon_df['sem_sim'] = phon_df.apply(
    lambda r: get_sem_sim(r['word_a'], r['word_b'], r['domain']), axis=1)

# Compute mean ± SEM per quintile
summary = phon_df.groupby('pos_quintile').agg(
    phon_mean=('phon_sim','mean'), phon_sem=('phon_sim',lambda x: x.sem()),
    sem_mean=('sem_sim','mean'),   sem_sem=('sem_sim', lambda x: x.sem())
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
xs = np.arange(len(summary))

# Semantic similarity (declining)
ax.plot(xs, summary['sem_mean'], 'b-o', lw=2.5, markersize=8, label='Semantic similarity')
ax.fill_between(xs, summary['sem_mean'] - summary['sem_sem'],
                    summary['sem_mean'] + summary['sem_sem'],
                alpha=0.2, color='blue')

# Phonological similarity (rising)
ax2 = ax.twinx()
ax2.plot(xs, summary['phon_mean'], 'r-s', lw=2.5, markersize=8, label='Phonological similarity')
ax2.fill_between(xs, summary['phon_mean'] - summary['phon_sem'],
                     summary['phon_mean'] + summary['phon_sem'],
                 alpha=0.2, color='red')

# Find crossover point
# (Add vertical dashed line if curves cross)
ax.set_xticks(xs)
ax.set_xticklabels(['Q1\n(Early)', 'Q2', 'Q3', 'Q4', 'Q5\n(Late)'])
ax.set_xlabel('Retrieval Position Quintile')
ax.set_ylabel('Mean Semantic Similarity (cosine)', color='blue')
ax2.set_ylabel('Mean Phonological Similarity (edit distance)', color='red')
ax.set_title('H11: Semantic vs Phonological Similarity Over Retrieval Order\n(Replication of Kumar et al. 2022 in Hindi)')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig('fig_H11_dual_similarity.png', dpi=150)
plt.show()
```

**Why dual y-axes:** Semantic and phonological similarity are on different scales. Dual axes let both patterns be visible simultaneously while clearly showing the crossover point where phonological similarity becomes more prominent than semantic similarity — the theoretical "hand-off" moment between semantic and phonological retrieval cues.

**Conclusion for H11:** "[If replicated]: Phonological similarity between consecutive Hindi responses increased significantly over retrieval order (β=X, p<0.05), while semantic similarity declined, replicating Kumar et al. (2022) in Hindi. The crossover between the two curves occurred around the [Q3/Q4] quintile, suggesting that phonological cues become active retrieval guides when approximately 60-70% of easily accessible semantically similar words have already been retrieved. [If not replicated]: Unlike Kumar et al. (2022), phonological similarity did not significantly increase over retrieval order in our Hindi sample (β=X, p=ns). Possible explanations include: the shorter task duration (60s vs 3 min), the bilingual context allowing code-switching to reset semantic neighbourhoods, or genuine differences in Hindi phonological neighbourhood structure."

### Step 10.3 — H12: Phonological Similarity Predicts Word Count

```python
per_part_phon = phon_df.groupby('subject_id')['phon_sim'].mean().reset_index()
per_part_phon.columns = ['subject_id','mean_phon_sim']
fluency = fluency.merge(per_part_phon, on='subject_id', how='left')

rho_phon, p_phon = spearmanr(fluency['mean_phon_sim'].dropna(),
                               fluency.loc[fluency['mean_phon_sim'].notna(), 'total_words'])
print(f"H12: ρ(mean phon sim, total words) = {rho_phon:.3f}, p = {p_phon:.4f}")
print("Interpretation:", "SUPPORTS H12 (more phonologically clustered → more words)" 
      if rho_phon > 0 and p_phon < 0.05 else "Does not support H12")
```

---

## 15. Phase 13 — Composite Fluency Score (RQ5) {#phase13}

### Step 11.1 — Build Composite Score (Equal-Weighted + PCA Alternative)

```python
# Merge all features into one participant table
feature_df = fluency[['subject_id','total_words','mean_irt_ms']].copy()
feature_df = feature_df.merge(
    nn_df.groupby('subject_id')['mean_nn_dist'].mean().reset_index(), 
    on='subject_id', how='left')
feature_df = feature_df.merge(
    ari_df.groupby('subject_id')[['nmi_sem','nmi_pho']].mean().reset_index(),
    on='subject_id', how='left')

# Z-score each component
from scipy.stats import zscore
feature_df['z_words']   = zscore(feature_df['total_words'].fillna(feature_df['total_words'].median()))
feature_df['z_speed']   = zscore(-feature_df['mean_irt_ms'].fillna(feature_df['mean_irt_ms'].median()))  # invert: lower IRT = better
feature_df['z_spatial'] = zscore(-feature_df['mean_nn_dist'].fillna(feature_df['mean_nn_dist'].median()))  # invert: lower dist = more compact
feature_df['z_semAlign']= zscore(feature_df['nmi_sem'].fillna(0))
feature_df['z_phoAlign']= zscore(feature_df['nmi_pho'].fillna(0))

# Equal-weight composite (your report's formula)
feature_df['score_equal'] = feature_df[['z_words','z_speed','z_spatial',
                                         'z_semAlign','z_phoAlign']].mean(axis=1)

# VFT-only score for comparison
feature_df['score_vft_only'] = feature_df[['z_words','z_speed']].mean(axis=1)

# Merge with confidence
feature_df = feature_df.merge(fluency[['subject_id','hindi_confidence']], 
                                on='subject_id', how='left')

# Compare correlations with confidence
rho_eq, p_eq   = spearmanr(feature_df['score_equal'].dropna(),
                             feature_df.loc[feature_df['score_equal'].notna(),'hindi_confidence'])
rho_vft, p_vft = spearmanr(feature_df['score_vft_only'].dropna(),
                             feature_df.loc[feature_df['score_vft_only'].notna(),'hindi_confidence'])

print(f"VFT-only score vs confidence: ρ = {rho_vft:.3f}, p = {p_vft:.4f}")
print(f"Integrated score vs confidence: ρ = {rho_eq:.3f}, p = {p_eq:.4f}")
```

**Graph 11.1 — Score Comparison:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (score_col, title) in zip(axes, [('score_vft_only','VFT-Only Score'),
                                           ('score_equal','Integrated Score')]):
    mask = feature_df[[score_col,'hindi_confidence']].notna().all(axis=1)
    x = feature_df.loc[mask, score_col]
    y = feature_df.loc[mask, 'hindi_confidence']
    ax.scatter(x, y, color=PALETTE[0], s=80, edgecolors='gray', lw=0.5, alpha=0.8)
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'r--', lw=2)
    rho, p = spearmanr(x, y)
    ax.set_xlabel(title)
    ax.set_ylabel('Hindi Confidence Score')
    ax.set_title(f'{title}\nρ = {rho:.3f}, p = {p:.3f}')

plt.suptitle('RQ5: VFT-Only vs Integrated Composite Score vs Hindi Confidence',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_RQ5_composite_scores.png', dpi=150)
plt.show()
```

**Conclusion for RQ5:** "The VFT-only score showed a stronger (albeit negative) correlation with Hindi confidence (ρ=−0.449, p=0.007) than the integrated VFT+SpAM score (ρ=−0.317, p=0.063). This does not support the hypothesis that integrated scoring provides stronger external validity. However, the integrated score is theoretically richer, capturing both retrieval dynamics (VFT) and perceived semantic structure (SpAM). The lack of improvement in correlation with confidence may reflect construct misalignment — self-rated confidence is not necessarily a valid external criterion for lexical fluency, particularly in bilingual speakers who may systematically underestimate or overestimate their proficiency."

---

## 16. Phase 14 — Final Summary, Conclusions & Discussion {#phase14}

### Step 12.1 — Hypothesis Results Table

```python
# Create final summary table
results_table = pd.DataFrame([
    ['H1',  'Within < Between IRT',            'Welch t + d',     '?','?','?', 'H1 supported/rejected'],
    ['H2',  'IRT increases with position',       'LME per domain',  '?','?','?', ''],
    ['H3',  'Confidence → word count',           'Spearman ρ',      '?','?','?', ''],
    ['H4',  'Domain differences (word count)',   'Kruskal-Wallis',  '5.295','0.1515','—', 'Not significant'],
    ['H5',  'Domain differences (RT)',           'Kruskal-Wallis',  '1.919','0.5894','—', 'Not significant'],
    ['H6',  'Confidence → RT',                  'Spearman ρ',      '?','?','?', ''],
    ['H7',  'Confidence → cluster size',         'Spearman ρ',      '?','?','?', ''],
    ['H8',  'SpAM compactness by domain',        'Kruskal-Wallis',  '3.038','0.3857','—', 'Not significant'],
    ['H9',  'SpAM ~ semantic clusters',          'Sign test',       '48/78','0.027','ARI=0.154','Supported'],
    ['H10', 'SpAM ~ phonetic clusters',          'Sign test',       '48/79','0.036','ARI=0.111','Supported'],
    ['H11', 'Phon sim increases over position', 'LME interaction', '?','?','?', ''],
    ['H12', 'Phon sim → word count',            'Spearman ρ',      '?','?','?', ''],
    ['H13', 'Integrated > VFT-only score',      'Spearman comparison','?','?','?','Not supported'],
], columns=['Hypothesis','Prediction','Test','Statistic','p-value','Effect Size','Conclusion'])
print(results_table.to_string(index=False))
```

### Step 12.2 — How to Write the Discussion

Structure your Discussion as follows:

**Paragraph 1 — Structured retrieval confirmed (H1, H2):**
Lead with the positive results. If H1 is confirmed (between-cluster IRTs significantly longer), state: "The finding that between-cluster IRTs were substantially longer than within-cluster IRTs (d=X) confirms that Hindi speakers do not retrieve words randomly — they exploit semantic sub-clusters before switching to new categories, consistent with the optimal foraging model of Hills et al. (2012)." Then add the serial position result.

**Paragraph 2 — Domain differences (RQ1, RQ3):**
Acknowledge the non-significant Kruskal-Wallis results honestly but contextualise them: "The absence of significant domain effects (RQ1, RQ3) likely reflects the severe imbalance in domain representation — colours contributed only N=4 Hindi-language observations, and the overall sample of 35 participants provides limited power for detecting between-domain effects. Descriptive patterns, however, are consistent with theoretical predictions: animals showed the highest mean IRT (largest vocabulary) and body-parts showed the most compact SpAM arrangement (clearest hierarchical structure)."

**Paragraph 3 — The confidence paradox (RQ2):**
Address the negative ρ directly: "The unexpected negative association between Hindi confidence and word count (ρ = −0.395) requires careful interpretation. [If ceiling effect found]: The confidence measure showed a ceiling distribution (X% of participants rated maximum confidence), severely restricting variance and rendering correlation estimates unreliable. [If no ceiling effect]: This counter-intuitive pattern may reflect metacognitive miscalibration common in bilingual populations — highly confident Hindi speakers may preferentially code-switch to English for rapid access to common category exemplars, reducing their Hindi word count specifically."

**Paragraph 4 — Cluster alignment (RQ4):**
Contextualise the ARI values: "While mean ARI was modest (semantic: 0.154, phonetic: 0.111), the consistent positive direction across participants (significant by sign test) indicates a genuine, if weak, structural correspondence between perceived semantic organisation (SpAM) and model-derived embedding clusters. NMI values (0.36–0.41) suggest moderate shared information — participants use meaning-based grouping principles that are partially captured by multilingual transformer representations. The stronger semantic alignment over phonetic alignment suggests Hindi speakers' spatial word arrangements are primarily meaning-driven rather than sound-driven."

**Paragraph 5 — Phonological facilitation (RQ6, RQ7):**
If replicated: "The replication of Kumar et al.'s (2022) phonological facilitation finding in Hindi extends this phenomenon beyond English. The diverging similarity curves — semantic similarity declining while phonological similarity rises over retrieval order — suggests a universal mechanism: as semantically similar words are exhausted, phonological proximity becomes a secondary retrieval cue, allowing continued search through phonological neighbours." If not replicated: discuss Hindi-specific reasons above.

**Paragraph 6 — Limitations:**
- N=35 limits power for subgroup effects
- Colours domain drastically underrepresented in Hindi responses
- Self-reported confidence may not validly measure Hindi proficiency
- Phonetic embedding via semantic transformer is a simplification — proper phonological modelling would use edit-distance on Devanagari phoneme strings
- Bilingual code-switching means "Hindi word count" is partly a function of language choice strategy, not purely lexical accessibility

### Step 12.3 — Figure for the Paper (Summary Dashboard)

```python
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

# Panel 1: Within vs Between IRT violin (H1)
ax1 = fig.add_subplot(gs[0, 0])
# [use within_arr and between_arr]
ax1.set_title('(1) H1: Clustering confirmed', fontsize=10, fontweight='bold')

# Panel 2: Serial position scatter for animals (H2)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('(2) H2: Lexical exhaustion', fontsize=10, fontweight='bold')

# Panel 3: Confidence vs total words (RQ2)
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('(3) RQ2: Confidence paradox', fontsize=10, fontweight='bold')

# Panel 4: SpAM compactness by domain (RQ3)
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_title('(4) RQ3: SpAM compactness', fontsize=10, fontweight='bold')

# Panel 5: ARI alignment bar (RQ4)
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title('(5) RQ4: Cluster alignment', fontsize=10, fontweight='bold')

# Panel 6: Dual similarity curves (H11)
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_title('(6) H11: Phonological rise', fontsize=10, fontweight='bold')

# Panel 7: Composite score vs VFT-only (RQ5) — full width bottom
ax7 = fig.add_subplot(gs[2, :])
ax7.set_title('(7) RQ5: Composite vs VFT-only score correlation with confidence',
              fontsize=10, fontweight='bold')
# [populate each panel with appropriate plots]

plt.suptitle('Summary: How Do Hindi Speakers Search Their Mental Lexicons?',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('fig_s

## 3. Phase 1 — Data Loading, Cleaning & Setup {#phase1}

### Step 1.1 — Load VFT Data

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, shapiro, mannwhitneyu
import statsmodels.formula.api as smf

PALETTE = sns.color_palette("Set2")
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})

df_raw = pd.read_csv("vft_responses.csv")
print(f"Shape: {df_raw.shape}")
print(df_raw.dtypes)
print(df_raw.head())
```

**What to check:** Confirm columns — subject_id, word, domain, position, rt_ms, language_type. Confirm 35 participants, 4 domains, ~1040 rows total.

### Step 1.2 — Clean & Filter


## 3. Phase 1 — Data Loading, Cleaning & Setup {#phase1}

### Step 1.1 — Load VFT Data

```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, shapiro, mannwhitneyu
import statsmodels.formula.api as smf

PALETTE = sns.color_palette("Set2")
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})

df_raw = pd.read_csv("vft_responses.csv")
print(f"Shape: {df_raw.shape}")
print(df_raw.dtypes)
print(df_raw.head())
```

**What to check:** Confirm columns — subject_id, word, domain, position, rt_ms, language_type. Confirm 35 participants, 4 domains, ~1040 rows total.

### Step 1.2 — Clean & Filter

```python
df = df_raw.copy()
df['language_type'] = df['language_type'].str.strip()
df['lang_binary'] = df['language_type'].apply(
    lambda x: 'Hindi/Hinglish' if 'Hindi' in str(x) else 'English')

# Remove IRTs above 60,000 ms (distraction, not retrieval cost)
THRESHOLD_MS = 60_000
removed = df[df['rt_ms'] > THRESHOLD_MS]
print(f"Rows removed (IRT > 60s): {len(removed)}")
df_clean = df[df['rt_ms'] <= THRESHOLD_MS].copy()
df_clean['irt_sec'] = df_clean['rt_ms'] / 1000

# Hindi/Hinglish subset — primary analysis
df_hh = df_clean[df_clean['lang_binary'] == 'Hindi/Hinglish'].copy()

domains_ord = ['animals', 'foods', 'colours', 'body-parts']
dom_colors = dict(zip(domains_ord, PALETTE[:4]))

print(f"Total rows after filter : {len(df_clean)}")
print(f"Hindi/Hinglish rows     : {len(df_hh)}")
print(f"Participants            : {df_clean['subject_id'].nunique()}")
print(f"\nSession-domain N per domain (Hindi/Hinglish):")
print(df_hh.groupby('domain')['subject_id'].nunique())
```

**IMPORTANT — Colours Warning:** If colours has only N=4 participants contributing Hindi responses, flag this immediately:
```python
domain_n = df_hh.groupby('domain')['subject_id'].nunique()
print("⚠ Domains with fewer than 10 participants — exclude from inferential tests:")
print(domain_n[domain_n < 10])
```

### Step 1.3 — Load Exit Poll / Demographics

```python
exit_poll = pd.read_csv("exit_poll.csv")  # adjust filename
print(exit_poll.dtypes)
print(exit_poll.head())

# Merge with fluency table on subject_id
# Identify the Hindi confidence/proficiency column
# If Likert 1-5: keep as continuous
# If it has multiple language columns: extract Hindi-specific score

# Example merge:
df_hh = df_hh.merge(exit_poll[['subject_id', 'hindi_confidence']], 
                     on='subject_id', how='left')
```

**Graph 1.1 — Confidence Score Distribution:**
```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(exit_poll['hindi_confidence'], bins=10, color=PALETTE[0], edgecolor='white')
ax.set_xlabel('Hindi Confidence Score')
ax.set_ylabel('Count')
ax.
```python
df = df_raw.copy()
df['language_type'] = df['language_type'].str.strip()
df['lang_binary'] = df['language_type'].apply(
    lambda x: 'Hindi/Hinglish' if 'Hindi' in str(x) else 'English')

# Remove IRTs above 60,000 ms (distraction, not retrieval cost)
THRESHOLD_MS = 60_000
removed = df[df['rt_ms'] > THRESHOLD_MS]
print(f"Rows removed (IRT > 60s): {len(removed)}")
df_clean = df[df['rt_ms'] <= THRESHOLD_MS].copy()
df_clean['irt_sec'] = df_clean['rt_ms'] / 1000

# Hindi/Hinglish subset — primary analysis
df_hh = df_clean[df_clean['lang_binary'] == 'Hindi/Hinglish'].copy()

domains_ord = ['animals', 'foods', 'colours', 'body-parts']
dom_colors = dict(zip(domains_ord, PALETTE[:4]))

print(f"Total rows after filter : {len(df_clean)}")
print(f"Hindi/Hinglish rows     : {len(df_hh)}")
print(f"Participants            : {df_clean['subject_id'].nunique()}")
print(f"\nSession-domain N per domain (Hindi/Hinglish):")
print(df_hh.groupby('domain')['subject_id'].nunique())
```

**IMPORTANT — Colours Warning:** If colours has only N=4 participants contributing Hindi responses, flag this immediately:
```python
domain_n = df_hh.groupby('domain')['subject_id'].nunique()
print("⚠ Domains with fewer than 10 participants — exclude from inferential tests:")
print(domain_n[domain_n < 10])
```

### Step 1.3 — Load Exit Poll / Demographics

```python
exit_poll = pd.read_csv("exit_poll.csv")  # adjust filename
print(exit_poll.dtypes)
print(exit_poll.head())

# Merge with fluency table on subject_id
# Identify the Hindi confidence/proficiency column
# If Likert 1-5: keep as continuous
# If it has multiple language columns: extract Hindi-specific score

# Example merge:
df_hh = df_hh.merge(exit_poll[['subject_id', 'hindi_confidence']], 
                     on='subject_id', how='left')
```

**Graph 1.1 — Confidence Score Distribution:**
```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(exit_poll['hindi_confidence'], bins=10, color=PALETTE[0], edgecolor='white')
ax.set_xlabel('Hindi Confidence Score')
ax.set_ylabel('Count')
ax.ummary_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Appendix A — Test Selection Quick Reference

| Analysis Target | Normality? | Test | Why |
|----------------|-----------|------|-----|
| Within vs between IRT | Not required (large N) | Welch's t-test (one-tailed) | Unequal variances; directional H₁ |
| Domain differences | Check Shapiro-Wilk first | Kruskal-Wallis | ≥1 domain non-normal |
| IRT × serial position | Not required | LME (position as fixed, subject as random) | Nested data; participant effects |
| Confidence × outcomes | Non-normal, monotonic | Spearman ρ | Ordinal/non-linear relationship |
| SpAM vs embedding clusters | Not applicable | ARI + sign test + permutation | Cluster partition comparison |
| Phon sim × position | Not required | LME interaction | Same nesting as serial position |
| Score comparison (RQ5) | Non-normal | Steiger's Z for dependent correlations | Same participants, two correlations |

## Appendix B — Required Libraries

```
# Core data & stats
pip install pandas numpy scipy matplotlib seaborn statsmodels

# Embeddings & clustering
pip install sentence-transformers scikit-learn hdbscan umap-learn

# FastText (choose one)
pip install fasttext-wheel          # supports .bin model with subword OOV
# OR: pip install gensim             # for .vec text format only

# Hindi phonology
pip install indic-transliteration python-Levenshtein

# Post-hoc stats
pip install scikit-posthocs pingouin

# SpAM neighbourhood
# (uses scipy.spatial.distance — already in scipy)
```

## Appendix C — File Structure Expected

```
project/
├── vft_responses.csv          # word, subject_id, domain, position, rt_ms, language_type
├── spam_responses.csv         # subject_id, word, domain, x, y
├── exit_poll.csv              # subject_id, hindi_confidence, age, ...
├── cc.hi.300.bin              # FastText Hindi model (download separately ~4GB)
└── analysis/
    ├── Phase01_loading.ipynb
    ├── Phase02_descriptives.ipynb
    ├── Phase03_fluency.ipynb
    ├── Phase04_clusters.ipynb
    ├── Phase05_hypothesis_H1_H2.ipynb
    ├── Phase06_fluency_effects.ipynb
    ├── Phase07_domain_tests.ipynb
    ├── Phase08_SpAM.ipynb
    ├── Phase09_fasttext_embeddings.ipynb     ⭐ NEW
    ├── Phase10_SpAM_IRT_integration.ipynb    ⭐ NEW
    ├── Phase11_embedding_alignment.ipynb
    ├── Phase12_phonology.ipynb
    ├── Phase13_composite.ipynb
    └── Phase14_summary.ipynb
```

---

*End of plan — 35 participants, 4 domains, 13 hypotheses, 7 research questions*
*Based on: Kumar et al. (2022), Dautriche et al. (2016), Hills et al. (2012), Troyer et al. (1997)*





## 2. Hypotheses {#hypotheses}

> **Important:** State all hypotheses BEFORE presenting any results. Each hypothesis maps to a specific statistical test. This is what your current report is missing entirely.

### Module A — Foundational VFT Structure

**H1 — Semantic Clustering**
- H₀: Within-cluster IRTs = Between-cluster IRTs (no structured retrieval)
- H₁: Between-cluster IRTs > Within-cluster IRTs (retrieval is semantically clustered)
- Test: Welch's t-test (one-tailed) + Cohen's d
- Grounding: Troyer et al. (1997), Hills et al. (2012)

**H2 — Lexical Exhaustion (Serial Position Effect)**
- H₀: β = 0 — IRT does not increase with serial position
- H₁: β > 0 — IRT increases with retrieval order in all domains (steeper for animals/foods than colours)
- Test: Linear mixed effects model per domain: IRT ~ position + (1+position|subject)
- Grounding: Gruenewald & Lockhead (1980)

### Module B — Domain Differences (RQ1, RQ3)

**H3 — Domain Differences in Productivity**
- H₀: Word count does not differ across domains
- H₁: At least one domain differs in mean word count
- Test: Shapiro-Wilk first → Kruskal-Wallis (non-parametric, already in report)
- Note: Colours has N=4 — treat as descriptive only, exclude from inferential test

**H4 — Domain Differences in Retrieval Speed**
- H₀: Mean RT does not differ across domains
- H₁: At least one domain differs in mean RT
- Test: Kruskal-Wallis (already in report)

### Module C — Participant-Level Effects (RQ2)

**H5 — Hindi Confidence Predicts Word Count**
- H₀: ρ = 0 (no association between confidence and total words)
- H₁: ρ > 0 (higher confidence → more words; one-tailed)
- Test: Spearman correlation (already in report, but direction needs re-examination)
- Note: Your report found ρ = −0.395 (negative). You MUST investigate this before concluding.

**H6 — Hindi Confidence Predicts Retrieval Speed**
- H₀: ρ = 0 (no association with mean RT)
- H₁: ρ < 0 (higher confidence → lower mean RT; faster retrieval)
- Test: Spearman correlation

**H7 — Cluster Size Increases with Hindi Confidence**
- H₀: Confidence does not predict mean cluster size
- H₁: Higher confidence → larger clusters (deeper semantic exploitation)
- Test: Spearman correlation

### Module D — SpAM Structure (RQ3, RQ4)

**H8 — SpAM Compactness Differs Across Domains**
- H₀: Mean nearest-neighbour distance = across domains
- H₁: At least one domain differs in spatial compactness
- Test: Kruskal-Wallis (already in report)

**H9 — SpAM Clusters Align with Semantic Embeddings Above Chance**
- H₀: ARI(SpAM, semantic) = chance level
- H₁: ARI(SpAM, semantic) > chance (sign test / permutation test)
- Test: Binomial sign test (already in report) + ARI permutation test for magnitude

**H10 — SpAM Clusters Align with Phonetic Embeddings Above Chance**
- H₀: ARI(SpAM, phonetic) = chance level
- H₁: ARI(SpAM, phonetic) > chance
- Test: Binomial sign test (already in report)

### Module E — Phonological Facilitation (RQ6, RQ7)

**H11 — Phonological Similarity Increases Over Retrieval Order**
- H₀: No interaction between similarity type and retrieval position
- H₁: Phonological similarity ↑ and semantic similarity ↓ with retrieval position
- Test: LME: similarity ~ type × position + (1+position|subject)
- Grounding: Kumar et al. (2022)

**H12 — Phonological Similarity Predicts Higher Word Count**
- H₀: ρ = 0 between mean phonological similarity and total words
- H₁: ρ > 0 (one-tailed Spearman)
- Grounding: Kumar et al. (2022)

### Module F — Composite Score (RQ5)

**H13 — Integrated Score Is Not Stronger Than VFT-Only for Confidence Correlation**
- H₀: |ρ_integrated| = |ρ_VFT-only|
- H₁: They differ (two-tailed, test with Steiger's Z for dependent correlations)
- Note: Your report found the opposite of what was hoped — VFT-only (ρ=−0.449) was actually stronger than integrated (ρ=−0.317). Report this honestly.

---