
# ============================================================
# HINDI VFT + SpAM COMPLETE ANALYSIS — PART 1
# Transformer-based semantic clustering + full statistics
# Python 3.14  |  All analyses, plots, interpretations
# ============================================================

import warnings, re, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import spearmanr, pearsonr, wilcoxon, kruskal
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sentence_transformers import SentenceTransformer

IMGDIR = "images"
os.makedirs(IMGDIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────
DOMAIN_COLORS = {
    "animals":    "#4E79A7",
    "foods":      "#F28E2B",
    "colours":    "#E15759",
    "body-parts": "#76B7B2",
}
DOMAINS = ["animals", "foods", "colours", "body-parts"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

print("✓ Imports OK")

# ============================================================
# § 1  LOAD DATA
# ============================================================

MERGED_PATH = "merged_vft_spam_responses.csv"
if not os.path.exists(MERGED_PATH):
    raise FileNotFoundError(f"Required input file not found: {MERGED_PATH}")

df_all = pd.read_csv(MERGED_PATH, encoding="utf-8")
required_vft_cols = {"subject_id", "session_id", "domain", "word", "rt_ms", "position", "language_type"}
missing_vft_cols = sorted(required_vft_cols - set(df_all.columns))
if missing_vft_cols:
    raise ValueError(f"Missing required columns in {MERGED_PATH}: {missing_vft_cols}")

# Standardise domain names and language labels
df_all["domain"] = df_all["domain"].str.strip().str.lower()
df_all["language_type"] = df_all["language_type"].astype(str).str.strip()
df_all["subject_id"] = df_all["subject_id"].astype(str)
df_all["session_id"] = df_all["session_id"].astype(str)
df_all["rt_ms"] = pd.to_numeric(df_all["rt_ms"], errors="coerce")
df_all["position"] = pd.to_numeric(df_all["position"], errors="coerce")

# For language-mix comparison, use the same merged source.
df_lang_all = df_all[["domain", "language_type"]].copy()

# Compute per-subject hi_fluency composite
fl_cols = [c for c in df_all.columns if c in ("Hi_Read","Hi_Write","hi_confidence")]
for c in fl_cols:
    df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
if fl_cols:
    df_all["hi_fluency"] = df_all[fl_cols].mean(axis=1)
else:
    df_all["hi_fluency"] = np.nan

# Keep all-language summary for EH3/domain language comparison.
lang_domain_cmp = (
    df_lang_all.groupby(["domain", "language_type"]).size().unstack(fill_value=0)
)
if "Hindi/Hinglish" not in lang_domain_cmp.columns:
    lang_domain_cmp["Hindi/Hinglish"] = 0
if "English" not in lang_domain_cmp.columns:
    lang_domain_cmp["English"] = 0
lang_domain_cmp["total_responses"] = lang_domain_cmp.sum(axis=1)
lang_domain_cmp["pct_hindi"] = np.where(
    lang_domain_cmp["total_responses"] > 0,
    (lang_domain_cmp["Hindi/Hinglish"] / lang_domain_cmp["total_responses"]) * 100,
    np.nan,
)
lang_domain_cmp["pct_english"] = np.where(
    lang_domain_cmp["total_responses"] > 0,
    (lang_domain_cmp["English"] / lang_domain_cmp["total_responses"]) * 100,
    np.nan,
)

# Core VFT analyses are Hindi-only.
df = df_all[df_all["language_type"].str.lower().eq("hindi/hinglish")].copy()

if "language_type" in df.columns:
    langs = set(df["language_type"].astype(str).str.lower().unique())
    if not langs.issubset({"hindi/hinglish"}):
        raise ValueError("Core analysis dataset is not Hindi-only.")

print(f"✓ Loaded merged source (all responses): {len(df_all)} rows")
print(f"✓ Hindi-only subset for core analysis: {len(df)} rows")
print(f"✓ Language-mix source rows: {len(df_lang_all)}")
print("✓ All inferential tests will run on Hindi/Hinglish rows only")
print(df["domain"].value_counts())

# ============================================================
# § 2  EXTRACT SpAM COORDINATES FROM MERGED SOURCE
# ============================================================

print("\n── Reading SpAM coordinates from merged source ──")
required_spam_cols = {"subject_id", "session_id", "domain", "word", "x", "y"}
if required_spam_cols.issubset(set(df_all.columns)):
    spam_df = df_all[list(required_spam_cols)].copy()
    spam_df["domain"] = spam_df["domain"].astype(str).str.lower().str.strip()
    spam_df["word"] = spam_df["word"].astype(str).str.strip()
    spam_df["x"] = pd.to_numeric(spam_df["x"], errors="coerce")
    spam_df["y"] = pd.to_numeric(spam_df["y"], errors="coerce")
    spam_df = spam_df.dropna(subset=["x", "y"])
else:
    spam_df = pd.DataFrame(columns=["subject_id", "session_id", "domain", "word", "x", "y"])
print(f"SpAM records extracted: {len(spam_df)}")

SPAM_AVAILABLE = len(spam_df) > 10
print(f"SpAM data available: {SPAM_AVAILABLE}")

# ============================================================
# § 3  TRANSFORMER EMBEDDINGS (multilingual MiniLM)
# ============================================================

print("\n── Loading transformer model ──")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)
print(f"✓ Model loaded: {MODEL_NAME}")

# Get all unique words per domain
domain_words = {}
for dom in DOMAINS:
    words = df[df["domain"] == dom]["word"].dropna().unique().tolist()
    domain_words[dom] = words
    print(f"  {dom}: {len(words)} unique words")

# Compute embeddings
print("── Computing embeddings ──")
domain_embeddings = {}
for dom, words in domain_words.items():
    if len(words) == 0:
        continue
    embs = model.encode(words, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    domain_embeddings[dom] = {"words": words, "embeddings": embs}
    print(f"  {dom}: {embs.shape}")

print("✓ Embeddings computed")

# ============================================================
# § 4  EMBEDDING-BASED SEMANTIC CLUSTERING
# ============================================================

print("\n── Embedding-based hierarchical clustering ──")

domain_clusters = {}   # word → cluster_label per domain
domain_sim_matrices = {}
OPTIMAL_K = {"animals": 5, "foods": 6, "colours": 3, "body-parts": 4}

for dom in DOMAINS:
    if dom not in domain_embeddings:
        continue
    words = domain_embeddings[dom]["words"]
    embs  = domain_embeddings[dom]["embeddings"]
    
    if len(words) < 4:
        domain_clusters[dom] = {w: 0 for w in words}
        continue
    
    # Cosine similarity → distance
    sim_matrix = embs @ embs.T
    np.fill_diagonal(sim_matrix, 1.0)
    dist_matrix = 1.0 - np.clip(sim_matrix, -1, 1)
    domain_sim_matrices[dom] = (words, sim_matrix)
    
    # Hierarchical clustering (Ward)
    condensed = pdist(embs, metric="cosine")
    Z = linkage(condensed, method="ward")
    
    # Silhouette-based optimal k
    k_range = range(2, min(len(words), 9))
    best_k, best_sil = OPTIMAL_K.get(dom, 4), -1
    for k in k_range:
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        try:
            sil = silhouette_score(embs, labels, metric="cosine")
            if sil > best_sil:
                best_sil, best_k = sil, k
        except Exception:
            pass
    
    labels = fcluster(Z, best_k, criterion="maxclust")
    domain_clusters[dom] = dict(zip(words, labels))
    print(f"  {dom}: k={best_k}, silhouette={best_sil:.3f}")

print("✓ Embedding clusters assigned")

# Map clusters back to df
df["emb_cluster"] = df.apply(
    lambda r: domain_clusters.get(r["domain"], {}).get(r["word"], 0), axis=1
)

# ============================================================
# § 5  WITHIN vs BETWEEN CLUSTER IRTs (using EMBEDDING clusters)
# ============================================================

print("\n── §5 Within vs Between Cluster IRTs ──")

# For each participant+domain, sort by position, then flag switch when cluster changes
df = df.sort_values(["subject_id", "domain", "position"]).reset_index(drop=True)

# Compute IRT values (already in rt_ms column)
df["log_irt"] = np.log(df["rt_ms"].clip(lower=100))

# Determine within vs between using embedding cluster transitions
within_irts, between_irts = [], []
per_subj = []

for (sid, dom), grp in df.groupby(["subject_id", "domain"]):
    grp = grp.sort_values("position").reset_index(drop=True)
    clusters = grp["emb_cluster"].values
    irts     = grp["rt_ms"].values
    
    w_list, b_list = [], []
    for i in range(1, len(clusters)):
        if np.isnan(irts[i]) or irts[i] <= 0:
            continue
        if clusters[i] == clusters[i-1]:
            w_list.append(irts[i])
            within_irts.append(irts[i])
        else:
            b_list.append(irts[i])
            between_irts.append(irts[i])
    
    if w_list and b_list:
        per_subj.append({
            "subject_id": sid, "domain": dom,
            "mean_within":  np.mean(w_list),
            "mean_between": np.mean(b_list),
            "n_within":  len(w_list),
            "n_between": len(b_list),
        })

per_subj_df = pd.DataFrame(per_subj)
subj_means  = per_subj_df.groupby("subject_id")[["mean_within","mean_between"]].mean().reset_index()

w = subj_means["mean_within"].values
b = subj_means["mean_between"].values

# Tests
ttest_res = pg.ttest(w, b, paired=True, alternative="less")
# Normalise column names (pingouin ≥0.5 uses p_val not p-val)
ttest_res.columns = [c.replace('-','_') for c in ttest_res.columns]
wilcoxon_res = wilcoxon(w, b, alternative="less")
d_cohen = float(ttest_res['cohen_d'].values[0]) if 'cohen_d' in ttest_res.columns else pg.compute_effsize(b, w, eftype='cohen')

print(f"  N participants with both w+b: {len(subj_means)}")
print(f"  Within  IRT: M={w.mean():.0f} ms, SD={w.std():.0f}")
print(f"  Between IRT: M={b.mean():.0f} ms, SD={b.std():.0f}")
print(f"  Paired t: t={ttest_res['T'].values[0]:.2f}, p={ttest_res['p_val'].values[0]:.4f}, d={d_cohen:.2f}")
print(f"  Wilcoxon: W={wilcoxon_res.statistic:.1f}, p={wilcoxon_res.pvalue:.4f}")

# Mixed-effects model
df2 = per_subj_df.melt(id_vars=["subject_id","domain"],
                        value_vars=["mean_within","mean_between"],
                        var_name="irt_type", value_name="irt")
df2["is_between"] = (df2["irt_type"] == "mean_between").astype(int)
try:
    lme_rq1 = smf.mixedlm("irt ~ is_between + domain", data=df2,
                           groups=df2["subject_id"]).fit(reml=True)
    print(f"  LME is_between coef={lme_rq1.params['is_between']:.1f}, p={lme_rq1.pvalues['is_between']:.4f}")
except Exception as e:
    print(f"  LME skipped: {e}")

# ── Figure RQ1 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: paired violin
pairs = pd.DataFrame({"Within": w, "Between": b})
ax = axes[0]
vp = ax.violinplot([w, b], positions=[1, 2], widths=0.6,
                    showmedians=True, showextrema=False)
for pc, col in zip(vp["bodies"], ["#4E79A7", "#E15759"]):
    pc.set_facecolor(col); pc.set_alpha(0.7)
for xi_w, xi_b in zip(w, b):
    ax.plot([1, 2], [xi_w, xi_b], "gray", alpha=0.3, lw=0.8)
ax.scatter(np.ones(len(w)), w, color="#4E79A7", zorder=3, s=25, alpha=0.8)
ax.scatter(np.full(len(b), 2), b, color="#E15759", zorder=3, s=25, alpha=0.8)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Within-cluster", "Between-cluster"])
ax.set_ylabel("Mean IRT (ms)")
ax.set_title("RQ1 — Within vs Between Cluster IRTs\n(embedding-based clusters)")
ax.legend(
    handles=[
        mpatches.Patch(color="#4E79A7", label="Within-cluster"),
        mpatches.Patch(color="#E15759", label="Between-cluster"),
    ],
    loc="upper left",
    fontsize=9,
)
p_val = ttest_res['p_val'].values[0]
pstr = f"p < .001" if p_val < 0.001 else f"p = {p_val:.3f}"
ax.annotate(f"t = {ttest_res['T'].values[0]:.2f}, {pstr}\nd = {abs(d_cohen):.2f}",
            xy=(1.5, max(b)*0.85), ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

# Right: bar by domain
dom_df = per_subj_df.groupby("domain")[["mean_within","mean_between"]].mean().reset_index()
x = np.arange(len(dom_df))
ax2 = axes[1]
bars1 = ax2.bar(x - 0.2, dom_df["mean_within"], 0.35, label="Within", color="#4E79A7", alpha=0.85)
bars2 = ax2.bar(x + 0.2, dom_df["mean_between"], 0.35, label="Between", color="#E15759", alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels([d.capitalize() for d in dom_df["domain"]], rotation=20)
ax2.set_ylabel("Mean IRT (ms)"); ax2.set_title("Within vs Between IRTs by Domain")
ax2.legend()
fig.suptitle("Clustering-and-Switching Model — Hindi VFT\n(Transformer Embedding Clusters)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/rq1_within_between_emb.png")
plt.close()
print(f"  ✓ Saved rq1_within_between_emb.png")

# ============================================================
# § 6  RQ2: CLUSTER METRICS → FLUENCY SCORE
# ============================================================

print("\n── §6 Cluster Metrics → Fluency ──")

# Per-participant fluency + cluster metrics
subj_stats = []
for sid, grp in df.groupby("subject_id"):
    total_words = len(grp)
    mean_irt    = grp["rt_ms"].mean()
    # embedding-cluster-based metrics per domain, then average
    by_dom = []
    for dom, dgrp in grp.groupby("domain"):
        dgrp = dgrp.sort_values("position").reset_index(drop=True)
        labels = dgrp["emb_cluster"].values
        n = len(labels)
        switches = sum(labels[i] != labels[i-1] for i in range(1, n)) if n > 1 else 0
        # cluster sizes = run-lengths of same cluster
        sizes = []
        cur_size = 1
        for i in range(1, n):
            if labels[i] == labels[i-1]:
                cur_size += 1
            else:
                sizes.append(cur_size)
                cur_size = 1
        sizes.append(cur_size)
        by_dom.append({"domain": dom, "cluster_size": np.mean(sizes),
                        "n_switches": switches, "n_clusters": len(sizes)})
    dom_df2 = pd.DataFrame(by_dom)
    hi_flu = grp["hi_fluency"].mean()
    hi_conf = grp["hi_confidence"].mean() if "hi_confidence" in grp.columns else np.nan
    subj_stats.append({
        "subject_id": sid,
        "total_words": total_words,
        "mean_irt": mean_irt,
        "mean_cluster_size": dom_df2["cluster_size"].mean(),
        "mean_switches": dom_df2["n_switches"].mean(),
        "mean_n_clusters": dom_df2["n_clusters"].mean(),
        "hi_fluency": hi_flu,
        "hi_confidence": hi_conf,
        "language_count": grp["language_count"].iloc[0] if "language_count" in grp.columns else np.nan,
        "age": grp["age"].iloc[0] if "age" in grp.columns else np.nan,
    })

subj_df = pd.DataFrame(subj_stats).dropna(subset=["mean_cluster_size"])
for c in ["total_words", "mean_irt", "mean_cluster_size", "mean_switches", "mean_n_clusters", "hi_fluency", "hi_confidence", "language_count", "age"]:
    if c in subj_df.columns:
        subj_df[c] = pd.to_numeric(subj_df[c], errors="coerce")

# Pearson + Spearman
r_p, p_p = pearsonr(subj_df["mean_cluster_size"], subj_df["total_words"])
r_s, p_s = spearmanr(subj_df["mean_cluster_size"], subj_df["total_words"])
print(f"  Pearson r = {r_p:.3f}, p = {p_p:.4f}")
print(f"  Spearman ρ = {r_s:.3f}, p = {p_s:.4f}")

# OLS regression
X = sm.add_constant(subj_df["mean_cluster_size"])
ols_rq2 = sm.OLS(subj_df["total_words"], X).fit()
print(f"  OLS: β={ols_rq2.params.iloc[1]:.3f}, R²={ols_rq2.rsquared:.3f}, p={ols_rq2.pvalues.iloc[1]:.4f}")

# Multiple regression with fluency
valid_fl = subj_df.dropna(subset=["hi_fluency","language_count"])
if len(valid_fl) > 10:
    for c in ["mean_cluster_size", "hi_fluency", "language_count", "total_words"]:
        valid_fl[c] = pd.to_numeric(valid_fl[c], errors="coerce")
    valid_fl = valid_fl.dropna(subset=["mean_cluster_size", "hi_fluency", "language_count", "total_words"])
    X2 = sm.add_constant(valid_fl[["mean_cluster_size","hi_fluency","language_count"]])
    ols_multi = sm.OLS(valid_fl["total_words"], X2).fit()
    print(f"  Multi-OLS R²={ols_multi.rsquared:.3f}")

# ── Figure RQ2 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(subj_df["mean_cluster_size"], subj_df["total_words"],
           alpha=0.75, s=60, color="#4E79A7", edgecolors="white", lw=0.5,
           label="Participants")
m, b_r = np.polyfit(subj_df["mean_cluster_size"], subj_df["total_words"], 1)
xs = np.linspace(subj_df["mean_cluster_size"].min(), subj_df["mean_cluster_size"].max(), 100)
ax.plot(xs, m*xs + b_r, color="#E15759", lw=2, label="Linear fit")
ax.set_xlabel("Mean Cluster Size"); ax.set_ylabel("Total Words")
ax.set_title(f"RQ2 — Cluster Size vs Fluency\nr = {r_p:.2f}, p {'< .001' if p_p < 0.001 else f'= {p_p:.3f}'}")
ax.legend(fontsize=9)

ax2 = axes[1]
metrics = ["mean_cluster_size", "mean_switches", "mean_n_clusters"]
labels_ = ["Cluster Size", "Switch Count", "# Clusters"]
colors_ = ["#4E79A7", "#F28E2B", "#76B7B2"]
corrs, pvals = [], []
for m_ in metrics:
    r_, p_ = pearsonr(subj_df[m_].dropna(), subj_df.loc[subj_df[m_].notna(), "total_words"])
    corrs.append(r_); pvals.append(p_)
bars = ax2.barh(labels_, corrs, color=colors_, alpha=0.85)
ax2.axvline(0, color="black", lw=0.8)
for bar, p_ in zip(bars, pvals):
    sig = "***" if p_ < 0.001 else "**" if p_ < 0.01 else "*" if p_ < 0.05 else "ns"
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             sig, va="center", fontweight="bold")
ax2.set_xlabel("Pearson r with Total Words")
ax2.set_title("Cluster Metric Correlations with Fluency")
ax2.set_xlim(-0.1, max(corrs) + 0.15)
ax2.legend(bars, labels_, loc="lower right", fontsize=8, frameon=True, title="Metrics")
fig.suptitle("RQ2 — Semantic Cluster Structure & Lexical Fluency", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/rq2_cluster_fluency_emb.png")
plt.close()
print(f"  ✓ Saved rq2_cluster_fluency_emb.png")

print("\n✓ Part 1 complete — data, embeddings, clusters, RQ1, RQ2 done")
pd.to_pickle({"df": df, "subj_df": subj_df, "per_subj_df": per_subj_df,
               "df_all": df_all,
               "df_lang_all": df_lang_all,
               "language_domain_comparison": lang_domain_cmp.reset_index(),
               "domain_clusters": domain_clusters, "domain_embeddings": domain_embeddings,
               "domain_sim_matrices": domain_sim_matrices, "spam_df": spam_df,
               "SPAM_AVAILABLE": SPAM_AVAILABLE,
               "rq1": {"t": ttest_res, "wilcoxon": wilcoxon_res, "d": d_cohen},
               "rq2": {"r": r_p, "p": p_p, "ols": ols_rq2}},
              "analysis_state.pkl")
print("✓ State saved to analysis_state.pkl")
