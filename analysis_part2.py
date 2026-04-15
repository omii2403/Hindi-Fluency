
# ============================================================
# HINDI VFT + SpAM COMPLETE ANALYSIS — PART 2
# EH1-EH4, SpAM, RQ3, Hindi Fluency Predictor, Figures
# ============================================================

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import spearmanr, pearsonr, kruskal, mannwhitneyu
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

IMGDIR = "images"
os.makedirs(IMGDIR, exist_ok=True)

DOMAIN_COLORS = {
    "animals":    "#4E79A7",
    "foods":      "#F28E2B",
    "colours":    "#E15759",
    "body-parts": "#76B7B2",
}
DOMAINS = ["animals", "foods", "colours", "body-parts"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

# ── Load state from Part 1 ──────────────────────────────────
state = pd.read_pickle("analysis_state.pkl")
df               = state["df"]
df_all           = state.get("df_all", df.copy())
df_lang_all      = state.get("df_lang_all", None)
subj_df          = state["subj_df"]
per_subj_df      = state["per_subj_df"]
domain_clusters  = state["domain_clusters"]
domain_embeddings= state["domain_embeddings"]
domain_sim_matrices = state["domain_sim_matrices"]
spam_df          = state["spam_df"]
SPAM_AVAILABLE   = state["SPAM_AVAILABLE"]
rq1_stats        = state["rq1"]
rq2_stats        = state["rq2"]
if df_lang_all is None:
    if os.path.exists("merged_vft_spam_responses.csv"):
        df_lang_all = pd.read_csv("merged_vft_spam_responses.csv", encoding="utf-8")
    else:
        df_lang_all = df_all.copy()

if "domain" in df_lang_all.columns:
    df_lang_all["domain"] = df_lang_all["domain"].astype(str).str.strip().str.lower()
if "language_type" in df_lang_all.columns:
    df_lang_all["language_type"] = df_lang_all["language_type"].astype(str).str.strip()

print(f"✓ State loaded — Hindi-only rows: {len(df)}, enriched rows: {len(df_all)}, language-mix rows: {len(df_lang_all)}, SpAM available: {SPAM_AVAILABLE}")
if "language_type" in df.columns:
    langs = set(df["language_type"].astype(str).str.lower().unique())
    if not langs.issubset({"hindi/hinglish"}):
        raise ValueError("Part 2 inferential dataset is not Hindi-only.")
print("✓ All statistical tests in Part 2 use Hindi/Hinglish rows only")

# ============================================================
# § 7  EXPLORATORY HYPOTHESES (EH1–EH4)
# ============================================================

# ── EH1: Domain IRT differences ─────────────────────────────
print("\n── EH1: Domain IRT differences (Kruskal-Wallis) ──")
dom_groups = [df[df["domain"]==d]["rt_ms"].dropna().values for d in DOMAINS]
H, p_kw = kruskal(*dom_groups)
print(f"  Kruskal-Wallis H={H:.3f}, p={p_kw:.4f}")

# Dunn post-hoc (manual pairwise MannWhitney + BH correction)
pairs_eh1 = []
for i, d1 in enumerate(DOMAINS):
    for j, d2 in enumerate(DOMAINS):
        if j <= i: continue
        g1 = df[df["domain"]==d1]["rt_ms"].dropna().values
        g2 = df[df["domain"]==d2]["rt_ms"].dropna().values
        U, p = mannwhitneyu(g1, g2, alternative="two-sided")
        r_eff = 1 - (2*U)/(len(g1)*len(g2))
        pairs_eh1.append({"d1":d1,"d2":d2,"U":U,"p_raw":p,"r":r_eff})
pairs_eh1_df = pd.DataFrame(pairs_eh1)
_, pairs_eh1_df["p_bh"], _, _ = multipletests(pairs_eh1_df["p_raw"], method="fdr_bh")
print(pairs_eh1_df[["d1","d2","U","r","p_bh"]].to_string(index=False))

# Domain descriptives
dom_desc = df.groupby("domain")["rt_ms"].agg(
    N="count", Mean=np.mean, Median=np.median, SD=np.std,
    Skew=lambda x: float(stats.skew(x.dropna()))
).round(1)
print("\nDomain descriptives:\n", dom_desc)

# ── EH2: Serial position / lexical exhaustion ───────────────
print("\n── EH2: Serial Position Effect (Mixed LM) ──")
try:
    lme_sp = smf.mixedlm("rt_ms ~ position * C(domain)",
                          data=df.dropna(subset=["rt_ms","position"]),
                          groups=df.dropna(subset=["rt_ms","position"])["subject_id"]).fit(reml=True)
    print(f"  position coef={lme_sp.params.get('position',np.nan):.2f}, "
          f"p={lme_sp.pvalues.get('position',np.nan):.4f}")
except Exception as e:
    print(f"  LME error: {e}")

# OLS per domain serial slopes
print("  Per-domain OLS slopes (position→IRT):")
for dom in DOMAINS:
    sub = df[df["domain"]==dom].dropna(subset=["rt_ms","position"])
    if len(sub) < 5: continue
    m, b_, r, p_, se = stats.linregress(sub["position"], sub["rt_ms"])
    print(f"    {dom}: β={m:.1f} ms/pos, r²={r**2:.3f}, p={p_:.4f}")

# ── EH3: Descriptive language mix (no inferential test) ──────
print("\n── EH3: Descriptive language mix by domain (no inferential test) ──")
lang_dom = df_lang_all.groupby(["domain","language_type"]).size().unstack(fill_value=0)
if "Hindi/Hinglish" not in lang_dom.columns:
    lang_dom["Hindi/Hinglish"] = 0
if "English" not in lang_dom.columns:
    lang_dom["English"] = 0
lang_dom["total_responses"] = lang_dom.sum(axis=1)
lang_dom["pct_hindi"] = np.where(
    lang_dom["total_responses"] > 0,
    (lang_dom["Hindi/Hinglish"] / lang_dom["total_responses"]) * 100,
    np.nan,
)
lang_dom["pct_english"] = np.where(
    lang_dom["total_responses"] > 0,
    (lang_dom["English"] / lang_dom["total_responses"]) * 100,
    np.nan,
)
language_domain_cmp = lang_dom.reset_index()[
    ["domain", "Hindi/Hinglish", "English", "total_responses", "pct_hindi", "pct_english"]
].sort_values("domain")
print(language_domain_cmp.round(1).to_string(index=False))
language_domain_cmp.to_csv("table_language_domain_comparison.csv", index=False)
print("✓ Saved table_language_domain_comparison.csv")

# ── EH4: Cluster profiles ────────────────────────────────────
print("\n── EH4: Cluster profiles ──")
for m_ in ["mean_cluster_size","mean_switches","mean_n_clusters"]:
    r_, p_ = pearsonr(subj_df[m_].dropna(),
                       subj_df.loc[subj_df[m_].notna(),"total_words"])
    print(f"  {m_}: r={r_:.3f}, p={p_:.4f}")

# ── EH Figures ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# EH1 — Raincloud
ax = axes[0, 0]
for i, dom in enumerate(DOMAINS):
    vals = df[df["domain"]==dom]["rt_ms"].dropna().values / 1000  # to seconds
    parts = ax.violinplot(vals, positions=[i], widths=0.6, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(DOMAIN_COLORS[dom]); pc.set_alpha(0.6)
    ax.boxplot(vals, positions=[i], widths=0.18, patch_artist=True,
               boxprops=dict(facecolor=DOMAIN_COLORS[dom], alpha=0.8),
               medianprops=dict(color="white", lw=2),
               whiskerprops=dict(lw=1), capprops=dict(lw=1),
               flierprops=dict(marker=".", ms=3, alpha=0.3))
ax.set_xticks(range(4)); ax.set_xticklabels([d.capitalize() for d in DOMAINS])
ax.set_ylabel("IRT (seconds)")
ax.set_title(f"EH1 — IRT by Domain\nKruskal-Wallis H={H:.1f}, p {'< .001' if p_kw<0.001 else f'= {p_kw:.3f}'}")
ax.legend(
    handles=[
        mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize(), alpha=0.7)
        for d in DOMAINS
    ],
    fontsize=8,
    loc="upper right",
    title="Domains",
)

# EH2 — Serial position slopes
ax2 = axes[0, 1]
for dom in DOMAINS:
    sub = df[df["domain"]==dom].dropna(subset=["rt_ms","position"])
    grp_pos = sub.groupby("position")["rt_ms"].mean()
    ax2.plot(grp_pos.index, grp_pos.values/1000, marker="o", ms=4,
             label=dom.capitalize(), color=DOMAIN_COLORS[dom], alpha=0.85)
    m, b_, _, _, _ = stats.linregress(grp_pos.index, grp_pos.values)
    xs = np.array([grp_pos.index.min(), grp_pos.index.max()])
    ax2.plot(xs, (m*xs+b_)/1000, "--", color=DOMAIN_COLORS[dom], alpha=0.5, lw=1.5)
ax2.set_xlabel("Serial Position"); ax2.set_ylabel("Mean IRT (s)")
ax2.set_title("EH2 — Serial Position Effect (Lexical Exhaustion)")
ax2.legend(fontsize=9)

# EH3 — Code switching
ax3 = axes[1, 0]
lang_dom2 = df_lang_all.groupby(["domain","language_type"]).size().unstack(fill_value=0).reset_index()
x3 = np.arange(4)
hindi_pct = []
eng_pct = []
for dom in DOMAINS:
    row = lang_dom2[lang_dom2["domain"]==dom]
    h = row.get("Hindi/Hinglish", pd.Series([0])).values[0]
    e = row.get("English", pd.Series([0])).values[0]
    tot = h + e if h + e > 0 else 1
    hindi_pct.append(h/tot*100); eng_pct.append(e/tot*100)
bars_h = ax3.bar(x3, hindi_pct, 0.5, label="Hindi/Hinglish", color="#4E79A7", alpha=0.85)
bars_e = ax3.bar(x3, eng_pct, 0.5, bottom=hindi_pct, label="English", color="#E15759", alpha=0.85)
ax3.axhline(50, color="white", lw=1.5, ls="--")
ax3.set_xticks(x3); ax3.set_xticklabels([d.capitalize() for d in DOMAINS])
ax3.set_ylabel("% of responses"); ax3.set_title("EH3 — Code-Switching by Domain")
ax3.legend(fontsize=9); ax3.set_ylim(0, 105)

# EH4 — Cluster metrics scatter
ax4 = axes[1, 1]
sc1 = ax4.scatter(subj_df["mean_cluster_size"], subj_df["total_words"],
                  c=subj_df["mean_n_clusters"], cmap="viridis", s=60,
                  alpha=0.8, edgecolors="white", lw=0.5)
plt.colorbar(sc1, ax=ax4, label="# Clusters")
ax4.set_xlabel("Mean Cluster Size"); ax4.set_ylabel("Total Words")
ax4.set_title("EH4 — Cluster Profile Predictors of Fluency")
r1, p1 = pearsonr(subj_df["mean_cluster_size"].dropna(),
                   subj_df.loc[subj_df["mean_cluster_size"].notna(),"total_words"])
ax4.text(0.05, 0.95, f"r = {r1:.2f}", transform=ax4.transAxes, va="top", fontsize=10)
ax4.legend(
    handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4E79A7", markersize=8,
               markeredgecolor="white", markeredgewidth=0.5, label="Participant"),
    ],
    fontsize=8,
    loc="lower right",
)

fig.suptitle("Exploratory Hypotheses EH1–EH4 — Hindi VFT Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/eh1_eh4_panel.png")
plt.close()
print(f"✓ Saved eh1_eh4_panel.png")

# ============================================================
# § 8  EMBEDDING VISUALISATION FIGURES
# ============================================================

print("\n── Embedding Visualisation ──")

# ── t-SNE across all domains ─────────────────────────────────
all_words, all_embs, all_doms = [], [], []
for dom in DOMAINS:
    if dom not in domain_embeddings:
        continue
    for w, e in zip(domain_embeddings[dom]["words"], domain_embeddings[dom]["embeddings"]):
        all_words.append(w)
        all_embs.append(e)
        all_doms.append(dom)

all_embs_arr = np.array(all_embs)
perp = min(30, len(all_embs_arr) - 1)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
tsne_coords = tsne.fit_transform(all_embs_arr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for dom in DOMAINS:
    idx = [i for i, d in enumerate(all_doms) if d == dom]
    ax.scatter(tsne_coords[idx, 0], tsne_coords[idx, 1],
               label=dom.capitalize(), color=DOMAIN_COLORS[dom],
               s=40, alpha=0.75, edgecolors="white", lw=0.3)
ax.set_title("t-SNE of Word Embeddings (by Domain)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=9)

# Cosine similarity heatmaps (average across domains)
ax2 = axes[1]
dom_avg_sim = []
dom_labels_ax2 = []
for dom in DOMAINS:
    if dom not in domain_sim_matrices:
        continue
    words_, sim_ = domain_sim_matrices[dom]
    triu_vals = sim_[np.triu_indices_from(sim_, k=1)]
    dom_avg_sim.append(triu_vals)
    dom_labels_ax2.append(dom.capitalize())

vp = ax2.violinplot(dom_avg_sim, positions=range(len(dom_avg_sim)),
                    showmedians=True, showextrema=False)
for pc, dom_label in zip(vp["bodies"], dom_labels_ax2):
    dom_key = dom_label.lower()
    pc.set_facecolor(DOMAIN_COLORS.get(dom_key, "#999999")); pc.set_alpha(0.7)
ax2.set_xticks(range(len(dom_labels_ax2)))
ax2.set_xticklabels(dom_labels_ax2)
ax2.set_ylabel("Pairwise Cosine Similarity")
ax2.set_title("Semantic Neighbourhood Density by Domain")
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.legend(
    handles=[
        mpatches.Patch(color=DOMAIN_COLORS.get(lbl.lower(), "#999999"), label=lbl, alpha=0.7)
        for lbl in dom_labels_ax2
    ],
    fontsize=8,
    loc="best",
    title="Domains",
)

fig.suptitle("Transformer Embedding Space — Hindi VFT Words", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/embedding_tsne_similarity.png")
plt.close()
print(f"✓ Saved embedding_tsne_similarity.png")

# ── Per-domain similarity heatmaps ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, dom in zip(axes.flatten(), DOMAINS):
    if dom not in domain_sim_matrices:
        ax.set_visible(False); continue
    words_, sim_ = domain_sim_matrices[dom]
    # Sort by cluster
    clust_map = domain_clusters.get(dom, {})
    word_clust = [(w, clust_map.get(w,0)) for w in words_]
    word_clust.sort(key=lambda x: x[1])
    sorted_words = [w for w, _ in word_clust]
    idx_order = [words_.index(w) for w in sorted_words if w in words_]
    sim_sorted = sim_[np.ix_(idx_order, idx_order)]
    # Limit display to 40 words max
    n_show = min(40, len(sorted_words))
    sim_show = sim_sorted[:n_show, :n_show]
    words_show = sorted_words[:n_show]
    im = ax.imshow(sim_show, cmap="RdYlBu", vmin=-0.2, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xticks(range(n_show)); ax.set_yticks(range(n_show))
    ax.set_xticklabels(words_show, rotation=90, fontsize=5)
    ax.set_yticklabels(words_show, fontsize=5)
    ax.set_title(f"{dom.capitalize()} — Cosine Similarity (sorted by cluster)")

fig.suptitle("Semantic Similarity Heatmaps (Transformer Embeddings)\nWords sorted by embedding cluster", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/similarity_heatmaps_emb.png")
plt.close()
print(f"✓ Saved similarity_heatmaps_emb.png")

# ── Neighbourhood density → IRT ──────────────────────────────
print("── Neighbourhood density vs IRT ──")
word_density = {}
for dom in DOMAINS:
    if dom not in domain_sim_matrices:
        continue
    words_, sim_ = domain_sim_matrices[dom]
    k = min(3, len(words_)-1)
    for i, w in enumerate(words_):
        sims_i = sim_[i].copy()
        sims_i[i] = -1
        top_k = np.sort(sims_i)[::-1][:k]
        word_density[(dom, w)] = float(np.mean(top_k))

df["neighbourhood_density"] = df.apply(
    lambda r: word_density.get((r["domain"], r["word"]), np.nan), axis=1)

# LME: log_irt ~ neighbourhood_density + position + domain
valid_nd = df.dropna(subset=["neighbourhood_density","rt_ms","position"])
if len(valid_nd) > 50:
    try:
        lme_nd = smf.mixedlm("rt_ms ~ neighbourhood_density + position + C(domain)",
                              data=valid_nd,
                              groups=valid_nd["subject_id"]).fit(reml=True)
        nd_coef = lme_nd.params.get("neighbourhood_density", np.nan)
        nd_p    = lme_nd.pvalues.get("neighbourhood_density", np.nan)
        print(f"  LME neighbourhood_density: β={nd_coef:.1f}, p={nd_p:.4f}")
    except Exception as e:
        print(f"  LME error: {e}")
        nd_coef, nd_p = np.nan, np.nan
else:
    nd_coef, nd_p = np.nan, np.nan

# ============================================================
# § 9  SpAM ANALYSIS (RQ3)
# ============================================================

print(f"\n── §9 SpAM Analysis (available: {SPAM_AVAILABLE}) ──")

# Defaults used in final summary even if SpAM data are missing.
rq3_df = pd.DataFrame()
H_rq4, p_rq4 = np.nan, np.nan
rq4_posthoc_df = pd.DataFrame()
rq4_density_word_df = pd.DataFrame()
rho_rq5, p_rq5 = np.nan, np.nan
lme_rq5_coef, lme_rq5_p = np.nan, np.nan
rq5_pair_df = pd.DataFrame()

if SPAM_AVAILABLE and len(spam_df) > 20:
    spam_df["domain"] = spam_df["domain"].str.strip().str.lower()
    spam_df = spam_df.dropna(subset=["x","y"])
    
    # ── Consensus distance matrices ──────────────────────────
    spam_consensus = {}
    for dom in DOMAINS:
        sdom = spam_df[spam_df["domain"]==dom]
        if len(sdom) < 5: continue
        # Words appearing in ≥3 participants
        word_counts = sdom.groupby("word")["subject_id"].nunique()
        common_words = word_counts[word_counts >= 3].index.tolist()
        if len(common_words) < 3: continue
        
        subj_mats = {}
        for sid, sgrp in sdom[sdom["word"].isin(common_words)].groupby("subject_id"):
            coords = sgrp.set_index("word")[["x","y"]].loc[
                [w for w in common_words if w in sgrp["word"].values]]
            if len(coords) < 3: continue
            d_mat = squareform(pdist(coords.values, metric="euclidean"))
            subj_mats[sid] = (coords.index.tolist(), d_mat)
        
        if len(subj_mats) < 3: continue
        # Average across participants
        all_words_spam = common_words
        avg_mat = np.zeros((len(all_words_spam), len(all_words_spam)), dtype=float)
        count_mat = np.zeros_like(avg_mat, dtype=float)
        for sid, (words_s, d_mat) in subj_mats.items():
            for i, w1 in enumerate(all_words_spam):
                for j, w2 in enumerate(all_words_spam):
                    if w1 in words_s and w2 in words_s:
                        ii, jj = words_s.index(w1), words_s.index(w2)
                        avg_mat[i,j] += d_mat[ii,jj]
                        count_mat[i,j] += 1
        avg_mat = np.divide(
            avg_mat,
            count_mat,
            out=np.full_like(avg_mat, np.nan, dtype=float),
            where=count_mat > 0,
        )
        np.fill_diagonal(avg_mat, 0.0)
        spam_consensus[dom] = {"words": all_words_spam, "dist": avg_mat}
        print(f"  {dom}: {len(all_words_spam)} common words, {len(subj_mats)} participants")
    
    # ── RQ3: SpAM distance → VFT IRT ─────────────────────────
    print("\n── RQ3: SpAM distance → VFT IRT (Spearman) ──")
    rq3_results = []
    all_spam_dist, all_vft_irt = [], []
    
    for dom in DOMAINS:
        if dom not in spam_consensus: continue
        cons = spam_consensus[dom]
        cwords = cons["words"]
        dist_mat = cons["dist"]
        
        dom_df_vft = df[df["domain"]==dom].dropna(subset=["rt_ms"])
        word_mean_irt = dom_df_vft.groupby("word")["rt_ms"].mean().to_dict()
        
        pair_spam, pair_irt = [], []
        for i, w1 in enumerate(cwords):
            for j, w2 in enumerate(cwords):
                if j <= i: continue
                d_spam = dist_mat[i,j]
                irt1 = word_mean_irt.get(w1, np.nan)
                irt2 = word_mean_irt.get(w2, np.nan)
                if np.isnan(d_spam) or np.isnan(irt1) or np.isnan(irt2): continue
                pair_spam.append(d_spam)
                pair_irt.append((irt1 + irt2) / 2)
        
        if len(pair_spam) < 5: continue
        rho, p_rq3 = spearmanr(pair_spam, pair_irt)
        rq3_results.append({"domain": dom, "rho": rho, "p": p_rq3, "n": len(pair_spam)})
        all_spam_dist.extend(pair_spam)
        all_vft_irt.extend(pair_irt)
        print(f"  {dom}: ρ={rho:.3f}, p={p_rq3:.4f}, n={len(pair_spam)} pairs")
    
    # BH correction
    if rq3_results:
        rq3_df = pd.DataFrame(rq3_results)
        _, rq3_df["p_bh"], _, _ = multipletests(rq3_df["p"], method="fdr_bh")
        print("\nRQ3 with BH correction:")
        print(rq3_df[["domain","rho","p","p_bh","n"]].to_string(index=False))
        
        # Overall Spearman
        if all_spam_dist:
            rho_all, p_all = spearmanr(all_spam_dist, all_vft_irt)
            print(f"\n  Overall ρ={rho_all:.3f}, p={p_all:.4f}")
    
    # ── SpAM MDS + Hierarchical clustering figures ───────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, dom in zip(axes.flatten(), DOMAINS):
        if dom not in spam_consensus:
            ax.set_visible(False); continue
        cons = spam_consensus[dom]
        cwords = cons["words"]
        dist_mat = cons["dist"]
        
        n_items = len(cwords)
        if n_items < 4:
            ax.set_visible(False); continue

        # MDS and linkage require finite distances. Fill missing pairs with a large value
        # only for plotting/clustering; inferential analyses already skip NaNs explicitly.
        dist_viz = dist_mat.copy().astype(float)
        if np.isnan(dist_viz).any():
            finite_vals = dist_viz[np.isfinite(dist_viz)]
            fill_val = (np.nanmax(finite_vals) * 1.25) if len(finite_vals) else 1.0
            dist_viz = np.where(np.isfinite(dist_viz), dist_viz, fill_val)
            np.fill_diagonal(dist_viz, 0.0)
        
        # MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        coords_mds = mds.fit_transform(dist_viz)
        
        # Hierarchical clustering on SpAM
        Z_sp = linkage(squareform((dist_viz + dist_viz.T)/2), method="ward")
        k_sp = min(4, n_items-1)
        labels_sp = fcluster(Z_sp, k_sp, criterion="maxclust")
        
        cmap = plt.cm.get_cmap("tab10", k_sp)
        for i, (word, lbl) in enumerate(zip(cwords, labels_sp)):
            ax.scatter(coords_mds[i,0], coords_mds[i,1],
                       color=cmap(lbl-1), s=80, alpha=0.9, edgecolors="white", lw=0.5)
            ax.annotate(word, (coords_mds[i,0], coords_mds[i,1]),
                        fontsize=6, alpha=0.85,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"SpAM MDS — {dom.capitalize()} (k={k_sp} clusters)")
        ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
        cluster_handles = [
            mpatches.Patch(color=cmap(i), label=f"Cluster {i+1}")
            for i in range(k_sp)
        ]
        ax.legend(handles=cluster_handles, fontsize=7, loc="best", title="Clusters")
    
    fig.suptitle("SpAM Spatial Arrangement — Consensus MDS Maps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMGDIR}/spam_mds_consensus.png")
    plt.close()
    print(f"✓ Saved spam_mds_consensus.png")
    
    # RQ3 forest plot
    if rq3_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        rq3_df_plot = rq3_df.copy() if "rq3_df" in locals() else pd.DataFrame(rq3_results)
        y_pos = range(len(rq3_df_plot))
        colors_rq3 = [DOMAIN_COLORS[d] for d in rq3_df_plot["domain"]]
        ax.barh(y_pos, rq3_df_plot["rho"], color=colors_rq3, alpha=0.85, height=0.5)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([d.capitalize() for d in rq3_df_plot["domain"]])
        for i, row in rq3_df_plot.iterrows():
            sig = "***" if row["p_bh"]<0.001 else "**" if row["p_bh"]<0.01 else \
                  "*" if row["p_bh"]<0.05 else "ns"
            ax.text(row["rho"] + 0.01, i, f"ρ={row['rho']:.2f} {sig}", va="center", fontsize=10)
        ax.set_xlabel("Spearman ρ (SpAM distance vs mean VFT IRT)")
        ax.set_title("RQ3 — SpAM Semantic Distance → VFT IRT\n(BH-corrected)")
        ax.legend(
            handles=[
                mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize())
                for d in rq3_df_plot["domain"]
            ],
            fontsize=8,
            loc="lower right",
            title="Domains",
        )
        plt.tight_layout()
        plt.savefig(f"{IMGDIR}/rq3_spam_vft_correlation.png")
        plt.close()
        print(f"✓ Saved rq3_spam_vft_correlation.png")

    # ── RQ4: SpAM compactness differences across domains ─────
    print("\n── RQ4: SpAM neighbourhood compactness across domains ──")
    density_rows = []
    for dom in DOMAINS:
        if dom not in spam_consensus:
            continue
        words_ = spam_consensus[dom]["words"]
        dist_ = spam_consensus[dom]["dist"]
        if len(words_) < 3:
            continue
        for i, w in enumerate(words_):
            row = dist_[i].copy().astype(float)
            row[i] = np.inf
            finite_row = row[np.isfinite(row)]
            if len(finite_row) == 0:
                continue
            nn_dist = float(np.min(finite_row))
            mean_dist = float(np.mean(finite_row))
            density_rows.append({
                "domain": dom,
                "word": w,
                "nn_dist": nn_dist,
                "mean_dist": mean_dist,
            })

    rq4_density_word_df = pd.DataFrame(density_rows)
    if len(rq4_density_word_df) > 8 and rq4_density_word_df["domain"].nunique() >= 2:
        rq4_groups = [rq4_density_word_df[rq4_density_word_df["domain"] == d]["nn_dist"].values
                      for d in DOMAINS if (rq4_density_word_df["domain"] == d).sum() > 2]
        if len(rq4_groups) >= 2:
            H_rq4, p_rq4 = kruskal(*rq4_groups)
            print(f"  Kruskal-Wallis on nearest-neighbour distance: H={H_rq4:.3f}, p={p_rq4:.4f}")

            # Pairwise post-hoc with BH correction
            post_rows = []
            present_domains = [d for d in DOMAINS if (rq4_density_word_df["domain"] == d).sum() > 2]
            for i, d1 in enumerate(present_domains):
                for j, d2 in enumerate(present_domains):
                    if j <= i:
                        continue
                    g1 = rq4_density_word_df[rq4_density_word_df["domain"] == d1]["nn_dist"].values
                    g2 = rq4_density_word_df[rq4_density_word_df["domain"] == d2]["nn_dist"].values
                    U, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")
                    post_rows.append({"d1": d1, "d2": d2, "U": U, "p_raw": p_raw})
            rq4_posthoc_df = pd.DataFrame(post_rows)
            if not rq4_posthoc_df.empty:
                _, rq4_posthoc_df["p_bh"], _, _ = multipletests(rq4_posthoc_df["p_raw"], method="fdr_bh")
                print(rq4_posthoc_df[["d1", "d2", "U", "p_bh"]].to_string(index=False))

            # Plot
            fig, ax = plt.subplots(figsize=(9, 5))
            data_plot = [rq4_density_word_df[rq4_density_word_df["domain"] == d]["nn_dist"].values
                         for d in DOMAINS if (rq4_density_word_df["domain"] == d).sum() > 0]
            dom_plot = [d for d in DOMAINS if (rq4_density_word_df["domain"] == d).sum() > 0]
            vp = ax.violinplot(data_plot, positions=np.arange(len(dom_plot)), widths=0.7,
                               showmedians=True, showextrema=False)
            for body, d in zip(vp["bodies"], dom_plot):
                body.set_facecolor(DOMAIN_COLORS[d])
                body.set_alpha(0.7)
            ax.set_xticks(np.arange(len(dom_plot)))
            ax.set_xticklabels([d.capitalize() for d in dom_plot])
            ax.set_ylabel("Nearest-neighbour SpAM distance")
            ax.set_title("RQ4 — SpAM Compactness by Domain\n"
                         f"Kruskal-Wallis H={H_rq4:.2f}, p={'< .001' if p_rq4 < 0.001 else f'= {p_rq4:.3f}'}")
            ax.legend(
                handles=[
                    mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize(), alpha=0.7)
                    for d in dom_plot
                ],
                fontsize=8,
                loc="best",
                title="Domains",
            )
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{IMGDIR}/rq4_spam_compactness_by_domain.png")
            plt.close()
            print(f"✓ Saved rq4_spam_compactness_by_domain.png")

    # ── RQ5: SpAM dispersion predicts VFT switch cost ────────
    print("\n── RQ5: SpAM dispersion vs VFT switching cost ──")
    spam_subj_rows = []
    for (sid, dom), grp in spam_df.groupby(["subject_id", "domain"]):
        coords = grp[["x", "y"]].dropna().values
        if len(coords) < 3:
            continue
        pdists = pdist(coords, metric="euclidean")
        if len(pdists) == 0:
            continue
        spam_subj_rows.append({
            "subject_id": sid,
            "domain": dom,
            "spam_dispersion": float(np.mean(pdists)),
            "spam_dispersion_median": float(np.median(pdists)),
        })
    spam_subj_df = pd.DataFrame(spam_subj_rows)

    rq5_pair_df = per_subj_df.copy()
    rq5_pair_df["subject_id"] = rq5_pair_df["subject_id"].astype(str)
    if not spam_subj_df.empty:
        spam_subj_df["subject_id"] = spam_subj_df["subject_id"].astype(str)
    rq5_pair_df["switch_cost"] = rq5_pair_df["mean_between"] - rq5_pair_df["mean_within"]
    if not spam_subj_df.empty:
        rq5_pair_df = rq5_pair_df.merge(
            spam_subj_df[["subject_id", "domain", "spam_dispersion"]],
            on=["subject_id", "domain"],
            how="inner"
        )
    rq5_pair_df = rq5_pair_df.dropna(subset=["spam_dispersion", "switch_cost"])

    if len(rq5_pair_df) > 10:
        rho_rq5, p_rq5 = spearmanr(rq5_pair_df["spam_dispersion"], rq5_pair_df["switch_cost"])
        print(f"  Spearman rho={rho_rq5:.3f}, p={p_rq5:.4f}, n={len(rq5_pair_df)}")

        try:
            lme_rq5 = smf.mixedlm(
                "switch_cost ~ spam_dispersion + C(domain)",
                data=rq5_pair_df,
                groups=rq5_pair_df["subject_id"]
            ).fit(reml=True)
            lme_rq5_coef = float(lme_rq5.params.get("spam_dispersion", np.nan))
            lme_rq5_p = float(lme_rq5.pvalues.get("spam_dispersion", np.nan))
            print(f"  MixedLM spam_dispersion coef={lme_rq5_coef:.1f}, p={lme_rq5_p:.4f}")
        except Exception as e:
            print(f"  MixedLM skipped: {e}")

        fig, ax = plt.subplots(figsize=(8, 6))
        for dom in DOMAINS:
            sub = rq5_pair_df[rq5_pair_df["domain"] == dom]
            if len(sub) == 0:
                continue
            ax.scatter(sub["spam_dispersion"], sub["switch_cost"],
                       s=55, alpha=0.75, color=DOMAIN_COLORS[dom],
                       edgecolors="white", lw=0.5, label=dom.capitalize())

        if len(rq5_pair_df) > 2:
            m_rq5, b_rq5, _, _, _ = stats.linregress(rq5_pair_df["spam_dispersion"], rq5_pair_df["switch_cost"])
            xs = np.linspace(rq5_pair_df["spam_dispersion"].min(), rq5_pair_df["spam_dispersion"].max(), 100)
            ax.plot(xs, m_rq5 * xs + b_rq5, color="black", lw=2, alpha=0.8)

        ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel("SpAM dispersion (mean pairwise distance)")
        ax.set_ylabel("VFT switching cost (between - within IRT, ms)")
        ax.set_title("RQ5 — Cross-task Link: SpAM Dispersion vs VFT Switching Cost\n"
                     f"Spearman rho={rho_rq5:.2f}, p={'< .001' if p_rq5 < 0.001 else f'= {p_rq5:.3f}'}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{IMGDIR}/rq5_spam_vft_switchcost.png")
        plt.close()
        print(f"✓ Saved rq5_spam_vft_switchcost.png")

else:
    print("  SpAM data not available — generating simulated RQ3 note figure")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "SpAM coordinate data could not be read\nfrom merged_vft_spam_responses.csv.\n\n"
            "RQ3 (SpAM ↔ VFT cross-task correlation)\nrequires valid x/y coordinate data.",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"))
    ax.set_axis_off()
    ax.set_title("RQ3 — SpAM Analysis: Data Unavailable")
    plt.savefig(f"{IMGDIR}/rq3_spam_vft_correlation.png")
    plt.close()

    # Keep placeholders for new SpAM-based RQs.
    for fig_name, title, msg in [
        ("rq4_spam_compactness_by_domain.png", "RQ4 — SpAM compactness: Data Unavailable",
         "RQ4 needs usable SpAM coordinates across domains."),
        ("rq5_spam_vft_switchcost.png", "RQ5 — SpAM-VFT link: Data Unavailable",
         "RQ5 needs matched SpAM + VFT records per subject and domain."),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"))
        ax.set_axis_off()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"{IMGDIR}/{fig_name}")
        plt.close()

# ============================================================
# § 10  HINDI FLUENCY AS PREDICTOR
# ============================================================

print("\n── §10 Hindi Fluency as Predictor ──")

valid_fl = subj_df.dropna(subset=["hi_fluency","hi_confidence"]).copy()
print(f"  Participants with fluency data: {len(valid_fl)}")

outcomes = ["total_words", "mean_irt", "mean_cluster_size"]
out_labels = ["Total Words", "Mean IRT (ms)", "Mean Cluster Size"]

fl_results = []
for outcome, label in zip(outcomes, out_labels):
    if outcome not in valid_fl.columns: continue
    valid_sub = valid_fl.dropna(subset=[outcome])
    if len(valid_sub) < 2:
        continue
    r_p, p_p = pearsonr(valid_sub["hi_fluency"], valid_sub[outcome])
    r_s, p_s = spearmanr(valid_sub["hi_fluency"], valid_sub[outcome])
    fl_results.append({"outcome": label, "r": r_p, "p_pearson": p_p,
                        "rho": r_s, "p_spearman": p_s, "n": len(valid_sub)})
    print(f"  {label}: r={r_p:.3f} (p={p_p:.3f}), ρ={r_s:.3f} (p={p_s:.3f})")

# Multiple regression
if len(valid_fl) > 10:
    pred_cols = [c for c in ["hi_fluency","language_count","age"] if c in valid_fl.columns]
    valid_fl2 = valid_fl.dropna(subset=pred_cols + ["total_words"])
    if len(valid_fl2) > 8:
        X_fl = sm.add_constant(valid_fl2[pred_cols])
        ols_fl = sm.OLS(valid_fl2["total_words"], X_fl).fit()
        print(f"\n  Multiple regression (total_words):")
        print(f"  R²={ols_fl.rsquared:.3f}, F({ols_fl.df_model:.0f},{ols_fl.df_resid:.0f})={ols_fl.fvalue:.2f}, p={ols_fl.f_pvalue:.4f}")
        for var in pred_cols:
            print(f"    {var}: β={ols_fl.params.get(var,np.nan):.3f}, p={ols_fl.pvalues.get(var,np.nan):.4f}")

# High vs Low fluency group comparison
if "hi_confidence" in valid_fl.columns:
    med_conf = valid_fl["hi_confidence"].median()
    hi_grp = valid_fl[valid_fl["hi_confidence"] > med_conf]["total_words"].dropna()
    lo_grp = valid_fl[valid_fl["hi_confidence"] <= med_conf]["total_words"].dropna()
    if len(hi_grp) > 3 and len(lo_grp) > 3:
        t_fl, p_fl = stats.ttest_ind(hi_grp, lo_grp)
        d_fl = (hi_grp.mean() - lo_grp.mean()) / np.sqrt(
            ((len(hi_grp)-1)*hi_grp.std()**2 + (len(lo_grp)-1)*lo_grp.std()**2)
            / (len(hi_grp)+len(lo_grp)-2))
        print(f"\n  High vs Low confidence: t={t_fl:.2f}, p={p_fl:.4f}, d={d_fl:.2f}")
        print(f"  Hi: M={hi_grp.mean():.1f}, Lo: M={lo_grp.mean():.1f}")

# ── Fluency figure ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, outcome, label in zip(axes, outcomes, out_labels):
    if outcome not in valid_fl.columns: continue
    valid_sub = valid_fl.dropna(subset=[outcome])
    if len(valid_sub) < 2:
        ax.text(0.5, 0.5, "Insufficient fluency data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Hi Fluency → {label}\nNot available")
        ax.set_xlabel("Hindi Fluency Score"); ax.set_ylabel(label)
        continue
    ax.scatter(valid_sub["hi_fluency"], valid_sub[outcome],
               alpha=0.75, s=55, color="#4E79A7", edgecolors="white", lw=0.5,
               label="Participants")
    if len(valid_sub) > 2:
        m_, b__, r_, p_, _ = stats.linregress(valid_sub["hi_fluency"], valid_sub[outcome])
        xs_ = np.linspace(valid_sub["hi_fluency"].min(), valid_sub["hi_fluency"].max(), 50)
        ax.plot(xs_, m_*xs_+b__, color="#E15759", lw=2, label="Linear fit")
        ax.set_title(f"Hi Fluency → {label}\nr={r_:.2f}, p={'< .05' if p_<0.05 else f'= {p_:.3f}'}")
    ax.set_xlabel("Hindi Fluency Score"); ax.set_ylabel(label)
    ax.legend(fontsize=8)

fig.suptitle("§10 — Hindi Fluency as Predictor of Retrieval Efficiency", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMGDIR}/hindi_fluency_predictor.png")
plt.close()
print(f"✓ Saved hindi_fluency_predictor.png")

# ============================================================
# § 11  SUMMARY STATISTICS TABLES
# ============================================================

print("\n── §11 Summary Tables ──")

# Table 1: IRT descriptives by domain
t1 = df.groupby("domain")["rt_ms"].agg(
    N="count",
    Mean=lambda x: round(x.mean(), 0),
    Median=lambda x: round(x.median(), 0),
    SD=lambda x: round(x.std(), 0),
    Skewness=lambda x: round(float(stats.skew(x.dropna())), 2),
    Kurtosis=lambda x: round(float(stats.kurtosis(x.dropna())), 2),
).reset_index()
print("\nTable 1 — IRT Descriptives by Domain:")
print(t1.to_string(index=False))

# Table 2: Cluster metrics by domain
t2_rows = []
for dom in DOMAINS:
    pdom = per_subj_df[per_subj_df["domain"]==dom]
    if len(pdom) == 0: continue
    t2_rows.append({
        "Domain": dom.capitalize(),
        "Mean Within IRT (ms)": round(pdom["mean_within"].mean(), 0),
        "Mean Between IRT (ms)": round(pdom["mean_between"].mean(), 0),
        "Ratio B/W": round(pdom["mean_between"].mean() / pdom["mean_within"].mean(), 2),
    })
t2 = pd.DataFrame(t2_rows)
print("\nTable 2 — Within vs Between IRT by Domain:")
print(t2.to_string(index=False))

rq1_t = float(rq1_stats["t"]["T"].values[0])
rq1_p_t = float(rq1_stats["t"]["p_val"].values[0])
rq1_w = float(rq1_stats["wilcoxon"].statistic)
rq1_p_w = float(rq1_stats["wilcoxon"].pvalue)
rq2_rho, rq2_p_s = spearmanr(subj_df["mean_cluster_size"], subj_df["total_words"])

def p_fmt(p):
    if pd.isna(p):
        return "--"
    return "< .001" if p < 0.001 else f"= {p:.3f}"

rq1_decision = "Reject H0" if rq1_p_w < 0.05 else "Not significant"
rq2_decision = "Reject H0" if rq2_p_s < 0.05 else "Not significant"
eh1_decision = "Domains differ" if p_kw < 0.05 else "Not significant"

if SPAM_AVAILABLE and "rq3_df" in locals() and not rq3_df.empty:
    rq3_min_p = float(rq3_df["p_bh"].min())
    rq3_effect = f"max |rho|={rq3_df['rho'].abs().max():.2f}"
    rq3_decision = "Reject H0" if rq3_min_p < 0.05 else "Not significant"
    rq3_stat = f"best rho={rq3_df.loc[rq3_df['p_bh'].idxmin(), 'rho']:.2f}"
    rq3_p_text = p_fmt(rq3_min_p)
else:
    rq3_effect = "--"
    rq3_decision = "SpAM pending"
    rq3_stat = "--"
    rq3_p_text = "--"

rq4_decision = "Domains differ" if (not pd.isna(p_rq4) and p_rq4 < 0.05) else ("SpAM pending" if pd.isna(p_rq4) else "Not significant")
rq4_stat = f"H={H_rq4:.2f}" if not pd.isna(H_rq4) else "--"
rq4_effect = "lower NN distance = denser semantic neighbourhood" if not pd.isna(H_rq4) else "--"

rq5_decision = "Reject H0" if (not pd.isna(p_rq5) and p_rq5 < 0.05) else ("SpAM pending" if pd.isna(p_rq5) else "Not significant")
rq5_stat = f"rho={rho_rq5:.2f}" if not pd.isna(rho_rq5) else "--"
if not pd.isna(lme_rq5_coef):
    rq5_effect = f"MixedLM beta={lme_rq5_coef:.1f}"
else:
    rq5_effect = "monotonic association" if not pd.isna(rho_rq5) else "--"

# Table 3: Hypothesis summary
t3 = pd.DataFrame([
    {"#": "RQ1", "Test": "Wilcoxon signed-rank", "Statistic": f"W={rq1_w:.1f}", "p": p_fmt(rq1_p_w), "Effect": f"t={rq1_t:.2f}, d={rq1_stats['d']:.2f}", "Decision": rq1_decision},
    {"#": "RQ2", "Test": "Spearman correlation", "Statistic": f"rho={rq2_rho:.2f}", "p": p_fmt(rq2_p_s), "Effect": "rank-based association", "Decision": rq2_decision},
    {"#": "RQ4", "Test": "Kruskal-Wallis on SpAM NN distance", "Statistic": rq4_stat, "p": p_fmt(p_rq4), "Effect": rq4_effect, "Decision": rq4_decision},
    {"#": "RQ5", "Test": "Spearman + MixedLM (SpAM dispersion -> switch cost)", "Statistic": rq5_stat, "p": p_fmt(p_rq5), "Effect": rq5_effect, "Decision": rq5_decision},
    {"#": "EH1", "Test": "Kruskal-Wallis", "Statistic": f"H={H:.2f}", "p": p_fmt(p_kw), "Effect": "eta2~0.05", "Decision": eh1_decision},
    {"#": "RQ3", "Test": "Spearman rho (BH-corrected)", "Statistic": rq3_stat, "p": rq3_p_text, "Effect": rq3_effect, "Decision": rq3_decision},
])
print("\nTable 3 — Hypothesis Test Summary:")
print(t3.to_string(index=False))

# Persist summary tables used in report/presentation.
t1.to_csv("table_irt_by_domain.csv", index=False)
t2.to_csv("table_cluster_metrics.csv", index=False)
t3.to_csv("table_hypothesis_summary.csv", index=False)
if SPAM_AVAILABLE and "rq3_df" in locals() and not rq3_df.empty:
    rq3_df.to_csv("table_spam_rq3_results.csv", index=False)

# RQ4/RQ5 auxiliary tables
rq45_rows = [
    {
        "rq": "RQ4",
        "test": "Kruskal-Wallis on domain-level SpAM nearest-neighbour distance",
        "statistic": rq4_stat,
        "p": p_fmt(p_rq4),
        "decision": rq4_decision,
    },
    {
        "rq": "RQ5",
        "test": "Spearman + MixedLM between SpAM dispersion and VFT switch cost",
        "statistic": rq5_stat,
        "p": p_fmt(p_rq5),
        "decision": rq5_decision,
    },
]
rq45_table = pd.DataFrame(rq45_rows)
rq45_table.to_csv("table_spam_rq45_results.csv", index=False)

if not rq4_posthoc_df.empty:
    rq4_posthoc_df.to_csv("table_spam_rq4_posthoc.csv", index=False)
if not rq4_density_word_df.empty:
    rq4_density_word_df.to_csv("table_spam_rq4_density_by_word.csv", index=False)
if not rq5_pair_df.empty:
    rq5_pair_df.to_csv("table_spam_rq5_pairlevel.csv", index=False)
print("\n✓ Saved table_irt_by_domain.csv, table_cluster_metrics.csv, table_hypothesis_summary.csv")
if "language_domain_cmp" in locals() and not language_domain_cmp.empty:
    print("✓ Saved table_language_domain_comparison.csv")
if SPAM_AVAILABLE and "rq3_df" in locals() and not rq3_df.empty:
    print("✓ Saved table_spam_rq3_results.csv")
print("✓ Saved table_spam_rq45_results.csv")
if not rq4_posthoc_df.empty:
    print("✓ Saved table_spam_rq4_posthoc.csv")
if not rq4_density_word_df.empty:
    print("✓ Saved table_spam_rq4_density_by_word.csv")
if not rq5_pair_df.empty:
    print("✓ Saved table_spam_rq5_pairlevel.csv")

print("\n✓ PART 2 COMPLETE")
print("Generated figures:")
import glob
for f in sorted(glob.glob(f"{IMGDIR}/*.png")):
    print(f"  {f}")
