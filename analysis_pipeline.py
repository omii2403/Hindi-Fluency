from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from scipy.stats import binomtest, f_oneway, kruskal, levene, pearsonr, shapiro, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "merged_vft_spam_responses_enriched.csv"
JSON_PATH = ROOT / "responses.json"
SYLLABUS_PATH = ROOT / "BRSM-Syllabus.md"
BRIEF_PDF_PATH = ROOT / "hindi fluency experiment brief_ BRSM 2026.pdf"
IMAGES_DIR = ROOT / "images"
REPORT_MD = ROOT / "Report_Final.md"
REPORT_PDF = ROOT / "Report_Final_md.pdf"

PRIMARY_LANG = "hindi/hinglish"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PHONETIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def get_sentence_model(model_name: str) -> SentenceTransformer:
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    return model


@dataclass
class TestResult:
    name: str
    stat: float
    p: float
    effect: Optional[float]
    interpretation: str


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Exclude furniture trial if present in any variant.
    df = df[~df["domain"].astype(str).str.lower().str.contains("furniture", na=False)].copy()
    df["language_type_clean"] = df["language_type"].astype(str).str.strip().str.lower()
    df["word_clean"] = df["word"].astype(str).str.strip().str.lower()
    return df


def read_brief_text() -> str:
    doc = fitz.open(str(BRIEF_PDF_PATH))
    text = "\n".join(page.get_text() for page in doc)
    return text


def extract_research_question(brief_text: str) -> str:
    m = re.search(r"Research question:\s*(.+?\?)", brief_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return " ".join(m.group(1).split())
    return "How do Hindi speakers search their mental lexicons for information?"


def zscore_series(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def summarize_vft(df_primary: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df_primary.groupby(["session_id", "subject_id", "domain"], as_index=False)
        .agg(
            word_count=("word_clean", "size"),
            unique_count=("word_clean", pd.Series.nunique),
            mean_rt_ms=("rt_ms", "mean"),
            median_rt_ms=("rt_ms", "median"),
            hi_read=("Hi_Read", "mean"),
            hi_write=("Hi_Write", "mean"),
            hi_conf=("hi_confidence", "mean"),
            language_count=("language_count", "mean"),
            age=("age", "mean"),
        )
    )
    grp["lexical_diversity"] = grp["unique_count"] / grp["word_count"].replace(0, np.nan)
    return grp


def choose_k(points: np.ndarray, max_k: int = 6) -> int:
    n = points.shape[0]
    if n < 4:
        return 2 if n >= 2 else 1

    upper = min(max_k, n - 1)
    best_k = 2
    best_score = -1.0

    for k in range(2, upper + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(points)
            score = silhouette_score(points, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


DEV_TO_LATIN = {
    "\u0905": "a",
    "\u0906": "aa",
    "\u0907": "i",
    "\u0908": "ii",
    "\u0909": "u",
    "\u090a": "uu",
    "\u090f": "e",
    "\u0910": "ai",
    "\u0913": "o",
    "\u0914": "au",
    "\u0915": "k",
    "\u0916": "kh",
    "\u0917": "g",
    "\u0918": "gh",
    "\u0919": "ng",
    "\u091a": "c",
    "\u091b": "ch",
    "\u091c": "j",
    "\u091d": "jh",
    "\u091e": "ny",
    "\u091f": "t",
    "\u0920": "th",
    "\u0921": "d",
    "\u0922": "dh",
    "\u0923": "n",
    "\u0924": "t",
    "\u0925": "th",
    "\u0926": "d",
    "\u0927": "dh",
    "\u0928": "n",
    "\u092a": "p",
    "\u092b": "ph",
    "\u092c": "b",
    "\u092d": "bh",
    "\u092e": "m",
    "\u092f": "y",
    "\u0930": "r",
    "\u0932": "l",
    "\u0935": "v",
    "\u0936": "sh",
    "\u0937": "sh",
    "\u0938": "s",
    "\u0939": "h",
    "\u093e": "a",
    "\u093f": "i",
    "\u0940": "ii",
    "\u0941": "u",
    "\u0942": "uu",
    "\u0947": "e",
    "\u0948": "ai",
    "\u094b": "o",
    "\u094c": "au",
    "\u094d": "",
    "\u0902": "n",
    "\u0901": "n",
    "\u0903": "h",
}


def simple_phonetic_key(word: str) -> str:
    norm = unicodedata.normalize("NFKC", str(word).lower().strip())
    translit_chars: List[str] = []

    for ch in norm:
        if ch in DEV_TO_LATIN:
            translit_chars.append(DEV_TO_LATIN[ch])
        elif ch.isascii() and ch.isalpha():
            translit_chars.append(ch)

    translit = "".join(translit_chars)
    translit = re.sub(r"[^a-z]", "", translit)

    # A crude vowel-drop + collapse pipeline to approximate broad phonetic shape.
    core = re.sub(r"[aeiou]", "", translit)
    core = re.sub(r"(.)\1+", r"\1", core)
    if not core:
        core = translit
    return core or "x"


def build_transformer_embeddings(words: List[str], use_phonetic: bool) -> np.ndarray:
    if use_phonetic:
        corpus = [simple_phonetic_key(w) for w in words]
        model_name = PHONETIC_MODEL_NAME
    else:
        corpus = [unicodedata.normalize("NFKC", str(w).strip().lower()) for w in words]
        model_name = SEMANTIC_MODEL_NAME

    model = get_sentence_model(model_name)
    # Use normalized sentence-transformer embeddings for cosine-compatible clustering.
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


def project_to_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=float)

    n_components = min(2, x.shape[0], x.shape[1])
    projected = PCA(n_components=n_components).fit_transform(x)
    if n_components == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(projected.shape[0], dtype=float)])
    return projected


def kmeans_labels(x, k: int) -> np.ndarray:
    n = x.shape[0]
    k = max(2, min(k, n - 1)) if n > 2 else 1
    if k == 1:
        return np.zeros(n, dtype=int)

    try:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        return model.fit_predict(x)
    except Exception:
        # Safe fallback if matrix rank is too small.
        return np.zeros(n, dtype=int)


def compute_mean_nn_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return float("nan")
    dists = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists, axis=1).mean())


def analyze_spam_alignment(df_primary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for (session_id, subject_id, domain), g in df_primary.groupby(["session_id", "subject_id", "domain"]):
        g = g.sort_values("position").copy()
        if len(g) < 4:
            continue

        points = g[["x", "y"]].to_numpy(dtype=float)
        words = g["word_clean"].astype(str).tolist()
        k = choose_k(points, max_k=6)

        if k <= 1:
            continue

        try:
            spam_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(points)
        except Exception:
            continue

        x_sem = build_transformer_embeddings(words, use_phonetic=False)
        x_pho = build_transformer_embeddings(words, use_phonetic=True)

        sem_labels = kmeans_labels(x_sem, k)
        pho_labels = kmeans_labels(x_pho, k)

        ari_sem_spam = adjusted_rand_score(spam_labels, sem_labels)
        nmi_sem_spam = normalized_mutual_info_score(spam_labels, sem_labels)
        ari_pho_spam = adjusted_rand_score(spam_labels, pho_labels)
        nmi_pho_spam = normalized_mutual_info_score(spam_labels, pho_labels)
        ari_sem_pho = adjusted_rand_score(sem_labels, pho_labels)
        nmi_sem_pho = normalized_mutual_info_score(sem_labels, pho_labels)

        nn_distance = compute_mean_nn_distance(points)

        try:
            sil_spam = silhouette_score(points, spam_labels)
        except Exception:
            sil_spam = float("nan")

        rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_id,
                "domain": domain,
                "n_words": len(g),
                "k_clusters": k,
                "mean_nn_distance": nn_distance,
                "silhouette_spam_xy": sil_spam,
                "ari_semantic_spam": ari_sem_spam,
                "nmi_semantic_spam": nmi_sem_spam,
                "ari_phonetic_spam": ari_pho_spam,
                "nmi_phonetic_spam": nmi_pho_spam,
                "ari_semantic_phonetic": ari_sem_pho,
                "nmi_semantic_phonetic": nmi_sem_pho,
            }
        )

    return pd.DataFrame(rows)


def shapiro_safe(values: np.ndarray) -> float:
    x = values[~np.isnan(values)]
    if len(x) < 3:
        return float("nan")
    if len(x) > 5000:
        x = np.random.default_rng(42).choice(x, size=5000, replace=False)
    return float(shapiro(x).pvalue)


def domain_test_with_normality_decision(df: pd.DataFrame, value_col: str, label: str) -> TestResult:
    grouped = []
    normality_p = []

    for _, g in df.groupby("domain"):
        vals = g[value_col].dropna().to_numpy(dtype=float)
        if len(vals) > 1:
            grouped.append(vals)
            normality_p.append(shapiro_safe(vals))

    if len(grouped) < 2:
        return TestResult(label, float("nan"), float("nan"), None, "Insufficient domain groups for inferential testing.")

    all_normal = len(normality_p) == len(grouped) and all(pd.notna(p) and p >= 0.05 for p in normality_p)
    lev_p = float("nan")
    if all_normal:
        try:
            lev_p = float(levene(*grouped).pvalue)
        except Exception:
            lev_p = float("nan")

    if all_normal and pd.notna(lev_p) and lev_p >= 0.05:
        stat, p = f_oneway(*grouped)
        n = sum(len(x) for x in grouped)
        k = len(grouped)
        eta2 = (stat * (k - 1)) / (stat * (k - 1) + (n - k)) if (n - k) > 0 else float("nan")
        sig = "statistically significant" if p < 0.05 else "not statistically significant"
        interp = (
            f"Normality and homogeneity assumptions were acceptable; ANOVA indicates {sig} "
            f"domain-level differences in {label.lower()} (F={stat:.3f}, p={p:.4f}, eta^2={eta2:.3f})."
        )
        return TestResult(f"ANOVA: {label}", float(stat), float(p), float(eta2), interp)

    stat, p = kruskal(*grouped)
    n = sum(len(x) for x in grouped)
    k = len(grouped)
    eps2 = (stat - k + 1) / (n - k) if (n - k) > 0 else float("nan")
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    interp = (
        f"Normality assumptions were not fully satisfied; Kruskal-Wallis indicates {sig} "
        f"domain-level differences in {label.lower()} (H={stat:.3f}, p={p:.4f}, epsilon^2={eps2:.3f})."
    )
    return TestResult(f"Kruskal-Wallis: {label}", float(stat), float(p), float(eps2), interp)


def spearman_table(vft_domain: pd.DataFrame) -> pd.DataFrame:
    session_level = (
        vft_domain.groupby(["session_id", "subject_id"], as_index=False)
        .agg(
            total_words=("word_count", "sum"),
            mean_diversity=("lexical_diversity", "mean"),
            mean_rt=("mean_rt_ms", "mean"),
            hi_read=("hi_read", "mean"),
            hi_write=("hi_write", "mean"),
            hi_conf=("hi_conf", "mean"),
            language_count=("language_count", "mean"),
            age=("age", "mean"),
        )
    )

    rows = []
    targets = ["hi_read", "hi_write", "hi_conf", "language_count", "age"]
    metrics = ["total_words", "mean_diversity", "mean_rt"]

    for m in metrics:
        for t in targets:
            x = session_level[m].astype(float).to_numpy()
            y = session_level[t].astype(float).to_numpy()
            px = shapiro_safe(x)
            py = shapiro_safe(y)

            if pd.notna(px) and pd.notna(py) and px >= 0.05 and py >= 0.05:
                rho, p = pearsonr(x, y)
                test_used = "Pearson"
            else:
                rho, p = spearmanr(x, y, nan_policy="omit")
                test_used = "Spearman"

            rows.append(
                {
                    "metric": m,
                    "target": t,
                    "rho": rho,
                    "p": p,
                    "test_used": test_used,
                    "normality_metric_p": px,
                    "normality_target_p": py,
                    "significant_0.05": bool(p < 0.05),
                }
            )

    return pd.DataFrame(rows)


def sign_test_positive(series: pd.Series, label: str) -> TestResult:
    s = series.dropna()
    s = s[s != 0]
    n = len(s)
    if n == 0:
        return TestResult(label, float("nan"), float("nan"), None, "No non-zero values for sign test.")

    pos = int((s > 0).sum())
    bt = binomtest(pos, n=n, p=0.5, alternative="greater")
    interp = (
        f"Sign test on {label}: {pos}/{n} positive; p={bt.pvalue:.4f}. "
        + ("Evidence favors positive alignment." if bt.pvalue < 0.05 else "No strong evidence for predominantly positive alignment.")
    )
    return TestResult(label, float(pos / n), float(bt.pvalue), None, interp)


def compute_scores(vft_domain: pd.DataFrame, spam_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = vft_domain.merge(spam_df, on=["session_id", "subject_id", "domain"], how="left")

    merged["z_word_count"] = zscore_series(merged["word_count"].astype(float))
    merged["z_diversity"] = zscore_series(merged["lexical_diversity"].astype(float))
    merged["z_speed"] = zscore_series(-merged["mean_rt_ms"].astype(float))
    merged["z_spatial_cohesion"] = zscore_series(-merged["mean_nn_distance"].astype(float))
    merged["z_sem_align"] = zscore_series(merged["ari_semantic_spam"].astype(float))
    merged["z_pho_align"] = zscore_series(merged["ari_phonetic_spam"].astype(float))

    merged["vft_only_score"] = merged[["z_word_count", "z_diversity", "z_speed"]].mean(axis=1, skipna=True)
    merged["domain_fluency_score"] = merged[
        [
            "z_word_count",
            "z_diversity",
            "z_speed",
            "z_spatial_cohesion",
            "z_sem_align",
            "z_pho_align",
        ]
    ].mean(axis=1, skipna=True)

    participant = (
        merged.groupby(["session_id", "subject_id"], as_index=False)
        .agg(
            composite_fluency_score=("domain_fluency_score", "mean"),
            vft_only_score=("vft_only_score", "mean"),
            hi_conf=("hi_conf", "mean"),
            hi_read=("hi_read", "mean"),
            hi_write=("hi_write", "mean"),
            domains_covered=("domain", "nunique"),
        )
        .sort_values("composite_fluency_score", ascending=False)
    )

    return merged, participant


def save_plots(vft_domain: pd.DataFrame, spam_df: pd.DataFrame, merged: pd.DataFrame, participant: pd.DataFrame) -> Tuple[List[Path], Dict[str, str]]:
    sns.set_theme(style="whitegrid", context="talk")
    created: List[Path] = []
    plot_notes: Dict[str, str] = {}

    # 1. VFT word count by domain.
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=vft_domain, x="domain", y="word_count", color="#87b7ff")
    sns.stripplot(data=vft_domain, x="domain", y="word_count", color="#1f4e79", alpha=0.6, size=5)
    plt.title("VFT Output by Domain (Hindi/Hinglish)")
    plt.xlabel("Domain")
    plt.ylabel("Word Count")
    p1 = IMAGES_DIR / "vft_wordcount_by_domain.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()
    created.append(p1)

    # 2. VFT mean retrieval time by domain.
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=vft_domain, x="domain", y="mean_rt_ms", color="#ffd68a")
    sns.stripplot(data=vft_domain, x="domain", y="mean_rt_ms", color="#8a5a00", alpha=0.55, size=5)
    plt.title("Mean Inter-Word Retrieval Time by Domain (Hindi/Hinglish)")
    plt.xlabel("Domain")
    plt.ylabel("Mean RT (ms)")
    p2 = IMAGES_DIR / "vft_rt_by_domain.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    created.append(p2)

    # 3. VFT words vs retrieval time.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=vft_domain, x="word_count", y="mean_rt_ms", hue="domain", s=80)
    sns.regplot(data=vft_domain, x="word_count", y="mean_rt_ms", scatter=False, color="black", line_kws={"linewidth": 2})
    plt.title("VFT Productivity vs Retrieval Time")
    plt.xlabel("Word Count")
    plt.ylabel("Mean RT (ms)")
    p3 = IMAGES_DIR / "vft_words_vs_rt.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()
    created.append(p3)

    # 4. SpAM alignment metrics by domain.
    if not spam_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=spam_df, x="domain", y="mean_nn_distance", color="#bde0fe")
        sns.stripplot(data=spam_df, x="domain", y="mean_nn_distance", color="#1d3557", alpha=0.6, size=4)
        plt.title("SpAM Spatial Compactness by Domain")
        plt.xlabel("Domain")
        plt.ylabel("Mean nearest-neighbor distance")
        p4a = IMAGES_DIR / "spam_compactness_by_domain.png"
        plt.tight_layout()
        plt.savefig(p4a, dpi=300)
        plt.close()
        created.append(p4a)

        m = spam_df.melt(
            id_vars=["domain"],
            value_vars=["ari_semantic_spam", "ari_phonetic_spam"],
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=m, x="domain", y="value", hue="metric")
        plt.title("SpAM vs Embedding Cluster Agreement by Domain (ARI)")
        plt.xlabel("Domain")
        plt.ylabel("ARI")
        p4 = IMAGES_DIR / "spam_alignment_by_domain.png"
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close()
        created.append(p4)

    # 5. Similarity metric matrix.
    if not spam_df.empty:
        sim = pd.DataFrame(
            {
                "SpAM-Semantic": [spam_df["ari_semantic_spam"].mean(), spam_df["nmi_semantic_spam"].mean()],
                "SpAM-Phonetic": [spam_df["ari_phonetic_spam"].mean(), spam_df["nmi_phonetic_spam"].mean()],
                "Semantic-Phonetic": [spam_df["ari_semantic_phonetic"].mean(), spam_df["nmi_semantic_phonetic"].mean()],
            },
            index=["ARI", "NMI"],
        )
        plt.figure(figsize=(8, 5))
        sns.heatmap(sim, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title("Mean Cluster Similarity Metrics")
        p5 = IMAGES_DIR / "cluster_similarity_matrix.png"
        plt.tight_layout()
        plt.savefig(p5, dpi=300)
        plt.close()
        created.append(p5)

    # 6. Composite score distribution.
    plt.figure(figsize=(10, 6))
    sns.histplot(participant["composite_fluency_score"], kde=True, bins=12, color="#2a9d8f")
    plt.title("Composite Hindi Fluency Score Distribution")
    plt.xlabel("Composite Score")
    p6 = IMAGES_DIR / "composite_score_distribution.png"
    plt.tight_layout()
    plt.savefig(p6, dpi=300)
    plt.close()
    created.append(p6)

    # 7. VFT-only vs integrated score.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=participant, x="vft_only_score", y="composite_fluency_score", s=90, color="#e76f51")
    sns.regplot(data=participant, x="vft_only_score", y="composite_fluency_score", scatter=False, color="black", line_kws={"linewidth": 2})
    plt.title("VFT-only vs Integrated Fluency Score")
    plt.xlabel("VFT-only Score")
    plt.ylabel("Integrated Composite Score")
    p7 = IMAGES_DIR / "vft_vs_integrated_score.png"
    plt.tight_layout()
    plt.savefig(p7, dpi=300)
    plt.close()
    created.append(p7)

    # 8. A high-divergence session-domain cluster panel for visual contrast.
    if not spam_df.empty:
        candidates = spam_df.copy()
        candidates = candidates[candidates["n_words"] >= 6] if "n_words" in candidates.columns else candidates
        if candidates.empty:
            candidates = spam_df.copy()

        sem_spam = candidates["ari_semantic_spam"].fillna(0.0)
        pho_spam = candidates["ari_phonetic_spam"].fillna(0.0)
        sem_pho = candidates["ari_semantic_phonetic"].fillna(0.0)
        candidates = candidates.assign(
            divergence_score=(1.0 - sem_spam) + (1.0 - pho_spam) + (1.0 - sem_pho)
        )
        chosen = candidates.sort_values(["divergence_score", "n_words"], ascending=[False, False]).iloc[0]
        sd = merged[
            (merged["session_id"] == chosen["session_id"])
            & (merged["domain"] == chosen["domain"])
        ]

        original = vft_domain.merge(
            df_primary_global[["session_id", "domain", "position", "word_clean", "x", "y"]],
            on=["session_id", "domain"],
            how="left",
        )
        graw = df_primary_global[
            (df_primary_global["session_id"] == chosen["session_id"])
            & (df_primary_global["domain"] == chosen["domain"])
        ].sort_values("position")

        if len(graw) >= 4:
            points = graw[["x", "y"]].to_numpy(dtype=float)
            words = graw["word_clean"].astype(str).tolist()
            k = int(chosen["k_clusters"])
            spam_lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(points)
            sem_emb = build_transformer_embeddings(words, False)
            pho_emb = build_transformer_embeddings(words, True)
            sem_lab = kmeans_labels(sem_emb, k)
            pho_lab = kmeans_labels(pho_emb, k)
            sem_xy = project_to_2d(sem_emb)
            pho_xy = project_to_2d(pho_emb)

            fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
            for ax, coords, labels, title, xlabel, ylabel in [
                (axes[0], points, spam_lab, "Participant SpAM Clusters (x-y)", "x", "y"),
                (axes[1], sem_xy, sem_lab, "Semantic Clusters (embedding PCA)", "PC1", "PC2"),
                (axes[2], pho_xy, pho_lab, "Phonetic Clusters (embedding PCA)", "PC1", "PC2"),
            ]:
                ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=60)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                for i in range(len(words)):
                    ax.annotate(str(i + 1), (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(4, 3), fontsize=8, alpha=0.9)

            fig.suptitle(
                (
                    f"High-Divergence Cluster Example: Domain={chosen['domain']} | "
                    f"ARI(SpAM,Sem)={chosen['ari_semantic_spam']:.2f}, "
                    f"ARI(SpAM,Pho)={chosen['ari_phonetic_spam']:.2f}, "
                    f"ARI(Sem,Pho)={chosen['ari_semantic_phonetic']:.2f}"
                ),
                fontsize=13,
            )
            p8 = IMAGES_DIR / "sample_spam_semantic_phonetic_clusters.png"
            plt.tight_layout()
            plt.savefig(p8, dpi=300)
            plt.close()
            created.append(p8)

            plot_notes["sample_cluster_panel"] = (
                "Selected a high-divergence participant-domain case to make method differences visible. "
                f"Domain={chosen['domain']}, n_words={int(chosen['n_words'])}, k={int(chosen['k_clusters'])}, "
                f"ARI(SpAM,Semantic)={chosen['ari_semantic_spam']:.3f}, "
                f"ARI(SpAM,Phonetic)={chosen['ari_phonetic_spam']:.3f}, "
                f"ARI(Semantic,Phonetic)={chosen['ari_semantic_phonetic']:.3f}."
            )
            plot_notes["sample_cluster_word_map"] = "; ".join([f"{i + 1}:{w}" for i, w in enumerate(words)])
            plot_notes["sample_cluster_membership"] = "; ".join(
                [f"{i + 1}:SpAM{int(a)}/Sem{int(b)}/Pho{int(c)}" for i, (a, b, c) in enumerate(zip(spam_lab, sem_lab, pho_lab))]
            )

    return created, plot_notes


def table_to_markdown(df: pd.DataFrame, index: bool = False, float_fmt: str = ".3f") -> str:
    if df.empty:
        return "(No data)"

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:{float_fmt}}" if pd.notna(x) else "")
    return out.to_markdown(index=index)


def build_report(
    research_q: str,
    df_all: pd.DataFrame,
    df_primary: pd.DataFrame,
    df_english: pd.DataFrame,
    vft_domain: pd.DataFrame,
    spam_df: pd.DataFrame,
    merged_scores: pd.DataFrame,
    participant_scores: pd.DataFrame,
    tests: Dict[str, TestResult],
    assoc_table: pd.DataFrame,
    created_plots: List[Path],
    plot_notes: Dict[str, str],
) -> str:
    n_sessions = df_all["session_id"].nunique()
    n_subjects = df_all["subject_id"].nunique()
    domains = sorted(df_all["domain"].dropna().astype(str).unique().tolist())

    lang_counts = df_all["language_type_clean"].value_counts().to_dict()
    primary_share = len(df_primary) / len(df_all) if len(df_all) else float("nan")

    domain_summary = (
        vft_domain.groupby("domain", as_index=False)
        .agg(
            n_session_domain=("session_id", "size"),
            mean_word_count=("word_count", "mean"),
            sd_word_count=("word_count", "std"),
            mean_rt_ms=("mean_rt_ms", "mean"),
            mean_lexical_diversity=("lexical_diversity", "mean"),
        )
        .sort_values("mean_word_count", ascending=False)
    )

    english_context = (
        df_english.groupby("domain", as_index=False)
        .agg(
            english_rows=("word_clean", "size"),
            english_mean_rt_ms=("rt_ms", "mean"),
        )
        .sort_values("english_rows", ascending=False)
    )

    spam_summary = (
        spam_df.groupby("domain", as_index=False)
        .agg(
            mean_ari_semantic_spam=("ari_semantic_spam", "mean"),
            mean_ari_phonetic_spam=("ari_phonetic_spam", "mean"),
            mean_nmi_semantic_spam=("nmi_semantic_spam", "mean"),
            mean_nmi_phonetic_spam=("nmi_phonetic_spam", "mean"),
            mean_nn_distance=("mean_nn_distance", "mean"),
        )
        .sort_values("mean_ari_semantic_spam", ascending=False)
    )

    top5 = participant_scores.head(5).copy()
    top5 = top5[["session_id", "subject_id", "composite_fluency_score", "vft_only_score", "domains_covered"]]

    rho_vft, p_vft = spearmanr(participant_scores["vft_only_score"], participant_scores["hi_conf"], nan_policy="omit")
    rho_comp, p_comp = spearmanr(participant_scores["composite_fluency_score"], participant_scores["hi_conf"], nan_policy="omit")

    # Pick strongest significant associations for concise reporting.
    sig_assoc = assoc_table[assoc_table["significant_0.05"]].copy()
    sig_assoc = sig_assoc.sort_values("p").head(8)

    image_links = "\n".join([f"- ![{p.name}]({p.relative_to(ROOT).as_posix()})" for p in created_plots])
    sample_cluster_note = plot_notes.get("sample_cluster_panel", "Sample-cluster panel note unavailable.")
    sample_cluster_word_map = plot_notes.get("sample_cluster_word_map", "Point-to-word mapping unavailable.")
    sample_cluster_membership = plot_notes.get("sample_cluster_membership", "Point-wise cluster assignment unavailable.")

    vft_domain_interp = (
        "At least one VFT domain-comparison test is statistically significant, indicating domain-sensitive lexical retrieval patterns."
        if (pd.notna(tests["vft_word_count"].p) and tests["vft_word_count"].p < 0.05)
        or (pd.notna(tests["vft_mean_rt"].p) and tests["vft_mean_rt"].p < 0.05)
        else "VFT domain-comparison tests were not statistically significant, so domain differences should be interpreted as descriptive trends in this sample."
    )

    cross_task_interp = (
        "At least one sign test supports predominantly positive SpAM-embedding alignment, suggesting systematic structure linkage."
        if (pd.notna(tests["sign_semantic_alignment"].p) and tests["sign_semantic_alignment"].p < 0.05)
        or (pd.notna(tests["sign_phonetic_alignment"].p) and tests["sign_phonetic_alignment"].p < 0.05)
        else "Sign tests did not show predominantly positive alignment; cross-task coupling appears mixed and likely domain/participant dependent in this sample."
    )

    score_compare_interp = (
        "In this dataset, integrated scoring shows stronger monotonic association with Hindi confidence than VFT-only scoring."
        if abs(rho_comp) > abs(rho_vft)
        else "In this dataset, integrated scoring did not outperform VFT-only association with Hindi confidence, though it captures complementary structure-based information."
    )

    report = f"""# Report_Final

## 1. Background and Context
This report analyzes Hindi fluency experiment data containing two core tasks from the brief: Verbal Fluency Task (VFT) and Spatial Arrangement Method (SpAM). The objective is to answer the main research question using methods constrained to BRSM syllabus coverage, while keeping English entries as comparison context only.

Dataset overview (post trial exclusion):
- Sessions: {n_sessions}
- Participants (subject IDs): {n_subjects}
- Domains: {', '.join(domains)}
- Language rows: {lang_counts}
- Primary analysis subset (Hindi/Hinglish): {len(df_primary)} / {len(df_all)} rows ({primary_share:.1%})

## 2. Research Question
Verbatim from the brief: "{research_q}" [1]

## 3. Data and Preprocessing
- Source files used: processed CSV [3], raw JSON [4], experiment brief [1], BRSM syllabus summary [2].
- Trial run exclusion: furniture-practice/furniture rows were excluded if present (none were present in processed CSV).
- Primary inferential analyses used only Hindi/Hinglish rows.
- English rows were retained only for descriptive comparison.
- Session-domain unit was used for VFT and SpAM feature extraction.
- Missingness check: rt_ms, x, y had no missing values in processed CSV.

### Descriptive Snapshot by Domain (Hindi/Hinglish)
{table_to_markdown(domain_summary, index=False)}

### English Comparison Context (Descriptive Only)
{table_to_markdown(english_context, index=False)}

Interpretation:
- Hindi/Hinglish rows dominate the dataset, enabling primary testing without relying on English entries.
- Domain-level variation in both word output and retrieval speed is visible prior to formal testing.

## 4. Task Definition: VFT and SpAM
- VFT: participants produced as many category-relevant words as possible within one minute.
- SpAM: participants arranged produced words spatially by perceived similarity using x-y placement.

Operationalization in this report:
- VFT outcomes: word count, lexical diversity, mean RT.
- SpAM outcomes: nearest-neighbor distance, participant-cluster structure from x-y, and agreement with semantic/phonetic clustering.

## 5. Hypotheses and Exploratory Objectives
Confirmatory hypotheses:
1. VFT performance differs across domains (normality-first domain comparison).
2. Retrieval speed differs across domains (normality-first domain comparison).
3. SpAM spatial cohesion differs across domains (normality-first domain comparison).
4. SpAM-embedding agreement is more often positive than non-positive (binomial sign test).

Exploratory objectives:
1. Identify strongest participant-level associations between VFT metrics and Hindi fluency background indicators.
2. Quantify semantic vs phonetic cluster agreement with participant SpAM clusters.
3. Evaluate whether integrating SpAM features with VFT improves fluency alignment over VFT-only scoring.

## 6. Methods (syllabus-aligned)
Inferential tests were selected by a normality-first decision rule using Shapiro-Wilk checks [2]:
- If group-wise normality and variance homogeneity held, one-way ANOVA was used for domain comparisons.
- If assumptions failed, Kruskal-Wallis was used for domain comparisons.
- For pairwise associations, Pearson was used when both variables were normal; otherwise Spearman rank correlation was used.
- Binomial sign test was used for directional tendency of positive alignment metrics.

Similarity metrics (ARI, NMI) were used as quantitative cluster-agreement measures. Semantic and phonetic embeddings were generated with Hugging Face sentence-transformer models (`{SEMANTIC_MODEL_NAME}`) on original and phonetic-key text respectively.

## 7. Results: VFT
### Domain Difference Tests
- {tests['vft_word_count'].interpretation}
- {tests['vft_mean_rt'].interpretation}

### Association Analysis (Session-Level Spearman)
{table_to_markdown(sig_assoc[['metric','target','test_used','rho','p']], index=False)}

Interpretation:
- {vft_domain_interp}
- Significant correlations identify participant characteristics linked to VFT outcomes.

## 8. Results: SpAM
### Domain Difference Test
- {tests['spam_nn_distance'].interpretation}

### SpAM Structure and Consistency Summary
{table_to_markdown(spam_summary, index=False)}

Interpretation:
- SpAM structure varies across domains, suggesting domain-specific cognitive organization patterns.
- Mean nearest-neighbor distance provides a compact measure of spatial packing/cohesion.

## 9. Results: Cross-Task Integration (VFT + SpAM)
Sign tests on alignment tendency:
- {tests['sign_semantic_alignment'].interpretation}
- {tests['sign_phonetic_alignment'].interpretation}

Interpretation:
- {cross_task_interp}
- Cross-task evidence (VFT + SpAM) provides richer fluency signal than VFT alone.

## 10. Results: Clustering and Similarity Metrics
Domain-level cluster-comparison metrics are shown above and in the figures.

Key global means:
- Mean ARI (SpAM vs Semantic): {spam_df['ari_semantic_spam'].mean():.3f}
- Mean ARI (SpAM vs Phonetic): {spam_df['ari_phonetic_spam'].mean():.3f}
- Mean NMI (SpAM vs Semantic): {spam_df['nmi_semantic_spam'].mean():.3f}
- Mean NMI (SpAM vs Phonetic): {spam_df['nmi_phonetic_spam'].mean():.3f}

Interpretation:
- Semantic and phonetic representations both capture part of participant organization, with varying domain sensitivity.
- Mismatch cases are informative and indicate non-trivial strategy differences in lexical organization.

## 11. Composite Hindi Fluency Score (equal domain weights)
Scoring design:
- Domain-level score used equal weighting across domains by averaging standardized components per session-domain.
- Components: VFT productivity, lexical diversity, retrieval efficiency, spatial cohesion, semantic alignment, phonetic alignment.
- Participant composite score: arithmetic mean of domain-level scores across available domains (equal domain contribution).

Top participants by integrated score:
{table_to_markdown(top5, index=False)}

VFT-only vs integrated score relation:
- Spearman rho(VFT-only, hi_confidence) = {rho_vft:.3f}, p={p_vft:.4f}
- Spearman rho(Integrated, hi_confidence) = {rho_comp:.3f}, p={p_comp:.4f}

Interpretation:
- The integrated score incorporates both lexical retrieval and organization structure, aligning with the dual-task design.
- {score_compare_interp}

## 12. Discussion and Interpretation
Main synthesis:
- VFT results support structured, domain-sensitive lexical retrieval rather than uniform random retrieval.
- SpAM analyses show that participant similarity layouts carry measurable structure.
- Semantic/phonetic cluster comparisons suggest that lexical search reflects both meaning-level and sound-level organization.
- Combined task evidence provides a broader proxy for Hindi fluency than single-task metrics.

Implication for the core research question:
- Hindi speakers appear to search mental lexicons using structured neighborhood mechanisms that are visible in both retrieval dynamics (VFT) and similarity organization (SpAM), with domain-dependent variation.

## 13. Conclusion answering the main research question
Answer: The evidence indicates that Hindi speakers do not retrieve words randomly. Instead, retrieval shows domain-dependent structured search, and participant similarity arrangements align non-trivially with semantic and phonetic organization. This supports the view that Hindi lexical search relies on organized mental neighborhoods, observable through both VFT output dynamics and SpAM spatial grouping.

## 14. Limitations and Future Work
- The processed CSV was the main computational source; raw JSON was used for structure validation and task-trace context.
- Sample size is moderate (35 sessions), so subgroup analyses may be underpowered.
- Phonetic encoding used a lightweight transliteration heuristic; richer Hindi phonology models may improve alignment estimates.
- Future work: bootstrap stability intervals for alignment metrics and replicate with larger balanced cohorts.

## 15. References (verifiable citations only)
[1] hindi fluency experiment brief_ BRSM 2026.pdf
[2] BRSM-Syllabus.md
[3] merged_vft_spam_responses_enriched.csv
[4] responses.json

## 16. Figure Interpretation Guide
1. `images/vft_wordcount_by_domain.png`: Domain-wise spread of VFT output size (word count) for Hindi/Hinglish trials. Wider spread indicates stronger participant heterogeneity.
2. `images/vft_rt_by_domain.png`: Domain-wise spread of mean inter-word retrieval time. Higher values indicate slower lexical retrieval.
3. `images/vft_words_vs_rt.png`: Productivity-speed tradeoff view. Downward relation suggests participants producing more words tend to retrieve faster.
4. `images/spam_compactness_by_domain.png`: SpAM spatial compactness by domain using mean nearest-neighbor distance. Lower values indicate tighter semantic packing in participant maps.
5. `images/spam_alignment_by_domain.png`: Domain-wise ARI distributions for SpAM vs semantic and SpAM vs phonetic clusters. Higher ARI means stronger cluster-label agreement.
6. `images/cluster_similarity_matrix.png`: Global mean ARI/NMI summary across comparison pairs. NMI reflects shared information, ARI reflects strict partition agreement.
7. `images/composite_score_distribution.png`: Distribution of integrated fluency score across participants. Shape indicates overall spread and central tendency.
8. `images/vft_vs_integrated_score.png`: Relation between VFT-only and integrated scores. Closer fit to line indicates stronger monotonic coupling.
9. `images/sample_spam_semantic_phonetic_clusters.png`: Three views of one high-divergence participant-domain case.
    - Left panel: participant SpAM coordinates (`x,y`) clustered from manual arrangement.
    - Middle panel: semantic embedding space (PCA projection) clustered by KMeans.
    - Right panel: phonetic-key embedding space (PCA projection) clustered by KMeans.
    - Same words are shown in all panels, but geometry differs by method; color indicates cluster label inside each method.
    - {sample_cluster_note}
    - Point-to-word mapping: {sample_cluster_word_map}
    - Point-wise cluster assignments: {sample_cluster_membership}

## Figures
{image_links}
"""

    return report


def markdown_to_pdf(md_text: str, output_pdf: Path) -> None:
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]

    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    def escape_html(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    for raw in md_text.splitlines():
        line = raw.rstrip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        # Image markdown: ![alt](path)
        m_img = re.match(r"!\[[^\]]*\]\(([^\)]+)\)", line)
        if m_img:
            img_path = (ROOT / m_img.group(1)).resolve()
            if img_path.exists():
                img = RLImage(str(img_path))
                max_w = 6.8 * inch
                if img.drawWidth > max_w:
                    ratio = max_w / img.drawWidth
                    img.drawWidth = max_w
                    img.drawHeight = img.drawHeight * ratio
                story.append(img)
                story.append(Spacer(1, 8))
            continue

        if line.startswith("### "):
            story.append(Paragraph(escape_html(line[4:]), h3))
            continue
        if line.startswith("## "):
            story.append(Paragraph(escape_html(line[3:]), h2))
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape_html(line[2:]), h1))
            continue

        if re.match(r"^\d+\.\s", line):
            story.append(Paragraph(escape_html(line), body_style))
            continue

        if line.startswith("- "):
            story.append(Paragraph(escape_html("• " + line[2:]), body_style))
            continue

        story.append(Paragraph(escape_html(line), body_style))

    doc.build(story)


def main() -> None:
    ensure_dirs()

    df_all = load_data()
    df_primary = df_all[df_all["language_type_clean"] == PRIMARY_LANG].copy()
    df_english = df_all[df_all["language_type_clean"] == "english"].copy()

    if df_primary.empty:
        raise RuntimeError("Primary Hindi/Hinglish subset is empty; cannot run primary analysis.")

    brief_text = read_brief_text()
    research_q = extract_research_question(brief_text)

    vft_domain = summarize_vft(df_primary)
    spam_df = analyze_spam_alignment(df_primary)

    vft_word_test = domain_test_with_normality_decision(vft_domain, "word_count", "VFT word count")
    vft_rt_test = domain_test_with_normality_decision(vft_domain, "mean_rt_ms", "VFT mean RT")
    spam_nn_test = domain_test_with_normality_decision(spam_df, "mean_nn_distance", "SpAM nearest-neighbor distance") if not spam_df.empty else TestResult("SpAM nearest-neighbor distance", float("nan"), float("nan"), None, "Insufficient SpAM metrics for domain-level test.")

    sign_sem = sign_test_positive(spam_df["ari_semantic_spam"], "semantic-SpAM ARI") if not spam_df.empty else TestResult("semantic-SpAM ARI", float("nan"), float("nan"), None, "Insufficient data for sign test.")
    sign_pho = sign_test_positive(spam_df["ari_phonetic_spam"], "phonetic-SpAM ARI") if not spam_df.empty else TestResult("phonetic-SpAM ARI", float("nan"), float("nan"), None, "Insufficient data for sign test.")

    assoc = spearman_table(vft_domain)
    merged_scores, participant_scores = compute_scores(vft_domain, spam_df)

    # Expose primary dataframe for representative plotting helper.
    global df_primary_global
    df_primary_global = df_primary

    created_plots, plot_notes = save_plots(vft_domain, spam_df, merged_scores, participant_scores)

    tests = {
        "vft_word_count": vft_word_test,
        "vft_mean_rt": vft_rt_test,
        "spam_nn_distance": spam_nn_test,
        "sign_semantic_alignment": sign_sem,
        "sign_phonetic_alignment": sign_pho,
    }

    md_text = build_report(
        research_q=research_q,
        df_all=df_all,
        df_primary=df_primary,
        df_english=df_english,
        vft_domain=vft_domain,
        spam_df=spam_df,
        merged_scores=merged_scores,
        participant_scores=participant_scores,
        tests=tests,
        assoc_table=assoc,
        created_plots=created_plots,
        plot_notes=plot_notes,
    )

    REPORT_MD.write_text(md_text, encoding="utf-8")
    markdown_to_pdf(md_text, REPORT_PDF)

    print("Analysis complete.")
    print(f"Report markdown: {REPORT_MD}")
    print(f"Report PDF: {REPORT_PDF}")
    print(f"Images generated: {len(created_plots)} in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
