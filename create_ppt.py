"""
Single-page Hindi Fluency Poster Generator (A1 landscape)
Team: CTRL+ALT+DEL
Members: Akshat Kotadia (2025201005), Ankit Chavda (2025201045), Om Mehra (2025201008)

What this version fixes:
- Forces exactly one page with fixed panel layout (no auto overflow to page 2/3)
- Uses fewer, high-value figures from notebook outputs
- Increases readability with larger base font sizes
- Prevents table text overlap using wrapped paragraphs and KeepInFrame safety
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A1, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle


# -----------------------------------------------------------------------------
# Page geometry (A1 landscape) -- unchanged dimension
# -----------------------------------------------------------------------------
PAGE_W, PAGE_H = landscape(A1)  # 841 mm x 594 mm (approx)

MARGIN_X = 12 * mm
MARGIN_Y = 10 * mm
HEADER_H = 28 * mm
FOOTER_H = 12 * mm
COL_GAP = 6 * mm

CONTENT_TOP = PAGE_H - MARGIN_Y - HEADER_H
CONTENT_BOTTOM = MARGIN_Y + FOOTER_H
CONTENT_H = CONTENT_TOP - CONTENT_BOTTOM

CONTENT_W = PAGE_W - 2 * MARGIN_X
COL_W = (CONTENT_W - 2 * COL_GAP) / 3


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
DEEP_BLUE = colors.HexColor("#0D1B2A")
MID_BLUE = colors.HexColor("#1B3A5C")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E07B39")
LIGHT_BG = colors.HexColor("#F3F7FB")
WHITE = colors.white
TEXT_DARK = colors.HexColor("#1A1A2E")
TEXT_MID = colors.HexColor("#2C3E50")
LINE_SOFT = colors.HexColor("#B7C6D8")

PANEL_PAD = 4 * mm
PANEL_TITLE_H = 9 * mm
PANEL_GAP_Y = 4 * mm
INNER_W = COL_W - 2 * PANEL_PAD - 2 * mm


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, "Ankit Chavda", "images", "img")


def image_path(name: str) -> str:
    return os.path.join(IMG_DIR, name)


# -----------------------------------------------------------------------------
# Styles
# -----------------------------------------------------------------------------
styles = getSampleStyleSheet()


def _style(name: str, parent: str = "Normal", **kwargs) -> ParagraphStyle:
    return ParagraphStyle(name=name, parent=styles[parent], **kwargs)


S_BODY = _style(
    "Body",
    fontName="Helvetica",
    fontSize=10.2,
    leading=13.5,
    textColor=TEXT_DARK,
    alignment=TA_LEFT,
    spaceAfter=1.8 * mm,
)

S_BULLET = _style(
    "Bullet",
    fontName="Helvetica",
    fontSize=9.4,
    leading=12.4,
    textColor=TEXT_DARK,
    leftIndent=3 * mm,
    firstLineIndent=-2 * mm,
    spaceAfter=1.1 * mm,
)

S_NOTE = _style(
    "Note",
    fontName="Helvetica-Oblique",
    fontSize=8.4,
    leading=10.8,
    textColor=TEXT_MID,
)

S_CAPTION = _style(
    "Caption",
    fontName="Helvetica-Oblique",
    fontSize=8.4,
    leading=10.8,
    textColor=TEXT_MID,
    alignment=TA_CENTER,
)

S_TABLE_HEAD = _style(
    "TH",
    fontName="Helvetica-Bold",
    fontSize=8.8,
    leading=11.4,
    textColor=WHITE,
    alignment=TA_CENTER,
)

S_TABLE_CELL = _style(
    "TC",
    fontName="Helvetica",
    fontSize=8.2,
    leading=10.8,
    textColor=TEXT_DARK,
    alignment=TA_LEFT,
)

S_TABLE_CELL_C = _style(
    "TCC",
    fontName="Helvetica",
    fontSize=8.0,
    leading=10.6,
    textColor=TEXT_DARK,
    alignment=TA_CENTER,
)


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet> {text}", S_BULLET)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def fit_image(path: str, max_w: float, max_h: float):
    if not os.path.exists(path):
        return None
    try:
        with PILImage.open(path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return None
        ratio = min(max_w / w, max_h / h)
        return w * ratio, h * ratio
    except Exception:
        return None



def build_image_block(path: str, max_w: float, max_h: float, caption: str, interpretation: str | None = None):
    flow = []
    size = fit_image(path, max_w, max_h)
    if size is None:
        flow.append(Paragraph(f"[Missing image: {os.path.basename(path)}]", S_NOTE))
    else:
        iw, ih = size
        flow.append(Image(path, width=iw, height=ih))
    flow.append(Spacer(1, 1 * mm))
    flow.append(Paragraph(caption, S_CAPTION))
    if interpretation:
        flow.append(Spacer(1, 0.6 * mm))
        flow.append(Paragraph(f"<b>Interpretation:</b> {interpretation}", S_NOTE))
    return flow


@dataclass
class Panel:
    title: str
    height: float
    title_color: colors.Color
    body: list



def draw_header(c: canvas.Canvas):
    c.saveState()

    c.setFillColor(DEEP_BLUE)
    c.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)

    c.setFillColor(ORANGE)
    c.rect(0, PAGE_H - 2.5 * mm, PAGE_W, 2.5 * mm, fill=1, stroke=0)

    c.setFillColor(TEAL)
    c.rect(0, PAGE_H - HEADER_H, 5 * mm, HEADER_H, fill=1, stroke=0)

    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(
        PAGE_W / 2,
        PAGE_H - 11 * mm,
        "Hindi Verbal Fluency: VFT + SpAM Mental Lexicon Analysis",
    )

    c.setFont("Helvetica", 10.5)
    c.setFillColor(colors.HexColor("#D8E6F6"))
    c.drawCentredString(
        PAGE_W / 2,
        PAGE_H - 17.8 * mm,
        "Team CTRL+ALT+DEL | Akshat Kotadia (2025201005) | Ankit Chavda (2025201045) | Om Mehra (2025201008)",
    )

    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 23.3 * mm, "BRSM 2026 | IIIT Hyderabad")

    c.restoreState()



def draw_footer(c: canvas.Canvas):
    c.saveState()
    c.setFillColor(DEEP_BLUE)
    c.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, FOOTER_H - 1.5 * mm, PAGE_W, 1.5 * mm, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#D8E6F6"))
    c.setFont("Helvetica", 7.8)
    c.drawString(
        MARGIN_X,
        3.2 * mm,
        "VFT=Verbal Fluency Task | SpAM=Spatial Arrangement Method | IRT=Inter-Response Time | LME=Linear Mixed Effects",
    )
    c.restoreState()



def draw_panel(c: canvas.Canvas, x: float, y_top: float, w: float, panel: Panel):
    """Draw one fixed-height panel. y_top is top edge in canvas coordinates."""
    h = panel.height
    y = y_top - h

    # Panel body
    c.saveState()
    c.setFillColor(WHITE)
    c.setStrokeColor(LINE_SOFT)
    c.setLineWidth(0.9)
    c.roundRect(x, y, w, h, 2.2 * mm, fill=1, stroke=1)

    # Panel title strip
    c.setFillColor(panel.title_color)
    c.roundRect(x, y + h - PANEL_TITLE_H, w, PANEL_TITLE_H, 2.2 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 10.2)
    c.drawString(x + 3 * mm, y + h - PANEL_TITLE_H + 2.7 * mm, panel.title)
    c.restoreState()

    # Flowable region
    ix = x + PANEL_PAD
    iy = y + PANEL_PAD
    iw = w - 2 * PANEL_PAD
    ih = h - PANEL_TITLE_H - 1.2 * mm - PANEL_PAD

    kif = KeepInFrame(iw, ih, panel.body, mode="shrink")
    fw, fh = kif.wrapOn(c, iw, ih)
    # Top align
    kif.drawOn(c, ix, iy + max(0, ih - fh))


# -----------------------------------------------------------------------------
# Column content
# -----------------------------------------------------------------------------


def col1_panels() -> list[Panel]:
    # Introduction panel
    intro_table = Table(
        [
            [Paragraph("<b>Participants</b>", S_TABLE_CELL), Paragraph("35 Hindi-English bilinguals", S_TABLE_CELL)],
            [Paragraph("<b>Tasks</b>", S_TABLE_CELL), Paragraph("VFT (60 sec) + SpAM (2D arrangement)", S_TABLE_CELL)],
            [Paragraph("<b>Domains</b>", S_TABLE_CELL), Paragraph("Animals, Foods, Colours, Body-parts", S_TABLE_CELL)],
            [Paragraph("<b>Core dataset</b>", S_TABLE_CELL), Paragraph("723 Hindi/Hinglish responses", S_TABLE_CELL)],
            [Paragraph("<b>Important note</b>", S_TABLE_CELL), Paragraph("Colours has low N (n=4), mainly descriptive", S_TABLE_CELL)],
        ],
        colWidths=[INNER_W * 0.36, INNER_W * 0.64],
    )
    intro_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, LINE_SOFT),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    intro_body = [
        Paragraph(
            "This study checks how Hindi speakers search words in memory using timed verbal recall and spatial similarity mapping.",
            S_BODY,
        ),
        bullet("VFT captures retrieval speed and semantic switching behaviour."),
        bullet("SpAM captures perceived semantic geometry in 2D space."),
        bullet("Prior literature: Troyer (clustering), Dautriche (word-form), Kumar (phonological shift)."),
        bullet("Main goal: compare semantic vs phonetic organisation in Hindi lexical search."),
        Spacer(1, 1 * mm),
        intro_table,
    ]

    rq_body = [
        Paragraph("<b>Research Questions (RQ1-RQ7)</b>", S_BODY),
        bullet("RQ1: Do productivity and retrieval speed differ across domains?"),
        bullet("RQ2: Which participant variables (like confidence) relate to fluency outcomes?"),
        bullet("RQ3: Does SpAM compactness differ across domains?"),
        bullet("RQ4: Do participant SpAM clusters align with semantic and phonetic models?"),
        bullet("RQ5: Is integrated VFT+SpAM score better than VFT-only score?"),
        bullet("RQ6: Does phonological similarity increase across retrieval order?"),
        bullet("RQ7: Does phonological similarity predict higher productivity?"),
        Spacer(1, 0.8 * mm),
        Paragraph("<b>Hypotheses covered (H1-H13)</b>", S_BODY),
    ]

    hyp_table = Table(
        [
            [Paragraph("<b>H</b>", S_TABLE_HEAD), Paragraph("<b>Statement</b>", S_TABLE_HEAD)],
            [Paragraph("H1", S_TABLE_CELL_C), Paragraph("Between-cluster IRT > within-cluster IRT", S_TABLE_CELL)],
            [Paragraph("H2", S_TABLE_CELL_C), Paragraph("IRT increases with serial position", S_TABLE_CELL)],
            [Paragraph("H3", S_TABLE_CELL_C), Paragraph("Word count differs by domain", S_TABLE_CELL)],
            [Paragraph("H4", S_TABLE_CELL_C), Paragraph("IRT differs by domain", S_TABLE_CELL)],
            [Paragraph("H5", S_TABLE_CELL_C), Paragraph("Confidence positively predicts total words", S_TABLE_CELL)],
            [Paragraph("H6", S_TABLE_CELL_C), Paragraph("Confidence negatively predicts mean IRT", S_TABLE_CELL)],
            [Paragraph("H7", S_TABLE_CELL_C), Paragraph("Confidence positively predicts cluster size", S_TABLE_CELL)],
            [Paragraph("H8", S_TABLE_CELL_C), Paragraph("SpAM compactness differs by domain", S_TABLE_CELL)],
            [Paragraph("H9", S_TABLE_CELL_C), Paragraph("SpAM-semantic ARI is above chance", S_TABLE_CELL)],
            [Paragraph("H10", S_TABLE_CELL_C), Paragraph("SpAM-phonetic ARI is above chance", S_TABLE_CELL)],
            [Paragraph("H11", S_TABLE_CELL_C), Paragraph("Phonological-over-semantic trend increases over order", S_TABLE_CELL)],
            [Paragraph("H12", S_TABLE_CELL_C), Paragraph("Mean phonological similarity predicts productivity", S_TABLE_CELL)],
            [Paragraph("H13", S_TABLE_CELL_C), Paragraph("Integrated-score correlation differs from VFT-only", S_TABLE_CELL)],
        ],
        colWidths=[INNER_W * 0.12, INNER_W * 0.88],
    )
    hyp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE_SOFT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 1.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
            ]
        )
    )
    rq_body.append(hyp_table)

    # Methodology panel
    method_body = [
        Paragraph("Analysis pipeline used in notebook implementation:", S_BODY),
        bullet("Data merge + filtering to Hindi/Hinglish rows."),
        bullet("Compute total words, IRT, cluster size, switch count."),
        bullet("Shapiro-Wilk normality checks before inferential tests."),
        bullet("H1: Welch t-test for within vs between cluster IRT."),
        bullet("H2/H11: mixed-effects models over serial position."),
        bullet("H3/H4/H8: Kruskal-Wallis for domain-wise comparison."),
        bullet("H5-H7/H12: Spearman one-tailed correlations."),
        bullet("H9/H10: ARI, sign test, and permutation checks."),
        bullet("H13: Steiger test for dependent correlations."),
    ]

    # Models panel
    model_table = Table(
        [
            [Paragraph("<b>Model / Method</b>", S_TABLE_HEAD), Paragraph("<b>Purpose</b>", S_TABLE_HEAD)],
            [Paragraph("paraphrase-multilingual-MiniLM-L12-v2", S_TABLE_CELL), Paragraph("Semantic embedding for Hindi + Hinglish words (384-d)", S_TABLE_CELL)],
            [Paragraph("Phonetic key representation", S_TABLE_CELL), Paragraph("Sound-similarity signal for phonetic clustering/alignment", S_TABLE_CELL)],
            [Paragraph("K-Means + Agglomerative + HDBSCAN", S_TABLE_CELL), Paragraph("Independent clustering and robustness check", S_TABLE_CELL)],
            [Paragraph("ARI / NMI / Sign test", S_TABLE_CELL), Paragraph("Quantify semantic and phonetic cluster alignment", S_TABLE_CELL)],
        ],
        colWidths=[INNER_W * 0.45, INNER_W * 0.55],
    )
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE_SOFT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    model_body = [
        Paragraph("Semantic and phonetic similarity both included as required.", S_BODY),
        model_table,
        Spacer(1, 0.8 * mm),
        Paragraph("Interpretation: Multilingual transformer gives common space for Devanagari and Roman-script Hindi words.", S_NOTE),
    ]

    # Method figure panel
    fig_body = build_image_block(
        image_path("graph_9_4_umap_clusters.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=54 * mm,
        caption="Fig: UMAP view of embedding clusters by domain (semantic structure).",
        interpretation="Dense central groups show shared semantic neighbourhood; spread regions indicate domain-specific vocabulary pockets.",
    )

    return [
        Panel("INTRODUCTION + DATASET", 106 * mm, MID_BLUE, intro_body),
        Panel("RESEARCH QUESTIONS + HYPOTHESES", 142 * mm, TEAL, rq_body),
        Panel("METHODOLOGY", 112 * mm, MID_BLUE, method_body),
        Panel("TRANSFORMERS / MODELS USED", 90 * mm, TEAL, model_body),
        Panel("EMBEDDING MAP", 64 * mm, MID_BLUE, fig_body),
    ]



def _results_table() -> Table:
    # Wrapped cells to avoid overlap
    rows = [
        [
            Paragraph("<b>RQ</b>", S_TABLE_HEAD),
            Paragraph("<b>H</b>", S_TABLE_HEAD),
            Paragraph("<b>Test</b>", S_TABLE_HEAD),
            Paragraph("<b>Main Result</b>", S_TABLE_HEAD),
            Paragraph("<b>Decision</b>", S_TABLE_HEAD),
        ],
        [Paragraph("Found.", S_TABLE_CELL_C), Paragraph("H1", S_TABLE_CELL_C), Paragraph("Welch t", S_TABLE_CELL_C), Paragraph("Within 5841.9 vs between 10960.3 ms; t=9.2995; d=1.1206; p<0.0001", S_TABLE_CELL), Paragraph("Supported", S_TABLE_CELL_C)],
        [Paragraph("Found.", S_TABLE_CELL_C), Paragraph("H2", S_TABLE_CELL_C), Paragraph("MixedLM", S_TABLE_CELL_C), Paragraph("Serial-position slope not positive in any domain (0/4 domains support)", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ1", S_TABLE_CELL_C), Paragraph("H3", S_TABLE_CELL_C), Paragraph("Kruskal", S_TABLE_CELL_C), Paragraph("Word count by domain: H=3.2639, p=0.1956", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ1", S_TABLE_CELL_C), Paragraph("H4", S_TABLE_CELL_C), Paragraph("Kruskal", S_TABLE_CELL_C), Paragraph("IRT by domain: H=3.9028, p=0.1421", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ2", S_TABLE_CELL_C), Paragraph("H5", S_TABLE_CELL_C), Paragraph("Spearman", S_TABLE_CELL_C), Paragraph("Confidence vs total words: rho=-0.3951, p(one)=0.9906", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ2", S_TABLE_CELL_C), Paragraph("H6", S_TABLE_CELL_C), Paragraph("Spearman", S_TABLE_CELL_C), Paragraph("Confidence vs mean IRT: rho=+0.2755, p(one)=0.9454", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ2", S_TABLE_CELL_C), Paragraph("H7", S_TABLE_CELL_C), Paragraph("Spearman", S_TABLE_CELL_C), Paragraph("Confidence vs cluster size: rho=-0.2259, p(one)=0.8969", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ3", S_TABLE_CELL_C), Paragraph("H8", S_TABLE_CELL_C), Paragraph("Kruskal", S_TABLE_CELL_C), Paragraph("SpAM compactness: H=3.4451, p=0.3280", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ4", S_TABLE_CELL_C), Paragraph("H9", S_TABLE_CELL_C), Paragraph("ARI + sign", S_TABLE_CELL_C), Paragraph("Semantic ARI mean=0.1765; sign p<0.0001; permutation direction weak", S_TABLE_CELL), Paragraph("Mixed / weak", S_TABLE_CELL_C)],
        [Paragraph("RQ4", S_TABLE_CELL_C), Paragraph("H10", S_TABLE_CELL_C), Paragraph("ARI + sign", S_TABLE_CELL_C), Paragraph("Phonetic ARI mean=0.0745; sign p=0.0662", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ6", S_TABLE_CELL_C), Paragraph("H11", S_TABLE_CELL_C), Paragraph("LME", S_TABLE_CELL_C), Paragraph("type×position beta=-0.004144, p=0.1028", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ7", S_TABLE_CELL_C), Paragraph("H12", S_TABLE_CELL_C), Paragraph("Spearman", S_TABLE_CELL_C), Paragraph("Phonological similarity vs productivity: rho=+0.114, p=0.2570", S_TABLE_CELL), Paragraph("Not supported", S_TABLE_CELL_C)],
        [Paragraph("RQ5", S_TABLE_CELL_C), Paragraph("H13", S_TABLE_CELL_C), Paragraph("Steiger Z", S_TABLE_CELL_C), Paragraph("VFT rho=-0.461 vs integrated rho=-0.323; z=0.724, p=0.4689", S_TABLE_CELL), Paragraph("No diff.", S_TABLE_CELL_C)],
    ]

    col_widths = [INNER_W * 0.08, INNER_W * 0.07, INNER_W * 0.16, INNER_W * 0.54, INNER_W * 0.15]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DEEP_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE_SOFT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )

    return t



def col2_panels() -> list[Panel]:
    table_panel_body = [
        Paragraph(
            "All RQ/H values below are from notebook outputs only (no hallucinated statistics).",
            S_BODY,
        ),
        Spacer(1, 1 * mm),
        _results_table(),
        Spacer(1, 0.5 * mm),
        Paragraph("Note: Colours domain low N; interpret domain-level tests carefully.", S_NOTE),
    ]

    key_body = [
        Paragraph("<b>Main findings (simple summary):</b>", S_BODY),
        bullet("Strong evidence only for semantic cluster-boundary cost (H1)."),
        bullet("Domain effect, confidence effect, phonological facilitation mostly not supported."),
        bullet("Integrated score did not beat VFT-only score statistically (H13)."),
    ]

    fig_h1 = build_image_block(
        image_path("fig_moduleA_h1_cluster_comparison.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=51 * mm,
        caption="Fig: H1 cluster boundary effect (largest and clearest result).",
        interpretation="Most participant points stay above diagonal and violin distributions are clearly separated, so cluster-switch cost is strong.",
    )

    fig_rq2 = build_image_block(
        image_path("fig_moduleC_confidence_diagnostics.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=50 * mm,
        caption="Fig: Confidence diagnostics show ceiling compression.",
        interpretation="Many participants are at high confidence band, so correlation signal becomes unstable and opposite-direction trends can appear.",
    )

    fig_spam = build_image_block(
        image_path("fig_moduleD_h9_h10_alignment.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=50 * mm,
        caption="Fig: Semantic vs phonetic ARI alignment summary.",
        interpretation="Semantic ARI is generally above phonetic ARI, indicating meaning-based structure is stronger than sound-based structure here.",
    )

    return [
        Panel("RESULTS TABLE (RQ1-RQ7, H1-H13)", 222 * mm, MID_BLUE, table_panel_body),
        Panel("KEY FINDINGS", 62 * mm, TEAL, key_body),
        Panel("IMPORTANT FIGURE 1", 80 * mm, MID_BLUE, fig_h1),
        Panel("IMPORTANT FIGURE 2", 80 * mm, TEAL, fig_rq2),
        Panel("IMPORTANT FIGURE 3", 70 * mm, MID_BLUE, fig_spam),
    ]



def col3_panels() -> list[Panel]:
    sem_phon_table = Table(
        [
            [Paragraph("<b>Aspect</b>", S_TABLE_HEAD), Paragraph("<b>What We Saw</b>", S_TABLE_HEAD)],
            [Paragraph("Semantic", S_TABLE_CELL), Paragraph("H9 shows positive semantic ARI trend, but confirmatory directional evidence is limited.", S_TABLE_CELL)],
            [Paragraph("Phonetic", S_TABLE_CELL), Paragraph("H10 and H12 do not support strong phonetic facilitation in this sample.", S_TABLE_CELL)],
            [Paragraph("Serial trend", S_TABLE_CELL), Paragraph("H11 interaction not significant (p=0.1028).", S_TABLE_CELL)],
            [Paragraph("Composite score", S_TABLE_CELL), Paragraph("H13: integrated score does not significantly outperform VFT-only.", S_TABLE_CELL)],
        ],
        colWidths=[INNER_W * 0.28, INNER_W * 0.72],
    )
    sem_phon_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE_SOFT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )

    sem_phon_body = [
        Paragraph("Semantic and phonetic similarity coverage as required:", S_BODY),
        sem_phon_table,
    ]

    fig_phono = build_image_block(
        image_path("graph_12_1_h11_dual_similarity.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=54 * mm,
        caption="Fig: H11/H12 phonological results (no strong facilitation pattern).",
        interpretation="Curves do not show a reliable semantic-to-phonological handoff; uncertainty remains high.",
    )

    fig_h13 = build_image_block(
        image_path("graph_13_1_composite_vs_vft_confidence.png"),
        max_w=COL_W - 2 * PANEL_PAD - 2 * mm,
        max_h=54 * mm,
        caption="Fig: H13 comparison of VFT-only vs integrated score.",
        interpretation="Both scores show similar trend with confidence, and Steiger test says difference is not significant.",
    )

    discussion_body = [
        Paragraph("<b>Interpretation in simple Indian-English:</b>", S_BODY),
        bullet("In this sample, semantic clustering effect is very clear and strong."),
        bullet("Most other hypotheses are not getting support statistically."),
        bullet("This can happen due to small sample size and restricted confidence scale."),
        bullet("So we should report null results honestly, and not over-claim."),
        Spacer(1, 0.5 * mm),
        Paragraph("<b>Limitations:</b>", S_BODY),
        bullet("N=35 only; colours domain has low participation."),
        bullet("Confidence self-score has ceiling effect."),
        bullet("Some extended analysis blocks in notebook have inconsistent variants; we used core hypothesis outputs."),
        Spacer(1, 0.5 * mm),
        Paragraph("<b>Future work:</b>", S_BODY),
        bullet("Larger and balanced participant pool."),
        bullet("Objective Hindi proficiency test instead of only self-confidence."),
        bullet("More robust phonological feature design for Hindi words."),
    ]

    conclusion_body = [
        Paragraph(
            "<b>Final conclusion:</b> Hindi fluency retrieval shows strong semantic clustering (H1), but domain effect, confidence effect, and phonological facilitation are mostly not supported in this dataset.",
            S_BODY,
        )
    ]

    return [
        Panel("SEMANTIC VS PHONETIC SUMMARY", 104 * mm, MID_BLUE, sem_phon_body),
        Panel("IMPORTANT FIGURE 4", 90 * mm, TEAL, fig_phono),
        Panel("IMPORTANT FIGURE 5", 90 * mm, MID_BLUE, fig_h13),
        Panel("DISCUSSION", 150 * mm, TEAL, discussion_body),
        Panel("CONCLUSION", 80 * mm, MID_BLUE, conclusion_body),
    ]


# -----------------------------------------------------------------------------
# Build (single page)
# -----------------------------------------------------------------------------


def draw_column(c: canvas.Canvas, x: float, panels: list[Panel]):
    y = CONTENT_TOP
    for i, p in enumerate(panels):
        draw_panel(c, x, y, COL_W, p)
        y -= p.height
        if i < len(panels) - 1:
            y -= PANEL_GAP_Y



def build(output_pdf: str | None = None):
    if output_pdf is None:
        output_pdf = os.path.join(BASE_DIR, "hindi_fluency_poster.pdf")

    c = canvas.Canvas(output_pdf, pagesize=(PAGE_W, PAGE_H))

    draw_header(c)
    draw_footer(c)

    x1 = MARGIN_X
    x2 = x1 + COL_W + COL_GAP
    x3 = x2 + COL_W + COL_GAP

    draw_column(c, x1, col1_panels())
    draw_column(c, x2, col2_panels())
    draw_column(c, x3, col3_panels())

    # single page only
    c.showPage()
    c.save()

    print(f"DONE: Single-page poster saved -> {output_pdf}")
    print(f"Page size (A1 landscape): {PAGE_W:.2f} x {PAGE_H:.2f} points")


if __name__ == "__main__":
    build()
