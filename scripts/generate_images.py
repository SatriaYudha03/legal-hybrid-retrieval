"""Generate all visualization images for article/presentation."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns

OUT = "images"
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C_BM25    = "#4C72B0"   # blue
C_PRE     = "#C44E52"   # red
C_FT      = "#DD8452"   # orange
C_PREH    = "#55A868"   # green
C_FINEH   = "#8172B3"   # purple
C_NORM    = "#64B5CD"   # teal accent
C_BG      = "#F8F9FA"
C_GRID    = "#DEE2E6"

SYSTEM_COLORS = [C_BM25, C_PRE, C_FT, C_PREH, C_FINEH]
SYSTEM_LABELS = ["BM25", "IndoSBERT\nPretrained", "IndoSBERT\nFine-tuned",
                 "Pre-Hybrid\n(BM25+Pre+RRF)", "Fine-Hybrid\n(BM25+FT+RRF)"]
SYSTEM_KEYS   = ["bm25", "pretrained", "finetuned", "pre_hybrid", "fine_hybrid"]

METRIC_LABELS = {"recall@5": "Recall@5", "recall@10": "Recall@10",
                 "mrr": "MRR", "ndcg@10": "NDCG@10"}

# ── Load data ────────────────────────────────────────────────────────────────
with open("results/ablation.json") as f:
    abl = json.load(f)
with open("results/normalization_metrics.json") as f:
    norm_data = json.load(f)

tanpa = abl["tanpa_normalisasi"]
dgn   = abl["dengan_normalisasi"]
wilcoxon = abl.get("wilcoxon", {})


# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def pipeline_diagram():
    fig = plt.figure(figsize=(18, 14), facecolor=C_BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18); ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    def box(ax, x, y, w, h, text, fc, ec="#333", lw=1.5, fontsize=9, bold=False):
        r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.12", facecolor=fc,
                           edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(r)
        fw = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fw, zorder=4, wrap=True,
                multialignment="center",
                color="white" if fc not in ("#F2C14E", "#FFF9C4", C_BG) else "#222")

    def arr(ax, x0, y0, x1, y1, color="#555", style="-", lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, linestyle=style),
                    zorder=2)

    def zone(ax, x, y, w, h, label, color):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.2", facecolor=color,
                           edgecolor="#aaa", linewidth=1.2, alpha=0.15, zorder=1)
        ax.add_patch(r)
        ax.text(x + 0.2, y + h - 0.35, label, fontsize=10, fontweight="bold",
                color="#333", zorder=2, va="top")

    # ── Zones ──
    zone(ax,  0.3, 8.8,  6.5, 4.8,  "TAHAP INDEXING  (offline, sekali jalan)", "#4C72B0")
    zone(ax,  0.3, 4.0,  6.5, 4.5,  "TAHAP FINE-TUNING  (Google Colab, GPU)", "#DD8452")
    zone(ax,  7.2, 4.0,  5.4, 4.5,  "NORMALISASI QUERY  (opsional)", "#64B5CD")
    zone(ax,  7.2, 8.8, 10.4, 4.8,  "TAHAP RETRIEVAL  (saat ada query)", "#55A868")

    # ── INDEXING ──
    box(ax, 3.5, 13.1, 4.2, 0.7,
        "3 PDF Undang-Undang\nKetenagakerjaan · Konsumen · ITE", "#F2C14E", bold=True)
    box(ax, 3.5, 12.1, 4.0, 0.65,
        "Ekstraksi Teks PDF\n(pdfplumber · src/ingest.py)", "#4C72B0")
    box(ax, 3.5, 11.1, 4.0, 0.65,
        "Chunking per Pasal → 378 chunks\nsrc/chunk.py · chunks.jsonl", "#4C72B0")

    box(ax, 1.8, 10.0, 2.2, 0.6,  "Tokenisasi\n(rank-bm25)", "#4C72B0")
    box(ax, 1.8,  9.25, 2.2, 0.55, "BM25 Index\nbm25.pkl", "#2C4F8A", bold=True)

    box(ax, 5.2, 10.0, 2.2, 0.6,  "Encode IndoSBERT\nembedding 768-dim", "#C44E52")
    box(ax, 5.2,  9.25, 2.2, 0.55, "FAISS Index (pretrained)\nfaiss.faiss", "#8B2020", bold=True)

    arr(ax, 3.5, 12.75, 3.5, 12.43)
    arr(ax, 3.5, 11.78, 3.5, 11.43)
    arr(ax, 2.4, 11.1,  1.8, 10.3)
    arr(ax, 4.6, 11.1,  5.2, 10.3)
    arr(ax, 1.8, 9.7,   1.8, 9.53)
    arr(ax, 5.2, 9.7,   5.2, 9.53)

    # ── FINE-TUNING ──
    box(ax, 1.5, 7.8, 2.2, 0.6,
        "Generate Pseudo-Query\n(template per domain)", "#DD8452")
    box(ax, 3.5, 7.2, 2.5, 0.6,
        "Filter round-trip\n521 / 1127 pasangan lolos", "#DD8452")
    box(ax, 1.5, 6.6, 2.2, 0.6,
        "Hard Negative Mining\n4 negatif BM25/pasangan", "#DD8452")
    box(ax, 3.5, 6.0, 2.5, 0.65,
        "Fine-Tuning MNRL\nlr=2e-5, batch=32, 3 epoch", "#B05A20", bold=True)
    box(ax, 3.5, 5.1, 2.5, 0.6,
        "IndoSBERT Fine-tuned\nmodels/indosbert-legal-ft/", "#8B2020", bold=True)
    box(ax, 5.5, 4.35, 2.2, 0.55,
        "FAISS Index (fine-tuned)\ndata/index/faiss_ft/", "#8B2020", bold=True)

    arr(ax, 3.5, 11.1,  1.5,  8.1,  color="#DD8452", style="--")
    arr(ax, 1.5,  7.5,  2.8,  7.5)
    arr(ax, 4.2,  7.2,  3.8,  6.9)
    arr(ax, 2.2,  6.6,  2.8,  6.3)
    arr(ax, 3.5,  5.7,  3.5,  5.43)
    arr(ax, 4.5,  5.1,  5.5,  4.63)

    # ── NORMALISASI ──
    box(ax,  9.9, 7.75, 3.2, 0.6,
        "Kamus Akronim Hukum\ndata/normalization/legal_terms.json", "#3A8BA0")
    box(ax,  9.9, 7.0, 3.2, 0.6,
        "src/normalize.py\nEkspansi: asli + terminologi formal", "#3A8BA0")
    box(ax,  9.9, 6.2, 3.2, 0.6,
        "Query Ternormalisasi\n(masuk ke BM25 & FAISS)", "#2A6B80", bold=True)
    arr(ax, 9.9, 7.45,  9.9, 7.3)
    arr(ax, 9.9, 6.7,   9.9, 6.5)

    # ── RETRIEVAL ──
    box(ax, 12.5, 13.1, 3.5, 0.65,
        'QUERY PENGGUNA\n"berapa pesangon kalau dipecat?"',
        "#F2C14E", bold=True)
    box(ax,  9.5, 12.1, 3.0, 0.6,
        "Pencarian BM25\n→ Ranking Lexical", "#4C72B0")
    box(ax, 15.5, 12.1, 3.0, 0.6,
        "Pencarian IndoSBERT\n(cosine via FAISS)\n→ Ranking Semantic", "#C44E52")
    box(ax, 12.5, 11.1, 3.5, 0.6,
        "Reciprocal Rank Fusion\nRRF(d) = Σ 1/(k+rank),  k=60", "#8172B3", bold=True)
    box(ax, 12.5, 10.2, 3.5, 0.6,
        "Top-K Dokumen Hukum", "#F2C14E", bold=True)
    box(ax, 12.5,  9.3, 3.5, 0.65,
        "Evaluasi\nRecall@5/10 · MRR · NDCG@10\n+ Wilcoxon signed-rank", "#937860")

    arr(ax, 12.5, 12.78, 10.5, 12.4)
    arr(ax, 12.5, 12.78, 14.5, 12.4)
    arr(ax, 10.0, 11.8,  11.2, 11.4)
    arr(ax, 15.0, 11.8,  13.8, 11.4)
    arr(ax, 12.5, 10.8,  12.5, 10.5)
    arr(ax, 12.5,  9.9,  12.5,  9.63)

    # Query → Normalisasi → BM25/SEM
    arr(ax, 12.5, 12.78, 9.9, 7.3, color="#3A8BA0", style="--")
    arr(ax, 9.9,  6.2,  10.0, 12.1, color="#3A8BA0", style="--")

    # BM25/FAISS index dotted lines
    ax.annotate("", xy=(9.5, 12.1), xytext=(1.8, 9.53),
                arrowprops=dict(arrowstyle="-|>", color="#4C72B0",
                                lw=1.2, linestyle="dotted", connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(15.5, 12.1), xytext=(5.2, 9.53),
                arrowprops=dict(arrowstyle="-|>", color="#C44E52",
                                lw=1.2, linestyle="dotted", connectionstyle="arc3,rad=0.15"))
    ax.annotate("", xy=(15.5, 12.1), xytext=(5.5, 4.63),
                arrowprops=dict(arrowstyle="-|>", color="#8B2020",
                                lw=1.2, linestyle="dotted", connectionstyle="arc3,rad=0.2"))

    # ── Legend ──
    legend_els = [
        mpatches.Patch(fc="#F2C14E", ec="#555", label="Input / Output"),
        mpatches.Patch(fc="#4C72B0", label="Indexing (Lexical)"),
        mpatches.Patch(fc="#C44E52", label="Indexing (Semantic)"),
        mpatches.Patch(fc="#DD8452", label="Fine-Tuning"),
        mpatches.Patch(fc="#3A8BA0", label="Normalisasi Query"),
        mpatches.Patch(fc="#8172B3", label="Fusion (RRF)"),
        mpatches.Patch(fc="#937860", label="Evaluasi"),
    ]
    ax.legend(handles=legend_els, loc="lower left", bbox_to_anchor=(0.01, 0.01),
              fontsize=8.5, framealpha=0.9, ncol=2)

    ax.set_title("Pipeline Sistem Hybrid Retrieval Dokumen Hukum Indonesia\n"
                 "BM25 + IndoSBERT Fine-tuned + Reciprocal Rank Fusion",
                 fontsize=14, fontweight="bold", pad=8, y=0.99)

    fig.savefig(f"{OUT}/pipeline_diagram.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK pipeline_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ABLATION BAR CHART  (5 systems × 4 metrics, side-by-side tanpa/dgn)
# ══════════════════════════════════════════════════════════════════════════════
def ablation_bar():
    metrics = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    n_sys = len(SYSTEM_KEYS)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor=C_BG)
    fig.suptitle("Perbandingan Metrik Evaluasi — 5 Sistem × 2 Kondisi Normalisasi",
                 fontsize=14, fontweight="bold", y=1.01)

    x = np.arange(n_sys)
    width = 0.35

    for ax, metric in zip(axes, metrics):
        vals_t = [tanpa[k][metric] for k in SYSTEM_KEYS]
        vals_d = [dgn[k][metric]   for k in SYSTEM_KEYS]

        bars1 = ax.bar(x - width/2, vals_t, width, color=SYSTEM_COLORS,
                       alpha=0.6, label="Tanpa Norm.", edgecolor="#333", linewidth=0.7)
        bars2 = ax.bar(x + width/2, vals_d, width, color=SYSTEM_COLORS,
                       alpha=1.0, label="Dengan Norm.", edgecolor="#333", linewidth=0.7,
                       hatch="//")

        # value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SYSTEM_LABELS, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_facecolor(C_BG)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#888", alpha=0.6, label="Tanpa Normalisasi"),
                  Patch(facecolor="#888", alpha=1.0, hatch="//", label="Dengan Normalisasi")]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation_metrics.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK ablation_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. NDCG@10 HEATMAP  (5 systems × 4 metrics, two condition rows)
# ══════════════════════════════════════════════════════════════════════════════
def ablation_heatmap():
    metrics = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    rows, row_labels = [], []
    for k, lbl in zip(SYSTEM_KEYS, ["BM25", "IndoSBERT Pre", "IndoSBERT FT",
                                     "Pre-Hybrid", "Fine-Hybrid"]):
        rows.append([tanpa[k][m] for m in metrics])
        row_labels.append(f"{lbl} (tanpa norm)")
    for k, lbl in zip(SYSTEM_KEYS, ["BM25", "IndoSBERT Pre", "IndoSBERT FT",
                                     "Pre-Hybrid", "Fine-Hybrid"]):
        rows.append([dgn[k][m] for m in metrics])
        row_labels.append(f"{lbl} (dgn norm)")

    data = np.array(rows)
    col_labels = [METRIC_LABELS[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=C_BG)
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.45, vmax=0.85)

    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(rows)):
        for j in range(len(metrics)):
            v = data[i, j]
            fc = "white" if v > 0.68 else "#222"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=9, color=fc, fontweight="bold")

    # divider line between tanpa/dgn groups
    ax.axhline(4.5, color="#555", lw=2, ls="--")
    ax.text(3.55, 4.25, "── Tanpa Normalisasi ──", fontsize=8, color="#555", ha="right")
    ax.text(3.55, 4.75, "── Dengan Normalisasi ──", fontsize=8, color="#555", ha="right")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score", fontsize=10)

    ax.set_title("Heatmap Metrik Ablasi — Semua Sistem & Kondisi Normalisasi",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK ablation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. NORMALIZATION EFFECT CHART (delta per system)
# ══════════════════════════════════════════════════════════════════════════════
def normalization_effect():
    metrics = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    m_labels = [METRIC_LABELS[m] for m in metrics]
    n_sys = len(SYSTEM_KEYS)

    fig, axes = plt.subplots(1, n_sys, figsize=(20, 5), facecolor=C_BG)
    fig.suptitle("Efek Normalisasi Query — Δ Score (Dengan − Tanpa Normalisasi)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, key, label, color in zip(axes, SYSTEM_KEYS, SYSTEM_LABELS, SYSTEM_COLORS):
        deltas = [dgn[key][m] - tanpa[key][m] for m in metrics]
        bar_colors = ["#2ECC71" if d > 0 else "#E74C3C" for d in deltas]
        bars = ax.barh(m_labels, deltas, color=bar_colors, edgecolor="#333", linewidth=0.8)
        ax.axvline(0, color="#333", lw=1.2)

        for bar, d in zip(bars, deltas):
            sign = "+" if d >= 0 else ""
            xpos = d + 0.002 if d >= 0 else d - 0.002
            ha = "left" if d >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height()/2,
                    f"{sign}{d:.4f}", ha=ha, va="center", fontsize=8.5)

        ax.set_title(label.replace("\n", " "), fontsize=9, fontweight="bold",
                     color=color, pad=6)
        ax.set_xlim(-0.12, 0.12)
        ax.set_facecolor(C_BG)
        ax.grid(axis="x", color=C_GRID, linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/normalization_effect.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK normalization_effect.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYSTEM COMPARISON RADAR CHART  (best systems)
# ══════════════════════════════════════════════════════════════════════════════
def radar_chart():
    metrics = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]
    keys_m  = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    systems = [
        ("BM25 (dgn norm)",          dgn["bm25"],      C_BM25),
        ("IndoSBERT FT (dgn norm)",   dgn["finetuned"], C_FT),
        ("Fine-Hybrid (dgn norm)",    dgn["fine_hybrid"],C_FINEH),
        ("BM25 (tanpa norm)",         tanpa["bm25"],    C_BM25),
        ("Fine-Hybrid (tanpa norm)",  tanpa["fine_hybrid"], C_PREH),
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor=C_BG)
    ax.set_facecolor(C_BG)

    for label, vals, color in systems:
        v = [vals[k] for k in keys_m] + [vals[keys_m[0]]]
        ls = "-" if "dgn" in label else "--"
        ax.plot(angles, v, color=color, lw=2, ls=ls, label=label)
        ax.fill(angles, v, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.45, 0.90)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.50", "0.60", "0.70", "0.80"], fontsize=8)
    ax.grid(color=C_GRID, linewidth=0.8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)
    ax.set_title("Radar Chart — Perbandingan Sistem Terbaik", fontsize=13,
                 fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(f"{OUT}/radar_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK radar_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. WILCOXON SIGNIFICANCE TABLE
# ══════════════════════════════════════════════════════════════════════════════
def wilcoxon_table():
    w = abl["wilcoxon"]
    comparisons = list(w.keys())
    stats  = [w[k]["statistic"] for k in comparisons]
    pvals  = [w[k]["p_value"]   for k in comparisons]
    sig    = ["Signifikan (p<0.05)" if p < 0.05 else "Tidak Signifikan" for p in pvals]
    colors_sig = ["#2ECC71" if p < 0.05 else "#E74C3C" for p in pvals]

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=C_BG)
    ax.axis("off")

    col_labels = ["Perbandingan", "Statistik W", "p-value", "Signifikan (α=0.05)"]
    table_data = [[c, f"{s:.1f}", f"{p:.6f}", sg]
                  for c, s, p, sg in zip(comparisons, stats, pvals, sig)]

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 3 and row > 0:
            cell.set_facecolor(colors_sig[row - 1])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        cell.set_edgecolor("#CCC")

    ax.set_title("Hasil Uji Statistik Wilcoxon Signed-Rank",
                 fontsize=13, fontweight="bold", pad=14, y=0.98)

    fig.tight_layout()
    fig.savefig(f"{OUT}/wilcoxon_significance.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK wilcoxon_significance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY COMPARISON (single best-vs-baseline)
# ══════════════════════════════════════════════════════════════════════════════
def summary_comparison():
    metrics  = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    m_labels = [METRIC_LABELS[m] for m in metrics]

    systems = {
        "BM25\n(baseline)":          tanpa["bm25"],
        "IndoSBERT Pre\n(tanpa norm)": tanpa["pretrained"],
        "IndoSBERT FT\n(tanpa norm)": tanpa["finetuned"],
        "Fine-Hybrid\n(tanpa norm)":  tanpa["fine_hybrid"],
        "Fine-Hybrid\n(dgn norm) ★":  dgn["fine_hybrid"],
    }
    colors = [C_BM25, C_PRE, C_FT, C_PREH, C_FINEH]

    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    for i, (label, vals) in enumerate(systems.items()):
        ys = [vals[m] for m in metrics]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, ys, width, label=label,
                      color=colors[i], edgecolor="#333", linewidth=0.7,
                      alpha=0.9 if i < 4 else 1.0)
        if i == 4:  # highlight best
            for bar in bars:
                bar.set_linewidth(2)
                bar.set_edgecolor("#FFD700")

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=12)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Perbandingan Keseluruhan: Sistem Baseline vs Hybrid Terbaik",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # annotate best scores
    best = dgn["fine_hybrid"]
    for j, m in enumerate(metrics):
        ax.text(j + 2*width, best[m] + 0.015,
                f"{best[m]:.3f}", ha="center", fontsize=8,
                color="#FFD700", fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT}/summary_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK summary_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINE-TUNING IMPACT (pretrained vs finetuned, tanpa & dgn norm)
# ══════════════════════════════════════════════════════════════════════════════
def finetune_impact():
    metrics  = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    m_labels = [METRIC_LABELS[m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.2

    data = {
        "Pre (tanpa norm)":  [tanpa["pretrained"][m] for m in metrics],
        "FT  (tanpa norm)":  [tanpa["finetuned"][m]  for m in metrics],
        "Pre (dgn norm)":    [dgn["pretrained"][m]   for m in metrics],
        "FT  (dgn norm)":    [dgn["finetuned"][m]    for m in metrics],
    }
    colors = [C_PRE, C_FT, "#E88", "#FF9955"]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    for i, (label, vals) in enumerate(data.items()):
        offset = (i - 1.5) * w
        ax.bar(x + offset, vals, w, label=label, color=colors[i],
               edgecolor="#333", linewidth=0.8, hatch=hatches[i], alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=12)
    ax.set_ylim(0, 0.9); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Dampak Fine-Tuning IndoSBERT pada Retrieval Hukum",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/finetune_impact.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print("  OK finetune_impact.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    pipeline_diagram()
    ablation_bar()
    ablation_heatmap()
    normalization_effect()
    radar_chart()
    wilcoxon_table()
    summary_comparison()
    finetune_impact()
    print(f"\nDone! All images saved to '{OUT}/'")
