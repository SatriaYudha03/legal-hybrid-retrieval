"""Clean redesign of pipeline diagram - 22x14 canvas."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

C_BG = "#FAFAFA"

fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
ax  = fig.add_axes([0.01, 0.01, 0.98, 0.92])
ax.set_xlim(0, 22); ax.set_ylim(0, 14)
ax.axis("off"); ax.set_facecolor(C_BG)


def zone(x, y, w, h, label, color, alpha=0.12):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                       facecolor=color, edgecolor="#888", linewidth=1.5,
                       alpha=alpha, zorder=1)
    ax.add_patch(r)
    ax.text(x + 0.22, y + h - 0.25, label, fontsize=10, fontweight="bold",
            color="#333", zorder=2, va="top")


def box(x, y, w, h, line1, line2="", fc="#4C72B0", tc="white",
        dashed=False, fs1=9.5, fs2=8.5):
    ls = (0, (4, 3)) if dashed else "solid"
    ec = "#3A8BA0" if dashed else "white"
    lw = 2.2 if dashed else 1.6
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.14", facecolor=fc,
                       edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(r)
    dy = 0.17 if line2 else 0
    ax.text(x, y + dy, line1, ha="center", va="center", fontsize=fs1,
            fontweight="bold", color=tc, zorder=4)
    if line2:
        ax.text(x, y - 0.19, line2, ha="center", va="center", fontsize=fs2,
                color=tc, zorder=4, alpha=0.92, multialignment="center")


def arrv(x, y0, y1, color="#555", lw=1.9):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=5)


def fork(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.9), zorder=5)


def arrdiag(x0, y0, x1, y1, color="#888", lw=1.6, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=(0, (5, 3)),
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)


def note(x, y, text, ec="#888", fc="white"):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
            color=ec, fontstyle="italic",
            bbox=dict(fc=fc, ec=ec, pad=2.5, boxstyle="round,pad=0.3"),
            zorder=6)


# ── ZONES ────────────────────────────────────────────────────────────────────
zone(0.25, 0.25, 9.25, 13.5, "FASE OFFLINE  (dijalankan sekali)", "#4C72B0")
zone(0.45, 6.85, 8.85,  6.6, "INDEXING", "#4C72B0")
zone(0.45, 0.45, 8.85,  6.2, "FINE-TUNING  (Google Colab GPU)", "#DD8452")
zone(10.0, 0.25, 11.75, 13.5, "FASE ONLINE  (saat ada query)", "#55A868")
zone(10.2, 8.2,  11.35,  5.3, "RETRIEVAL", "#55A868")
zone(10.2, 0.45, 11.35,  7.5, "EVALUASI & HASIL", "#937860")

# ═══════════════════════════════════════════════════════════════════════════
# OFFLINE — INDEXING
# ═══════════════════════════════════════════════════════════════════════════
box(4.9, 13.05, 6.0, 0.75,
    "3 PDF Undang-Undang",
    "Ketenagakerjaan  |  Konsumen  |  ITE",
    fc="#F2C14E", tc="#222", fs1=10.5, fs2=9)

arrv(4.9, 12.68, 12.27)
box(4.9, 11.90, 5.5, 0.65,
    "Ekstraksi Teks PDF",
    "pdfplumber  |  src/ingest.py",
    fc="#4C72B0", fs2=8.5)

arrv(4.9, 11.57, 11.17)
box(4.9, 10.80, 5.5, 0.65,
    "Chunking per Pasal  ->  378 chunks",
    "src/chunk.py  |  chunks.jsonl",
    fc="#4C72B0", fs2=8.5)

# fork
fork(4.9, 10.47, 2.8, 9.87)
fork(4.9, 10.47, 7.0, 9.87)

# BM25 branch (left)
box(2.8, 9.50, 3.0, 0.65,  "Tokenisasi",       "rank-bm25",         fc="#2C4F8A", fs2=8.5)
arrv(2.8, 9.18, 8.57)
box(2.8, 8.20, 3.0, 0.70,  "BM25 Index",       "bm25.pkl",          fc="#1A336A", fs1=10.5)

# Semantic branch (right)
box(7.0, 9.50, 3.0, 0.65,  "Encode IndoSBERT", "embedding 768-dim", fc="#8B2020", fs2=8.5)
arrv(7.0, 9.18, 8.57)
box(7.0, 8.20, 3.0, 0.70,  "FAISS Index",      "(pretrained)",      fc="#6B0000", fs1=10.5)

# ═══════════════════════════════════════════════════════════════════════════
# OFFLINE — FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════
# dashed arrow from Chunking into fine-tuning zone
ax.annotate("", xy=(4.9, 6.4), xytext=(4.9, 10.47),
            arrowprops=dict(arrowstyle="-|>", color="#DD8452", lw=1.7,
                            linestyle=(0, (4, 3))), zorder=5)

box(4.9, 6.03, 5.8, 0.65,
    "Generate Pseudo-Query  (1127 kandidat)",
    "template per domain  |  scripts/07_build_synthetic_queries.py",
    fc="#DD8452", fs1=9, fs2=8)
arrv(4.9, 5.70, 5.27)

box(4.9, 4.90, 5.8, 0.65,
    "Filter Round-trip  ->  521 pasangan lolos",
    "query diregenerasi -> harus kembali ke dokumen asal",
    fc="#C04A10", fs1=9, fs2=8)
arrv(4.9, 4.57, 4.17)

box(4.9, 3.80, 5.8, 0.65,
    "Hard Negative Mining",
    "4 negatif BM25 per pasangan  |  meningkatkan diskriminasi model",
    fc="#DD8452", fs1=9, fs2=8)
arrv(4.9, 3.47, 3.07)

box(4.9, 2.70, 5.8, 0.65,
    "Fine-Tuning MNRL",
    "indo-sentence-bert-base  |  lr=2e-5  |  batch=32  |  3 epoch  |  FP16",
    fc="#B05A20", fs1=9, fs2=8)
arrv(4.9, 2.37, 1.92)

box(3.0, 1.55, 3.0, 0.65,  "IndoSBERT FT",   "models/indosbert-legal-ft/", fc="#7A1A00", fs2=8)
ax.annotate("", xy=(5.65, 1.55), xytext=(4.5, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6), zorder=5)
box(7.0, 1.55, 3.0, 0.65,  "FAISS FT Index", "data/index/faiss_ft/",       fc="#7A1A00", fs2=8)

# ═══════════════════════════════════════════════════════════════════════════
# ONLINE — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════
box(15.9, 13.05, 6.0, 0.75,
    "QUERY PENGGUNA",
    '"berapa pesangon kalau dipecat?"',
    fc="#F2C14E", tc="#222", fs1=11, fs2=9.5)

arrv(15.9, 12.68, 12.25)

box(15.9, 11.87, 6.0, 0.65,
    "Normalisasi Query  (opsional)",
    "ekspansi akronim hukum  |  src/normalize.py  |  legal_terms.json",
    fc="#3A8BA0", fs1=9.5, fs2=8.5, dashed=True)

fork(15.9, 11.54, 13.0, 10.90)
fork(15.9, 11.54, 18.8, 10.90)

box(13.0, 10.53, 3.8, 0.65,  "Pencarian BM25",      "Ranking Leksikal",  fc="#2C4F8A")
box(18.8, 10.53, 3.8, 0.65,  "Pencarian IndoSBERT", "Ranking Semantik",  fc="#8B2020")

fork(13.0, 10.20, 15.9, 9.63)
fork(18.8, 10.20, 15.9, 9.63)

box(15.9, 9.25, 6.0, 0.65,
    "Reciprocal Rank Fusion  (RRF)",
    "RRF(d) = sum  1 / (k + rank),   k = 60",
    fc="#8172B3", fs2=9)
arrv(15.9, 8.92, 8.50)

box(15.9, 8.13, 6.0, 0.65,
    "Top-K Dokumen Hukum",
    "pasal relevan dikembalikan ke pengguna",
    fc="#F2C14E", tc="#222", fs2=8.5)

# ═══════════════════════════════════════════════════════════════════════════
# ONLINE — EVALUASI
# ═══════════════════════════════════════════════════════════════════════════
arrv(15.9, 7.80, 7.35)

box(15.9, 6.98, 6.0, 0.65,
    "Evaluasi  (Ablasi 10 Konfigurasi)",
    "Recall@5/10  |  MRR  |  NDCG@10  |  Wilcoxon signed-rank (alpha=0.05)",
    fc="#937860", fs1=9.5, fs2=8)

box(15.9, 5.37, 6.0, 1.65,
    "5 Sistem  x  2 Kondisi Normalisasi",
    "BM25\nIndoSBERT Pretrained  |  IndoSBERT Fine-tuned\nPre-Hybrid (BM25+Pre+RRF)\nFine-Hybrid (BM25+FT+RRF)",
    fc="#4A6A8A", fs1=9.5, fs2=8.5)

box(15.9, 3.37, 6.0, 1.25,
    "Hasil Terbaik:  Fine-Hybrid + Normalisasi",
    "Recall@5: 0.696  |  Recall@10: 0.832\nMRR: 0.677  |  NDCG@10: 0.651",
    fc="#2C7A4B", fs1=10, fs2=9)

box(15.9, 1.82, 6.0, 0.70,
    "Peningkatan vs BM25 Baseline",
    "NDCG@10: +31.5%  |  MRR: +28.9%  |  Recall@10: +33.7%",
    fc="#1A5A3A", fs2=8.5)

box(15.9, 0.88, 6.0, 0.60,
    "Uji Wilcoxon: Fine-tuning signifikan (p<0.05)",
    "normalisasi efektif pada BM25 dan Fine-Hybrid",
    fc="#2C4F5A", fs1=9, fs2=8)

# ═══════════════════════════════════════════════════════════════════════════
# CONNECTING ARROWS  (offline artifacts -> online components)
# ═══════════════════════════════════════════════════════════════════════════
# BM25 Index -> BM25 Search
arrdiag(2.8, 7.84, 11.10, 10.53, color="#2C4F8A", lw=1.7, rad=-0.18)
note(7.2, 8.75, "BM25 Index", ec="#2C4F8A")

# FAISS pretrained -> IndoSBERT Search
arrdiag(7.0, 7.84, 16.9, 10.20, color="#8B2020", lw=1.7, rad=0.08)
note(12.5, 9.65, "FAISS pretrained", ec="#8B2020")

# FAISS FT -> IndoSBERT Search
arrdiag(8.5, 1.55, 16.9, 10.20, color="#7A1A00", lw=1.7, rad=0.20)
note(14.8, 5.2, "FAISS fine-tuned", ec="#7A1A00")

# ═══════════════════════════════════════════════════════════════════════════
# DIVIDER
# ═══════════════════════════════════════════════════════════════════════════
ax.plot([9.9, 9.9], [0.5, 13.6], color="#888", lw=1.5, ls="--", alpha=0.55, zorder=1)
ax.text(9.9, 13.72, "  OFFLINE  <-->  ONLINE  ", ha="center", va="center",
        fontsize=9, color="#555", fontstyle="italic",
        bbox=dict(fc="#EEE", ec="#AAA", pad=3, boxstyle="round"))

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
legend_els = [
    mpatches.Patch(fc="#F2C14E", ec="#333",  label="Input / Output"),
    mpatches.Patch(fc="#4C72B0",             label="Indexing Leksikal (BM25)"),
    mpatches.Patch(fc="#8B2020",             label="Indexing Semantik (IndoSBERT)"),
    mpatches.Patch(fc="#DD8452",             label="Fine-Tuning"),
    mpatches.Patch(fc="#3A8BA0",             label="Normalisasi Query (opsional, border putus-putus)"),
    mpatches.Patch(fc="#8172B3",             label="Fusion (RRF)"),
    mpatches.Patch(fc="#937860",             label="Evaluasi"),
    mpatches.Patch(fc="#2C7A4B",             label="Hasil Terbaik"),
]
ax.legend(handles=legend_els, loc="lower left", bbox_to_anchor=(0.005, 0.005),
          fontsize=8, framealpha=0.96, ncol=2, borderpad=0.9)

fig.suptitle(
    "Pipeline Sistem Hybrid Retrieval Dokumen Hukum Indonesia\n"
    "BM25 + IndoSBERT Fine-tuned + Reciprocal Rank Fusion + Normalisasi Query",
    fontsize=14, fontweight="bold", y=0.975)

fig.savefig("images/pipeline_diagram.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print("done")
