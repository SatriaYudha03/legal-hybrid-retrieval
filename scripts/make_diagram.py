"""Membuat diagram pipeline sistem (PNG) untuk presentasi/PowerPoint.

Jalankan:
    python scripts/make_diagram.py
Output: docs/pipeline_diagram.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Palet warna per fase
C_INPUT = "#F2C14E"   # input/output (kuning)
C_PREP = "#4C72B0"    # praproses (biru)
C_LEX = "#55A868"     # jalur lexical/BM25 (hijau)
C_SEM = "#C44E52"     # jalur semantic/IndoSBERT (merah)
C_FUSE = "#8172B3"    # fusion (ungu)
C_EVAL = "#937860"    # evaluasi (coklat)


def box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            linewidth=1.2, edgecolor="#333333", facecolor=color, zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight="bold", zorder=3)
    return (x, y, w, h)


def arrow(ax, p1, p2, style="-", color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=color, linestyle=style,
            shrinkA=2, shrinkB=2, zorder=1,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- Label fase ----
    ax.text(2, 97, "TAHAP INDEXING  (offline, sekali jalan)",
            fontsize=12, weight="bold", color="#333333")
    ax.text(2, 45.5, "TAHAP RETRIEVAL  (saat ada query)",
            fontsize=12, weight="bold", color="#333333")
    ax.axhline(47, color="#cccccc", linewidth=1, linestyle="--")

    # ---- INDEXING ----
    pdf = box(ax, 50, 92, 46, 6,
              "4 PDF UU\nKetenagakerjaan · Konsumen · ITE · Perlindungan Anak", C_INPUT,
              text_color="#333333")
    ext = box(ax, 50, 82, 40, 6,
              "Ekstraksi Teks PDF  (pdfplumber)\nsrc/ingest.py", C_PREP)
    chunk = box(ax, 50, 72, 40, 6,
                "Chunking per Pasal  →  859 chunks\nchunks.jsonl  (src/chunk.py)", C_PREP)

    tok = box(ax, 26, 60, 34, 6.5,
              "Tokenisasi\n(rank-bm25)", C_LEX)
    bm25idx = box(ax, 26, 51, 34, 6.5,
                  "BM25 Index\ndata/index/bm25.pkl", C_LEX)

    enc = box(ax, 74, 60, 36, 6.5,
              "Encode IndoSBERT\nembedding 768-dim", C_SEM)
    faiss = box(ax, 74, 51, 36, 6.5,
                "FAISS Index\ndata/index/faiss.faiss", C_SEM)

    arrow(ax, (50, 89), (50, 85))
    arrow(ax, (50, 79), (50, 75))
    arrow(ax, (40, 70), (26, 63.3))     # chunk -> tokenisasi
    arrow(ax, (60, 70), (74, 63.3))     # chunk -> encode
    arrow(ax, (26, 56.8), (26, 54.3))
    arrow(ax, (74, 56.8), (74, 54.3))

    # ---- RETRIEVAL ----
    query = box(ax, 50, 39, 52, 6.5,
                "QUERY PENGGUNA\n\"berapa pesangon kalau dipecat?\"", C_INPUT,
                text_color="#333333")

    lex = box(ax, 26, 28, 34, 7,
              "Pencarian BM25\n→ Ranking Lexical", C_LEX)
    sem = box(ax, 74, 28, 36, 7,
              "Pencarian IndoSBERT\n(cosine via FAISS)\n→ Ranking Semantic", C_SEM,
              fontsize=8.5)

    rrf = box(ax, 50, 16.5, 56, 7,
              "Reciprocal Rank Fusion (RRF)\nRRF(d) = Σ 1 / (k + rank),   k = 60", C_FUSE)
    topk = box(ax, 50, 7.5, 34, 5.5,
               "Top-K Dokumen Hukum", C_INPUT, text_color="#333333")

    ev = box(ax, 90, 16.5, 17, 9,
             "Evaluasi\nRecall@5/10\nMRR\nNDCG@10", C_EVAL, fontsize=8)

    # index -> pencarian (garis putus = 'dipakai oleh')
    arrow(ax, (26, 47.7), (26, 31.5), style="--", color="#55A868")
    arrow(ax, (74, 47.7), (74, 31.5), style="--", color="#C44E52")

    arrow(ax, (40, 37), (26, 31.5))     # query -> bm25 search
    arrow(ax, (60, 37), (74, 31.5))     # query -> indosbert search
    arrow(ax, (26, 24.5), (44, 19.5))   # bm25 -> rrf
    arrow(ax, (74, 24.5), (56, 19.5))   # indosbert -> rrf
    arrow(ax, (50, 13), (50, 10.3))     # rrf -> topk
    arrow(ax, (66, 9), (81.5, 14), style="--", color="#937860")  # topk -> eval

    ax.set_title("Pipeline Sistem Hybrid Retrieval Dokumen Hukum Indonesia\n"
                 "BM25 + IndoSBERT + Reciprocal Rank Fusion",
                 fontsize=14, weight="bold", pad=14)

    out_dir = Path(__file__).resolve().parent.parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_diagram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Diagram tersimpan -> {out_path}")


if __name__ == "__main__":
    main()
