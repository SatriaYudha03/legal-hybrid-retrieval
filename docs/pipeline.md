# Diagram Pipeline Sistem

Sumber diagram dalam format **Mermaid** (mudah diedit, render otomatis di GitHub
& VSCode dengan ekstensi Markdown Preview Mermaid).

## Pipeline lengkap (indexing + fine-tuning + retrieval)

```mermaid
flowchart TB
    subgraph IDX["TAHAP INDEXING (offline, sekali jalan)"]
        direction TB
        PDF["3 PDF UU\nKonsumen · ITE · Anak"]
        EXT["Ekstraksi Teks PDF (pdfplumber)\nsrc/ingest.py"]
        CHK["Chunking per Pasal → 378 chunks\nchunks.jsonl  (src/chunk.py)"]

        TOK["Tokenisasi\n(rank-bm25)"]
        BM25IDX[("BM25 Index\nbm25.pkl")]

        ENC["Encode IndoSBERT pretrained\nembedding 768-dim"]
        FAISS[("FAISS Index\n(pretrained)\nfaiss.faiss")]

        PDF --> EXT --> CHK
        CHK --> TOK --> BM25IDX
        CHK --> ENC --> FAISS
    end

    subgraph FT["TAHAP FINE-TUNING (sekali, di Google Colab GPU)"]
        direction TB
        SYN["Bangkitkan pseudo-query\ntemplate per domain\nscripts/07_build_synthetic_queries.py"]
        FILT["Filter round-trip\n521 pasangan lolos / 1127 kandidat"]
        HNEG["Hard negative mining\n4 negatif BM25 per pasangan"]
        PAIRS[("data/train/pairs.jsonl\n521 pasangan latih")]
        MNRL["Fine-Tuning MNRL\nbase: firqaaa/indo-sentence-bert-base\nlr=2e-5, batch=32, 3 epoch, FP16\nnotebooks/06_finetune_colab.ipynb"]
        FTMODEL[("IndoSBERT fine-tuned\nmodels/indosbert-legal-ft/")]
        FAISSFT[("FAISS Index\n(fine-tuned)\ndata/index/faiss_ft")]

        CHK --> SYN --> FILT --> HNEG --> PAIRS --> MNRL --> FTMODEL
        FTMODEL --> FAISSFT
    end

    subgraph NORM["NORMALISASI QUERY (opsional, on/off via config)"]
        direction LR
        DICT[("Kamus akronim\ndata/normalization/legal_terms.json")]
        NQ["src/normalize.py\nEkspansi query: asli + terminologi formal"]
        DICT --> NQ
    end

    subgraph RET["TAHAP RETRIEVAL (saat ada query)"]
        direction TB
        Q["QUERY PENGGUNA\n'berapa pesangon kalau dipecat?'"]
        LEX["Pencarian BM25\n→ Ranking Lexical"]
        SEM["Pencarian IndoSBERT\n(cosine via FAISS)\n→ Ranking Semantic"]
        RRF["Reciprocal Rank Fusion\nRRF(d) = Σ 1/(k+rank), k=60"]
        TOPK["Top-K Dokumen Hukum"]
        EVAL["Evaluasi Ablasi\nRecall@5/10 · MRR · NDCG@10\n+ Wilcoxon signed-rank"]

        Q --> NQ
        NQ --> LEX
        NQ --> SEM
        LEX --> RRF
        SEM --> RRF
        RRF --> TOPK --> EVAL
    end

    BM25IDX -.dipakai oleh.-> LEX
    FAISS -.dipakai oleh (pretrained).-> SEM
    FAISSFT -.dipakai oleh (fine-tuned).-> SEM

    classDef input fill:#F2C14E,stroke:#333,color:#222;
    classDef prep fill:#4C72B0,stroke:#333,color:#fff;
    classDef lex fill:#55A868,stroke:#333,color:#fff;
    classDef sem fill:#C44E52,stroke:#333,color:#fff;
    classDef fuse fill:#8172B3,stroke:#333,color:#fff;
    classDef eval fill:#937860,stroke:#333,color:#fff;
    classDef ft fill:#DD8452,stroke:#333,color:#fff;
    classDef norm fill:#64B5CD,stroke:#333,color:#fff;

    class PDF,Q,TOPK input;
    class EXT,CHK prep;
    class TOK,BM25IDX,LEX lex;
    class ENC,FAISS,FAISSFT,SEM sem;
    class RRF fuse;
    class EVAL eval;
    class SYN,FILT,HNEG,PAIRS,MNRL,FTMODEL ft;
    class DICT,NQ norm;
```

## Versi ringkas (untuk slide ikhtisar)

```mermaid
flowchart LR
    PDF[3 UU PDF\n378 chunk] --> IDX[Indexing\nBM25 + FAISS]
    PDF --> FT[Fine-Tuning IndoSBERT\nMNRL · 521 pairs]
    FT --> IDX

    Q[Query Pengguna] --> NORM[Normalisasi Query\nKamus akronim]
    NORM --> BM25[BM25\nleksikal]
    NORM --> SBERT[IndoSBERT FT\nsemantik]
    IDX -.-> BM25
    IDX -.-> SBERT
    BM25 --> RRF[RRF Fusion]
    SBERT --> RRF
    RRF --> R[Top-K Dokumen\n+ Evaluasi Wilcoxon]
```

## Konfigurasi ablasi (5 sistem × 2 kondisi normalisasi)

| Sistem | Cabang FAISS | Normalisasi |
|---|---|---|
| BM25 | — | tanpa / dengan |
| IndoSBERT pretrained | `faiss.faiss` | tanpa / dengan |
| IndoSBERT fine-tuned | `faiss_ft/` | tanpa / dengan |
| Pre-hybrid (BM25 + pretrained + RRF) | `faiss.faiss` | tanpa / dengan |
| Fine-hybrid (BM25 + fine-tuned + RRF) | `faiss_ft/` | tanpa / dengan |
