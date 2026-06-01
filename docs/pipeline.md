# Diagram Pipeline Sistem

Sumber diagram dalam format **Mermaid** (mudah diedit, render otomatis di GitHub
& VSCode dengan ekstensi Markdown Preview Mermaid). Versi gambar siap-PowerPoint
ada di [`pipeline_diagram.png`](pipeline_diagram.png).

```mermaid
flowchart TB
    subgraph IDX["TAHAP INDEXING (offline, sekali jalan)"]
        direction TB
        PDF["4 PDF UU<br/>Ketenagakerjaan · Konsumen · ITE · Anak"]
        EXT["Ekstraksi Teks PDF (pdfplumber)<br/>src/ingest.py"]
        CHK["Chunking per Pasal → 859 chunks<br/>chunks.jsonl (src/chunk.py)"]

        TOK["Tokenisasi<br/>(rank-bm25)"]
        BM25IDX[("BM25 Index<br/>bm25.pkl")]

        ENC["Encode IndoSBERT<br/>embedding 768-dim"]
        FAISS[("FAISS Index<br/>faiss.faiss")]

        PDF --> EXT --> CHK
        CHK --> TOK --> BM25IDX
        CHK --> ENC --> FAISS
    end

    subgraph RET["TAHAP RETRIEVAL (saat ada query)"]
        direction TB
        Q["QUERY PENGGUNA<br/>'berapa pesangon kalau dipecat?'"]
        LEX["Pencarian BM25<br/>→ Ranking Lexical"]
        SEM["Pencarian IndoSBERT<br/>(cosine via FAISS)<br/>→ Ranking Semantic"]
        RRF["Reciprocal Rank Fusion<br/>RRF(d) = Σ 1/(k+rank), k=60"]
        TOPK["Top-K Dokumen Hukum"]
        EVAL["Evaluasi<br/>Recall@5/10 · MRR · NDCG@10"]

        Q --> LEX
        Q --> SEM
        LEX --> RRF
        SEM --> RRF
        RRF --> TOPK --> EVAL
    end

    BM25IDX -.dipakai oleh.-> LEX
    FAISS -.dipakai oleh.-> SEM

    classDef input fill:#F2C14E,stroke:#333,color:#222;
    classDef prep fill:#4C72B0,stroke:#333,color:#fff;
    classDef lex fill:#55A868,stroke:#333,color:#fff;
    classDef sem fill:#C44E52,stroke:#333,color:#fff;
    classDef fuse fill:#8172B3,stroke:#333,color:#fff;
    classDef eval fill:#937860,stroke:#333,color:#fff;

    class PDF,Q,TOPK input;
    class EXT,CHK prep;
    class TOK,BM25IDX,LEX lex;
    class ENC,FAISS,SEM sem;
    class RRF fuse;
    class EVAL eval;
```

## Versi ringkas (untuk slide ikhtisar)

```mermaid
flowchart LR
    Q[Query] --> BM25[BM25<br/>lexical]
    Q --> SBERT[IndoSBERT<br/>semantic]
    BM25 --> RRF[RRF Fusion]
    SBERT --> RRF
    RRF --> R[Top-K Dokumen]
```
