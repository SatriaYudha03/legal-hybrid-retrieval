# Struktur Proyek & Cara Kerja

```
legal-hybrid-retrieval/
├── README.md                       # deskripsi penelitian lengkap
├── STRUCTURE.md                    # dokumen ini
├── requirements.txt                # dependencies
├── config.yaml                     # semua konfigurasi (path, model, parameter)
│
├── data/
│   ├── raw/                        # PDF UU (tidak di-commit ke git)
│   ├── processed/                  # chunks.jsonl + teks bersih per domain
│   ├── index/
│   │   ├── bm25.pkl                # indeks BM25
│   │   ├── faiss.faiss             # FAISS index (IndoSBERT pretrained)
│   │   └── faiss_ft/               # FAISS index (IndoSBERT fine-tuned)
│   ├── normalization/
│   │   └── legal_terms.json        # kamus akronim & sinonim per domain
│   ├── train/
│   │   └── pairs.jsonl             # 521 pasangan latih sintetis (query, positive, hard_negatives)
│   └── eval/
│       ├── queries.json            # 74 query evaluasi
│       └── qrels.json              # ground truth relevansi (bertingkat: 2=sangat rel, 1=pendukung)
│
├── src/                            # logika inti (importable)
│   ├── config.py                   # loader config.yaml
│   ├── ingest.py                   # ekstraksi teks dari PDF (pdfplumber)
│   ├── chunk.py                    # chunking per pasal
│   ├── bm25_retriever.py           # BM25 (rank-bm25)
│   ├── semantic_retriever.py       # IndoSBERT + FAISS (pretrained & fine-tuned)
│   ├── fusion.py                   # Reciprocal Rank Fusion
│   ├── normalize.py                # normalisasi query (ekspansi kamus)
│   └── evaluate.py                 # metrik retrieval + uji Wilcoxon
│
├── scripts/                        # entry point pipeline (urut angka = urutan eksekusi)
│   ├── 00_baseline.py              # evaluasi baseline 5 sistem (simpan perquery)
│   ├── 01_extract.py               # PDF → data/processed/<domain>.txt
│   ├── 02_chunk.py                 # txt → data/processed/chunks.jsonl
│   ├── 03_build_index.py           # chunks → BM25 + FAISS index (pretrained)
│   ├── 04_evaluate.py              # evaluasi dasar (BM25, SBERT, Hybrid)
│   ├── 05_evaluate_normalization.py# evaluasi dengan/tanpa normalisasi query
│   ├── 07_build_synthetic_queries.py # bangkitkan pasangan latih sintetis + filter round-trip
│   ├── 08_finetune_sbert.py        # fine-tuning IndoSBERT (fallback CPU lokal)
│   ├── 08b_rebuild_index_ft.py     # rebuild FAISS dari model fine-tuned
│   └── 09_evaluate_ablation.py     # ablasi 5 sistem × 2 normalisasi + Wilcoxon
│
├── models/
│   ├── indosbert-legal-ft/         # model fine-tuned (tidak di-commit ke git)
│   └── README.md
│
├── notebooks/
│   ├── 06_finetune_colab.ipynb     # fine-tuning di Google Colab GPU (T4, FP16) — direkomendasikan
│   └── 06_finetuning_evaluasi.ipynb # presentasi: tabel ablasi + grafik + boxplot + Wilcoxon
│
├── docs/
│   ├── pipeline.md                 # diagram pipeline Mermaid (indexing + retrieval + fine-tuning)
│   └── penjelasan_transformer.md   # teori Transformer → dikaitkan ke studi kasus
│
└── results/
    ├── metrics.json                # baseline rata-rata 3 sistem
    ├── baseline_perquery.json      # baseline NDCG per-query (74 query)
    ├── normalization_metrics.json  # rata-rata semua kondisi normalisasi
    ├── normalization_perquery.json # normalisasi per-query
    ├── ablation.json               # ablasi rata-rata (5 sistem × 2 kondisi)
    └── ablation_perquery.json      # ablasi per-query (untuk Wilcoxon)
```

## Alur menjalankan pipeline lengkap

```bash
pip install -r requirements.txt

# Taruh PDF di data/raw/
python scripts/01_extract.py               # ekstraksi teks
python scripts/02_chunk.py                 # chunking per pasal
python scripts/03_build_index.py           # bangun BM25 + FAISS (pretrained)

# Isi data/eval/qrels.json (ground truth) berdasarkan chunk_id yang dihasilkan

python scripts/00_baseline.py              # evaluasi baseline
python scripts/05_evaluate_normalization.py # evaluasi efek normalisasi

python scripts/07_build_synthetic_queries.py # bangkitkan data latih sintetis

# Fine-tuning (Google Colab direkomendasikan):
#   buka notebooks/06_finetune_colab.ipynb di Colab, jalankan semua sel
#   unduh hasil ke models/indosbert-legal-ft/

python scripts/08b_rebuild_index_ft.py     # rebuild FAISS dari model fine-tuned
python scripts/09_evaluate_ablation.py     # ablasi lengkap + Wilcoxon

jupyter notebook notebooks/06_finetuning_evaluasi.ipynb  # lihat hasil
```

## Catatan penting

- **Ground truth (qrels.json) adalah pekerjaan terberat.** Tanpa label relevansi, metrik Recall/MRR/NDCG tidak bisa dihitung. Isi `chunk_id` asli setelah langkah chunking.
- **Model dan PDF tidak di-commit ke git** karena ukurannya besar. Lihat `.gitignore`.
- **Fine-tuning di CPU lokal sangat lambat** (dataset 521 pasangan × 3 epoch). Gunakan `notebooks/06_finetune_colab.ipynb` di Google Colab (GPU T4) — selesai dalam ~2 menit.
- `src/fusion.py` dapat dipakai dan diuji unit secara independen.
