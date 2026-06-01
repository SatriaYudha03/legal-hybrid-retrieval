# Struktur Proyek & Cara Kerja

```
legal-hybrid-retrieval/
├── README.md              # deskripsi penelitian (sudah ada)
├── STRUCTURE.md           # dokumen ini
├── requirements.txt       # dependencies
├── config.yaml            # SEMUA konfigurasi (path, model, parameter)
│
├── data/
│   ├── raw/               # >> taruh PDF UU di sini <<
│   ├── processed/         # teks bersih + chunks.jsonl (hasil olahan)
│   ├── index/             # bm25.pkl + FAISS index
│   └── eval/
│       ├── queries.json   # query evaluasi (target 100)
│       └── qrels.json     # GROUND TRUTH relevansi (wajib diisi manual)
│
├── src/                   # logika inti (importable)
│   ├── config.py          # loader config  [SELESAI]
│   ├── ingest.py          # ekstraksi PDF   [TODO]
│   ├── chunk.py           # chunking pasal  [TODO]
│   ├── bm25_retriever.py  # BM25            [TODO]
│   ├── semantic_retriever.py # IndoSBERT+FAISS [TODO]
│   ├── fusion.py          # RRF             [SELESAI]
│   └── evaluate.py        # metrik          [TODO]
│
├── scripts/               # entry point pipeline (urut)
│   ├── 01_extract.py      # PDF  -> data/processed/<domain>.txt
│   ├── 02_chunk.py        # txt  -> data/processed/chunks.jsonl
│   ├── 03_build_index.py  # chunk-> BM25 + FAISS index
│   └── 04_evaluate.py     # query+qrels -> results/metrics.json
│
└── results/               # output metrik & ranking
```

## Alur menjalankan (setelah implementasi TODO selesai)

```bash
pip install -r requirements.txt          # 1. install
# taruh PDF di data/raw/
python scripts/01_extract.py             # 2. ekstraksi teks
python scripts/02_chunk.py               # 3. chunking
python scripts/03_build_index.py         # 4-5. bangun index BM25 & FAISS
# isi data/eval/qrels.json (ground truth) berdasarkan chunk yang dihasilkan
python scripts/04_evaluate.py            # 6 + evaluasi 3 model
```

## Urutan implementasi yang disarankan

1. `src/ingest.py` — pastikan ekstraksi PDF bersih (paling makan waktu).
2. `src/chunk.py` — chunking per pasal; pakai struktur "Pasal N".
3. `src/bm25_retriever.py` — paling cepat memberi hasil, untuk validasi pipeline.
4. `src/semantic_retriever.py` — IndoSBERT + FAISS.
5. `src/evaluate.py` — metrik (atau pakai pytrec_eval).
6. Isi `data/eval/qrels.json` & lengkapi `queries.json` ke 100 query.

## Catatan penting

- **Ground truth (qrels.json) adalah pekerjaan terberat.** Tanpa label relevansi,
  metrik Recall/MRR/NDCG tidak bisa dihitung. Isi `chunk_id` asli setelah Tahap 3.
- **`src/fusion.py` sudah jadi** dan bisa langsung dipakai/diuji unit.
- Untuk korpus kecil ini, FAISS sebenarnya opsional (numpy+cosine cukup),
  tapi dipertahankan agar sesuai spesifikasi README.
```
