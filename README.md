# Hybrid Retrieval Dokumen Hukum Indonesia: Integrasi BM25, Fine-Tuned IndoSBERT, dan Normalisasi Query dengan Reciprocal Rank Fusion

## Ringkasan Proyek

Proyek ini membangun dan mengevaluasi sistem pencarian dokumen hukum Indonesia yang menggabungkan tiga komponen utama: retrieval leksikal berbasis **BM25**, retrieval semantik berbasis **IndoSBERT yang dilatih ulang (fine-tuned)**, dan **normalisasi query** berbasis kamus. Hasil akhir ketiga komponen digabungkan menggunakan **Reciprocal Rank Fusion (RRF)**.

Dataset terdiri dari **378 chunk pasal** dari tiga undang-undang multi-domain: Perlindungan Konsumen, Informasi dan Transaksi Elektronik (ITE), dan Perlindungan Anak. Evaluasi dilakukan pada **74 query berlabel** menggunakan metrik Recall@5, Recall@10, MRR, dan NDCG@10, dengan uji signifikansi statistik **Wilcoxon signed-rank**.

---

# Latar Belakang

Pencarian dokumen hukum merupakan permasalahan yang menantang karena bahasa yang digunakan dalam undang-undang sering kali berbeda dengan bahasa yang digunakan masyarakat umum.

Sebagai contoh:

Query pengguna:

> "berapa pesangon kalau dipecat?"

Dokumen hukum:

> "Pemutusan hubungan kerja dan hak pekerja atas uang pesangon..."

Meskipun keduanya membahas topik yang sama, kata-kata yang digunakan berbeda sehingga metode berbasis kata kunci sering gagal menemukan dokumen yang relevan.

Model Transformer seperti IndoSBERT mampu memahami hubungan semantik antara query dan dokumen sehingga berpotensi memberikan hasil pencarian yang lebih baik. Namun, metode semantic retrieval terkadang kurang baik dalam menangani istilah hukum yang sangat spesifik, nomor pasal, atau frasa yang harus dicocokkan secara eksak.

Selain gap semantik, ada juga **gap kosakata**: pengguna awam memakai akronim dan istilah informal (misalnya "PHK", "dipecat", "medsos") yang tidak muncul verbatim di teks UU. Normalisasi query menjembatani gap ini sebelum pencarian dilakukan.

Karena itu, proyek ini menggabungkan BM25, fine-tuned IndoSBERT, dan normalisasi query agar memperoleh kelebihan dari seluruh pendekatan tersebut.

---

# Tujuan

1. Membangun baseline sistem retrieval dokumen hukum Indonesia (BM25 + IndoSBERT pretrained + Hybrid RRF).
2. Menerapkan **normalisasi query** berbasis kamus akronim dan sinonim per domain hukum.
3. **Fine-tuning IndoSBERT** menggunakan data sintetis berformat pertanyaan dan `MultipleNegativesRankingLoss`.
4. Membandingkan performa 5 konfigurasi sistem × 2 kondisi normalisasi (ablasi lengkap).
5. Membuktikan signifikansi statistik setiap peningkatan menggunakan uji **Wilcoxon signed-rank**.

---

# Dataset

## Dokumen yang Digunakan

Pipeline aktif menggunakan **3 UU** (UU Ketenagakerjaan dikecualikan pada tahap baseline untuk menjaga integritas evaluasi):

### 1. UU Perlindungan Konsumen

Domain:
- Hak konsumen
- Ganti rugi
- Barang cacat
- Perlindungan pembeli

Jumlah chunk: 135 pasal

---

### 2. UU Informasi dan Transaksi Elektronik (ITE)

Domain:
- Informasi elektronik
- Dokumen elektronik
- Transaksi elektronik
- Kejahatan siber

Jumlah chunk: 69 pasal

---

### 3. UU Perlindungan Anak

Domain:
- Hak anak
- Pendidikan
- Kekerasan terhadap anak
- Perlindungan khusus

Jumlah chunk: 174 pasal

---

**Total: 378 chunk pasal · 74 query berlabel · qrels bertingkat (2 = sangat relevan, 1 = pendukung)**

---

# Arsitektur Sistem

## Komponen 1: BM25 (Retrieval Leksikal)

Input: Query pengguna

Proses: Tokenisasi → pencarian BM25

Output: Ranking dokumen berdasarkan kemiripan kata

BM25 andal untuk istilah hukum spesifik dan nomor pasal, tetapi buta terhadap sinonim dan parafrasa.

---

## Komponen 2: IndoSBERT (Retrieval Semantik)

Input: Query pengguna + chunk pasal

Proses:
1. Query dan dokumen diubah menjadi embedding 768-dim.
2. Kemiripan dihitung menggunakan cosine similarity (via FAISS).

Output: Ranking dokumen berdasarkan kemiripan makna.

IndoSBERT dapat mencocokkan "dipecat" dengan "pemutusan hubungan kerja" meskipun kata-katanya berbeda.

---

## Komponen 3: Normalisasi Query

Input: Query pengguna mentah

Proses: Ekspansi query menggunakan kamus akronim dan sinonim per domain (`data/normalization/legal_terms.json`). Query asli tetap utuh — terminologi formal *ditambahkan* di belakang.

Contoh:
- `"PHK"` → `"PHK pemutusan hubungan kerja"`
- `"medsos"` → `"medsos media sosial"`
- `"BPSK"` → `"BPSK Badan Penyelesaian Sengketa Konsumen"`

Strategi ekspansi (bukan penggantian) menguntungkan BM25 (exact match tambahan) sekaligus SBERT (sinyal semantik diperkaya).

---

## Komponen 4: Fine-Tuning IndoSBERT

Lihat bagian [Fine-Tuning IndoSBERT](#fine-tuning-indosbert) untuk penjelasan lengkap.

---

## Komponen 5: Reciprocal Rank Fusion (RRF)

BM25 menghasilkan ranking leksikal.

IndoSBERT (pretrained atau fine-tuned) menghasilkan ranking semantik.

Kedua ranking digabung menggunakan RRF:

```
RRF(d) = Σ  1 / (k + rank_r(d))     k = 60
```

Semakin tinggi skor RRF, semakin tinggi posisi dokumen pada hasil akhir.

---

# Fine-Tuning IndoSBERT

## Motivasi

IndoSBERT pretrained (`firqaaa/indo-sentence-bert-base`) dilatih pada korpus umum, bukan teks hukum. Embedding yang dihasilkan belum optimal untuk memetakan pertanyaan awam ke pasal undang-undang. Fine-tuning bertujuan **menggeser embedding** agar query awam makin dekat ke pasal relevan di ruang vektor.

## Pembangunan Data Latih Sintetis

Karena tidak ada dataset query-pasal hukum Indonesia yang tersedia publik, data latih dibangkitkan secara sintetis menggunakan **template tanpa LLM** (offline dan reproducible — menjaga integritas skripsi).

Proses (`scripts/07_build_synthetic_queries.py`):
1. Tiap chunk pasal → hingga 3 pseudo-query berformat pertanyaan awam (template per domain).
2. **Filter round-trip**: buang pseudo-query yang gagal menemukan pasal sumbernya di top-10 hybrid BM25+SBERT.
3. **Hard negative mining**: ambil 4 pasal teratas BM25 yang bukan pasal sumber sebagai negatif eksplisit.

Hasil:

| Metrik | Nilai |
|---|---|
| Kandidat dibangkitkan | 1127 |
| Lolos filter round-trip | **521 (46.2%)** |
| Chunk tercakup | 232 / 378 (61.4%) |
| Hard negative per pasangan | 4 (rata-rata) |

Data tersimpan di `data/train/pairs.jsonl` berformat `{query, positive, positive_id, domain, hard_negatives[]}`.

## Prosedur Fine-Tuning

| Aspek | Nilai |
|---|---|
| Base model | `firqaaa/indo-sentence-bert-base` |
| Loss | `MultipleNegativesRankingLoss` + 4 hard negative |
| Platform | Google Colab (GPU T4) |
| Batch size efektif | 32 (batch=8, grad_accum=4) |
| Learning rate | 2e-5 |
| Epochs | 3 |
| FP16 | Ya |
| Training time | ~2 menit (Colab T4) |

Script fine-tuning tersedia di `scripts/08_finetune_sbert.py` (CPU lokal) dan `notebooks/06_finetune_colab.ipynb` (Google Colab GPU, direkomendasikan).

Model tersimpan di `models/indosbert-legal-ft/`.

## Keunggulan atas Pendekatan Acuan (Kodri et al., TEKNIKA 2025)

| Aspek | Paper acuan | Proyek ini |
|---|---|---|
| Bentuk data sintetis | Ringkasan pasal (formal→formal) | **Pertanyaan awam (awam→formal)** |
| Loss | `CosineSimilarityLoss` (tanpa negatif) | **MNRL + 4 hard negative** |
| Uji signifikansi | Tidak ada | **Wilcoxon signed-rank** |
| Data latih distribusi | ≠ distribusi uji | ≈ distribusi uji nyata |

---

# Pipeline Sistem

## Tahap 1 — Pengumpulan Dokumen

Taruh PDF UU di `data/raw/`.

## Tahap 2 — Ekstraksi Teks

```bash
python scripts/01_extract.py
```

Output: `data/processed/<domain>.txt`

## Tahap 3 — Chunking per Pasal

```bash
python scripts/02_chunk.py
```

Output: `data/processed/chunks.jsonl` (378 chunk)

Format tiap chunk:

```json
{
  "id": "CONS_001",
  "domain": "konsumen",
  "text": "Pasal 4 Hak konsumen adalah: ..."
}
```

## Tahap 4 — Bangun Indeks BM25 dan FAISS (pretrained)

```bash
python scripts/03_build_index.py
```

Output: `data/index/bm25.pkl` + `data/index/faiss.faiss`

## Tahap 5 — Data Sintetis + Fine-Tuning

```bash
python scripts/07_build_synthetic_queries.py   # bangkitkan pasangan latih
# Fine-tuning di Google Colab: notebooks/06_finetune_colab.ipynb
python scripts/08b_rebuild_index_ft.py         # rebuild FAISS dari model FT
```

Output: `data/train/pairs.jsonl` · `models/indosbert-legal-ft/` · `data/index/faiss_ft`

## Tahap 6 — Evaluasi Ablasi

```bash
python scripts/00_baseline.py                  # baseline 5 sistem
python scripts/05_evaluate_normalization.py    # efek normalisasi
python scripts/09_evaluate_ablation.py         # matriks ablasi lengkap + Wilcoxon
```

Output: `results/ablation.json` · `results/ablation_perquery.json`

---

# Evaluasi

## Sistem yang Dibandingkan (Ablasi 5 × 2)

| Sistem | Tanpa Normalisasi | Dengan Normalisasi |
|---|:---:|:---:|
| BM25 | ✓ | ✓ |
| IndoSBERT pretrained | ✓ | ✓ |
| IndoSBERT fine-tuned | ✓ | ✓ |
| Pre-hybrid (BM25 + pretrained + RRF) | ✓ | ✓ |
| **Fine-hybrid (BM25 + fine-tuned + RRF)** | ✓ | ✓ |

## Metrik

- **Recall@5** — apakah dokumen relevan muncul di 5 hasil teratas
- **Recall@10** — apakah dokumen relevan muncul di 10 hasil teratas
- **MRR** — Mean Reciprocal Rank; seberapa cepat dokumen relevan pertama ditemukan
- **NDCG@10** — kualitas urutan ranking secara keseluruhan

## Uji Signifikansi

Wilcoxon signed-rank test (one-sided) pada vektor NDCG@10 per-query (74 sampel).

---

# Hasil

## Baseline (74 query, 3 UU, tanpa normalisasi)

| Sistem | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 | 0.5417 | 0.6227 | 0.5250 | 0.4947 |
| IndoSBERT pretrained | 0.5212 | 0.7061 | 0.5437 | 0.5043 |
| **Hybrid (BM25+pretrained+RRF)** | **0.6002** | **0.7775** | **0.5722** | **0.5632** |

## Efek Normalisasi Query

| Sistem | NDCG@10 (tanpa) | NDCG@10 (dengan) | Delta | Wilcoxon p |
|---|---|---|---|---|
| BM25 | 0.4947 | 0.5778 | +8.3% | **0.0027** |
| IndoSBERT pretrained | 0.5043 | 0.5225 | +1.8% | 0.23 (n.s.) |
| Hybrid pretrained | 0.5632 | 0.6355 | +7.2% | **0.0065** |

IndoSBERT sudah resilient terhadap variasi kosakata (embedding menangkap semantik), sehingga gain normalisasi kecil. BM25 dan Hybrid mendapat gain terbesar.

## Ablasi Lengkap — NDCG@10

| Sistem | Tanpa Normalisasi | Dengan Normalisasi |
|---|---|---|
| BM25 | 0.4947 | 0.5778 |
| IndoSBERT pretrained | 0.5043 | 0.5225 |
| IndoSBERT fine-tuned | 0.5781 | 0.6149 |
| Pre-hybrid (BM25+pretrained) | 0.5632 | 0.6355 |
| **Fine-hybrid (BM25+fine-tuned)** | **0.5879** | **0.6512** |

## Uji Wilcoxon — Pasangan Kunci

| Perbandingan | W | p | Signifikan? |
|---|---|---|---|
| pretrained vs fine-tuned [tanpa norm] | 636.5 | 0.0171 | Ya |
| pretrained vs fine-tuned [dengan norm] | 522.0 | 0.0014 | Ya |
| pre-hybrid vs fine-hybrid [tanpa norm] | 504.5 | 0.0618 | Tidak |
| pre-hybrid vs fine-hybrid [dengan norm] | 589.5 | 0.2646 | Tidak |
| efek normalisasi pada BM25 | 51.0 | 0.0027 | Ya |
| efek normalisasi pada fine-hybrid | 98.5 | 0.0297 | Ya |

## Temuan Utama

1. **Fine-tuning MNRL terbukti signifikan** pada IndoSBERT sendirian (p=0.017 tanpa norm, p=0.001 dengan norm). Menggunakan pertanyaan awam sebagai data sintetis + loss kontrastif menghasilkan peningkatan nyata.

2. **Fine-tuning tidak signifikan pada level hybrid** (p=0.06/0.26). RRF meredam gain SBERT karena BM25 sudah kuat — insight jujur yang tidak ada di paper acuan.

3. **Normalisasi query adalah lever terbesar** dan paling konsisten: signifikan untuk BM25 (+8.3%, p=0.003) dan Hybrid (+6.3%, p=0.030).

4. **Sistem terbaik = Fine-hybrid + Normalisasi**: NDCG@10 = **0.6512** vs baseline **0.5632** → **+15.7% total**.

5. Kontribusi terpisah:
   - Fine-tuning saja (tanpa norm): +7.4% NDCG atas pretrained
   - Normalisasi saja (tanpa FT): +7.2% NDCG atas pre-hybrid
   - Keduanya bersama: **+15.7%** atas baseline

---

# Teknologi

| Kategori | Library / Tool |
|---|---|
| Bahasa | Python 3.12 |
| Retrieval leksikal | rank-bm25 |
| Retrieval semantik | sentence-transformers, FAISS |
| Fine-tuning | sentence-transformers (MultipleNegativesRankingLoss) |
| Model base | `firqaaa/indo-sentence-bert-base` (768-dim) |
| Evaluasi metrik | pytrec_eval, scikit-learn |
| Uji statistik | scipy (Wilcoxon) |
| Ekstraksi PDF | pdfplumber |
| GPU fine-tuning | Google Colab (T4, FP16) |

---

# Cara Menjalankan

```bash
pip install -r requirements.txt

# 1. Taruh PDF UU di data/raw/
# 2. Ekstraksi dan chunking
python scripts/01_extract.py
python scripts/02_chunk.py

# 3. Bangun indeks baseline (BM25 + FAISS pretrained)
python scripts/03_build_index.py

# 4. Evaluasi baseline
python scripts/00_baseline.py

# 5. Evaluasi normalisasi query
python scripts/05_evaluate_normalization.py

# 6. Bangkitkan data sintetis
python scripts/07_build_synthetic_queries.py

# 7. Fine-tuning (direkomendasikan di Google Colab: notebooks/06_finetune_colab.ipynb)
#    atau CPU lokal (lambat):
python scripts/08_finetune_sbert.py

# 8. Rebuild FAISS index dari model fine-tuned
python scripts/08b_rebuild_index_ft.py

# 9. Evaluasi ablasi lengkap + Wilcoxon
python scripts/09_evaluate_ablation.py

# 10. Lihat hasil & visualisasi
jupyter notebook notebooks/06_finetuning_evaluasi.ipynb
```

---

# Struktur Direktori

```
legal-hybrid-retrieval/
├── README.md
├── STRUCTURE.md
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── raw/                        # PDF UU (tidak di-commit ke git)
│   ├── processed/                  # chunks.jsonl + teks bersih
│   ├── index/                      # bm25.pkl · faiss.faiss · faiss_ft/
│   ├── normalization/
│   │   └── legal_terms.json        # kamus akronim per domain
│   ├── train/
│   │   └── pairs.jsonl             # 521 pasangan latih sintetis
│   └── eval/
│       ├── queries.json            # 74 query evaluasi
│       └── qrels.json              # ground truth relevansi
│
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── chunk.py
│   ├── bm25_retriever.py
│   ├── semantic_retriever.py
│   ├── fusion.py
│   ├── normalize.py                # normalisasi query
│   └── evaluate.py                 # metrik + Wilcoxon
│
├── scripts/
│   ├── 00_baseline.py
│   ├── 01_extract.py
│   ├── 02_chunk.py
│   ├── 03_build_index.py
│   ├── 04_evaluate.py
│   ├── 05_evaluate_normalization.py
│   ├── 07_build_synthetic_queries.py
│   ├── 08_finetune_sbert.py
│   ├── 08b_rebuild_index_ft.py
│   └── 09_evaluate_ablation.py
│
├── models/
│   ├── indosbert-legal-ft/         # model fine-tuned (tidak di-commit)
│   └── README.md
│
├── notebooks/
│   ├── 06_finetune_colab.ipynb     # fine-tuning di Google Colab GPU
│   └── 06_finetuning_evaluasi.ipynb # tabel ablasi + grafik + Wilcoxon
│
├── docs/
│   ├── pipeline.md                 # diagram pipeline (Mermaid)
│   └── penjelasan_transformer.md   # teori Transformer → studi kasus
│
└── results/
    ├── metrics.json
    ├── baseline_perquery.json
    ├── normalization_metrics.json
    ├── normalization_perquery.json
    ├── ablation.json
    └── ablation_perquery.json
```
