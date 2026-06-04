# Progress Fine-Tuning IndoSBERT

Terakhir diperbarui: 2026-06-04

---

## Status Keseluruhan

```
Fase 0  [✅ SELESAI] Kunci Baseline
Fase 1  [✅ SELESAI] Query Normalization
Fase 2  [✅ SELESAI] Data Latih Sintetis
Fase 3  [✅ SELESAI] Fine-Tuning MNRL (Google Colab GPU, 3 epoch)
Fase 4  [✅ SELESAI] Evaluasi Ablasi + Wilcoxon
Fase 5  [✅ SELESAI] Integrasi & Presentasi
```

---

## Fase 0 — Kunci Baseline ✅

**Tujuan:** bekukan angka "sebelum" agar perbandingan setelah fine-tuning sah.

- [x] Hapus UU Ketenagakerjaan dari `config.yaml`
- [x] Rebuild pipeline: extract → chunk → build index (3 UU tersisa)
  - Chunk: 378 (dari 859 sebelumnya)
  - Domain: Konsumen (135), ITE (69), Anak (174)
- [x] Filter `queries.json` & `qrels.json`: hapus Q001–Q025 (ketenagakerjaan)
  - Query: 100 → 75 (aktif 74, satu kosong di qrels)
  - Qrels: 98 → 74 query berlabel
- [x] Tambah fungsi `evaluate_run_perquery()` & `wilcoxon_test()` ke `src/evaluate.py`
- [x] Jalankan `scripts/00_baseline.py` → simpan hasil ke `results/`

### Hasil Baseline (74 query, 3 UU)

| Sistem | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 | 0.5417 | 0.6227 | 0.5250 | 0.4947 |
| IndoSBERT (pretrained) | 0.5212 | 0.7061 | 0.5437 | 0.5043 |
| **Hybrid (BM25+SBERT+RRF)** | **0.6002** | **0.7775** | **0.5722** | **0.5632** |

**File yang dihasilkan:**
- `results/metrics.json` — rata-rata per sistem
- `results/baseline_perquery.json` — skor per-query (untuk Wilcoxon nanti)

---

## Fase 1 — Query Normalization ✅

**Tujuan:** bridging gap antara bahasa awam query dengan terminologi formal UU.
Diambil dari paper acuan (Kodri et al.) — terbukti lever terbesar di paper tersebut.

**Keputusan:** kamus manual (semi-terstruktur per domain) + ekspansi query (bukan replacement).
Strategi ekspansi: query asli tetap utuh, terminologi formal *ditambahkan* di belakang.
Cara ini menguntungkan BM25 (exact match tambahan) sekaligus SBERT (sinyal semantik baru).

- [x] Buat `data/normalization/legal_terms.json` — kamus akronim & sinonim per domain:
  - Konsumen: `BPSK`, `kadaluarsa`, `komplain`, `barang palsu`, dst.
  - ITE: `medsos`, `diretas`, `hoax`, `email`, `judi online`, dst.
  - Anak: `KPAI`, `pelecehan seksual`, `narkoba`, `adopsi`, dst.
- [x] Buat `src/normalize.py` — fungsi `normalize_query(text, domain)`
- [x] Buat `scripts/05_evaluate_normalization.py` — evaluasi langsung dengan/tanpa normalisasi
- [x] Evaluasi ulang: bandingkan dengan/tanpa normalisasi

### Hasil Fase 1 — Query Normalization (74 query, 3 UU)

| Sistem | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| BM25 (tanpa norm) | 0.5417 | 0.6227 | 0.5250 | 0.4947 |
| **BM25 (dengan norm)** | **0.6025** | **0.6971** | **0.6119** | **0.5778** |
| IndoSBERT (tanpa norm) | 0.5212 | 0.7061 | 0.5437 | 0.5043 |
| IndoSBERT (dengan norm) | 0.5392 | 0.7358 | 0.5522 | 0.5225 |
| Hybrid (tanpa norm) | 0.6002 | 0.7775 | 0.5722 | 0.5632 |
| **Hybrid (dengan norm)** | **0.6453** | **0.8027** | **0.6602** | **0.6355** |

**Delta normalisasi:**
- BM25: R@5 +6.1%, R@10 +7.4%, MRR +8.7%, NDCG@10 +8.3% — **Wilcoxon p=0.0027 (signifikan)**
- IndoSBERT: R@5 +1.8%, MRR +0.9%, NDCG@10 +1.8% — Wilcoxon p=0.23 (tidak signifikan)
- Hybrid: R@5 +4.5%, R@10 +2.5%, MRR +8.8%, NDCG@10 +7.2% — **Wilcoxon p=0.0065 (signifikan)**

**Insight:** Normalisasi paling efektif untuk BM25 (vocabulary bridging) dan menular ke Hybrid via RRF.
IndoSBERT sudah resilient terhadap variasi kosakata (embedding tangkap semantik), sehingga gain kecil.

**File yang dihasilkan:**
- `results/normalization_metrics.json` — rata-rata semua kondisi
- `results/normalization_perquery.json` — skor per-query (untuk Wilcoxon)

---

## Fase 2 — Data Latih Sintetis ✅

**Tujuan:** bangkitkan pseudo-query berbentuk pertanyaan awam dari 378 chunk.
Ini perbaikan atas paper acuan yang memakai ringkasan (distribusi latih ≠ distribusi uji).

**Keputusan:** **template tanpa LLM** (tak ada API key; offline & reproducible — bagus untuk integritas skripsi).
Sumber pertanyaan: judul BAB (di-derive ulang dari teks `.txt`) + istilah kunci pasal (frekuensi kata isi, stopword Sastrawi+kurасi dibuang) + frame pertanyaan per domain.

- [x] Buat `scripts/07_build_synthetic_queries.py`
  - Tiap chunk → s.d. 3 pseudo-query awam (template per domain)
  - **Filter round-trip** (hybrid BM25+SBERT, top-10): buang query yang tak menemukan pasal sumbernya
  - Validasi judul BAB: tolak judul "sampah" dari UU perubahan (klausa sisipan ber-angka)
- [x] Tambang **4 hard negative**/pasangan dari BM25 (pasal top BM25 yang bukan pasal sumber)
- [x] Simpan ke `data/train/pairs.jsonl` format `{query, positive, positive_id, domain, hard_negatives[]}`

### Hasil Fase 2 — Data Latih Sintetis

| Metrik | Nilai |
|---|---|
| Kandidat dibangkitkan | 1127 |
| Lolos round-trip | **521 (46.2%)** |
| Chunk tercakup | 232/378 (61.4%) |
| Per domain | konsumen 193, ite 137, anak 191 |
| Hard negative / pasangan | 4 (rata-rata) |

**Catatan kualitas (untuk skripsi):** pendekatan template menghasilkan data yang **lebih
bising** daripada LLM (sebagian query generik, mis. dari Pasal 1 berisi definisi). Mitigasi:
(a) filter round-trip menjamin tiap pasangan benar-benar relevan; (b) dedup kata, validasi
judul BAB, & daftar stopword tambahan menekan artefak. MNRL + hard negative cukup tahan
terhadap sisa noise. Trade-off ini adalah konsekuensi sah dari keputusan "tanpa LLM".

**File yang dihasilkan:**
- `data/train/pairs.jsonl` — 521 pasangan latih

---

## Fase 3 — Fine-Tuning MNRL ✅

**Tujuan:** geser embedding IndoSBERT agar query awam makin dekat ke pasal relevan.
Perbaikan atas paper: pakai `MultipleNegativesRankingLoss` + hard negative,
bukan `CosineSimilarityLoss` tanpa negatif.

**Keputusan akhir:** **Google Colab GPU** (T4, batch=8, grad_accum=4, fp16, 3 epoch).
Model disimpan ke `models/indosbert-legal-ft/` dan index FT ke `data/index/faiss_ft`.

- [x] Buat `scripts/08_finetune_sbert.py` (fallback CPU lokal)
- [x] Buat `notebooks/06_finetune_colab.ipynb` (siap-pakai di Google Colab GPU)
- [x] Fine-tune di Colab: base `firqaaa/indo-sentence-bert-base`, MNRL + 4 hard-neg, lr 2e-5, 3 epoch
- [x] Simpan model ke `models/indosbert-legal-ft/` (unduh dari Colab)
- [x] Rebuild FAISS index FT via `scripts/08b_rebuild_index_ft.py` → `data/index/faiss_ft`

---

## Fase 4 — Evaluasi Ablasi + Signifikansi ✅

**Tujuan:** ukur kontribusi tiap komponen secara terpisah + buktikan signifikansi statistik.

- [x] Buat `scripts/09_evaluate_ablation.py` — matriks 5 sistem × 2 kondisi normalisasi:

  | Sistem | Tanpa norm | Dengan norm |
  |---|:---:|:---:|
  | Sistem | Tanpa norm | Dengan norm |
  |---|---|---|
  | BM25 | R@5=0.5417 NDCG=0.4947 | R@5=0.6025 NDCG=0.5778 |
  | IndoSBERT pretrained | R@5=0.5212 NDCG=0.5043 | R@5=0.5392 NDCG=0.5225 |
  | IndoSBERT fine-tuned | R@5=0.5860 NDCG=0.5781 | R@5=0.6000 NDCG=0.6149 |
  | Pre-hybrid (BM25+pretrained) | R@5=0.6002 NDCG=0.5632 | R@5=0.6453 NDCG=0.6355 |
  | **Fine-hybrid (BM25+FT)** | R@5=0.6149 NDCG=0.5879 | **R@5=0.6959 NDCG=0.6512** |

- [x] Uji Wilcoxon signed-rank (NDCG@10):
  - pretrained vs FT [tanpa norm]: W=636.5, **p=0.0171** (signifikan)
  - pretrained vs FT [dengan norm]: W=522.0, **p=0.0014** (sangat signifikan)
  - pre_hybrid vs fine_hybrid [tanpa norm]: W=504.5, p=0.0618 (tidak signifikan)
  - pre_hybrid vs fine_hybrid [dengan norm]: W=589.5, p=0.2646 (tidak signifikan)
  - efek normalisasi BM25: W=51.0, **p=0.0027** (sangat signifikan)
  - efek normalisasi fine_hybrid: W=98.5, **p=0.0297** (signifikan)
- [x] Simpan ke `results/ablation.json` & `results/ablation_perquery.json`

### Insight Kunci Fase 4 (untuk skripsi)

1. **Fine-tuning MNRL terbukti signifikan** pada SBERT sendirian (p=0.017 tanpa norm, p=0.001 dengan norm) — jauh lebih baik dari paper acuan (+0.46%, tidak diuji signifikansi).
2. **Namun fine-tuning tidak signifikan pada level hybrid** (p=0.06/0.26). RRF meredam gain SBERT karena BM25 sudah kuat. Ini insight jujur yang tidak ada di paper.
3. **Normalisasi tetap lever terbesar** untuk BM25 (+8.3%, p=0.003) dan hybrid (+6.3%, p=0.030).
4. **Sistem terbaik = Fine-hybrid + Normalisasi**: NDCG@10=0.6512 vs baseline 0.5632 → **+15.7% total**.
5. Kontribusi komponen terpisah (untuk bab metodologi/analisis):
   - Fine-tuning saja (tanpa norm): +7.4% NDCG atas pretrained
   - Normalisasi saja (tanpa FT): +7.2% NDCG atas pre-hybrid tanpa norm
   - Keduanya bersama: +15.7% atas baseline

---

## Fase 5 — Integrasi & Presentasi ✅

**Tujuan:** rapikan pipeline & siapkan materi skripsi/presentasi.

- [x] Update `config.yaml`: tambah bagian `normalization` (on/off) & `training`
- [x] Buat `notebooks/06_finetuning_evaluasi.ipynb` — tabel ablasi + grafik + Wilcoxon
- [x] Update `docs/RENCANA_FINETUNING.md` dengan hasil aktual

### Isi `notebooks/06_finetuning_evaluasi.ipynb`

Notebook presentasi (Jupyter lokal) — semua angka dibaca otomatis dari
`results/ablation.json` & `results/ablation_perquery.json` agar selalu sinkron:

1. **Tabel ablasi lengkap** — 5 sistem × 2 kondisi normalisasi (MultiIndex DataFrame).
2. **Grafik NDCG@10 berpasangan** — efek normalisasi per sistem (bar tanpa vs dengan).
3. **Grafik kontribusi komponen** — progresi baseline → +fine-tuning → +normalisasi → keduanya.
4. **Boxplot distribusi NDCG@10 per query** — sebaran 74 query, bukan sekadar rata-rata.
5. **Tabel Wilcoxon** — 6 perbandingan dengan kolom signifikansi (p<0.05).
6. **Ringkasan sistem terbaik** — Fine-hybrid+Norm vs baseline (semua metrik dihitung live).
7. **Insight kunci** — narasi siap-pakai untuk bab Analisis skripsi.

**Catatan:** notebook disimpan tanpa output pre-render (konsisten dgn notebook lain di
repo); jalankan di Jupyter lokal untuk merender tabel & grafik. Logika tiap sel sudah
diverifikasi bebas-error (matplotlib 3.9, pandas 2.2, scipy 1.14).

---

## Berkas yang Dibuat / Diubah

| Status | File | Keterangan |
|---|---|---|
| ✅ | `config.yaml` | Hapus ketenagakerjaan |
| ✅ | `src/evaluate.py` | Tambah `evaluate_run_perquery()` & `wilcoxon_test()` |
| ✅ | `scripts/00_baseline.py` | Script evaluasi baseline baru |
| ✅ | `scripts/_filter_ketenagakerjaan.py` | Filter eval data (sekali pakai) |
| ✅ | `results/metrics.json` | Baseline rata-rata |
| ✅ | `results/baseline_perquery.json` | Baseline per-query |
| ✅ | `src/normalize.py` | Query normalization |
| ✅ | `data/normalization/legal_terms.json` | Kamus akronim hukum |
| ✅ | `scripts/05_evaluate_normalization.py` | Evaluasi Fase 1 |
| ✅ | `results/normalization_metrics.json` | Hasil Fase 1 rata-rata |
| ✅ | `results/normalization_perquery.json` | Hasil Fase 1 per-query |
| ✅ | `scripts/07_build_synthetic_queries.py` | Pembuat data latih (template) |
| ✅ | `data/train/pairs.jsonl` | 521 pasangan latih sintetis |
| ✅ | `scripts/08_finetune_sbert.py` | Fine-tuning IndoSBERT (fallback CPU lokal) |
| ✅ | `scripts/08b_rebuild_index_ft.py` | Rebuild FAISS index dari model FT |
| ✅ | `notebooks/06_finetune_colab.ipynb` | Notebook fine-tuning Google Colab (GPU) |
| ✅ | `scripts/09_evaluate_ablation.py` | Evaluasi ablasi lengkap + Wilcoxon |
| ✅ | `results/ablation.json` | Hasil ablasi rata-rata 10 kombinasi |
| ✅ | `results/ablation_perquery.json` | Hasil ablasi per-query |
| ✅ | `config.yaml` | Tambah bagian `normalization` & `training` |
| ✅ | `notebooks/06_finetuning_evaluasi.ipynb` | Notebook presentasi Fase 5 |

---

## Keputusan Terbuka

| # | Keputusan | Pilihan |
|---|---|---|
| 1 | Sumber data latih sintetis (Fase 2) | ✅ **Template tanpa LLM** (tak ada API key; offline & reproducible) |
| 2 | Kompute fine-tuning (Fase 3) | ✅ **CPU lokal** (tak ada GPU; dataset kecil masih wajar) |
