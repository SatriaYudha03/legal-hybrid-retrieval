# Progress Fine-Tuning IndoSBERT

Terakhir diperbarui: 2026-06-03

---

## Status Keseluruhan

```
Fase 0  [✅ SELESAI] Kunci Baseline
Fase 1  [✅ SELESAI] Query Normalization
Fase 2  [✅ SELESAI] Data Latih Sintetis
Fase 3  [⬜ BELUM  ] Fine-Tuning MNRL (kompute: CPU lokal)
Fase 4  [⬜ BELUM  ] Evaluasi Ablasi + Wilcoxon
Fase 5  [⬜ BELUM  ] Integrasi & Presentasi
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

## Fase 3 — Fine-Tuning MNRL ⬜

**Tujuan:** geser embedding IndoSBERT agar query awam makin dekat ke pasal relevan.
Perbaikan atas paper: pakai `MultipleNegativesRankingLoss` + hard negative,
bukan `CosineSimilarityLoss` tanpa negatif.

**Keputusan:** **CPU lokal** (tak ada GPU; dataset kecil 521 pasangan, model base 768-dim,
MNRL 1–3 epoch masih wajar di CPU). Semua reproducible lokal tanpa setup eksternal.

- [ ] Buat `scripts/08_finetune_sbert.py`
  - Base model: `firqaaa/indo-sentence-bert-base`
  - Loss: `MultipleNegativesRankingLoss` (in-batch + 4 hard negative/pasangan)
  - Hyperparam: lr 2e-5, 1–3 epoch, warmup 10%, weight decay 0.01, batch sesuai memori
- [ ] Simpan model ke `models/indosbert-legal-ft/`
- [ ] Encode ulang 378 chunk → rebuild FAISS index dengan model baru

---

## Fase 4 — Evaluasi Ablasi + Signifikansi ⬜

**Tujuan:** ukur kontribusi tiap komponen secara terpisah + buktikan signifikansi statistik.

- [ ] Buat `scripts/09_evaluate_ablation.py` — matriks 5 sistem × 2 kondisi normalisasi:

  | Sistem | Tanpa norm | Dengan norm |
  |---|:---:|:---:|
  | BM25 | ⬜ | ⬜ |
  | IndoSBERT pretrained | ⬜ | ⬜ |
  | IndoSBERT fine-tuned | ⬜ | ⬜ |
  | Pre-hybrid (BM25+pretrained) | ⬜ | ⬜ |
  | Fine-hybrid (BM25+FT) | ⬜ | ⬜ |

- [ ] Uji Wilcoxon signed-rank: pretrained vs FT, pre-hybrid vs fine-hybrid
- [ ] Simpan tabel lengkap ke `results/ablation.json`

---

## Fase 5 — Integrasi & Presentasi ⬜

**Tujuan:** rapikan pipeline & siapkan materi skripsi/presentasi.

- [ ] Update `config.yaml`: tambah bagian `normalization` (on/off) & `training`
- [ ] Buat `notebooks/06_finetuning_evaluasi.ipynb` — tabel ablasi + grafik + Wilcoxon
- [ ] Update `docs/RENCANA_FINETUNING.md` dengan hasil aktual

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
| ⬜ | `scripts/08_finetune_sbert.py` | Fine-tuning IndoSBERT |
| ⬜ | `scripts/09_evaluate_ablation.py` | Evaluasi ablasi lengkap |
| ⬜ | `notebooks/06_finetuning_evaluasi.ipynb` | Notebook presentasi |

---

## Keputusan Terbuka

| # | Keputusan | Pilihan |
|---|---|---|
| 1 | Sumber data latih sintetis (Fase 2) | ✅ **Template tanpa LLM** (tak ada API key; offline & reproducible) |
| 2 | Kompute fine-tuning (Fase 3) | ✅ **CPU lokal** (tak ada GPU; dataset kecil masih wajar) |
