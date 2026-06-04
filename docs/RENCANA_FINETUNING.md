# Rencana Fine-Tuning IndoSBERT untuk Hybrid Legal Retrieval

Dokumen ini merangkum rencana menambahkan **fine-tuning model semantik (IndoSBERT)**
ke pipeline `BM25 + IndoSBERT + RRF` yang sudah ada, beserta perbandingan dengan
penelitian acuan dan keputusan desain yang melandasinya.

> **Status: SELESAI (Fase 0–5).** Dokumen ini menyimpan *rencana & rasional* desain;
> progres eksekusi dan angka final ada di [`docs/PROGRESS.md`](PROGRESS.md).

---

## 0. Status Akhir & Hasil Aktual

Eksekusi final memakai **378 chunk pasal dari 3 UU** (Perlindungan Konsumen, ITE,
Perlindungan Anak — UU Ketenagakerjaan dikeluarkan pada Fase 0) dan **74 query
berlabel** sebagai test set suci.

| Hasil kunci (NDCG@10) | Nilai |
|---|---|
| Baseline (Pre-hybrid, tanpa norm) | 0.5632 |
| **Sistem terbaik (Fine-hybrid + Normalisasi)** | **0.6512 (+15.6%)** |
| Fine-tuning MNRL pada SBERT (signifikansi) | p=0.017 tanpa norm, **p=0.001** dengan norm |
| Efek normalisasi (BM25 / fine-hybrid) | p=0.003 / p=0.030 |

**Tiga temuan untuk skripsi:**
1. **Fine-tuning MNRL kita signifikan** pada SBERT sendirian — mengungguli `CosineSimilarityLoss`
   paper acuan (+0.46%, tanpa uji). Bentuk data sintetis (pertanyaan, bukan ringkasan) + loss
   kontrastif terbukti menentukan.
2. **Pada level hybrid, fine-tuning tidak lagi signifikan** (p=0.06/0.26): RRF meredam gain SBERT
   karena BM25 sudah kuat — *insight jujur yang tak ada di paper*.
3. **Query normalization tetap lever terbesar** dan komplementer dengan fine-tuning.

Detail per-metrik & per-fase: lihat `results/ablation.json` dan `notebooks/06_finetuning_evaluasi.ipynb`.

---

## 1. Status Saat Ini (saat dokumen ini ditulis, pra-eksekusi)

- Pipeline memakai IndoSBERT (`firqaaa/indo-sentence-bert-base`, 768-dim) **murni
  sebagai inference** — tidak ada training. Lihat `src/semantic_retriever.py`.
- Korpus: **859 chunk** pasal dari 4 UU (Ketenagakerjaan, Konsumen, ITE, Anak).
- Evaluasi: **~98 query** berlabel + qrels bertingkat (2=sangat relevan, 1=pendukung)
  di `data/eval/queries.json` & `data/eval/qrels.json`.
- Metrik: Recall@5, Recall@10, MRR, NDCG@10 (`src/evaluate.py`).

**Pertanyaan yang dijawab dokumen ini:** apakah perlu fine-tuning, dan bagaimana
caranya, dengan tetap menjaga integritas evaluasi.

---

## 2. Acuan: Paper "Fine-Hybrid" (Kodri et al., TEKNIKA 2025)

Paper ini mengerjakan arsitektur yang **hampir identik** (BM25 + SBERT fine-tuned +
RRF) di domain hukum pajak Indonesia (UU KUP). Ringkasan metode mereka:

| Aspek | Pendekatan paper |
|---|---|
| Korpus | 367 paragraf (50 pasal) UU KUP |
| Test set | **Hanya 23 query** dari 5 ahli pajak |
| Data sintetis | Prompting 2 tahap ke ChatGPT → **ringkasan/"inti" tiap pasal** |
| Loss | `CosineSimilarityLoss` (tanpa contoh negatif) |
| Hyperparam | 20 epoch, lr 2e-5, batch 16, warmup 10% |
| Fitur kunci | **Query normalization** (kamus istilah + ekspansi akronim) |

### Temuan kritis dari membaca paper

1. **Fine-tuning sendiri nyaris tak berdampak.** Tabel 1 mereka (dgn normalisasi, k=5):
   - Pretrained SBERT: P@N 53.80% → Fine-tuned SBERT: P@N 54.26% (naik 0.46 poin;
     recall malah turun). Di dalam batas noise, apalagi dgn 23 query.
2. **Angka "12.08%" yang mereka banggakan = efek query normalization**, BUKAN
   fine-tuning (53.94% → 66.02% saat normalisasi dinyalakan pada model fine-hybrid).
3. **Penyebab fine-tuning lemah:** data latih berbentuk (pasal ↔ ringkasan pasal) —
   keduanya teks formal sisi-dokumen. Tugas nyatanya query awam → pasal. Distribusi
   latih ≠ distribusi uji.
4. **Tanpa uji signifikansi** pada 23 query → klaim rapuh.

---

## 3. Keputusan Desain (Gabungan Terbaik Dua Pendekatan)

| Aspek | Paper | Rencana kita | Alasan |
|---|---|---|---|
| Bentuk data sintetis | Ringkasan pasal | **Pseudo-query (pertanyaan)** | Cocok distribusi query nyata (praktik standar dense retrieval: doc2query/GPL) |
| Loss | CosineSimilarityLoss (tanpa negatif) | **MultipleNegativesRankingLoss + hard negative** | Retrieval butuh sinyal kontrastif; tanpa negatif rawan collapse |
| Query normalization | ✅ (lever terbesar) | **✅ Diadopsi** | Murah & berdampak besar untuk teks hukum penuh akronim |
| Desain ablasi 5-konfigurasi | ✅ | **✅ Diadopsi** | Memisahkan kontribusi tiap komponen |
| Anti-leakage query uji | ✅ (latih dari sintetis) | **✅ Dipertahankan** | 98 query berlabel = test set suci |
| Uji signifikansi | ❌ | **✅ Wilcoxon** | Kredibilitas yang tak dimiliki paper |

---

## 4. Prinsip Dasar (berlaku di semua fase)

- **98 query berlabel = test set suci.** Tidak pernah dipakai melatih model maupun
  "mengakali" kamus normalisasi. Pelanggaran = data leakage.
- **Setiap perubahan = variabel ablasi.** Normalisasi dan fine-tuning diuji terpisah.
- **BM25 tak tersentuh fine-tuning** — hanya leg semantik yang berubah.

---

## 5. Fase Eksekusi

### Fase 0 — Kunci Baseline
- Jalankan ulang evaluasi, simpan metrik **per-query** (bukan hanya rata-rata) ke
  `results/baseline_perquery.json` → wajib untuk uji Wilcoxon.
- **Ubah `src/evaluate.py`:** tambah fungsi metrik per-query.
- **Deliverable:** angka baseline 5 sistem terkunci + vektor per-query.

### Fase 1 — Query Normalization (pengungkit termurah, dari paper)
- **File baru:** `data/normalization/legal_terms.json` — peta akronim & istilah
  awam→formal per domain:
  - Ketenagakerjaan: `PHK → pemutusan hubungan kerja`, `PKWT → perjanjian kerja waktu tertentu`, `dipecat → pemutusan hubungan kerja`
  - Konsumen: `BPSK → Badan Penyelesaian Sengketa Konsumen`
  - ITE: `medsos → media sosial`, `diretas → akses ilegal`
  - Anak: dst.
- **File baru:** `src/normalize.py` — fungsi `normalize_query(text)`.
- **Desain:** untuk BM25 pakai **ekspansi** (tambah bentuk formal, jangan ganti);
  untuk SBERT, bentuk formal membantu.
- **Integritas:** kamus dibangun dari teks UU + glosarium hukum, BUKAN dengan
  mencoba ekspansi mana yang menaikkan skor query uji.

### Fase 2 — Data Latih Sintetis berbentuk Pertanyaan (perbaikan atas paper)
- **Script baru:** `scripts/07_build_synthetic_queries.py`
  - Tiap dari 859 chunk → 2–3 pertanyaan awam yang dijawab pasal itu (via LLM, atau
    template fallback memakai "judul alami pasal" dari `src/chunk.py`).
  - **Filter round-trip:** buang pseudo-query yang tak menemukan pasal sumbernya
    sendiri di top-k.
- **Hard negative mining dari BM25:** untuk tiap (pseudo-query, pasal positif),
  ambil pasal teratas BM25 yang bukan positif sebagai negatif eksplisit.
- **Output:** `data/train/pairs.jsonl` berisi `(query, positive, hard_negatives[])`.

### Fase 3 — Fine-Tuning dengan MNRL (perbaikan atas paper)
- **Script baru:** `scripts/08_finetune_sbert.py` — `sentence-transformers` +
  `MultipleNegativesRankingLoss`.
- **Base:** `firqaaa/indo-sentence-bert-base`.
- **Hyperparam:** lr 2e-5, 1–3 epoch, warmup 10%, weight decay 0.01, batch sebesar
  memori.
- **Output:** `models/indosbert-legal-ft/`.
- ⚠️ **Kompute:** `config.yaml` saat ini `device: cpu` → latih di CPU lambat.
  Disarankan **Google Colab (GPU)**.

### Fase 4 — Evaluasi Ablasi + Signifikansi
- **Script:** `scripts/09_evaluate_ablation.py`. Matriks:

  | Sistem | tanpa normalisasi | dengan normalisasi |
  |---|:---:|:---:|
  | BM25 | ✓ | ✓ |
  | IndoSBERT pretrained | ✓ | ✓ |
  | IndoSBERT fine-tuned | ✓ | ✓ |
  | Pre-hybrid (BM25+pretrained) | ✓ | ✓ |
  | Fine-hybrid (BM25+FT) | ✓ | ✓ |

- Metrik: R@5, R@10, MRR, NDCG@10.
- **Uji Wilcoxon signed-rank** antar pasangan kunci (pretrained vs FT; pre-hybrid vs
  fine-hybrid) memakai vektor per-query dari Fase 0.

### Fase 5 — Integrasi & Presentasi ✅
- ✅ `config.yaml`: bagian `normalization` (on/off) & `training` ditambahkan.
- ✅ Notebook `notebooks/06_finetuning_evaluasi.ipynb`: tabel ablasi + grafik NDCG@10 +
  grafik kontribusi komponen + boxplot per-query + tabel Wilcoxon + insight. Semua angka
  dibaca otomatis dari `results/ablation.json` agar sinkron.
- ✅ `docs/PROGRESS.md` & dokumen ini diperbarui dengan hasil aktual.

---

## 6. Berkas yang Akan Dibuat / Diubah

**Baru:** (semua ✅ selesai)
- ✅ `src/normalize.py`
- ✅ `data/normalization/legal_terms.json`
- ✅ `scripts/07_build_synthetic_queries.py`
- ✅ `scripts/08_finetune_sbert.py` (+ `scripts/08b_rebuild_index_ft.py`, `notebooks/06_finetune_colab.ipynb`)
- ✅ `scripts/09_evaluate_ablation.py`
- ✅ `notebooks/06_finetuning_evaluasi.ipynb`

**Diubah:** (semua ✅ selesai)
- ✅ `src/evaluate.py` (metrik per-query + Wilcoxon)
- ✅ `config.yaml` (bagian `normalization` & `training`)

---

## 7. Estimasi Waktu (target 12 jam)

| Fase | Waktu | Catatan |
|---|---|---|
| 0 — Kunci baseline | ~45 mnt | Cepat |
| 1 — Normalization | ~2 jam | Titik lambat: review kamus oleh peneliti |
| 2 — Data sintetis | ~1.5 jam | Generasi LLM di latar |
| 3 — Fine-tuning | ~1 jam (GPU) / +2-4 jam (CPU) | Bergantung kompute |
| 4 — Evaluasi + Wilcoxon | ~1.5 jam | Otomatis |
| 5 — Integrasi + notebook | ~2 jam | Notebook paling makan waktu |
| **Total** | **~9 jam (+buffer)** | Muat 12 jam **jika GPU + LLM API tersedia** |

**Versi-dipangkas (jika waktu mepet, ~7 jam):**
- Ablasi 4 sistem inti (lewati pretrained-SBERT-alone).
- Kamus normalisasi cukup 10–15 akronim utama per domain.
- Notebook ringkas (tabel + 1 grafik).

---

## 8. Risiko

- **CPU-only fine-tuning** → +2–4 jam.
- **Review kamus & sampel sintetis** bergantung kecepatan peneliti.
- **Dependency `sentence-transformers`/CUDA di Windows** kadang rewel.

---

## 9. Keputusan yang Dibutuhkan Sebelum Eksekusi

1. **Sumber data latih:** LLM via API / template tanpa LLM / hibrida?
2. **Kompute fine-tuning:** Google Colab (GPU) / CPU lokal / GPU lokal?
3. **Scope:** versi penuh (5 sistem × 2 normalisasi) atau versi-dipangkas?
4. **Titik mulai:** Fase 0 (paling rapi) / Fase 1 (quick win) / langsung fine-tuning?

---

## 10. Nilai untuk Skripsi

Apa pun hasilnya, ada cerita kuat:
1. Bila **normalisasi > fine-tuning** → "untuk korpus hukum kecil, query
   normalization lebih sepadan daripada fine-tuning."
2. Bila **fine-tuning MNRL kita > Cosine paper** → "bentuk data sintetis (pertanyaan
   vs ringkasan) dan pilihan loss menentukan keberhasilan fine-tuning retrieval."
3. **Uji Wilcoxon** → kredibilitas statistik yang tak dimiliki paper pembanding.
