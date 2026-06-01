# Integrasi BM25 dan IndoSBERT Menggunakan Reciprocal Rank Fusion untuk Pencarian Dokumen Hukum Indonesia Multi-Domain

## Ringkasan Proyek

Proyek ini bertujuan membangun sistem pencarian dokumen hukum Indonesia yang mampu memahami baik pencocokan kata kunci maupun makna dari sebuah pertanyaan. Sistem akan menggabungkan pendekatan lexical retrieval menggunakan BM25 dan semantic retrieval menggunakan IndoSBERT, kemudian mengombinasikan hasil keduanya menggunakan Reciprocal Rank Fusion (RRF).

Dataset terdiri dari empat undang-undang yang mewakili domain hukum yang berbeda, yaitu Ketenagakerjaan, Perlindungan Konsumen, Informasi dan Transaksi Elektronik (ITE), dan Perlindungan Anak. Dengan menggunakan beberapa domain hukum yang berbeda, penelitian ini dapat mengevaluasi kemampuan model dalam melakukan pencarian lintas domain hukum Indonesia.

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

Karena itu, proyek ini menggabungkan BM25 dan IndoSBERT agar memperoleh kelebihan dari kedua pendekatan tersebut.

---

# Tujuan

1. Membangun sistem pencarian dokumen hukum Indonesia berbasis BM25.
2. Membangun sistem pencarian dokumen hukum Indonesia berbasis IndoSBERT.
3. Menggabungkan BM25 dan IndoSBERT menggunakan Reciprocal Rank Fusion (RRF).
4. Membandingkan performa BM25, IndoSBERT, dan Hybrid Retrieval.
5. Mengevaluasi efektivitas Transformer untuk pencarian dokumen hukum Indonesia.

---

# Dataset

## Dokumen yang Digunakan

### 1. UU Ketenagakerjaan

Domain:
- Hubungan kerja
- PHK
- Upah
- Pesangon
- Serikat pekerja

Jumlah halaman:
- ±230 halaman

---

### 2. UU Perlindungan Konsumen

Domain:
- Hak konsumen
- Ganti rugi
- Barang cacat
- Perlindungan pembeli

Jumlah halaman:
- ±46 halaman

---

### 3. UU Informasi dan Transaksi Elektronik (ITE)

Domain:
- Informasi elektronik
- Dokumen elektronik
- Transaksi elektronik
- Kejahatan siber

Jumlah halaman:
- ±38 halaman

---

### 4. UU Perlindungan Anak

Domain:
- Hak anak
- Pendidikan
- Kekerasan terhadap anak
- Perlindungan khusus

Jumlah halaman:
- ±66 halaman

---

# Arsitektur Sistem

## Pendekatan 1: BM25

Input:
- Query pengguna

Proses:
- Tokenisasi
- Pencarian BM25

Output:
- Ranking dokumen berdasarkan kemiripan kata

Contoh:

Query:

> hak konsumen atas barang rusak

BM25 akan mencari dokumen yang mengandung kata:

- hak
- konsumen
- barang
- rusak

---

## Pendekatan 2: IndoSBERT

Input:
- Query pengguna
- Dokumen hukum

Proses:

1. Query diubah menjadi embedding.
2. Dokumen diubah menjadi embedding.
3. Similarity dihitung menggunakan cosine similarity.

Output:
- Ranking dokumen berdasarkan kemiripan makna.

Contoh:

Query:

> barang yang dibeli tidak sesuai pesanan

IndoSBERT dapat menemukan pasal yang membahas:

> kompensasi atas barang yang tidak sesuai perjanjian

meskipun kata-katanya berbeda.

---

## Pendekatan 3: Hybrid Retrieval

BM25 menghasilkan ranking pertama.

IndoSBERT menghasilkan ranking kedua.

Kedua ranking digabung menggunakan:

### Reciprocal Rank Fusion (RRF)

Formula:

RRF(d) = Σ 1 / (k + rank(d))

Umumnya:

k = 60

Semakin tinggi skor RRF, semakin tinggi posisi dokumen pada hasil akhir.

---

# Pipeline Sistem

## Tahap 1

Mengumpulkan dokumen hukum.

Output:

PDF UU.

---

## Tahap 2

Ekstraksi teks dari PDF.

Output:

Raw text.

---

## Tahap 3

Chunking dokumen.

Contoh:

Pasal 1
Pasal 2
Pasal 3

atau

Chunk per paragraf.

Output:

```json
{
  "id": "DOC_001",
  "domain": "ketenagakerjaan",
  "text": "Isi pasal..."
}
```

---

## Tahap 4

Membangun indeks BM25.

Output:

BM25 index.

---

## Tahap 5

Membuat embedding menggunakan IndoSBERT.

Output:

Vector database.

---

## Tahap 6

Menjalankan retrieval.

Input:

Query pengguna.

Output:

Top-K dokumen.

---

# Evaluasi

## Model yang Dibandingkan

### Model 1

BM25

---

### Model 2

IndoSBERT

---

### Model 3

BM25 + IndoSBERT + RRF

---

# Query Evaluasi

Contoh query:

## Ketenagakerjaan

- hak pekerja yang dipecat
- pesangon PHK
- jam kerja maksimum

## Konsumen

- barang rusak setelah dibeli
- hak pembeli online
- kompensasi produk cacat

## ITE

- bukti elektronik di pengadilan
- transaksi elektronik yang sah
- pencemaran nama baik online

## Perlindungan Anak

- hak anak memperoleh pendidikan
- perlindungan anak dari kekerasan
- kewajiban orang tua terhadap anak

Target:

100 query total.

---

# Metrik Evaluasi

## Recall@5

Apakah dokumen relevan muncul pada 5 hasil teratas.

---

## Recall@10

Apakah dokumen relevan muncul pada 10 hasil teratas.

---

## MRR

Mean Reciprocal Rank.

Mengukur seberapa cepat dokumen relevan pertama ditemukan.

---

## NDCG@10

Mengukur kualitas urutan ranking hasil pencarian.

---

# Teknologi

## Bahasa Pemrograman

Python

## Library Retrieval

- rank-bm25

## Transformer

- sentence-transformers
- transformers

## Model

- IndoSBERT

## Evaluasi

- scikit-learn
- pytrec_eval

## Penyimpanan Vector

- FAISS

---

# Hasil yang Diharapkan

Hipotesis penelitian:

1. IndoSBERT memberikan hasil yang lebih baik daripada BM25 pada query bahasa alami.
2. BM25 lebih baik untuk istilah hukum yang spesifik.
3. Hybrid BM25 + IndoSBERT menggunakan RRF menghasilkan performa terbaik secara keseluruhan.
4. Integrasi retrieval leksikal dan semantic retrieval mampu meningkatkan kualitas pencarian dokumen hukum Indonesia multi-domain.