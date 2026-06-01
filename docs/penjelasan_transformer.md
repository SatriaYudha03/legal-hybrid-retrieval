# Materi Transformer — dari Teori ke Studi Kasus Pencarian Dokumen Hukum

Dokumen ini menjelaskan konsep **Transformer** dan mengaitkannya dengan studi kasus
proyek: pencarian dokumen hukum Indonesia menggunakan **IndoSBERT** (sebuah model
Transformer) yang dipadukan dengan **BM25** melalui **Reciprocal Rank Fusion**.

---

## 1. Motivasi: Dari Pencocokan Kata ke Pemahaman Makna

Metode pencarian klasik seperti **BM25** bekerja dengan **mencocokkan kata**. Ia
menghitung seberapa sering kata pada query muncul di dokumen. Masalahnya, bahasa
manusia kaya akan **sinonim dan parafrasa**:

| Query pengguna | Teks Undang-Undang | Cocok secara kata? |
|---|---|---|
| "berapa pesangon kalau **dipecat**?" | "**pemutusan hubungan kerja** dan hak atas uang pesangon" | ❌ kata "dipecat" ≠ "pemutusan hubungan kerja" |

BM25 bisa gagal di sini karena tidak ada kata yang sama persis. Kita butuh model
yang memahami bahwa "dipecat" dan "pemutusan hubungan kerja" **bermakna sama**.
Di sinilah **Transformer** berperan.

---

## 2. Apa Itu Transformer?

**Transformer** adalah arsitektur jaringan saraf yang diperkenalkan oleh Vaswani
dkk. (2017) dalam paper *"Attention is All You Need"*. Ia dirancang untuk memproses
**data berurutan** (seperti teks) dan menjadi fondasi hampir semua model bahasa
modern (BERT, GPT, dll).

Inovasi utamanya: mengganti pemrosesan berurutan (seperti pada RNN/LSTM) dengan
mekanisme **attention** yang memproses **seluruh kata sekaligus** dan secara
eksplisit memodelkan **hubungan antar-kata**, sejauh apa pun jaraknya dalam kalimat.

### Komponen utama
1. **Input Embedding** — tiap kata/token diubah jadi vektor angka.
2. **Positional Encoding** — menambahkan informasi *urutan* kata (karena attention
   sendiri tidak peka urutan).
3. **Self-Attention** — inti Transformer (lihat bagian 3).
4. **Multi-Head Attention** — beberapa "sudut pandang" attention sekaligus.
5. **Feed-Forward Network** — transformasi non-linear per token.
6. **Residual connection + Layer Normalization** — menstabilkan pelatihan.

---

## 3. Self-Attention — Jantung Transformer

**Self-attention** memungkinkan tiap kata "memperhatikan" kata lain dalam kalimat
untuk membangun maknanya berdasarkan **konteks**.

Contoh: pada kalimat *"bank itu berada di tepi sungai"* vs *"saya menabung di bank"*,
kata **"bank"** memiliki makna berbeda. Self-attention membuat representasi "bank"
menyesuaikan diri dengan kata-kata di sekitarnya (sungai vs menabung).

### Cara kerja (intuisi Query–Key–Value)
Tiap token menghasilkan tiga vektor:
- **Query (Q)** — "apa yang saya cari?"
- **Key (K)** — "informasi apa yang saya tawarkan?"
- **Value (V)** — "isi informasi saya."

Bobot perhatian dihitung dengan mencocokkan Query satu kata terhadap Key semua kata,
lalu dipakai untuk merata-rata Value:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √dₖ ) · V
```

- `Q · Kᵀ` → skor kemiripan tiap pasang kata.
- `√dₖ` → penskalaan agar gradien stabil.
- `softmax` → mengubah skor jadi bobot (total 1).
- Hasil akhir → representasi tiap kata yang sudah "diperkaya konteks".

### Multi-Head Attention
Alih-alih satu attention, Transformer menjalankan beberapa "kepala" (*head*)
secara paralel. Tiap head bisa menangkap relasi berbeda (mis. sintaksis,
semantik, koreferensi). Hasilnya digabung — sehingga model menangkap banyak jenis
hubungan sekaligus.

---

## 4. BERT — Transformer untuk Memahami Bahasa

**BERT** (Bidirectional Encoder Representations from Transformers, Google 2018)
adalah Transformer yang **hanya memakai bagian encoder**. Ia dilatih pada teks
masif dengan dua tugas:
- **Masked Language Modeling** — menebak kata yang disembunyikan.
- **Next Sentence Prediction** — menebak apakah dua kalimat berurutan.

Karena membaca konteks dari **dua arah** (kiri & kanan sekaligus), BERT
menghasilkan pemahaman makna kata yang sangat kaya. **IndoBERT** adalah BERT yang
dilatih pada korpus bahasa Indonesia.

---

## 5. Sentence-BERT (SBERT) — dari Kata ke *Kalimat*

Masalah BERT untuk pencarian: BERT biasa menghasilkan embedding **per token**, dan
untuk membandingkan dua kalimat ia harus memproses keduanya bersamaan
(*cross-encoder*) — sangat lambat bila harus membandingkan 1 query dengan ribuan
dokumen.

**Sentence-BERT (SBERT)** menyelesaikan ini: ia memodifikasi BERT (dengan struktur
*siamese* dan **pooling**) agar menghasilkan **satu vektor embedding untuk seluruh
kalimat**. Dengan begitu:

1. Tiap dokumen diubah jadi **satu vektor** (sekali, di awal — disimpan di indeks).
2. Query diubah jadi **satu vektor**.
3. Kemiripan dihitung cepat dengan **cosine similarity**.

**IndoSBERT** = Sentence-BERT untuk bahasa Indonesia. Inilah model Transformer yang
dipakai di proyek ini (`firqaaa/indo-sentence-bert-base`, menghasilkan embedding
**768 dimensi**).

### Embedding + Cosine Similarity = Pencarian Semantik
Dua teks yang **bermakna mirip** akan memiliki vektor yang **berdekatan** di ruang
768 dimensi, walau kata-katanya berbeda:

```
cosine(A, B) = (A · B) / (||A|| · ||B||)     →  nilai 1 = identik, 0 = tak terkait
```

Inilah alasan IndoSBERT bisa mencocokkan "dipecat" dengan "pemutusan hubungan kerja".

> Catatan implementasi: di proyek ini embedding dinormalisasi L2, sehingga
> **inner product == cosine similarity**, dan pencarian dipercepat dengan **FAISS**.

---

## 6. Kaitan ke Studi Kasus

| Pendekatan | Berbasis | Kekuatan | Kelemahan |
|---|---|---|---|
| **BM25** | Statistik kata | Istilah hukum spesifik, nomor pasal, frasa eksak | Buta terhadap sinonim/parafrasa |
| **IndoSBERT** (Transformer) | Makna (embedding) | Memahami bahasa awam & parafrasa | Bisa meleset pada istilah teknis yang harus eksak |
| **Hybrid (RRF)** | Gabungan keduanya | Mengambil kelebihan keduanya | Sensitif terhadap kedalaman kandidat fusi |

Studi kasus ini **mengajarkan secara konkret** apa yang Transformer tambahkan:
ketika BM25 gagal karena kata berbeda, IndoSBERT (Transformer) menyelamatkan
pencarian lewat pemahaman makna — dan sebaliknya. **Reciprocal Rank Fusion**
menggabungkan kedua ranking:

```
RRF(d) = Σ  1 / (k + rank_r(d)),    k = 60
```

---

## 7. Ringkasan Alur Berpikir

```
Teks  →  [Transformer: embedding + self-attention]  →  Vektor makna (768-dim)
Query →  [Transformer: embedding]                    →  Vektor makna (768-dim)
                         │
              cosine similarity  →  ranking semantik (IndoSBERT)
                         │
        digabung (RRF) dengan ranking leksikal (BM25)
                         │
                  Top-K dokumen hukum
```

**Kesimpulan:** Transformer (melalui IndoSBERT) memungkinkan sistem memahami
*makna* pertanyaan, bukan sekadar mencocokkan kata. Proyek ini mendemonstrasikan
dan mengevaluasi manfaat tersebut pada domain dokumen hukum Indonesia.

---

## Referensi
- Vaswani et al. (2017), *Attention is All You Need*.
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers*.
- Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Cormack et al. (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*.
