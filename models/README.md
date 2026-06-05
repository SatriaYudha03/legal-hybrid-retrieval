# models/

Folder ini menyimpan **model IndoSBERT** yang dipakai pipeline — baik cache pretrained
maupun model hasil fine-tuning.

> Isi model **tidak di-commit ke git** (lihat `.gitignore`) karena ukurannya besar.
> Yang di-commit hanya README ini.

---

## Model pretrained (cache HuggingFace)

Unduh sekali (butuh internet):

```bash
python scripts/00_download_model.py
```

Model (`firqaaa/indo-sentence-bert-base`, ~500 MB) tersimpan di cache HuggingFace
lokal. Setelah itu seluruh pipeline pretrained bisa berjalan **offline**.

Model yang dipakai: lihat `embedding.model_name` di `config.yaml`.

---

## Model fine-tuned (`indosbert-legal-ft/`)

Model IndoSBERT yang telah dilatih ulang pada 521 pasangan sintetis (query awam ↔ pasal hukum)
menggunakan `MultipleNegativesRankingLoss` + 4 hard negative.

**Cara mengisi:**

1. Jalankan fine-tuning di Google Colab (direkomendasikan — selesai ~2 menit di GPU T4):
   - Buka `notebooks/06_finetune_colab.ipynb`
   - Jalankan semua sel
   - Unduh folder model yang dihasilkan
2. Taruh hasil unduhan di `models/indosbert-legal-ft/`
3. Rebuild FAISS index:
   ```bash
   python scripts/08b_rebuild_index_ft.py
   ```

Atau fine-tuning di CPU lokal (lebih lambat, ~30–60 menit):

```bash
python scripts/08_finetune_sbert.py
```

**Detail pelatihan:**

| Aspek | Nilai |
|---|---|
| Base model | `firqaaa/indo-sentence-bert-base` |
| Loss | `MultipleNegativesRankingLoss` |
| Hard negatives per pasangan | 4 |
| Batch size efektif | 32 (batch=8, grad_accum=4) |
| Learning rate | 2e-5 |
| Epochs | 3 |
| FP16 | Ya (Colab GPU) |
| Training time | ~2 menit (T4) |
| Output dimensi | 768 |
