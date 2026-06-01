# models/

Folder ini menyimpan **cache lokal model IndoSBERT** agar:
- Tidak perlu internet saat presentasi/demo (anti-gagal).
- Proyek portabel (pindah laptop tanpa download ulang).
- Versi model jelas & reproducible.

## Cara mengisi

Jalankan sekali (butuh internet):

```bash
python scripts/00_download_model.py
```

Model (~500 MB) akan terunduh dari HuggingFace dan tersimpan di sini.
Setelah itu, seluruh pipeline bisa berjalan **offline**.

> Folder isi model **tidak di-commit ke git** (lihat .gitignore) karena besar.
> Yang di-commit hanya README ini.

## Model yang dipakai

Lihat `embedding.model_name` di `config.yaml`. Default:
`firqaaa/indo-sentence-bert-base`
