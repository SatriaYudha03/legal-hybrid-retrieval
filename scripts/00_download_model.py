"""Tahap 0: download model IndoSBERT ke folder models/ (cache lokal).

Jalankan SEKALI saat ada internet:
    python scripts/00_download_model.py

Setelah ini, seluruh pipeline bisa jalan offline (penting saat demo presentasi).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Arahkan cache HuggingFace ke folder proyek SEBELUM import library.
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir)

    model_name = cfg["embedding"]["model_name"]
    print(f"Mengunduh model: {model_name}")
    print(f"Tujuan cache   : {models_dir}")
    print("Ini butuh internet & mungkin beberapa menit (~500 MB)...\n")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=str(models_dir))

    # Uji cepat: encode satu kalimat untuk memastikan model benar-benar siap.
    vec = model.encode(["uji coba kalimat hukum Indonesia"])
    print(f"\nBerhasil. Dimensi embedding: {vec.shape[1]}")
    print("Model tersimpan lokal — pipeline siap dijalankan offline.")


if __name__ == "__main__":
    main()
