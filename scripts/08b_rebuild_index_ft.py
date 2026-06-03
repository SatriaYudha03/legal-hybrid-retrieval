"""Rebuild FAISS index lokal menggunakan model fine-tuned dari Colab.

Jalankan SETELAH mengekstrak indosbert-legal-ft.zip ke models/indosbert-legal-ft/:
    python scripts/08b_rebuild_index_ft.py

Output: data/index/faiss_ft.faiss + data/index/faiss_ft.meta.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Paksa mode offline — cegah SentenceTransformer mencoba kontak HuggingFace Hub
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]      = "1"

# Output langsung tampil (hindari buffering saat di-pipe / redirect)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print(">>> Memuat library (numpy, faiss, sentence-transformers)... mohon tunggu ~10-30s",
      flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import load_config, resolve_path

print(">>> Library siap.\n", flush=True)


def main() -> None:
    cfg       = load_config()
    tcfg      = cfg["training"]
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    ft_model  = resolve_path(tcfg["output_dir"])

    # --- validasi model ---
    if not ft_model.exists():
        print(f"ERROR: model fine-tuned tidak ditemukan di {ft_model}")
        sys.exit(1)

    model_files = [f.name for f in ft_model.iterdir()]
    print(f"[1/4] Model FT ditemukan: {ft_model}")
    print(f"      File: {model_files}")

    # --- muat chunk ---
    doc_ids, texts = [], []
    for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            doc_ids.append(obj["id"])
            texts.append(obj["text"])
    print(f"\n[2/4] Chunk dimuat: {len(doc_ids)}")

    # --- muat model (lokal, offline) ---
    print(f"\n[3/4] Memuat model dari disk (offline)...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(
        str(ft_model),
        device=cfg["embedding"]["device"],
        local_files_only=True,   # ← jangan kontak HF Hub
    )
    print(f"      Model dimuat ({time.time()-t0:.1f}s)", flush=True)

    # --- encode chunk dengan progress manual ---
    batch_size = cfg["embedding"]["batch_size"]
    n          = len(texts)
    n_batches  = (n + batch_size - 1) // batch_size
    print(f"\n      Encoding {n} chunk dalam {n_batches} batch (batch_size={batch_size})...",
          flush=True)

    all_emb = []
    t_enc   = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        all_emb.append(emb)
        done    = min(i + batch_size, n)
        pct     = done / n * 100
        elapsed = time.time() - t_enc
        eta     = (elapsed / done * (n - done)) if done > 0 else 0
        print(f"      [{done:3d}/{n}] {pct:5.1f}%  elapsed={elapsed:.1f}s  ETA={eta:.0f}s",
              flush=True)

    embeddings = np.vstack(all_emb)
    print(f"      Encoding selesai ({time.time()-t_enc:.1f}s) — shape: {embeddings.shape}",
          flush=True)

    # --- bangun & simpan FAISS ---
    print(f"\n[4/4] Membangun FAISS index...", flush=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    ft_index_path = index_dir / tcfg["ft_index_name"]
    faiss.write_index(index, str(ft_index_path.with_suffix(".faiss")))
    meta = {
        "doc_ids":      doc_ids,
        "model_name":   str(ft_model),
        "batch_size":   batch_size,
        "device":       cfg["embedding"]["device"],
        "cache_folder": None,
    }
    ft_index_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    total = time.time() - t0
    print(f"\nFAISS index FT tersimpan: {ft_index_path}.faiss  ({len(doc_ids)} chunk)")
    print(f"Total waktu: {total:.1f}s")
    print("\nSiap untuk Fase 4: python scripts/09_evaluate_ablation.py")


if __name__ == "__main__":
    main()
