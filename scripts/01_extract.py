"""Tahap 1-2: ekstrak teks dari PDF UU di data/raw/ -> data/processed/<domain>.txt

Jalankan:
    python scripts/01_extract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.ingest import extract_text, clean_text    # noqa: E402


def main() -> None:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for doc in cfg["documents"]:
        pdf_path = raw_dir / doc["file"]
        if not pdf_path.exists():
            print(f"[SKIP] PDF tidak ditemukan: {pdf_path}")
            continue
        print(f"[EXTRACT] {doc['file']} -> domain={doc['domain']}")
        text = clean_text(extract_text(pdf_path))
        (out_dir / f"{doc['domain']}.txt").write_text(text, encoding="utf-8")

    print("Selesai. Cek data/processed/")


if __name__ == "__main__":
    main()
