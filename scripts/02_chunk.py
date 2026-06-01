"""Tahap 3: chunking teks bersih -> data/processed/chunks.jsonl

Jalankan:
    python scripts/02_chunk.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.chunk import chunk_by_pasal               # noqa: E402


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["processed_dir"])
    max_tokens = cfg["chunking"]["max_tokens"]

    all_chunks = []
    for doc in cfg["documents"]:
        txt_path = processed / f"{doc['domain']}.txt"
        if not txt_path.exists():
            print(f"[SKIP] belum diekstrak: {txt_path}")
            continue
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_by_pasal(text, doc["domain"], max_tokens=max_tokens)
        print(f"[CHUNK] {doc['domain']}: {len(chunks)} chunk")
        all_chunks.extend(chunks)

    out_path = processed / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"Total {len(all_chunks)} chunk -> {out_path}")


if __name__ == "__main__":
    main()
