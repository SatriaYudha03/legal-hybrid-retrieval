"""Tahap 4-5: bangun index BM25 & embedding IndoSBERT (FAISS) dari chunks.jsonl

Jalankan:
    python scripts/03_build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path        # noqa: E402
from src.bm25_retriever import BM25Retriever            # noqa: E402
from src.semantic_retriever import SemanticRetriever    # noqa: E402


def load_chunks(path: Path) -> tuple[list[str], list[str]]:
    doc_ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj["id"])
            texts.append(obj["text"])
    return doc_ids, texts


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["processed_dir"])
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, texts = load_chunks(processed / "chunks.jsonl")
    print(f"Memuat {len(doc_ids)} chunk")

    # BM25
    bm25 = BM25Retriever(
        k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"],
        use_sastrawi=(cfg["bm25"]["tokenizer"] == "sastrawi"),
    )
    bm25.build(doc_ids, texts)
    bm25.save(index_dir / "bm25.pkl")
    print("BM25 index tersimpan.")

    # IndoSBERT + FAISS
    sem = SemanticRetriever(
        model_name=cfg["embedding"]["model_name"],
        batch_size=cfg["embedding"]["batch_size"],
        device=cfg["embedding"]["device"],
        cache_folder=str(resolve_path(cfg["paths"]["models_dir"])),
    )
    sem.build(doc_ids, texts)
    sem.save(index_dir / "faiss")
    print("FAISS index tersimpan.")


if __name__ == "__main__":
    main()
