"""Pendekatan 2 — Semantic retrieval dengan IndoSBERT + FAISS.

Alur:
  1. Encode seluruh chunk menjadi embedding (sentence-transformers),
     ter-normalisasi L2 sehingga inner product == cosine similarity.
  2. Simpan ke FAISS IndexFlatIP.
  3. Saat query: encode query -> cari nearest neighbor di FAISS.

Cache model diarahkan ke folder lokal models/ (lihat models_dir di config)
agar bisa berjalan offline saat demo presentasi.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class SemanticRetriever:
    """Index & pencarian semantik berbasis embedding IndoSBERT."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        cache_folder: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache_folder = cache_folder
        self.model = None             # SentenceTransformer
        self.index = None             # faiss.Index
        self.doc_ids: list[str] = []

    def _load_model(self) -> None:
        """Lazy-load SentenceTransformer (mahal, hanya saat dibutuhkan)."""
        if self.model is not None:
            return
        if self.cache_folder:
            os.environ.setdefault("HF_HOME", self.cache_folder)
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", self.cache_folder)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name, device=self.device, cache_folder=self.cache_folder
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode daftar teks menjadi matriks (N x D) float32 ter-normalisasi L2."""
        self._load_model()
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,     # -> cosine == dot product
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return emb.astype("float32")

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        """Bangun FAISS index dari embedding seluruh chunk."""
        import faiss

        self.doc_ids = list(doc_ids)
        emb = self.encode(texts)
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (ternormalisasi)
        self.index.add(emb)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Kembalikan [(chunk_id, cosine_score), ...] terurut menurun."""
        if self.index is None:
            raise RuntimeError("Index belum dibangun. Panggil build() atau load().")
        q = self.encode([query])
        scores, idx = self.index.search(q, top_k)
        return [
            (self.doc_ids[i], float(s))
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]

    def save(self, path: str | Path) -> None:
        """Simpan FAISS index + doc_ids + metadata model (prefix path)."""
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        meta = {
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "cache_folder": self.cache_folder,
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "SemanticRetriever":
        import faiss

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(
            model_name=meta["model_name"],
            batch_size=meta["batch_size"],
            device=meta["device"],
            cache_folder=meta.get("cache_folder"),
        )
        obj.index = faiss.read_index(str(path.with_suffix(".faiss")))
        obj.doc_ids = meta["doc_ids"]
        return obj
