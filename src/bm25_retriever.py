"""Pendekatan 1 — Lexical retrieval dengan BM25.

Membungkus rank-bm25 (BM25Okapi) + tokenisasi Bahasa Indonesia. Index
dipersist sebagai pickle agar tidak perlu dibangun ulang tiap kali.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Stemmer Sastrawi di-load lazy (mahal) & hanya bila use_sastrawi=True.
_stemmer = None
_stopwords: set[str] = set()


def _ensure_sastrawi() -> None:
    global _stemmer, _stopwords
    if _stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        _stemmer = StemmerFactory().create_stemmer()
        _stopwords = set(StopWordRemoverFactory().get_stop_words())


def tokenize(text: str, use_sastrawi: bool = False) -> list[str]:
    """Tokenisasi untuk BM25: lowercase + ambil kata; opsional stemming ID."""
    tokens = _TOKEN_RE.findall(text.lower())
    if use_sastrawi:
        _ensure_sastrawi()
        tokens = [_stemmer.stem(t) for t in tokens if t not in _stopwords]
    return tokens


class BM25Retriever:
    """Index & pencarian BM25 atas korpus chunk."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_sastrawi: bool = False):
        self.k1 = k1
        self.b = b
        self.use_sastrawi = use_sastrawi
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []

    def build(self, doc_ids: list[str], texts: list[str]) -> None:
        """Bangun index BM25 dari daftar dokumen."""
        self.doc_ids = list(doc_ids)
        corpus = [tokenize(t, self.use_sastrawi) for t in texts]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Kembalikan [(chunk_id, score), ...] terurut menurun."""
        if self.bm25 is None:
            raise RuntimeError("Index belum dibangun. Panggil build() atau load().")
        scores = self.bm25.get_scores(tokenize(query, self.use_sastrawi))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_k]]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "doc_ids": self.doc_ids,
                 "k1": self.k1, "b": self.b, "use_sastrawi": self.use_sastrawi},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(k1=data["k1"], b=data["b"], use_sastrawi=data["use_sastrawi"])
        obj.bm25 = data["bm25"]
        obj.doc_ids = data["doc_ids"]
        return obj
