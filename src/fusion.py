"""Pendekatan 3 — Reciprocal Rank Fusion (RRF).

Menggabungkan beberapa ranking (BM25 & IndoSBERT) menjadi satu ranking akhir.

Formula (Cormack et al., 2009):
    RRF(d) = Σ_r  1 / (k + rank_r(d))

dengan rank dimulai dari 1 dan k konstanta (umumnya 60).

Bagian ini diimplementasikan penuh karena formulanya kecil & tidak
bergantung pada data — sekaligus sebagai referensi gaya kode modul lain.
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Gabungkan beberapa ranking memakai RRF.

    Args:
        rankings: daftar hasil retrieval, tiap elemen berupa daftar
            (doc_id, score) yang SUDAH terurut dari paling relevan.
            Score tidak dipakai oleh RRF (RRF hanya memakai posisi/rank),
            tetapi formatnya dibuat sama dengan output retriever lain.
        k: konstanta RRF.
        top_k: bila diisi, potong hasil akhir ke top_k teratas.

    Returns:
        Daftar (doc_id, rrf_score) terurut menurun berdasarkan skor RRF.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k] if top_k else fused
