"""Evaluasi retrieval: Recall@K, MRR, NDCG@K.

Diimplementasi mandiri (tanpa pytrec_eval) agar mudah dijalankan di Windows
tanpa C compiler. Struktur data:

    qrels: {query_id: {doc_id: relevance}}   # ground truth (relevance >= 1 = relevan)
    run:   {query_id: {doc_id: score}}       # hasil sistem
"""
from __future__ import annotations

import math


def _ranked_ids(doc_scores: dict[str, float]) -> list[str]:
    """Urutkan doc_id berdasarkan skor menurun."""
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Proporsi dokumen relevan yang muncul di top-k."""
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / posisi dokumen relevan pertama (0 bila tidak ada)."""
    for i, d in enumerate(ranked_ids, start=1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain pada top-k (mendukung relevansi bertingkat)."""
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k], start=1):
        rel = relevance.get(d, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _parse_metric(name: str) -> tuple[str, int | None]:
    """'recall@5' -> ('recall', 5); 'mrr' -> ('mrr', None)."""
    if "@" in name:
        base, k = name.split("@")
        return base.lower(), int(k)
    return name.lower(), None


def evaluate_run(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metrics: list[str],
) -> dict[str, float]:
    """Hitung rata-rata metrik di seluruh query (hanya query yang ada di qrels)."""
    totals = {m: 0.0 for m in metrics}
    n = 0
    for qid, relevance in qrels.items():
        if qid.startswith("_"):          # lewati kunci komentar seperti "_comment"
            continue
        relevant_ids = {d for d, r in relevance.items() if r > 0}
        ranked = _ranked_ids(run.get(qid, {}))
        n += 1
        for m in metrics:
            base, k = _parse_metric(m)
            if base == "recall":
                totals[m] += recall_at_k(ranked, relevant_ids, k)
            elif base == "mrr":
                totals[m] += reciprocal_rank(ranked, relevant_ids)
            elif base == "ndcg":
                totals[m] += ndcg_at_k(ranked, relevance, k)
            else:
                raise ValueError(f"Metrik tidak dikenal: {m}")

    return {m: (totals[m] / n if n else 0.0) for m in metrics}


def evaluate_run_perquery(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Kembalikan metrik per-query sebagai {query_id: {metrik: nilai}}.

    Dipakai untuk uji Wilcoxon signed-rank antar sistem.
    """
    results: dict[str, dict[str, float]] = {}
    for qid, relevance in qrels.items():
        if qid.startswith("_"):
            continue
        relevant_ids = {d for d, r in relevance.items() if r > 0}
        ranked = _ranked_ids(run.get(qid, {}))
        row: dict[str, float] = {}
        for m in metrics:
            base, k = _parse_metric(m)
            if base == "recall":
                row[m] = recall_at_k(ranked, relevant_ids, k)
            elif base == "mrr":
                row[m] = reciprocal_rank(ranked, relevant_ids)
            elif base == "ndcg":
                row[m] = ndcg_at_k(ranked, relevance, k)
            else:
                raise ValueError(f"Metrik tidak dikenal: {m}")
        results[qid] = row
    return results


def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Uji Wilcoxon signed-rank antara dua vektor skor per-query.

    Kembalikan {'statistic': W, 'p_value': p}.
    Butuh scipy; bila tidak tersedia kembalikan NaN.
    """
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(scores_a, scores_b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    except ImportError:
        return {"statistic": float("nan"), "p_value": float("nan"), "error": "scipy not installed"}
    except Exception as e:
        return {"statistic": float("nan"), "p_value": float("nan"), "error": str(e)}
