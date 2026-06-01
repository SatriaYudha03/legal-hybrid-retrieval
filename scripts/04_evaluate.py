"""Tahap 6 + Evaluasi: jalankan ketiga model (BM25, IndoSBERT, Hybrid-RRF)
atas seluruh query evaluasi, lalu hitung metrik.

Jalankan:
    python scripts/04_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path        # noqa: E402
from src.bm25_retriever import BM25Retriever            # noqa: E402
from src.semantic_retriever import SemanticRetriever    # noqa: E402
from src.fusion import reciprocal_rank_fusion           # noqa: E402
from src.evaluate import evaluate_run                   # noqa: E402


def main() -> None:
    cfg = load_config()
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    eval_dir = resolve_path(cfg["paths"]["eval_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["retrieval"]["top_k"]
    rrf_k = cfg["fusion"]["k"]
    depth = cfg["fusion"]["candidate_depth"]  # kedalaman kandidat yang difusikan

    # Muat query evaluasi & ground truth
    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))
    qrels = json.loads((eval_dir / "qrels.json").read_text(encoding="utf-8"))

    # Muat index
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem = SemanticRetriever.load(index_dir / "faiss")

    runs: dict[str, dict] = {"bm25": {}, "indosbert": {}, "hybrid": {}}
    for q in queries:
        qid, text = q["id"], q["text"]
        # Ambil kandidat dalam (depth) lalu fusikan; potong ke top_k hanya di akhir.
        bm25_res = bm25.search(text, top_k=depth)
        sem_res = sem.search(text, top_k=depth)
        hybrid_res = reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k, top_k=top_k)

        runs["bm25"][qid] = dict(bm25_res[:top_k])
        runs["indosbert"][qid] = dict(sem_res[:top_k])
        runs["hybrid"][qid] = dict(hybrid_res)

    # Evaluasi tiap model
    summary = {}
    for name, run in runs.items():
        summary[name] = evaluate_run(run, qrels, cfg["evaluation"]["metrics"])
        print(f"{name:>10}: {summary[name]}")

    (results_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Metrik tersimpan -> {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
