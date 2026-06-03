"""Fase 0 — Kunci baseline sebelum fine-tuning.

Jalankan tiga sistem (BM25, IndoSBERT, Hybrid) pada 74 query (3 UU),
simpan:
  - results/metrics.json          : rata-rata per sistem
  - results/baseline_perquery.json: skor per-query per sistem (untuk Wilcoxon)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.fusion import reciprocal_rank_fusion
from src.evaluate import evaluate_run, evaluate_run_perquery


def main() -> None:
    cfg = load_config()
    index_dir   = resolve_path(cfg["paths"]["index_dir"])
    eval_dir    = resolve_path(cfg["paths"]["eval_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    top_k = cfg["retrieval"]["top_k"]
    rrf_k = cfg["fusion"]["k"]
    depth = cfg["fusion"]["candidate_depth"]
    metrics = cfg["evaluation"]["metrics"]

    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))
    qrels   = json.loads((eval_dir / "qrels.json").read_text(encoding="utf-8"))

    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem  = SemanticRetriever.load(index_dir / "faiss")

    runs: dict[str, dict] = {"bm25": {}, "indosbert": {}, "hybrid": {}}
    for q in queries:
        qid, text = q["id"], q["text"]
        bm25_res   = bm25.search(text, top_k=depth)
        sem_res    = sem.search(text, top_k=depth)
        hybrid_res = reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k, top_k=top_k)

        runs["bm25"][qid]      = dict(bm25_res[:top_k])
        runs["indosbert"][qid] = dict(sem_res[:top_k])
        runs["hybrid"][qid]    = dict(hybrid_res)

    # --- rata-rata ---
    print("\n=== Baseline (rata-rata) ===")
    summary: dict[str, dict] = {}
    for name, run in runs.items():
        summary[name] = evaluate_run(run, qrels, metrics)
        row = "  ".join(f"{m}={v:.4f}" for m, v in summary[name].items())
        print(f"  {name:>10}: {row}")

    (results_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- per-query (untuk Wilcoxon) ---
    perquery: dict[str, dict] = {}
    for name, run in runs.items():
        perquery[name] = evaluate_run_perquery(run, qrels, metrics)

    (results_dir / "baseline_perquery.json").write_text(
        json.dumps(perquery, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nTersimpan:")
    print(f"  {results_dir / 'metrics.json'}")
    print(f"  {results_dir / 'baseline_perquery.json'}")
    print(f"\nTotal query dievaluasi: {len([k for k in qrels if not k.startswith('_')])}")


if __name__ == "__main__":
    main()
