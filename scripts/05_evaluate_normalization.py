"""Fase 1 — Query Normalization.

Bandingkan tiga sistem (BM25, IndoSBERT, Hybrid) dengan dan tanpa normalisasi query.
Simpan ke:
  - results/normalization_metrics.json   : rata-rata semua kondisi
  - results/normalization_perquery.json  : skor per-query (untuk Wilcoxon)

Jalankan:
    python scripts/05_evaluate_normalization.py
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
from src.evaluate import evaluate_run, evaluate_run_perquery, wilcoxon_test
from src.normalize import normalize_query


def _build_runs(
    queries: list[dict],
    bm25: BM25Retriever,
    sem: SemanticRetriever,
    top_k: int,
    depth: int,
    rrf_k: int,
    use_norm: bool,
) -> dict[str, dict]:
    runs: dict[str, dict] = {"bm25": {}, "indosbert": {}, "hybrid": {}}
    for q in queries:
        qid = q["id"]
        text = q["text"]
        domain = q.get("domain")

        if use_norm:
            text = normalize_query(text, domain=domain)

        bm25_res   = bm25.search(text, top_k=depth)
        sem_res    = sem.search(text, top_k=depth)
        hybrid_res = reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k, top_k=top_k)

        runs["bm25"][qid]      = dict(bm25_res[:top_k])
        runs["indosbert"][qid] = dict(sem_res[:top_k])
        runs["hybrid"][qid]    = dict(hybrid_res)

    return runs


def _print_table(label: str, summary: dict[str, dict]) -> None:
    header = f"\n=== {label} ==="
    print(header)
    col_w = 14
    metric_keys = list(next(iter(summary.values())).keys())
    header_row = f"  {'Sistem':>12}  " + "  ".join(f"{m:>{col_w}}" for m in metric_keys)
    print(header_row)
    for name, metrics in summary.items():
        row = "  ".join(f"{v:>{col_w}.4f}" for v in metrics.values())
        print(f"  {name:>12}  {row}")


def main() -> None:
    cfg         = load_config()
    index_dir   = resolve_path(cfg["paths"]["index_dir"])
    eval_dir    = resolve_path(cfg["paths"]["eval_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    top_k   = cfg["retrieval"]["top_k"]
    rrf_k   = cfg["fusion"]["k"]
    depth   = cfg["fusion"]["candidate_depth"]
    metrics = cfg["evaluation"]["metrics"]

    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))
    qrels   = json.loads((eval_dir / "qrels.json").read_text(encoding="utf-8"))

    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem  = SemanticRetriever.load(index_dir / "faiss")

    print(f"Query: {len(queries)} | Qrels: {len([k for k in qrels if not k.startswith('_')])} berlabel")

    # --- jalankan kedua kondisi ---
    print("\nMenjalankan tanpa normalisasi...")
    runs_plain = _build_runs(queries, bm25, sem, top_k, depth, rrf_k, use_norm=False)
    print("Menjalankan dengan normalisasi...")
    runs_norm  = _build_runs(queries, bm25, sem, top_k, depth, rrf_k, use_norm=True)

    # --- rata-rata ---
    summary_plain = {n: evaluate_run(r, qrels, metrics) for n, r in runs_plain.items()}
    summary_norm  = {n: evaluate_run(r, qrels, metrics) for n, r in runs_norm.items()}

    _print_table("Tanpa Normalisasi (baseline)", summary_plain)
    _print_table("Dengan Normalisasi (Fase 1)", summary_norm)

    # --- delta ---
    print("\n=== Delta (norm - baseline) ===")
    for name in summary_plain:
        deltas = {m: summary_norm[name][m] - summary_plain[name][m] for m in metrics}
        row = "  ".join(f"{m}={v:+.4f}" for m, v in deltas.items())
        print(f"  {name:>12}: {row}")

    # --- Wilcoxon per sistem pada NDCG@10 ---
    print("\n=== Uji Wilcoxon (NDCG@10, tanpa vs dengan normalisasi) ===")
    perquery_plain = {n: evaluate_run_perquery(r, qrels, metrics) for n, r in runs_plain.items()}
    perquery_norm  = {n: evaluate_run_perquery(r, qrels, metrics) for n, r in runs_norm.items()}

    wtest_results: dict[str, dict] = {}
    for name in runs_plain:
        qids = [qid for qid in qrels if not qid.startswith("_")]
        scores_plain = [perquery_plain[name].get(qid, {}).get("ndcg@10", 0.0) for qid in qids]
        scores_norm  = [perquery_norm[name].get(qid, {}).get("ndcg@10", 0.0)  for qid in qids]
        wtest = wilcoxon_test(scores_plain, scores_norm)
        wtest_results[name] = wtest
        p_str = f"{wtest.get('p_value', float('nan')):.4f}" if "error" not in wtest else wtest["error"]
        print(f"  {name:>12}: W={wtest.get('statistic', float('nan')):.1f}, p={p_str}")

    # --- simpan hasil ---
    output: dict = {
        "tanpa_normalisasi": summary_plain,
        "dengan_normalisasi": summary_norm,
        "wilcoxon_ndcg10":   wtest_results,
    }
    (results_dir / "normalization_metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    perquery_output: dict = {
        "tanpa_normalisasi": perquery_plain,
        "dengan_normalisasi": perquery_norm,
    }
    (results_dir / "normalization_perquery.json").write_text(
        json.dumps(perquery_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nTersimpan:")
    print(f"  {results_dir / 'normalization_metrics.json'}")
    print(f"  {results_dir / 'normalization_perquery.json'}")

    # --- contoh ekspansi query untuk inspeksi ---
    print("\n=== Contoh Ekspansi Query (5 pertama) ===")
    for q in queries[:5]:
        expanded = normalize_query(q["text"], domain=q.get("domain"))
        if expanded != q["text"]:
            print(f"  [{q['id']}] {q['text']}")
            print(f"         => {expanded}")
        else:
            print(f"  [{q['id']}] (tidak berubah) {q['text']}")


if __name__ == "__main__":
    main()
