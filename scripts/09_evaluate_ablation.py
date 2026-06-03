"""Fase 4 — Evaluasi Ablasi Lengkap + Uji Signifikansi Wilcoxon.

Matriks 5 sistem x 2 kondisi normalisasi (10 kombinasi):
  Sistem       : BM25 | IndoSBERT-pretrained | IndoSBERT-FT | Pre-hybrid | Fine-hybrid
  Normalisasi  : tanpa | dengan

Wilcoxon signed-rank (NDCG@10) untuk pasangan kunci:
  - pretrained vs FT (IndoSBERT saja)
  - pre-hybrid vs fine-hybrid
  - tanpa norm vs dengan norm (per sistem)

Output:
  results/ablation.json        -- rata-rata semua 10 kombinasi
  results/ablation_perquery.json -- skor per-query (untuk tabel skripsi)

Jalankan:
    python scripts/09_evaluate_ablation.py
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


def run_retrieval(
    queries: list[dict],
    bm25: BM25Retriever,
    sem_pre: SemanticRetriever,
    sem_ft: SemanticRetriever,
    top_k: int,
    depth: int,
    rrf_k: int,
    use_norm: bool,
) -> dict[str, dict]:
    """Jalankan 5 sistem sekaligus untuk satu kondisi normalisasi."""
    runs = {
        "bm25": {}, "pretrained": {}, "finetuned": {},
        "pre_hybrid": {}, "fine_hybrid": {},
    }
    for q in queries:
        qid    = q["id"]
        text   = normalize_query(q["text"], q.get("domain")) if use_norm else q["text"]
        domain = q.get("domain")
        _ = domain  # domain sudah dipakai di normalize_query

        bm25_res   = bm25.search(text, top_k=depth)
        pre_res    = sem_pre.search(text, top_k=depth)
        ft_res     = sem_ft.search(text, top_k=depth)

        pre_hybrid = reciprocal_rank_fusion([bm25_res, pre_res], k=rrf_k, top_k=top_k)
        ft_hybrid  = reciprocal_rank_fusion([bm25_res, ft_res],  k=rrf_k, top_k=top_k)

        runs["bm25"][qid]        = dict(bm25_res[:top_k])
        runs["pretrained"][qid]  = dict(pre_res[:top_k])
        runs["finetuned"][qid]   = dict(ft_res[:top_k])
        runs["pre_hybrid"][qid]  = dict(pre_hybrid)
        runs["fine_hybrid"][qid] = dict(ft_hybrid)

    return runs


def _print_table(label: str, summary: dict[str, dict], metrics: list[str]) -> None:
    COL = 12
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    header = f"  {'Sistem':<16}" + "".join(f"{m:>{COL}}" for m in metrics)
    print(header)
    print(f"  {'-'*16}" + "-"*COL*len(metrics))
    for name, vals in summary.items():
        row = f"  {name:<16}" + "".join(f"{vals[m]:>{COL}.4f}" for m in metrics)
        print(row)


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
    qids    = [k for k in qrels if not k.startswith("_")]

    print(f"Query: {len(queries)} | Berlabel: {len(qids)}")

    bm25     = BM25Retriever.load(index_dir / "bm25.pkl")
    sem_pre  = SemanticRetriever.load(index_dir / "faiss")
    ft_path  = index_dir / cfg["training"]["ft_index_name"]
    if not ft_path.with_suffix(".faiss").exists():
        print(f"ERROR: FAISS index FT tidak ditemukan di {ft_path}.faiss")
        print("Jalankan dulu: python scripts/08b_rebuild_index_ft.py")
        sys.exit(1)
    sem_ft = SemanticRetriever.load(ft_path)

    # --- jalankan dua kondisi ---
    print("\nMenjalankan tanpa normalisasi (5 sistem)...")
    runs_plain = run_retrieval(queries, bm25, sem_pre, sem_ft, top_k, depth, rrf_k, use_norm=False)
    print("Menjalankan dengan normalisasi (5 sistem)...")
    runs_norm  = run_retrieval(queries, bm25, sem_pre, sem_ft, top_k, depth, rrf_k, use_norm=True)

    # --- rata-rata ---
    summ_plain = {n: evaluate_run(r, qrels, metrics) for n, r in runs_plain.items()}
    summ_norm  = {n: evaluate_run(r, qrels, metrics) for n, r in runs_norm.items()}

    _print_table("TANPA Normalisasi", summ_plain, metrics)
    _print_table("DENGAN Normalisasi", summ_norm, metrics)

    # --- delta ---
    print(f"\n{'='*60}")
    print("  Delta (dengan - tanpa normalisasi)")
    print(f"{'='*60}")
    COL = 12
    header = f"  {'Sistem':<16}" + "".join(f"{m:>{COL}}" for m in metrics)
    print(header)
    for name in summ_plain:
        deltas = {m: summ_norm[name][m] - summ_plain[name][m] for m in metrics}
        row = f"  {name:<16}" + "".join(f"{deltas[m]:>+{COL}.4f}" for m in metrics)
        print(row)

    # --- per-query ---
    pq_plain = {n: evaluate_run_perquery(r, qrels, metrics) for n, r in runs_plain.items()}
    pq_norm  = {n: evaluate_run_perquery(r, qrels, metrics) for n, r in runs_norm.items()}

    # --- Wilcoxon ---
    METRIC = "ndcg@10"

    def scores(pq: dict, name: str) -> list[float]:
        return [pq[name].get(qid, {}).get(METRIC, 0.0) for qid in qids]

    wilcoxon_results: dict[str, dict] = {}

    pairs_to_test = [
        # (label, pq_dict_A, system_A, pq_dict_B, system_B)
        ("pretrained_vs_FT [tanpa_norm]",   pq_plain, "pretrained",  pq_plain, "finetuned"),
        ("pretrained_vs_FT [dengan_norm]",  pq_norm,  "pretrained",  pq_norm,  "finetuned"),
        ("pre_hybrid_vs_fine_hybrid [tanpa]", pq_plain,"pre_hybrid",  pq_plain, "fine_hybrid"),
        ("pre_hybrid_vs_fine_hybrid [dgn]",  pq_norm,  "pre_hybrid",  pq_norm,  "fine_hybrid"),
        ("norm_effect BM25",                 pq_plain, "bm25",        pq_norm,  "bm25"),
        ("norm_effect fine_hybrid",          pq_plain, "fine_hybrid", pq_norm,  "fine_hybrid"),
    ]

    print(f"\n{'='*60}")
    print(f"  Uji Wilcoxon signed-rank ({METRIC})")
    print(f"{'='*60}")
    for label, pq_a, sys_a, pq_b, sys_b in pairs_to_test:
        w = wilcoxon_test(scores(pq_a, sys_a), scores(pq_b, sys_b))
        wilcoxon_results[label] = w
        p    = w.get("p_value", float("nan"))
        stat = w.get("statistic", float("nan"))
        sig  = "**SIGNIFIKAN**" if isinstance(p, float) and p < 0.05 else ""
        p_str = f"{p:.4f}" if isinstance(p, float) and p == p else w.get("error", "nan")
        print(f"  {label:<42}  W={stat:.1f}  p={p_str}  {sig}")

    # --- simpan ---
    ablation_out = {
        "tanpa_normalisasi":  summ_plain,
        "dengan_normalisasi": summ_norm,
        "wilcoxon": wilcoxon_results,
    }
    (results_dir / "ablation.json").write_text(
        json.dumps(ablation_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pq_out = {
        "tanpa_normalisasi":  pq_plain,
        "dengan_normalisasi": pq_norm,
    }
    (results_dir / "ablation_perquery.json").write_text(
        json.dumps(pq_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nTersimpan:")
    print(f"  {results_dir / 'ablation.json'}")
    print(f"  {results_dir / 'ablation_perquery.json'}")


if __name__ == "__main__":
    main()
