"""Helper internal: kumpulkan pool kandidat (BM25+IndoSBERT) untuk semua query
ke satu file teks RINGKAS untuk ditinjau/dilabeli. Output: data/eval/_pools.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.fusion import reciprocal_rank_fusion

POOL = 12
SNIPPET = 85


def main() -> None:
    cfg = load_config()
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    eval_dir = resolve_path(cfg["paths"]["eval_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    depth = cfg["fusion"]["candidate_depth"]
    rrf_k = cfg["fusion"]["k"]

    chunks = {}
    with open(processed / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            chunks[o["id"]] = o

    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem = SemanticRetriever.load(index_dir / "faiss")

    lines = []
    for q in queries:
        pooled = reciprocal_rank_fusion(
            [bm25.search(q["text"], depth), sem.search(q["text"], depth)],
            k=rrf_k, top_k=POOL,
        )
        lines.append(f"### {q['id']} [{q['domain']}/{q['type']}] :: {q['text']}")
        for i, (cid, _s) in enumerate(pooled, 1):
            snip = re.sub(r"\s+", " ", chunks.get(cid, {}).get("text", ""))[:SNIPPET]
            lines.append(f"  {i:2}. {cid} :: {snip}")
        lines.append("")

    (eval_dir / "_pools.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Pools ringkas tersimpan: {len(queries)} query -> {eval_dir / '_pools.txt'}")


if __name__ == "__main__":
    main()
