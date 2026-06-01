"""Tool bantu pelabelan ground truth (qrels) secara interaktif.

Untuk tiap query, menampilkan kandidat top-N hasil GABUNGAN BM25 + IndoSBERT
(pooling ala TREC), lalu Anda tinggal menandai pasal mana yang relevan.
Hasil disimpan bertahap ke data/eval/qrels.json (bisa berhenti & lanjut nanti).

Jalankan di terminal Anda sendiri (bukan otomatis):
    python scripts/label_qrels.py            # hanya query yang belum dilabeli
    python scripts/label_qrels.py --all      # ulangi semua, termasuk yang sudah ada
    python scripts/label_qrels.py --pool 20  # tampilkan 20 kandidat (default 15)

Cara menilai (per query):
    - Ketik nomor kandidat PALING relevan (grade 2), pisah spasi.  Contoh: 1 3
    - Lalu nomor relevan pendukung (grade 1).                      Contoh: 5
    - Enter kosong = lewati bagian itu.
    - 's' = skip query ini,  'q' = simpan & keluar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.fusion import reciprocal_rank_fusion


def load_chunks(path: Path) -> dict[str, dict]:
    """id -> {text, metadata}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["id"]] = o
    return out


def parse_nums(raw: str, n: int) -> list[int]:
    """Ubah '1 3 5' menjadi indeks valid [0..n-1]."""
    idxs = []
    for tok in raw.replace(",", " ").split():
        if tok.isdigit():
            i = int(tok) - 1
            if 0 <= i < n:
                idxs.append(i)
    return idxs


def save_qrels(path: Path, qrels: dict) -> None:
    comment = qrels.get("_comment", "")
    body = {k: v for k, v in qrels.items() if k != "_comment"}
    ordered = {"_comment": comment, **dict(sorted(body.items()))}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="ulangi termasuk yang sudah dilabeli")
    ap.add_argument("--pool", type=int, default=15, help="jumlah kandidat ditampilkan")
    args = ap.parse_args()

    cfg = load_config()
    index_dir = resolve_path(cfg["paths"]["index_dir"])
    eval_dir = resolve_path(cfg["paths"]["eval_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    depth = cfg["fusion"]["candidate_depth"]
    rrf_k = cfg["fusion"]["k"]

    chunks = load_chunks(processed / "chunks.jsonl")
    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))

    qrels_path = eval_dir / "qrels.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8")) if qrels_path.exists() else {}

    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem = SemanticRetriever.load(index_dir / "faiss")

    todo = [q for q in queries if args.all or q["id"] not in qrels]
    print(f"Akan melabeli {len(todo)} dari {len(queries)} query. "
          f"(s=skip, q=simpan&keluar)\n")

    for qi, q in enumerate(todo, 1):
        qid, text = q["id"], q["text"]
        # pooling: gabungkan kandidat BM25 + IndoSBERT
        pooled = reciprocal_rank_fusion(
            [bm25.search(text, depth), sem.search(text, depth)], k=rrf_k, top_k=args.pool
        )
        print("=" * 78)
        print(f"[{qi}/{len(todo)}]  {qid}  ({q.get('domain','')}/{q.get('type','')})")
        print(f"QUERY: {text}\n")
        for i, (cid, _score) in enumerate(pooled, 1):
            meta = chunks.get(cid, {}).get("metadata", {})
            snippet = chunks.get(cid, {}).get("text", "")[:95]
            print(f"  {i:2}. [{cid}]  {meta.get('bab','')}")
            print(f"      {snippet}")

        g2 = input("\n  Nomor PALING relevan (grade 2): ").strip()
        if g2.lower() == "q":
            break
        if g2.lower() == "s":
            continue
        g1 = input("  Nomor relevan pendukung (grade 1): ").strip()

        judgment = {}
        for i in parse_nums(g2, len(pooled)):
            judgment[pooled[i][0]] = 2
        for i in parse_nums(g1, len(pooled)):
            judgment.setdefault(pooled[i][0], 1)

        qrels[qid] = judgment
        save_qrels(qrels_path, qrels)
        print(f"  -> tersimpan ({len(judgment)} pasal relevan).\n")

    save_qrels(qrels_path, qrels)
    labeled = sum(1 for q in queries if q["id"] in qrels)
    print(f"\nSelesai. Total {labeled}/{len(queries)} query sudah punya ground truth.")
    print(f"File: {qrels_path}")


if __name__ == "__main__":
    main()
