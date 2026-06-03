"""Pencarian interaktif: ketik pertanyaan/pasal, dapatkan pasal yang relevan.

Memakai pipeline yang sama dengan evaluasi: BM25 + IndoSBERT difusikan
dengan Reciprocal Rank Fusion (RRF). Kandidat diambil sedalam
``fusion.candidate_depth`` lalu dipotong ke top_k hanya di akhir.

Jalankan:
    python scripts/05_search.py                 # mode hybrid (default)
    python scripts/05_search.py --method bm25    # hanya lexical
    python scripts/05_search.py --method indosbert
    python scripts/05_search.py --top-k 5
    python scripts/05_search.py -q "hak cuti pekerja perempuan"   # sekali jalan, lalu keluar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Terminal Windows default sering cp1252; paksa UTF-8 agar "—" / "…" tak jadi "�".
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path        # noqa: E402
from src.bm25_retriever import BM25Retriever            # noqa: E402
from src.semantic_retriever import SemanticRetriever    # noqa: E402
from src.fusion import reciprocal_rank_fusion           # noqa: E402

# Nama domain yang lebih ramah dibaca daripada slug internal.
DOMAIN_LABEL = {
    "ketenagakerjaan": "UU Ketenagakerjaan",
    "konsumen": "UU Perlindungan Konsumen",
    "ite": "UU Informasi dan Transaksi Elektronik",
    "anak": "UU Perlindungan Anak",
}


def load_chunk_map(path: Path) -> dict[str, dict]:
    """Muat chunks.jsonl menjadi peta id -> {domain, text, metadata}."""
    chunks: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks[obj["id"]] = obj
    return chunks


def snippet(text: str, n: int = 320) -> str:
    """Potong teks panjang menjadi cuplikan agar output terminal rapi."""
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " …"


def format_hit(rank: int, chunk_id: str, score: float, chunk: dict, full: bool) -> str:
    """Susun satu baris hasil pencarian untuk ditampilkan."""
    meta = chunk.get("metadata", {})
    domain = DOMAIN_LABEL.get(chunk.get("domain", ""), chunk.get("domain", "-"))
    pasal = meta.get("pasal", "?")
    bab = meta.get("bab", "")
    header = f"  {rank}. Pasal {pasal} — {domain}"
    if bab:
        header += f" ({bab})"
    header += f"   [skor {score:.4f}]"
    body = chunk["text"] if full else snippet(chunk["text"])
    return f"{header}\n     {body}"


def search(
    query: str,
    method: str,
    bm25: BM25Retriever,
    sem: SemanticRetriever | None,
    depth: int,
    rrf_k: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Jalankan retrieval sesuai metode terpilih dan kembalikan top_k hasil."""
    if method == "bm25":
        return bm25.search(query, top_k=top_k)
    if method == "indosbert":
        return sem.search(query, top_k=top_k)
    # hybrid: ambil kandidat dalam dari kedua retriever, fusikan, potong di akhir
    bm25_res = bm25.search(query, top_k=depth)
    sem_res = sem.search(query, top_k=depth)
    return reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k, top_k=top_k)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Pencarian pasal hukum (BM25 + IndoSBERT + RRF).")
    parser.add_argument(
        "--method", choices=["hybrid", "bm25", "indosbert"], default="hybrid",
        help="Metode retrieval (default: hybrid).",
    )
    parser.add_argument(
        "--top-k", type=int, default=cfg["retrieval"]["top_k"],
        help="Jumlah pasal yang ditampilkan.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Tampilkan teks pasal penuh (default: cuplikan).",
    )
    parser.add_argument(
        "-q", "--query", default=None,
        help="Langsung cari query ini lalu keluar (tanpa mode interaktif).",
    )
    args = parser.parse_args()

    index_dir = resolve_path(cfg["paths"]["index_dir"])
    processed = resolve_path(cfg["paths"]["processed_dir"])
    depth = cfg["fusion"]["candidate_depth"]
    rrf_k = cfg["fusion"]["k"]

    # Muat index & metadata chunk.
    print("Memuat index & data pasal …")
    chunk_map = load_chunk_map(processed / "chunks.jsonl")
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem: SemanticRetriever | None = None
    if args.method in ("hybrid", "indosbert"):
        print("Memuat model IndoSBERT (sekali di awal, mohon tunggu) …")
        sem = SemanticRetriever.load(index_dir / "faiss")
        # paksa model termuat sekarang agar query pertama tidak terasa lambat
        sem._load_model()

    def run_and_show(query: str) -> None:
        hits = search(query, args.method, bm25, sem, depth, rrf_k, args.top_k)
        if not hits:
            print("  (tidak ada hasil)")
            return
        print(f"\n=== {len(hits)} pasal paling relevan (metode: {args.method}) ===")
        for rank, (cid, score) in enumerate(hits, start=1):
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            print(format_hit(rank, cid, score, chunk, args.full))

    # Mode sekali jalan.
    if args.query is not None:
        run_and_show(args.query)
        return

    # Mode interaktif.
    print("\n" + "=" * 60)
    print(" Pencarian Pasal Hukum  —  ketik pertanyaan lalu tekan Enter")
    print(f" Metode: {args.method} | top-k: {args.top_k}")
    print(" Ketik 'keluar', 'exit', atau Ctrl+C untuk berhenti.")
    print("=" * 60)
    while True:
        try:
            query = input("\nTanya pasal > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa.")
            break
        if not query:
            continue
        if query.lower() in {"keluar", "exit", "quit", "q"}:
            print("Sampai jumpa.")
            break
        run_and_show(query)


if __name__ == "__main__":
    main()
