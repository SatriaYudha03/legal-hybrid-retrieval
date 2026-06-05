"""Pencarian interaktif: ketik pertanyaan/pasal, dapatkan pasal yang relevan.

Default: Fine-Hybrid + Normalisasi (konfigurasi terbaik dari penelitian).
  - Model: IndoSBERT fine-tuned (indosbert-legal-ft)
  - Index FAISS: faiss_ft
  - Fusion: Reciprocal Rank Fusion (k=60)
  - Normalisasi query: aktif (ekspansi akronim + sinonim hukum)

Jalankan:
    python scripts/05_search.py                        # Fine-Hybrid + normalisasi (default)
    python scripts/05_search.py --no-norm              # Fine-Hybrid tanpa normalisasi
    python scripts/05_search.py --method bm25          # hanya lexical
    python scripts/05_search.py --method indosbert     # hanya semantik (fine-tuned)
    python scripts/05_search.py --model pretrained     # gunakan FAISS pretrained
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
from src.normalize import normalize_query               # noqa: E402

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
    parser = argparse.ArgumentParser(
        description="Pencarian pasal hukum — Fine-Hybrid + Normalisasi (default terbaik).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", choices=["hybrid", "bm25", "indosbert"], default="hybrid",
        help="Metode retrieval (default: hybrid).",
    )
    parser.add_argument(
        "--model", choices=["finetuned", "pretrained"], default="finetuned",
        help="Model IndoSBERT yang digunakan (default: finetuned).",
    )
    parser.add_argument(
        "--no-norm", action="store_true",
        help="Matikan normalisasi query (default: normalisasi aktif).",
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
        "--reject-threshold", type=float, default=0.35, metavar="T",
        help="Tolak query jika max cosine similarity < T (default: 0.35). "
             "Set 0 untuk matikan filter.",
    )
    parser.add_argument(
        "-q", "--query", default=None,
        help="Langsung cari query ini lalu keluar (tanpa mode interaktif).",
    )
    args = parser.parse_args()

    use_norm      = not args.no_norm
    threshold     = args.reject_threshold  # 0 = nonaktif
    index_dir     = resolve_path(cfg["paths"]["index_dir"])
    processed     = resolve_path(cfg["paths"]["processed_dir"])
    depth         = cfg["fusion"]["candidate_depth"]
    rrf_k         = cfg["fusion"]["k"]

    # Pilih index FAISS sesuai model.
    faiss_index = "faiss_ft" if args.model == "finetuned" else "faiss"

    # Muat index & metadata chunk.
    print("Memuat index & data pasal …")
    chunk_map = load_chunk_map(processed / "chunks.jsonl")
    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem: SemanticRetriever | None = None
    if args.method in ("hybrid", "indosbert"):
        model_label = "fine-tuned" if args.model == "finetuned" else "pretrained"
        print(f"Memuat model IndoSBERT {model_label} (sekali di awal, mohon tunggu) …")
        sem = SemanticRetriever.load(index_dir / faiss_index)
        sem._load_model()

    CAKUPAN = "UU ITE  |  UU Perlindungan Konsumen  |  UU Perlindungan Anak"

    def run_and_show(query: str) -> None:
        # 1. Normalisasi query.
        query_input = normalize_query(query) if use_norm else query
        if use_norm and query_input != query:
            print(f"  [normalisasi] {query!r}  ->  {query_input!r}")

        # 2. Out-of-scope detection via max cosine similarity.
        if threshold > 0 and sem is not None:
            top1 = sem.search(query_input, top_k=1)
            if top1:
                max_sim = top1[0][1]
                if max_sim < threshold:
                    print(f"\n  [di luar cakupan]  skor relevansi tertinggi: {max_sim:.4f}  "
                          f"(threshold: {threshold})")
                    print(f"  Sistem ini hanya mencakup: {CAKUPAN}")
                    print("  Coba ajukan pertanyaan seputar transaksi elektronik, "
                          "perlindungan konsumen, atau hak anak.")
                    return
                # Tampilkan skor agar pengguna tahu seberapa relevan query-nya.
                print(f"  [relevansi] max cosine similarity: {max_sim:.4f}  "
                      f"(threshold: {threshold})  -> dalam cakupan")

        # 3. Jalankan retrieval, tampilkan 3 hasil teratas.
        hits = search(query_input, args.method, bm25, sem, depth, rrf_k, top_k=3)
        if not hits:
            print("  (tidak ada hasil)")
            return
        norm_label   = "+ normalisasi" if use_norm else "tanpa normalisasi"
        model_lbl    = args.model
        print(f"\n=== Top-3 pasal paling relevan  "
              f"[{args.method} | {model_lbl} | {norm_label}] ===")
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
    norm_status = "AKTIF" if use_norm else "nonaktif"
    model_label = "fine-tuned (terbaik)" if args.model == "finetuned" else "pretrained"
    thr_status  = f"{threshold}" if threshold > 0 else "nonaktif"
    print("\n" + "=" * 65)
    print("  Pencarian Pasal Hukum Indonesia")
    print(f"  Metode    : {args.method}  |  Model: {model_label}")
    print(f"  Normalisasi: {norm_status}  |  Threshold reject: {thr_status}")
    print(f"  Cakupan   : {CAKUPAN}")
    print("  Ketik 'keluar' atau tekan Ctrl+C untuk berhenti.")
    print("=" * 65)
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
