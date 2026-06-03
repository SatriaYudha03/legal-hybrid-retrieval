"""Fase 2 — Bangkitkan data latih sintetis (template, TANPA LLM).

Untuk tiap chunk pasal, hasilkan beberapa pseudo-query berbahasa awam, lalu:
  1. FILTER ROUND-TRIP : buang query yang tidak menemukan pasal sumbernya
     sendiri di top-k (hybrid BM25+SBERT). Menjamin pasangan (query, positif)
     benar-benar relevan.
  2. HARD-NEGATIVE MINING : ambil pasal teratas BM25 yang BUKAN pasal sumber
     sebagai contoh negatif yang sulit (untuk MultipleNegativesRankingLoss).

Sumber pertanyaan:
  - Judul BAB (di-derive ulang dari teks .txt; tidak tersimpan di index).
  - Istilah kunci pasal (frekuensi kata isi, stopword Sastrawi dibuang).
  - Frame pertanyaan per domain.

INTEGRITAS: 74 query uji TIDAK pernah dilihat di sini. Data ini murni
diturunkan dari teks UU → tak ada kebocoran test set.

Output: data/train/pairs.jsonl  →  {"query", "positive", "hard_negatives": [...]}

Jalankan:
    python scripts/07_build_synthetic_queries.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.fusion import reciprocal_rank_fusion
from src.chunk import _BAB_RE, _extract_bab_title  # re-derive judul BAB

# ---- parameter generasi ----
MAX_PER_CHUNK   = 3     # maksimal pseudo-query yang disimpan per chunk
N_HARD_NEG      = 4     # jumlah hard negative per pasangan
ROUNDTRIP_TOPK  = 10    # query dianggap valid bila pasal sumber ada di top-k ini
BM25_DEPTH      = 30    # kedalaman BM25 untuk mining hard negative
MIN_TERM_LEN    = 5     # panjang minimal kata isi yang dianggap "istilah kunci"
N_KEY_TERMS     = 4     # jumlah istilah kunci yang diambil per chunk

# Frame pertanyaan awam per domain. "{t}" diisi istilah kunci / topik BAB.
_FRAMES: dict[str, list[str]] = {
    "konsumen": [
        "hak konsumen terkait {t}",
        "kewajiban pelaku usaha soal {t}",
        "aturan tentang {t}",
        "bagaimana ketentuan {t}",
    ],
    "ite": [
        "aturan {t} dalam transaksi elektronik",
        "sanksi terkait {t}",
        "ketentuan tentang {t}",
        "bagaimana hukum mengatur {t}",
    ],
    "anak": [
        "perlindungan anak dari {t}",
        "hak anak atas {t}",
        "kewajiban orang tua soal {t}",
        "aturan tentang {t}",
    ],
}
# Frame umum (dipakai untuk topik BAB, lintas domain).
_GENERIC_FRAMES = ["apa aturan tentang {t}", "ketentuan {t}", "{t}"]

# Kata yang sudah tersirat di frame / terlalu pervasif per domain. Istilah kunci
# semacam ini dilewati agar tak lahir query degeneratif ("hak konsumen terkait konsumen").
_STOP_TERMS: dict[str, set[str]] = {
    "konsumen": {"konsumen", "pelaku", "usaha", "barang"},
    "ite":      {"elektronik", "transaksi", "sistem", "informasi"},
    "anak":     {"anak", "orang"},
}


# Kata fungsi/penghubung yang sering lolos stopword Sastrawi namun tak bermakna
# sebagai istilah kunci hukum.
_EXTRA_STOP = {
    "maupun", "antara", "terhadap", "mengenai", "dimaksud", "tersebut",
    "dilakukan", "sebagaimana", "berdasarkan", "meliputi", "adalah",
    "dalam", "untuk", "yang", "atau", "dengan", "pada", "setiap",
}


def _load_stopwords() -> set[str]:
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        return set(StopWordRemoverFactory().get_stop_words()) | _EXTRA_STOP
    except Exception:
        return set(_EXTRA_STOP)


def base_pasal_id(chunk_id: str) -> str:
    """KONSUMEN_PASAL_5_p1 / _dup1 -> KONSUMEN_PASAL_5 (identitas pasal sebenarnya)."""
    return re.sub(r"_(p|dup)\d+$", "", re.sub(r"_(p|dup)\d+$", "", chunk_id))


def _valid_bab_title(title: str) -> bool:
    """Tolak judul BAB sampah (UU perubahan punya klausa sisipan ber-angka)."""
    if not title or len(title.split()) > 6:
        return False
    low = title.lower()
    if re.search(r"\d", title):                       # judul asli tak berangka
        return False
    if any(bad in low for bad in ("pasal", "disisipkan", "diubah", "dihapus")):
        return False
    return True


def build_bab_titles(processed_dir: Path, domain: str) -> dict[str, str]:
    """Peta {'BAB I': 'Ketentuan Umum', ...} dari teks .txt domain."""
    txt_path = processed_dir / f"{domain}.txt"
    if not txt_path.exists():
        return {}
    text = txt_path.read_text(encoding="utf-8")
    titles: dict[str, str] = {}
    for m in _BAB_RE.finditer(text):
        label = f"BAB {m.group(1)}"
        title = _extract_bab_title(text, m.end())
        if _valid_bab_title(title) and label not in titles:
            titles[label] = title
    return titles


def extract_key_terms(text: str, stopwords: set[str]) -> list[str]:
    """Istilah kunci pasal: kata isi terpadat (frekuensi), stopword dibuang."""
    # buang header "Pasal N" dan angka penomoran
    body = re.sub(r"(?i)^pasal\s+\d+[a-z]*", " ", text)
    words = re.findall(r"[A-Za-z]+", body.lower())
    content = [w for w in words if len(w) >= MIN_TERM_LEN and w not in stopwords]
    if not content:
        return []
    freq = Counter(content)
    top = [w for w, _ in freq.most_common(N_KEY_TERMS)]

    # tambahkan satu bigram natural (dua kata isi berdekatan paling sering)
    bigrams = Counter()
    prev = None
    for w in content:
        if prev:
            bigrams[f"{prev} {w}"] += 1
        prev = w
    if bigrams:
        bg, c = bigrams.most_common(1)[0]
        if c >= 2:
            top.insert(0, bg)
    return top


def generate_candidates(
    chunk: dict, bab_title: str, stopwords: set[str]
) -> list[str]:
    """Hasilkan daftar kandidat pseudo-query untuk satu chunk."""
    domain = chunk["domain"]
    terms = extract_key_terms(chunk["text"], stopwords)
    frames = _FRAMES.get(domain, _GENERIC_FRAMES)

    # buang istilah pervasif yang sudah tersirat di frame domain
    stop_terms = _STOP_TERMS.get(domain, set())
    terms = [t for t in terms if not all(w in stop_terms for w in t.split())]

    cands: list[str] = []

    # 1. dari topik BAB (jika ada & bukan "Ketentuan Umum"/penutup yang generik)
    if bab_title and bab_title.lower() not in {"ketentuan umum", "ketentuan penutup", "ketentuan peralihan"}:
        topic = bab_title.lower()
        cands.append(topic)
        # frame generik hanya untuk topik singkat (hindari pengulangan janggal)
        if len(topic.split()) <= 2:
            cands.append(random.choice(_GENERIC_FRAMES).format(t=topic))

    # 2. dari istilah kunci pasal
    for term in terms[:3]:
        frame = random.choice(frames)
        cands.append(frame.format(t=term))

    # 3. gabungan dua istilah kunci (lebih spesifik ke pasal)
    if len(terms) >= 2:
        cands.append(f"{terms[0]} {terms[1]}")

    # rapikan & deduplikasi
    seen: set[str] = set()
    unique: list[str] = []
    for c in cands:
        c = re.sub(r"\s+", " ", c).strip().lower()
        # buang kata berulang (jaga urutan kemunculan pertama)
        tok_seen: set[str] = set()
        c = " ".join(w for w in c.split() if not (w in tok_seen or tok_seen.add(w)))
        # tolak query yang berakhir kata penghubung (istilah ter-dedup habis)
        if c.split() and c.split()[-1] in {"soal", "tentang", "terkait", "atas", "dari", "dalam", "mengatur"}:
            continue
        if c and c not in seen and len(c.split()) >= 2:  # query minimal 2 kata
            seen.add(c)
            unique.append(c)
    return unique


def main() -> None:
    cfg = load_config()
    random.seed(cfg.get("seed", 42))

    index_dir     = resolve_path(cfg["paths"]["index_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    train_dir     = resolve_path(cfg["paths"].get("train_dir", "data/train"))
    train_dir.mkdir(parents=True, exist_ok=True)

    rrf_k = cfg["fusion"]["k"]

    chunks = [
        json.loads(line)
        for line in (processed_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {c["id"]: c for c in chunks}

    bm25 = BM25Retriever.load(index_dir / "bm25.pkl")
    sem  = SemanticRetriever.load(index_dir / "faiss")
    stopwords = _load_stopwords()

    # judul BAB per domain
    domains = sorted({c["domain"] for c in chunks})
    bab_titles = {d: build_bab_titles(processed_dir, d) for d in domains}

    print(f"Chunk: {len(chunks)} | Domain: {domains}")
    print(f"Membangkitkan & memfilter pseudo-query (round-trip top-{ROUNDTRIP_TOPK})...\n")

    pairs: list[dict] = []
    n_cand_total = 0
    n_accepted = 0
    used_queries: set[str] = set()  # cegah query identik lintas chunk

    for ci, chunk in enumerate(chunks, start=1):
        src_id = chunk["id"]
        src_base = base_pasal_id(src_id)
        bab_title = bab_titles.get(chunk["domain"], {}).get(chunk["metadata"].get("bab", ""), "")

        candidates = generate_candidates(chunk, bab_title, stopwords)
        n_cand_total += len(candidates)

        kept = 0
        for query in candidates:
            if kept >= MAX_PER_CHUNK:
                break
            if query in used_queries:
                continue

            # --- round-trip: hybrid harus menemukan pasal sumber di top-k ---
            bm25_res = bm25.search(query, top_k=ROUNDTRIP_TOPK)
            sem_res  = sem.search(query, top_k=ROUNDTRIP_TOPK)
            fused = reciprocal_rank_fusion([bm25_res, sem_res], k=rrf_k, top_k=ROUNDTRIP_TOPK)
            found_bases = {base_pasal_id(d) for d, _ in fused}
            if src_base not in found_bases:
                continue  # query tidak menuntun ke sumbernya → buang

            # --- hard negative dari BM25 (pasal lain teratas) ---
            hard_negs: list[str] = []
            for did, _ in bm25.search(query, top_k=BM25_DEPTH):
                if base_pasal_id(did) == src_base:
                    continue
                hard_negs.append(by_id[did]["text"])
                if len(hard_negs) >= N_HARD_NEG:
                    break

            pairs.append({
                "query": query,
                "positive": chunk["text"],
                "positive_id": src_id,
                "domain": chunk["domain"],
                "hard_negatives": hard_negs,
            })
            used_queries.add(query)
            kept += 1
            n_accepted += 1

        if ci % 50 == 0:
            print(f"  {ci}/{len(chunks)} chunk diproses, {n_accepted} pasangan diterima")

    out_path = train_dir / "pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- statistik ---
    by_dom = Counter(p["domain"] for p in pairs)
    avg_neg = sum(len(p["hard_negatives"]) for p in pairs) / len(pairs) if pairs else 0
    chunks_covered = len({p["positive_id"] for p in pairs})

    print("\n=== Statistik Data Latih Sintetis ===")
    print(f"  Kandidat dibangkitkan : {n_cand_total}")
    print(f"  Lolos round-trip      : {n_accepted} ({n_accepted/n_cand_total*100:.1f}%)")
    print(f"  Chunk tercakup        : {chunks_covered}/{len(chunks)} ({chunks_covered/len(chunks)*100:.1f}%)")
    print(f"  Per domain            : {dict(by_dom)}")
    print(f"  Rata-rata hard neg    : {avg_neg:.2f}")
    print(f"\nTersimpan: {out_path}")

    # contoh
    print("\n=== Contoh 6 pasangan ===")
    for p in pairs[:6]:
        print(f"  Q: {p['query']}")
        print(f"     -> {p['positive_id']}  (hard_neg: {len(p['hard_negatives'])})")


if __name__ == "__main__":
    main()
