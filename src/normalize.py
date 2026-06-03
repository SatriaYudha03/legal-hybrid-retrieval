"""Query normalization: ekspansi akronim & sinonim awam → terminologi formal UU.

Strategi: query expansion (append sinonim formal di belakang query asli).
Dengan cara ini BM25 tetap bisa mencocokkan kata-kata asli, sementara
SBERT mendapat sinyal tambahan dari terminologi formal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "normalization" / "legal_terms.json"
_TERMS: dict | None = None


def _load_terms() -> dict:
    global _TERMS
    if _TERMS is None:
        _TERMS = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    return _TERMS


def _expand_abbreviations(text: str, abbrevs: dict[str, str]) -> str:
    for abbr, expansion in abbrevs.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)
    return text


def _expand_synonyms(text: str, synonyms: dict[str, list[str]]) -> str:
    """Tambahkan sinonim formal di akhir teks (tanpa duplikat kata yang sudah ada)."""
    text_lower = text.lower()
    additions: list[str] = []

    for informal, formals in synonyms.items():
        # Cek apakah frasa informal ada di teks (cocokkan frasa utuh)
        if informal.lower() in text_lower:
            for formal in formals:
                # Hanya tambahkan jika belum ada di teks
                if formal.lower() not in text_lower:
                    additions.append(formal)

    if additions:
        # Deduplicate sambil menjaga urutan
        seen: set[str] = set()
        unique: list[str] = []
        for term in additions:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                unique.append(term)
        text = text + " " + " ".join(unique)

    return text


def normalize_query(text: str, domain: str | None = None) -> str:
    """Normalisasi query dengan ekspansi akronim dan sinonim hukum.

    Args:
        text:   teks query asli
        domain: 'konsumen' | 'ite' | 'anak' | None

    Returns:
        teks query diperluas dengan terminologi formal UU
    """
    terms = _load_terms()

    # Tahap 1: ekspansi akronim (selalu diterapkan)
    text = _expand_abbreviations(text, terms.get("abbreviations", {}))

    # Tahap 2: sinonim global
    text = _expand_synonyms(text, terms.get("global", {}))

    # Tahap 3: sinonim per domain
    synonyms = terms.get("synonyms", {})
    if domain and domain in synonyms:
        text = _expand_synonyms(text, synonyms[domain])
    else:
        # Domain tidak diketahui → coba semua domain
        for d_syns in synonyms.values():
            text = _expand_synonyms(text, d_syns)

    return text
