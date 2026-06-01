"""Tahap 2 — Ekstraksi teks dari PDF UU.

Mengubah PDF undang-undang menjadi raw text yang bersih:
  - ekstrak per halaman dengan pdfplumber,
  - buang nomor halaman & footer/header berulang (mis. "KETENAGAKERJAAN"),
  - buang preamble ("Menimbang"/"Mengingat") sebelum "BAB I",
  - rapikan baris kosong berlebih.

Catatan: newline antar-baris SENGAJA dipertahankan agar penanda "Pasal N"
tetap berada di awal baris (dipakai oleh src/chunk.py). Reflow paragraf
dilakukan saat chunking, bukan di sini.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pdfplumber

# Baris yang hanya berisi angka -> nomor halaman.
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
# Penanda awal isi UU (setelah preamble). Toleran spasi/typo "BAB I".
_BAB_SATU_RE = re.compile(r"^BAB\s+I\b", re.MULTILINE)
# Fallback untuk UU perubahan (mis. UU 35/2014) yang tidak punya "BAB I".
_MEMUTUSKAN_RE = re.compile(r"^MEMUTUSKAN", re.MULTILINE)


def extract_text(pdf_path: str | Path) -> str:
    """Ekstrak seluruh teks dari satu PDF, halaman dipisah newline ganda."""
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
    return "\n\n".join(pages)


def _repeated_short_lines(lines: list[str], min_count: int = 5, max_len: int = 40) -> set[str]:
    """Deteksi header/footer berulang (mis. nama UU yang muncul tiap halaman)."""
    counts = Counter(
        ln for ln in lines
        if 0 < len(ln) <= max_len and not ln.lower().startswith(("pasal", "bab", "ayat"))
    )
    return {ln for ln, c in counts.items() if c >= min_count}


def clean_text(raw: str) -> str:
    """Bersihkan teks hasil ekstraksi dari artefak PDF.

    Membuang: nomor halaman, header/footer berulang, baris kosong, dan
    preamble sebelum "BAB I". Mengembalikan teks dengan newline dipertahankan.
    """
    lines = [ln.strip() for ln in raw.splitlines()]

    footers = _repeated_short_lines(lines)

    cleaned: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if _PAGE_NUM_RE.match(ln):       # nomor halaman
            continue
        if ln in footers:                # header/footer berulang
            continue
        cleaned.append(ln)

    text = "\n".join(cleaned)

    # Buang preamble: mulai dari "BAB I"; fallback ke "MEMUTUSKAN" untuk UU perubahan.
    m = _BAB_SATU_RE.search(text)
    if m is None:
        m = _MEMUTUSKAN_RE.search(text)
    if m:
        text = text[m.start():]

    return text
