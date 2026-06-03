"""Tahap 3 — Chunking dokumen menjadi unit retrieval (per Pasal).

Strategi: pecah teks bersih berdasarkan penanda "Pasal N" yang berada di awal
baris. Tiap chunk berisi satu pasal utuh (header + isi), dengan konteks BAB
sebagai metadata. Pasal yang terlalu panjang (> max_tokens) dipecah lagi
menjadi sub-chunk berbasis kata dengan sedikit overlap.

Output tiap chunk:
    {
      "id": "KETENAGAKERJAAN_PASAL_156",
      "domain": "ketenagakerjaan",
      "text": "Pasal 156 ...",
      "metadata": {"pasal": "156", "bab": "BAB XII", "part": 0}
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Penanda pasal & bab di awal baris.
#
# PENTING: heading pasal harus berupa baris yang HANYA berisi "Pasal N"
# (diakhiri akhir baris, lihat ``\s*$``). Tanpa anchor ini, rujukan silang
# yang ter-wrap oleh PDF — mis. baris "Pasal 140, pengusaha dilarang:" di
# tengah badan Pasal 144 — akan salah dikira batas pasal baru, sehingga
# badan pasal terpotong dan lahir chunk sampah. (Bug ini sempat membuat
# ~36% chunk rusak.)
_PASAL_RE = re.compile(r"(?m)^Pasal\s+(\d+[A-Za-z]*)\s*$")
_BAB_RE = re.compile(r"(?m)^BAB\s+([A-Z]+)\b")
# Penanda yang mengakhiri judul BAB (judul ada di antara "BAB X" dan penanda ini).
_BAB_TITLE_STOP_RE = re.compile(r"^(Pasal|Bagian|Paragraf|BAB)\b", re.IGNORECASE)


@dataclass
class Chunk:
    """Satu unit dokumen yang akan diindeks & di-retrieve."""

    id: str
    domain: str
    text: str
    metadata: dict = field(default_factory=dict)  # mis. {"pasal": "156", "bab": "BAB XII"}


def _normalize(text: str) -> str:
    """Reflow: gabungkan baris-baris wrap menjadi satu spasi, rapikan whitespace."""
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _count_tokens(text: str) -> int:
    """Aproksimasi jumlah token (berbasis kata) untuk cek batas IndoSBERT."""
    return len(text.split())


def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Pecah teks panjang menjadi beberapa bagian berbasis kata + overlap."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    parts, start = [], 0
    step = max(1, max_tokens - overlap)
    while start < len(words):
        parts.append(" ".join(words[start:start + max_tokens]))
        start += step
    return parts


def _extract_bab_title(text: str, after: int) -> str:
    """Ambil judul BAB (baris-baris setelah "BAB X" sampai penanda berikutnya).

    Pada UU Indonesia, "BAB X" diikuti judul deskriptif di baris berikutnya,
    mis. "BAB V\\nPELATIHAN KERJA\\nPasal 9". Judul ini dipakai sebagai
    "judul alami" pasal untuk Strategy A pada fine-tuning.
    """
    lines: list[str] = []
    for raw in text[after:].splitlines():
        line = raw.strip()
        if not line:
            if lines:
                break
            continue
        if _BAB_TITLE_STOP_RE.match(line):
            break
        lines.append(line)
        if len(lines) >= 3:  # judul wajar maksimal beberapa baris
            break
    return " ".join(lines).title()


def _bab_at(pos: int, babs: list[tuple[int, str, str]]) -> tuple[str, str]:
    """Kembalikan (label, judul) BAB terdekat sebelum posisi `pos`."""
    current_label, current_title = "", ""
    for bab_pos, label, title in babs:
        if bab_pos <= pos:
            current_label, current_title = label, title
        else:
            break
    return current_label, current_title


def chunk_by_pasal(
    text: str,
    domain: str,
    max_tokens: int = 256,
    overlap: int = 0,
) -> list[Chunk]:
    """Pecah teks satu UU menjadi daftar Chunk berdasarkan pasal."""
    babs = [
        (m.start(), f"BAB {m.group(1)}", _extract_bab_title(text, m.end()))
        for m in _BAB_RE.finditer(text)
    ]
    matches = list(_PASAL_RE.finditer(text))

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}  # untuk menjamin id unik (UU perubahan punya pasal berulang)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pasal_no = m.group(1)
        body = _normalize(text[start:end])
        bab, bab_title = _bab_at(start, babs)

        for part_idx, part_text in enumerate(_split_long(body, max_tokens, overlap)):
            base_id = f"{domain.upper()}_PASAL_{pasal_no}"
            if part_idx > 0:
                base_id += f"_p{part_idx}"
            # dedupe id bila pasal yang sama muncul lebih dari sekali
            count = seen.get(base_id, 0)
            seen[base_id] = count + 1
            chunk_id = base_id if count == 0 else f"{base_id}_dup{count}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    domain=domain,
                    text=part_text,
                    metadata={"pasal": pasal_no, "bab": bab, "part": part_idx},
                )
            )
    return chunks
