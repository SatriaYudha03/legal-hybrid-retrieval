"""Fase 3 — Fine-Tuning IndoSBERT dengan MultipleNegativesRankingLoss (CPU lokal).

Perbaikan atas paper acuan: memakai MNRL + hard negative (sinyal kontrastif),
bukan CosineSimilarityLoss tanpa negatif. Data latih: pseudo-query awam (Fase 2).

Alur:
  1. Muat data/train/pairs.jsonl → Dataset (anchor, positive, negative_1..N).
  2. Latih `firqaaa/indo-sentence-bert-base` (dari cache lokal models/) dengan MNRL.
  3. Simpan model ke models/indosbert-legal-ft/.
  4. Encode ulang 378 chunk → bangun FAISS index FT terpisah (data/index/faiss_ft)
     agar Fase 4 bisa membandingkan pretrained vs fine-tuned.

Jalankan:
    python scripts/08_finetune_sbert.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.semantic_retriever import SemanticRetriever


def load_pairs(path: Path, num_neg: int) -> "list[dict]":
    """Baca pairs.jsonl; pastikan tiap baris punya >= num_neg hard negative."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        negs = p.get("hard_negatives", [])
        if len(negs) < num_neg:
            continue  # lewati pasangan yang kekurangan negatif (jaga kolom seragam)
        row = {"anchor": p["query"], "positive": p["positive"]}
        for i in range(num_neg):
            row[f"negative_{i+1}"] = negs[i]
        rows.append(row)
    return rows


def main() -> None:
    cfg = load_config()
    seed       = cfg.get("seed", 42)
    tcfg       = cfg["training"]
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    train_dir  = resolve_path(cfg["paths"]["train_dir"])
    index_dir  = resolve_path(cfg["paths"]["index_dir"])
    processed  = resolve_path(cfg["paths"]["processed_dir"])
    out_dir    = resolve_path(tcfg["output_dir"])
    num_neg    = int(tcfg["num_hard_negatives"])

    # cache offline (sama seperti SemanticRetriever)
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(models_dir))

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    # --- data latih ---
    rows = load_pairs(train_dir / tcfg["pairs_file"], num_neg)
    train_dataset = Dataset.from_list(rows)
    print(f"Data latih: {len(rows)} pasangan (anchor + positive + {num_neg} hard negatives)")

    # --- model base (dari cache lokal) ---
    model = SentenceTransformer(
        cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
        cache_folder=str(models_dir),
    )
    model.max_seq_length = int(tcfg["max_seq_length"])
    print(f"Model base dimuat: {cfg['embedding']['model_name']} (max_seq_len={model.max_seq_length})")

    # --- loss & argumen ---
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tcfg["epochs"]),
        per_device_train_batch_size=int(tcfg["batch_size"]),
        learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        weight_decay=float(tcfg["weight_decay"]),
        fp16=False, bf16=False,                       # CPU
        batch_sampler=BatchSamplers.NO_DUPLICATES,    # cegah duplikat in-batch (false negative)
        logging_steps=10,
        save_strategy="no",                           # simpan manual di akhir
        report_to=[],                                 # tanpa wandb dsb.
        seed=seed,
        dataloader_num_workers=0,                     # Windows aman
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss
    )
    print(f"\nMulai fine-tuning: {tcfg['epochs']} epoch, batch {tcfg['batch_size']}, lr {tcfg['learning_rate']} (CPU)\n")
    trainer.train()

    # --- simpan model ---
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"\nModel fine-tuned tersimpan: {out_dir}")

    # --- rebuild FAISS index dengan model FT (terpisah dari pretrained) ---
    print("\nMembangun ulang FAISS index dengan model fine-tuned...")
    doc_ids, texts = [], []
    for line in (processed / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            doc_ids.append(obj["id"])
            texts.append(obj["text"])

    sem_ft = SemanticRetriever(
        model_name=str(out_dir),                      # path lokal model FT
        batch_size=cfg["embedding"]["batch_size"],
        device=cfg["embedding"]["device"],
    )
    sem_ft.build(doc_ids, texts)
    ft_index_path = index_dir / tcfg["ft_index_name"]
    sem_ft.save(ft_index_path)
    print(f"FAISS index FT tersimpan: {ft_index_path}.faiss ({len(doc_ids)} chunk)")
    print("\nFase 3 selesai. Index pretrained ('faiss') tetap utuh untuk perbandingan ablasi.")


if __name__ == "__main__":
    main()
