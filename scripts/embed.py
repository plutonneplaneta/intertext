"""Векторное представление (multilingual-e5-base) — семантический слой поверх
лексического. E5 требует префикс "query: "/"passage: " перед текстом;
здесь роман — запрос (ищем, откуда взято), Библия — паспорт (что ищем).

Карта ищется по имени: индексы torch.cuda и nvidia-smi на сервере не совпадают
(torch видит V100 под нулём, а у неё нет ядер в torch cu130). Боевую V100 не трогаем.
Без подходящей карты -- CPU (около 14 минут на всю Библию).
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def find_gpu_by_name(substr: str) -> str:
    for i in range(torch.cuda.device_count()):
        if substr in torch.cuda.get_device_name(i):
            return f"cuda:{i}"
    return "cpu"


def load_tsv(path: str, id_col: str = "verse_id", text_col: str = "text") -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(row[id_col])
            texts.append(row[text_col])
    return ids, texts


def main(tsv_path: str, prefix: str, out_npz: str, id_col: str = "verse_id") -> None:
    ids, texts = load_tsv(tsv_path, id_col=id_col)
    print(f"единиц: {len(ids)}", file=sys.stderr)

    model = SentenceTransformer("intfloat/multilingual-e5-base", device=find_gpu_by_name("3060"))
    prefixed = [f"{prefix} {t}" for t in texts]

    t0 = time.time()
    embeddings = model.encode(
        prefixed, batch_size=32, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    print(f"готово: {embeddings.shape} за {time.time()-t0:.0f}с", file=sys.stderr)

    np.savez(out_npz, ids=np.array(ids, dtype=object), embeddings=embeddings)
    print(f"сохранено -> {out_npz}")


if __name__ == "__main__":
    tsv_path, prefix, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]
    id_col = sys.argv[4] if len(sys.argv) > 4 else "verse_id"
    main(tsv_path, prefix, out_npz, id_col=id_col)
