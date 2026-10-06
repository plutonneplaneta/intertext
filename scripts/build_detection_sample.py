"""Выборка абзацев для проверки ОБНАРУЖЕНИЯ (а не атрибуции) и «пустой» эталон для пула.

Состав тот же, что в fusion_detection.py: все эталонные абзацы и 600 случайных не
эталонных (seed 0), чтобы числа были сравнимы с прошлой итерацией. Для пула кандидатов
нужен эталон-пустышка: у ВСЕХ абзацев ссылка на несуществующую книгу XXX, поэтому
fusion_retrieval.py build не подмешивает в пул правильные стихи -- иначе у эталонных
абзацев кандидаты оказались бы искусственно лучше, и признак «абзац цитатный» стал бы
утечкой разметки. Пул -- объединение топ-K (POOL_K) по каждому скореру.

Выход: pseudo_gold.tsv (для build) и sample_meta.tsv (target_id, length, is_gold).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from oracle_diagnostics import load_gold

N_NEG = 600


def main(novel_ann: str, gold_tsv: str, pseudo_gold_out: str, meta_out: str) -> None:
    ids, raws = [], []
    with open(novel_ann, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(r["id"])
                raws.append(r.get("raw", ""))
    gold_pars = {g[0] for g in load_gold(gold_tsv)}
    rng = np.random.default_rng(0)
    negs = [t for t in ids if t not in gold_pars]
    sample = sorted(gold_pars) + sorted(rng.choice(negs, size=min(N_NEG, len(negs)), replace=False).tolist())
    length = {i: len(r.split()) for i, r in zip(ids, raws)}
    with open(pseudo_gold_out, "w", encoding="utf-8") as f:
        f.write("novel_verse_id\tmatch_score\tfragment\tbible_refs\n")
        for p in sample:
            f.write(f"{p}\t1.00\tx\tXXX.1.1\n")
    with open(meta_out, "w", encoding="utf-8") as f:
        f.write("target_id\tlength\tis_gold\n")
        for p in sample:
            f.write(f"{p}\t{length[p]}\t{int(p in gold_pars)}\n")
    print(f"абзацев {len(sample)} (эталонных {len(gold_pars)}) -> {pseudo_gold_out}, {meta_out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
