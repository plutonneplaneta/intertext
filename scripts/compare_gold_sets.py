"""Сопоставление двух эталонов: Тихомиров (v7) и академическое ПСС.

Для каждого примечания академического эталона: тот же ли абзац есть у Тихомирова и
пересекаются ли их стихи. Три исхода: согласие (тот же абзац, общий стих), расхождение
(тот же абзац, другие стихи), новое место (абзаца у Тихомирова нет) -- только новые места
годятся как независимая проверка. Печатает и обратную долю: сколько записей Тихомирова
подтверждено вторым комментатором.
"""
from __future__ import annotations

import sys

from oracle_diagnostics import load_gold, verse_matches


def verses(group, all_ids):
    return {s for s in all_ids if verse_matches(s, group)}


def main(tikh: str, acad: str, bible_ids_tsv: str) -> None:
    import csv
    with open(bible_ids_tsv, encoding="utf-8", newline="") as f:
        ids = [r["verse_id"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    T, A = load_gold(tikh), load_gold(acad, min_match=0.6)
    t_by = {}
    for par, g, fr in T:
        t_by.setdefault(par, []).append((verses(g, ids), fr))
    agree = diverge = new = 0
    print(f"{'абзац':14s} {'исход':10s} фрагмент")
    new_recs = []
    for k, (par, g, fr) in enumerate(A):
        av = verses(g, ids)
        if par not in t_by:
            new += 1
            new_recs.append(k)
            print(f"{par:14s} {'НОВОЕ':10s} {fr[:50]}")
        elif any(av & tv for tv, _ in t_by[par]):
            agree += 1
            print(f"{par:14s} {'согласие':10s} {fr[:50]}")
        else:
            diverge += 1
            print(f"{par:14s} {'расхождение':10s} {fr[:50]}")
    print(f"\nакадемических записей {len(A)}: согласие {agree}, расхождение {diverge}, новых мест {new}")
    a_pars = {}
    for par, g, _ in A:
        a_pars.setdefault(par, []).append(verses(g, ids))
    conf = sum(1 for par, g, _ in T if par in a_pars and any(verses(g, ids) & av for av in a_pars[par]))
    print(f"записей Тихомирова, подтверждённых академическим комментарием: {conf} из {len(T)}")
    print("индексы новых мест (по порядку в академическом файле):", new_recs)


if __name__ == "__main__":
    main(*sys.argv[1:4])
