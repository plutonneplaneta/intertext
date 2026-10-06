"""Контрольные признаки для проверки surprisal: чем ещё может объясняться эффект.

Для каждой пары пула:
  vlen_raw / vlen_ppct   длина стиха в символах и её процентиль внутри пула абзаца.
                         Контекст длиннее -- больше выигрыш в логвероятности,
                         а правильные стихи могут быть систематически длиннее.
  trare_ppct             процентиль tfidf_rare ВНУТРИ пула. Признаки surprisal
                         нормированы внутри пула, прежние скореры -- среди всех
                         31 102 стихов; это отделяет эффект нормировки от эффекта
                         самого сигнала.
  shuf_delta_raw/_ppct   sur_delta, перемешанный между стихами одного абзаца
                         (плацебо): то же распределение значений, связи с парой нет.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict


def ppct(vals):
    m: dict[float, float] = {}
    for i, x in enumerate(sorted(vals)):
        m.setdefault(x, i / len(vals))
    return [m[v] for v in vals]


def main(table_tsv: str, bible_tsv: str, out_tsv: str, seed: int = 0) -> None:
    with open(bible_tsv, encoding="utf-8", newline="") as f:
        vlen = {r["verse_id"]: len(r["text"]) for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    pool = defaultdict(dict)
    with open(table_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            pool[r["target_id"]][r["source_id"]] = (float(r["tfidf_rare_raw"]), float(r["sur_delta_raw"]))
    rng = random.Random(seed)
    with open(out_tsv, "w", encoding="utf-8") as g:
        g.write("target_id\tsource_id\tvlen_raw\tvlen_ppct\ttrare_ppct\tshuf_delta_raw\tshuf_delta_ppct\n")
        for par, d in pool.items():
            sids = list(d)
            lens = [float(vlen.get(s, 0)) for s in sids]
            tr = ppct([d[s][0] for s in sids])
            sur = [d[s][1] for s in sids]
            rng.shuffle(sur)
            sp = ppct(sur)
            lp = ppct(lens)
            for k, s in enumerate(sids):
                g.write(f"{par}\t{s}\t{lens[k]:.0f}\t{lp[k]:.6f}\t{tr[k]:.6f}\t{sur[k]:.4f}\t{sp[k]:.6f}\n")
    print(f"-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
