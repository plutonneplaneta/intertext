"""Почему места «с мостиком» не взяты: ранг правильного стиха у каждого скорера.

В таблице fusion_retrieval.py у каждой пары (абзац, стих) есть процентиль балла
скорера среди всех стихов, а правильные стихи подмешаны в пул принудительно.
Значит глобальный ранг правильного стиха у скорера равен (1 - pct) * N, и
пересчитывать оценку заново не нужно.

Вход: таблица пула, TSV мест из gold_findability.py (rec_id в первой колонке).
Вывод: по каждому месту -- лучший ранг среди его правильных стихов у каждого
скорера, и класс причины:
  (a) глубина пула -- лучший ранг у какого-то скорера в пределах POOL_K_NEW,
      но вне текущего пула (топ-300 по каждому);
  (b) невидимо -- лучший ранг везде хуже POOL_K_NEW.
"""
from __future__ import annotations

import csv
import sys

N_VERSES = 31102
POOL_K_NOW = 300
POOL_K_NEW = 1000


def main(table_tsv: str, unreached_tsv: str, out_tsv: str) -> None:
    with open(unreached_tsv, encoding="utf-8") as f:
        want = {int(r["rec_id"]): r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}

    best: dict[int, dict[str, float]] = {}
    with open(table_tsv, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        pct_cols = [c for c in rd.fieldnames if c.endswith("_pct")]
        for row in rd:
            rec = int(row["rec_id"])
            if rec not in want or row["label"] != "1":
                continue
            cur = best.setdefault(rec, {c: float("inf") for c in pct_cols})
            for c in pct_cols:
                rank = (1.0 - float(row[c])) * N_VERSES + 1
                if rank < cur[c]:
                    cur[c] = rank

    names = [c[:-4] for c in pct_cols]
    with open(out_tsv, "w", encoding="utf-8") as out:
        out.write("rec_id\tpar\tverse\tclass\tbest_scorer\tbest_rank\t"
                  + "\t".join(names) + "\n")
        print(f"{'rec':>4} {'абзац':16s} {'класс':6s} {'лучший скорер':14s} {'ранг':>6s}   фрагмент")
        for rec in sorted(want):
            ranks = best.get(rec)
            if not ranks:
                print(f"{rec:4d} нет положительных строк в таблице")
                continue
            sc, rk = min(ranks.items(), key=lambda kv: kv[1])
            sc = sc[:-4]
            cls = "a" if rk <= POOL_K_NEW else "b"
            w = want[rec]
            print(f"{rec:4d} {w['par']:16s} {cls:6s} {sc:14s} {rk:6.0f}   {w['fragment'][:40]}")
            out.write(f"{rec}\t{w['par']}\t{w['verse']}\t{cls}\t{sc}\t{rk:.0f}\t"
                      + "\t".join(f"{ranks[c]:.0f}" for c in pct_cols) + "\n")
    print(f"\nПорог пула сейчас: топ-{POOL_K_NOW} у каждого скорера; проверяется топ-{POOL_K_NEW}.")


if __name__ == "__main__":
    main(*sys.argv[1:4])
