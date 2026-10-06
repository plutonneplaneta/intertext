"""Сводка по результатам llm_contamination.py: pair, repeat, blind, complete."""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

import numpy as np


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(pair, repeat, blind, complete):
    print("== pair: оценка по типу пары (G золотые; A/B/C/R обманки) ==")
    by = defaultdict(list)
    types = defaultdict(Counter)
    bad = 0
    for r in rows(pair):
        s = fnum(r["score"])
        if s is None:
            bad += 1
            continue
        by[r["decoy_type"]].append(s)
        types[r["decoy_type"]][r["type"]] += 1
    print(f"{'тип':4s} {'n':>4s} {'ср.балл':>8s} {'доля>=2':>8s} {'доля=3':>7s}")
    for t in "GABCR":
        v = np.array(by[t])
        if len(v):
            print(f"{t:4s} {len(v):4d} {v.mean():8.2f} {np.mean(v>=2):8.2f} {np.mean(v==3):7.2f}")
    print(f"не разобрано: {bad}")
    g = np.array(by["G"])
    for t in "ABC":
        v = np.array(by[t])
        if len(v) and len(g):
            print(f"  G против {t}: разница доли>=2 {np.mean(g>=2)-np.mean(v>=2):+.2f}")
    print("тип заимствования у G:", dict(types["G"].most_common(5)))
    print("тип заимствования у A:", dict(types["A"].most_common(5)))

    print("\n== repeat: устойчивость при температуре 0,7, три прогона ==")
    det = {(r["target_id"], r["source_id"]): fnum(r["score"]) for r in rows(pair)}
    grp = defaultdict(list)
    for r in rows(repeat):
        s = fnum(r["score"])
        if s is not None:
            grp[(r["decoy_type"], r["target_id"], r["source_id"])].append(s)
    for t in "GA":
        flips, agree, n = 0, 0, 0
        spread = []
        for (ty, tid, sid), v in grp.items():
            if ty != t or len(v) < 2:
                continue
            n += 1
            flips += int(len(set(x >= 2 for x in v)) > 1)
            spread.append(max(v) - min(v))
            d = det.get((tid, sid))
            if d is not None:
                agree += int(round(np.mean(v)) == d)
        print(f"{t}: пар {n}, вердикт «>=2» перекидывается между прогонами у {flips/max(n,1):.0%}, "
              f"средний размах {np.mean(spread):.2f}, среднее по прогонам == детерминированный у {agree/max(n,1):.0%}")

    print("\n== blind: выбор из пяти стихов без подсказки (шанс 1/5 = 0,20) ==")
    b = rows(blind)
    ch = Counter(r["chosen_type"] for r in b)
    n = len(b)
    print(f"абзацев {n}; выбрано: " + ", ".join(f"{k}={v}" for k, v in ch.most_common()))
    print(f"золотой выбран в {ch['G']/max(n,1):.0%}, «нет» — {ch['none']/max(n,1):.0%}")
    gold_scores = [fnum(r["score"]) for r in b if r["chosen_type"] == "G" and fnum(r["score"]) is not None]
    if gold_scores:
        print(f"средний балл, когда золотой выбран: {np.mean(gold_scores):.2f}")
    paired = [(r["target_id"]) for r in b]
    gpair = defaultdict(list)
    for r in rows(pair):
        if r["decoy_type"] == "G" and fnum(r["score"]) is not None:
            gpair[r["target_id"]].append(fnum(r["score"]))
    both = [(max(gpair[r["target_id"]]) >= 2, r["chosen_type"] == "G") for r in b if r["target_id"] in gpair]
    if both:
        a_ok = sum(x for x, _ in both)
        b_ok = sum(y for _, y in both)
        print(f"золотая пара с баллом>=2 при показе пары: {a_ok}/{len(both)}; "
              f"золотой выбран вслепую: {b_ok}/{len(both)}")

    print("\n== complete: из какой книги и главы, без стиха ==")
    c = rows(complete)
    print(f"записей {len(c)}; книга угадана {np.mean([int(r['book_ok']) for r in c]):.0%}, "
          f"глава угадана {np.mean([int(r['chapter_ok']) for r in c]):.0%}")
    gospel = [r for r in c]
    print("(для сравнения: у 18 книг в эталоне случайный выбор книги даёт порядка 5-10%)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
