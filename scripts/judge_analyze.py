"""Итоги слепой ручной проверки: точность находок систем, надёжность читающего.

Вход: ключ (item_id, novel, paragraph_id, verse_id, type, systems), разметка первого прохода
(item_id, score, reason), повторная разметка (item_id, score_retest, score_first).
Правила -- в reports/prereg_manual_check.md (заданы до чтения).

Печатает: распределение оценок по типам пар; чувствительность на золотых и долю оценок >= 2 на
случайных (условие допустимости); точность P@40 по системам и романам с интервалами Уилсона;
сравнение систем по парам, попавшим только в одну из них (точный критерий Фишера); согласие
повторной разметки; список находок, названных аллюзиями.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Точный двусторонний критерий Фишера для таблицы [[a, b], [c, d]]."""
    n1, n2, k = a + b, c + d, a + c
    total = n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)

    lo, hi = max(0, k - n2), min(n1, k)
    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs + 1e-12))


def main(key_tsv: str, labels_tsv: str, retest_tsv: str) -> None:
    key = {}
    with open(key_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sysd = {}
            for part in filter(None, r["systems"].split(",")):
                s, rank = part.split(":")
                sysd[s] = int(rank)
            key[r["item_id"]] = {**r, "systems": sysd}
    lab, reason = {}, {}
    with open(labels_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            lab[r["item_id"]] = int(r["score"])
            reason[r["item_id"]] = r["reason"]
    assert set(lab) == set(key), "разметка и ключ не совпадают"

    print("== распределение оценок по типам пар")
    by_type = defaultdict(list)
    for i, k in key.items():
        by_type[k["type"]].append(lab[i])
    for t in ("gold", "random", "hard", "find"):
        v = by_type[t]
        print(f"  {t:7s} n={len(v):3d}  оценки 0/1/2/3: {sum(x==0 for x in v)}/{sum(x==1 for x in v)}/"
              f"{sum(x==2 for x in v)}/{sum(x==3 for x in v)}")

    g = by_type["gold"]
    rnd = by_type["random"] + by_type["hard"]   # раунд 1: случайные, раунд 2: трудные случайные
    fpr_max = 0.10 if by_type["random"] else 0.15
    sens = sum(x >= 2 for x in g) / len(g)
    fpr = sum(x >= 2 for x in rnd) / len(rnd)
    lo, hi = wilson(sum(x >= 2 for x in g), len(g))
    print(f"\n== надёжность читающего (условия: чувствительность >= 0,70, случайные <= {fpr_max:.2f})")
    print(f"  чувствительность на золотых: {sens:.3f} ({sum(x>=2 for x in g)}/{len(g)}; 95%: {lo:.2f}..{hi:.2f})")
    for novel in ("C&P", "BK"):
        gg = [lab[i] for i, k in key.items() if k["type"] == "gold" and k["novel"] == novel]
        print(f"    {novel}: {sum(x>=2 for x in gg)}/{len(gg)}")
    print(f"  оценок >= 2 на случайных: {fpr:.3f} ({sum(x>=2 for x in rnd)}/{len(rnd)})")
    ok = sens >= 0.70 and fpr <= fpr_max
    print(f"  условия допустимости: {'ВЫПОЛНЕНЫ' if ok else 'НЕ ВЫПОЛНЕНЫ -- точности ненадёжны'}")
    missed = [(i, key[i]["verse_id"], key[i]["paragraph_id"]) for i, k in key.items()
              if k["type"] == "gold" and lab[i] < 2]
    print(f"  золотые пары с оценкой < 2: {missed}")

    print("\n== точность находок P@40 (положительная = оценка >= 2; в скобках строгая = 3)")
    sets = {s: defaultdict(set) for s in ("S1", "S2", "S3")}
    for i, k in key.items():
        for s in k["systems"]:
            sets[s][k["novel"]].add(i)
            sets[s]["ALL"].add(i)
    for s in ("S1", "S2", "S3"):
        row = []
        for scope in ("C&P", "BK", "ALL"):
            ids = sets[s][scope]
            k2 = sum(lab[i] >= 2 for i in ids)
            k3 = sum(lab[i] == 3 for i in ids)
            lo, hi = wilson(k2, len(ids))
            row.append(f"{scope}: {k2}/{len(ids)} = {k2/max(1,len(ids)):.3f} [{lo:.2f}..{hi:.2f}] ({k3})")
        print(f"  {s}  " + " | ".join(row))

    def compare(a: str, b: str, primary: bool = False):
        only_a = sets[a]["ALL"] - sets[b]["ALL"]
        only_b = sets[b]["ALL"] - sets[a]["ALL"]
        pa = sum(lab[i] >= 2 for i in only_a)
        pb = sum(lab[i] >= 2 for i in only_b)
        p = fisher_two_sided(pa, len(only_a) - pa, pb, len(only_b) - pb)
        shared = sets[a]["ALL"] & sets[b]["ALL"]
        print(f"  {a} против {b}{' (основное)' if primary else ''}: общих пар {len(shared)} "
              f"(из них положительных {sum(lab[i] >= 2 for i in shared)}); только {a}: {pa}/{len(only_a)}, "
              f"только {b}: {pb}/{len(only_b)}; Фишер p = {p:.3f}")

    print("\n== сравнение систем по парам, попавшим только в одну из них")
    compare("S2", "S1", primary=True)
    compare("S3", "S1")
    compare("S2", "S3")

    print("\n== число пар в списках систем и их пересечения")
    for s in ("S1", "S2", "S3"):
        print(f"  {s}: {len(sets[s]['ALL'])} пар")

    with open(retest_tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    a = [int(r["score_first"]) for r in rows]
    b = [int(r["score_retest"]) for r in rows]
    ba, bb = [x >= 2 for x in a], [y >= 2 for y in b]
    agree = sum(x == y for x, y in zip(ba, bb)) / len(rows)
    p1, p2 = sum(ba) / len(rows), sum(bb) / len(rows)
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (agree - pe) / (1 - pe)
    print(f"\n== повторная разметка: {len(rows)} пар, совпадение 0-3 {sum(x == y for x, y in zip(a, b))/len(rows):.3f}, "
          f"бинарное {agree:.3f}, kappa {kappa:.3f}")

    print("\n== находки (неэталонные абзацы), названные аллюзиями (>= 2)")
    for i, k in sorted(key.items(), key=lambda kv: -lab[kv[0]]):
        if k["type"] == "find" and lab[i] >= 2:
            print(f"  {i} {k['novel']:3s} {k['paragraph_id']:16s} {k['verse_id']:12s} оценка {lab[i]} "
                  f"системы {','.join(f'{s}:{r}' for s, r in sorted(k['systems'].items()))} -- {reason[i][:90]}")

    print("\n== какие пары системы ставили в топ, но читающий признал не аллюзией (оценка 0), по типу пустышек")
    trivial = [i for i, k in key.items() if k["type"] == "find" and lab[i] == 0]
    print(f"  всего находок с оценкой 0: {len(trivial)} из {sum(k['type']=='find' for k in key.values())}")
    short = defaultdict(int)
    for i in trivial:
        for s in key[i]["systems"]:
            short[s] += 1
    print("  по системам (находок с оценкой 0 в её списке):", dict(short))


if __name__ == "__main__":
    main(*sys.argv[1:4])
