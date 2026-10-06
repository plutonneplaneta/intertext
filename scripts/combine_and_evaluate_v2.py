"""Комбинированная модель атрибуции: leave-one-target-out, произвольные наборы
признаков, с доверительным интервалом на разницу между наборами.

Отличия от combine_and_evaluate.py:

  * набор признаков задаётся в вызове, так что «база из трёх слоёв» и
    расширенный набор меряются одним и тем же кодом на одной и той же
    таблице -- иначе разницу нельзя приписать признакам;
  * стандартизация считается ТОЛЬКО по обучающей части каждого сгиба.
    В первой версии mu и sigma брались по всей таблице, включая отложенный
    абзац: утечка небольшая, но при 46 абзацах её не отличить от эффекта;
  * разница медианных рангов между наборами сопровождается бутстрапом по
    абзацам -- на 20-30 проверяемых местах медиана скачет на десятки
    рангов от перестановки одного абзаца.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def load_table(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        names = [c for c in reader.fieldnames
                 if c not in ("target_id", "source_id", "label")]
        rows = [(r["target_id"], r["source_id"],
                 [float(r[c]) for c in names], int(r["label"])) for r in reader]
    return names, rows


def fit_logreg(X, y, l2=2.0, lr=0.5, iters=2000):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return w


def rank_of_best_positive(scores, labels):
    order = np.argsort(-scores, kind="stable")
    hits = np.where(labels[order] == 1)[0]
    return int(hits[0]) + 1 if len(hits) else None


def loo_ranks(rows, names, feat_names, l2=2.0):
    cols = [names.index(f) for f in feat_names]
    by_t = defaultdict(list)
    for r in rows:
        by_t[r[0]].append(r)
    targets = [t for t, rs in by_t.items() if any(r[3] == 1 for r in rs)]

    ranks = {}
    for held in targets:
        tr = [r for t, rs in by_t.items() if t != held for r in rs]
        te = by_t[held]
        Xtr = np.array([[r[2][c] for c in cols] for r in tr])
        ytr = np.array([r[3] for r in tr], dtype=float)
        mu, sd = Xtr.mean(0), Xtr.std(0)   # только по обучающей части
        sd[sd == 0] = 1.0
        w = fit_logreg((Xtr - mu) / sd, ytr, l2=l2)
        Xte = (np.array([[r[2][c] for c in cols] for r in te]) - mu) / sd
        s = np.hstack([Xte, np.ones((len(te), 1))]) @ w
        ranks[held] = rank_of_best_positive(s, np.array([r[3] for r in te]))
    return ranks, targets


def summarize(name, ranks: dict, n_targets: int):
    """Медиана здесь -- настоящая (среднее двух средних при чётном числе
    абзацев), а не sorted(ranks)[n//2], как в combine_and_evaluate.py: на
    16 абзацах разница между этими двумя «медианами» была 158 против 90,
    то есть больше любого эффекта, который мы пытаемся измерить.

    И всё же медиана здесь -- плохая сводка: распределение рангов двугорбое
    (часть мест находится в первой десятке, остальные теряются в сотнях),
    поэтому рядом печатаются квартили и доли топ-k."""
    v = np.array([r for r in ranks.values() if r is not None])
    q1, q3 = np.percentile(v, [25, 75])
    print(f"{name:34s} мед.ранг={np.median(v):5.0f} [кв.{q1:.0f}..{q3:.0f}]  "
          f"топ-1={np.mean(v<=1):.2f}  топ-5={np.mean(v<=5):.2f}  "
          f"топ-10={np.mean(v<=10):.2f}  топ-50={np.mean(v<=50):.2f}  "
          f"(абзацев {len(v)}/{n_targets})")


def boot_median_diff(a: dict, b: dict, n_boot=5000, seed=0):
    keys = [k for k in a if a[k] is not None and b.get(k) is not None]
    av = np.array([a[k] for k in keys], float)
    bv = np.array([b[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    d = [np.median(av[i]) - np.median(bv[i])
         for i in (rng.integers(0, len(keys), len(keys)) for _ in range(n_boot))]
    return np.median(av) - np.median(bv), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main(argv: list[str]) -> None:
    table_path = argv[0]
    names, rows = load_table(table_path)
    print(f"признаков в таблице: {len(names)}", file=sys.stderr)
    print(f"  {', '.join(names)}", file=sys.stderr)

    sets: list[tuple[str, list[str]]] = []
    for spec in argv[1:]:
        label, cols = spec.split("=", 1)
        sets.append((label, cols.split(",")))
    if not sets:
        sets = [("все признаки", names)]

    print(f"\nleave-one-target-out по таблице {table_path}\n")
    results = {}
    n_targets = None
    for label, cols in sets:
        missing = [c for c in cols if c not in names]
        if missing:
            print(f"{label}: нет колонок {missing} -- пропущено")
            continue
        ranks, targets = loo_ranks(rows, names, cols)
        n_targets = len(targets)
        results[label] = ranks
        summarize(label, ranks, n_targets)

    if len(results) > 1:
        base_label = sets[0][0]
        print()
        for label in list(results)[1:]:
            d, lo, hi = boot_median_diff(results[label], results[base_label])
            print(f"{label} против «{base_label}»: медиана ранга "
                  f"{d:+.0f} (бутстрап 95%: {lo:+.0f}..{hi:+.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
