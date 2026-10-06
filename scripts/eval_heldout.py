"""Независимая проверка: обучение на одном романе, тест на других.

Обучающая таблица -- пул «Преступления и наказания» (эталон Тихомирова, 64 места), тестовые --
пулы других романов (эталон академического ПСС). Модель (та же логрегрессия fit_logreg, те же
стандартизация и колонки) обучается ОДИН раз на всех строках обучающей таблицы, затем
ранжирует кандидатов каждого тестового места. Ничто из тестовых таблиц в подбор признаков,
порогов и весов не участвует.

Тестовых таблиц может быть несколько (через запятую, `имя=путь`): ранги считаются по всем местам
вместе (основной анализ) и описательно по каждой таблице отдельно.

Метрика та же, что везде: ранг лучшего правильного стиха среди кандидатов пула места;
медиана, топ-10/50/200 и средний log-ранг. Для каждого набора -- p против СЛУЧАЙНОГО порядка
в пуле (симуляция по реальным размерам пулов и числу правильных стихов, 20 000 прогонов), и для
парных сравнений с первым набором (базой) -- перестановочный тест по знакам разностей log-рангов.

Использование:
  eval_heldout.py <train.tsv> <test.tsv | имя=путь,имя=путь> "<название>=<колонки>" ...
Первый набор -- база для парных сравнений.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

from fusion_retrieval import fit_logreg


def load(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def matrix(rows, cols):
    return np.array([[float(r[c]) for c in cols] for r in rows]), np.array([int(r["label"]) for r in rows], float)


def main(train_path: str, test_arg: str, *specs: str) -> None:
    train = load(train_path)
    groups: dict[str, list] = {}
    for i, part in enumerate(test_arg.split(",")):
        name, _, path = part.partition("=")
        if not path:
            name, path = f"тест{i}", name
        groups[name] = load(path)
    by_rec = defaultdict(list)
    group_of = {}
    for gi, (name, rows) in enumerate(groups.items()):
        for r in rows:
            key = gi * 100000 + int(r["rec_id"])
            by_rec[key].append(r)
            group_of[key] = name
    recs = sorted(k for k, v in by_rec.items() if any(x["label"] == "1" for x in v))
    print(f"обучение: {len(train)} строк; тест: {len(recs)} мест "
          f"({', '.join(f'{n}: {sum(1 for k in recs if group_of[k] == n)}' for n in groups)})")

    rng = np.random.default_rng(0)
    pos_cache = {k: (len(by_rec[k]), sum(x["label"] == "1" for x in by_rec[k])) for k in recs}
    sims = []
    for _ in range(20000):
        ranks = [rng.choice(pos_cache[k][0], size=pos_cache[k][1], replace=False).min() + 1 for k in recs]
        sims.append(np.mean(np.log(ranks)))
    sims = np.array(sims)
    med_null = [np.median([rng.choice(pos_cache[k][0], size=pos_cache[k][1], replace=False).min() + 1
                           for k in recs]) for _ in range(3000)]
    print(f"случайный порядок: средний log-ранг {sims.mean():.2f} (95%: {np.percentile(sims, 2.5):.2f}.."
          f"{np.percentile(sims, 97.5):.2f}), медиана ранга {np.median(med_null):.0f}\n")

    results = {}
    print(f"{'набор':38s} {'мед.ранг':>8s} {'кв.25':>6s} {'кв.75':>6s} {'топ-10':>7s} {'топ-50':>7s} "
          f"{'топ-200':>8s} {'ср.log':>7s} {'p к случ.':>10s}")
    for spec in specs:
        label, cols = spec.split("=", 1)
        cols = cols.split(",")
        X, y = matrix(train, cols)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        w = fit_logreg((X - mu) / sd, y)
        ranks = {}
        for k in recs:
            Xt, yt = matrix(by_rec[k], cols)
            s = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))]) @ w
            order = np.argsort(-s, kind="stable")
            ranks[k] = int(np.where(yt[order] == 1)[0][0]) + 1
        v = np.array([ranks[k] for k in recs], float)
        results[label] = ranks
        p0 = (np.sum(sims <= np.mean(np.log(v))) + 1) / (len(sims) + 1)
        print(f"{label:38s} {np.median(v):8.0f} {np.percentile(v,25):6.0f} {np.percentile(v,75):6.0f} "
              f"{np.mean(v<=10):7.2f} {np.mean(v<=50):7.2f} {np.mean(v<=200):8.2f} {np.mean(np.log(v)):7.2f} "
              f"{p0:10.5f}")

    base = next(iter(results))
    print()
    for label in list(results)[1:]:
        a = np.log([results[label][k] for k in recs])
        b = np.log([results[base][k] for k in recs])
        d = a - b
        perm = (rng.choice([-1, 1], size=(20000, len(d))) * d).mean(1)
        p = (np.sum(np.abs(perm) >= abs(d.mean())) + 1) / 20001
        print(f"«{label}» против «{base}»: Δ log-ранга {d.mean():+.3f} (×{np.exp(d.mean()):.2f}), "
              f"перестановочный p = {p:.4f}")

    if len(groups) > 1:
        print("\nпо таблицам отдельно (описательно): мед. ранг / средний log-ранг")
        names = list(groups)
        print(f"{'набор':38s} " + " ".join(f"{n[:16]:>20s}" for n in names))
        for label, ranks in results.items():
            cells = []
            for n in names:
                v = np.array([ranks[k] for k in recs if group_of[k] == n], float)
                cells.append(f"{np.median(v):6.0f} / {np.mean(np.log(v)):5.2f} (n={len(v)})")
            print(f"{label:38s} " + " ".join(f"{c:>20s}" for c in cells))

    if os.environ.get("RANKS_OUT"):
        import json
        with open(os.environ["RANKS_OUT"], "a", encoding="utf-8") as fo:
            for label, ranks in results.items():
                fo.write(json.dumps({"label": label, "ranks": {str(k): v for k, v in ranks.items()}},
                                    ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
