"""Модели ранжирования поверх одной и той же таблицы признаков.

Измеренная слабость объединения -- не полнота шортлиста, а самый верх списка:
топ-200 доходит до 0,36, а топ-1 держится около нуля. Это ровно то, что
поточечная логрегрессия и должна давать: она учится отличать верную пару от
неверной В СРЕДНЕМ по всей таблице, а не ставить верную пару первой ВНУТРИ
своего абзаца. Здесь проверяются два средства против этого.

  поточечная    логрегрессия на парах (абзац, стих) с метками 0/1 -- как было
  попарная      логрегрессия на РАЗНОСТЯХ признаков между верной и неверной
                парой одного абзаца. Ровно та задача, которую мы измеряем:
                «поставь верную выше неверной внутри абзаца». Признаки-разности
                снимают и разброс между абзацами: длинному абзацу с высокими
                баллами у всех кандидатов больше не нужен свой порог
  каскад        приоритет по редкому, но точному признаку. Измерение потолка:
                когда срабатывает точная n-грамма, медиана ранга 6 из 31 102;
                когда пары по дереву -- 39; когда skipbigram -- 47. Такие
                кандидаты ставятся в начало списка до всякой регрессии, внутри
                яруса упорядочиваются своим баллом, остальные идут после по
                баллу модели

Оценка -- leave-one-target-out по абзацам: модель обучается без всех записей
своего абзаца, иначе соседняя запись того же абзаца подсказала бы ответ.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

TIERS = ("ngram_raw", "dep_pair2_raw", "skipbigram_raw", "soft_edges_raw")


def load(table: str):
    with open(table, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        names = [c for c in reader.fieldnames
                 if c not in ("target_id", "source_id", "rec_id", "label",
                              "in_natural_pool")]
        rows = [(r["target_id"], int(r["rec_id"]),
                 np.array([float(r[c]) for c in names], dtype=np.float64),
                 int(r["label"]), int(r.get("in_natural_pool", 1))) for r in reader]
    return names, rows


def fit_logreg(X, y, l2=2.0, lr=0.5, iters=2000, fit_bias=True):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))]) if fit_bias else X
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n
        if fit_bias:
            g[:-1] += l2 * w[:-1] / n
        else:
            g += l2 * w / n
        w -= lr * g
    return w


def pairwise_design(by_rec, recs, mu, sd, max_neg=60, seed=0):
    """Разности (верная - неверная) внутри одного абзаца, метка всегда 1,
    плюс те же разности со знаком минус и меткой 0 -- так модель без
    свободного члена учится знаку разности."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in recs:
        items = by_rec[r]
        pos = [f for f, l in items if l == 1]
        neg = [f for f, l in items if l == 0]
        if not pos or not neg:
            continue
        for p in pos:
            pick = neg if len(neg) <= max_neg else [
                neg[i] for i in rng.choice(len(neg), max_neg, replace=False)]
            for q in pick:
                d = (p - q) / sd
                rows.append((d, 1.0))
                rows.append((-d, 0.0))
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y


def tier_of(feats: np.ndarray, idx: dict[str, int]) -> int:
    for t, name in enumerate(TIERS):
        j = idx.get(name)
        if j is not None and feats[j] > 0:
            return t
    return len(TIERS)


def rank_of_positive(scores, labels, tiers=None):
    if tiers is None:
        order = np.argsort(-scores, kind="stable")
    else:
        order = np.lexsort((-scores, tiers))
    hits = np.where(np.asarray(labels)[order] == 1)[0]
    return int(hits[0]) + 1 if len(hits) else None


def main(table: str) -> None:
    names, rows = load(table)
    idx = {n: i for i, n in enumerate(names)}
    by_rec: dict[int, list] = defaultdict(list)
    par_of_rec: dict[int, str] = {}
    nat: dict[int, int] = {}
    for par, rec, feats, lab, n in rows:
        by_rec[rec].append((feats, lab))
        par_of_rec[rec] = par
        nat[rec] = n
    recs = [r for r, v in by_rec.items() if any(l == 1 for _, l in v)]
    print(f"признаков {len(names)}, мест эталона в таблице {len(recs)}, "
          f"из них извлечение достало само {sum(nat[r] for r in recs)}")

    allX = np.array([f for _, _, f, _, _ in rows])
    mu, sd = allX.mean(0), allX.std(0)
    sd[sd == 0] = 1.0

    results: dict[str, dict[int, int | None]] = {}

    def run(label, kind, use_cascade):
        ranks = {}
        for held in recs:
            tr_recs = [r for r in by_rec if par_of_rec[r] != par_of_rec[held]]
            if kind == "point":
                Xtr = np.array([f for r in tr_recs for f, _ in by_rec[r]])
                ytr = np.array([l for r in tr_recs for _, l in by_rec[r]], float)
                m, s = Xtr.mean(0), Xtr.std(0)
                s[s == 0] = 1.0
                w = fit_logreg((Xtr - m) / s, ytr)
                Xte = (np.array([f for f, _ in by_rec[held]]) - m) / s
                sc = np.hstack([Xte, np.ones((len(Xte), 1))]) @ w
            else:
                Xp, yp = pairwise_design(by_rec, tr_recs, mu, sd)
                w = fit_logreg(Xp, yp, fit_bias=False)
                sc = (np.array([f for f, _ in by_rec[held]]) / sd) @ w
            lab = [l for _, l in by_rec[held]]
            t = (np.array([tier_of(f, idx) for f, _ in by_rec[held]])
                 if use_cascade else None)
            ranks[held] = rank_of_positive(sc, lab, t)
        results[label] = ranks
        v = np.array([x for x in ranks.values() if x is not None], float)
        print(f"{label:34s} мед={np.median(v):6.0f} топ-1={np.mean(v<=1):.2f} "
              f"топ-5={np.mean(v<=5):.2f} топ-10={np.mean(v<=10):.2f} "
              f"топ-50={np.mean(v<=50):.2f} топ-200={np.mean(v<=200):.2f}")

    print()
    run("поточечная (как было)", "point", False)
    run("попарная", "pair", False)
    run("поточечная + каскад", "point", True)
    run("попарная + каскад", "pair", True)

    base = "поточечная (как было)"
    rng = np.random.default_rng(0)
    print()
    for label in results:
        if label == base:
            continue
        keys = [k for k in results[label]
                if results[label][k] is not None and results[base][k] is not None]
        a = np.array([results[label][k] for k in keys], float)
        b = np.array([results[base][k] for k in keys], float)
        d_med = [np.median(a[i]) - np.median(b[i])
                 for i in (rng.integers(0, len(keys), len(keys)) for _ in range(5000))]
        d_t10 = [float((a[i] <= 10).mean() - (b[i] <= 10).mean())
                 for i in (rng.integers(0, len(keys), len(keys)) for _ in range(5000))]
        print(f"{label} против «{base}»:")
        print(f"    медиана {np.median(a)-np.median(b):+.0f} "
              f"(95%: {np.percentile(d_med,2.5):+.0f}..{np.percentile(d_med,97.5):+.0f}); "
              f"топ-10 {(a<=10).mean()-(b<=10).mean():+.2f} "
              f"(95%: {np.percentile(d_t10,2.5):+.2f}..{np.percentile(d_t10,97.5):+.2f})")


if __name__ == "__main__":
    main(sys.argv[1])
