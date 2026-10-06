"""Оценка слоёв по двум разным задачам, которые пилот мерил вперемешку.

  ОТБОР (detection): какие абзацы романа вообще содержат заимствование.
      Здесь длина абзаца -- сама по себе сильнейший предсказатель (эталонные
      абзацы длиннее нецитатных в разы), поэтому единственный честный вопрос:
      добавляет ли слой хоть что-то СВЕРХ длины. Сравнение идёт не с случайным
      выбором, а с моделью «только длина».

  АТРИБУЦИЯ (attribution): внутри подозреваемого абзаца -- какой именно стих.
      Длина цели здесь постоянна (абзац один и тот же), так что нормировка на
      неё не имеет смысла вовсе. Зато имеет смысл нормировка на длину ИСТОЧНИКА:
      длинный стих легче задеть случайно.

Разделение существенно: сырой балл выглядит сильным в задаче отбора только
потому, что коррелирует с длиной, а длина коррелирует с эталоном. В задаче
атрибуции длины нет, и там виден настоящий сигнал слоя.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from normalize_experiments import (average_precision, load_gold_ids,
                                   load_lengths, median_rank, recall_at_k,
                                   roc_auc)


def load_gold_refs(path: str, min_match: float = 0.6):
    """novel_verse_id -> список диапазонов (книга, глава, стих_от, стих_до)."""
    refs = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row["novel_verse_id"] or float(row["match_score"]) < min_match:
                continue
            group = []
            for ref in row["bible_refs"].split(";"):
                if not ref:
                    continue
                book, chap, verses = ref.split(".", 2)
                if "-" in verses:
                    vf, vt = map(int, verses.split("-"))
                else:
                    vf = vt = int(verses)
                group.append((book, int(chap), vf, vt))
            if group:
                refs[row["novel_verse_id"]].append(group)
    return refs


def in_refs(source_id: str, group) -> bool:
    parts = source_id.split(".")
    if len(parts) != 4:
        return False
    _, book, chap, verse = parts
    try:
        chap, verse = int(chap), int(verse)
    except ValueError:
        return False
    return any(book == b and chap == c and vf <= verse <= vt for b, c, vf, vt in group)


def load_candidates(path: str, cols: list[str]):
    """target_id -> (список source_id, матрица баллов по колонкам)."""
    by_t: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            by_t[row["target_id"]].append(
                (row["source_id"], [float(row[c]) for c in cols])
            )
    return {t: ([r[0] for r in v], np.array([r[1] for r in v])) for t, v in by_t.items()}


# ------------------------------------------------------------------ отбор


def fit_logreg(X, y, l2=2.0, lr=0.5, iters=3000):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return w


def loo_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Оценка каждого абзаца моделью, обученной без него. 46 положительных
    на 2-4 признака переобучаются мгновенно, поэтому обучение на всех и
    применение к тем же данным дало бы завышенную AUC."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    out = np.empty(len(y))
    # исключать по одному все 3783 абзаца -- 3783 подгонки; отрицательных
    # много и они взаимозаменяемы, поэтому исключаем по одному только
    # положительные, для отрицательных хватает модели на всех данных
    w_all = fit_logreg(Xn, y)
    Xb = np.hstack([Xn, np.ones((len(y), 1))])
    out[:] = Xb @ w_all
    for i in np.where(y == 1)[0]:
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        w = fit_logreg(Xn[m], y[m])
        out[i] = np.hstack([Xn[i], 1.0]) @ w
    return out


def bootstrap_auc_diff(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                       n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Доверительный интервал разницы AUC: 46 положительных -- разница в
    третьем знаке ничего не значит, нужен размах."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        q = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([p, q])
        yy = y[idx]
        diffs[k] = roc_auc(a[idx], yy) - roc_auc(b[idx], yy)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def detection_report(ids, lens, y, layers: dict[str, np.ndarray]) -> None:
    loglen = np.log(np.maximum(lens, 1.0))[:, None]

    print("\n" + "=" * 78)
    print("ОТБОР АБЗАЦЕВ: 3783 абзаца, 46 эталонных")
    print("=" * 78)
    print(f"{'модель':34s} {'AUC':>6s} {'AP':>6s} {'r@50':>6s} {'r@100':>6s} "
          f"{'r@200':>6s} {'мед.ранг':>9s}")

    def line(name, s):
        print(f"{name:34s} {roc_auc(s,y):6.3f} {average_precision(s,y):6.3f} "
              f"{recall_at_k(s,y,50):6.2f} {recall_at_k(s,y,100):6.2f} "
              f"{recall_at_k(s,y,200):6.2f} {median_rank(s,y):9.0f}")

    line("только длина абзаца", lens)
    for name, s in layers.items():
        line(f"только {name} (сырой балл)", s)

    base = loo_scores(loglen, y)
    line("логрег: длина", base)
    for name, s in layers.items():
        both = loo_scores(np.hstack([loglen, s[:, None]]), y)
        line(f"логрег: длина + {name}", both)
        lo, hi = bootstrap_auc_diff(both, base, y)
        print(f"{'':34s} прирост AUC над «только длина»: "
              f"{roc_auc(both,y)-roc_auc(base,y):+.3f} (бутстрап 95%: {lo:+.3f}..{hi:+.3f})")

    if len(layers) > 1:
        allX = np.hstack([loglen] + [s[:, None] for s in layers.values()])
        both = loo_scores(allX, y)
        line("логрег: длина + все слои", both)
        lo, hi = bootstrap_auc_diff(both, base, y)
        print(f"{'':34s} прирост AUC над «только длина»: "
              f"{roc_auc(both,y)-roc_auc(base,y):+.3f} (бутстрап 95%: {lo:+.3f}..{hi:+.3f})")


# -------------------------------------------------------------- атрибуция


def attribution_report(cand_path: str, cols: list[str], gold_refs, label: str) -> None:
    cands = load_candidates(cand_path, cols)
    print("\n" + "=" * 78)
    print(f"АТРИБУЦИЯ ВНУТРИ АБЗАЦА — {label}")
    print("=" * 78)
    print(f"{'вариант балла':16s} {'найдено':>9s} {'мед.ранг':>9s} "
          f"{'топ-1':>6s} {'топ-5':>6s} {'топ-10':>7s} {'топ-50':>7s}")

    n_checkable = sum(len(v) for v in gold_refs.values())
    for ci, col in enumerate(cols):
        ranks = []
        for tid, groups in gold_refs.items():
            entry = cands.get(tid)
            if entry is None:
                continue
            sids, mat = entry
            order = np.argsort(-mat[:, ci], kind="stable")
            for group in groups:
                hit = None
                for rank, j in enumerate(order, start=1):
                    if in_refs(sids[j], group):
                        hit = rank
                        break
                if hit is not None:
                    ranks.append(hit)
        if not ranks:
            print(f"{col:16s} {'0':>9s}")
            continue
        r = np.array(ranks)
        print(f"{col:16s} {len(r)}/{n_checkable:<5d} {np.median(r):9.0f} "
              f"{np.mean(r<=1):6.2f} {np.mean(r<=5):6.2f} {np.mean(r<=10):7.2f} "
              f"{np.mean(r<=50):7.2f}")


def main(argv: list[str]) -> None:
    novel_tsv, gold_tsv = argv[0], argv[1]
    specs = argv[2:]  # label:col1,col2,...=path
    ids, lens = load_lengths(novel_tsv)
    gold_ids = load_gold_ids(gold_tsv)
    gold_refs = load_gold_refs(gold_tsv)
    y = np.array([1 if i in gold_ids else 0 for i in ids])
    idx = {t: i for i, t in enumerate(ids)}

    layers: dict[str, np.ndarray] = {}
    for spec in specs:
        head, path = spec.split("=", 1)
        label, colspec = head.split(":", 1)
        cols = colspec.split(",")
        cands = load_candidates(path, cols)
        for ci, col in enumerate(cols):
            arr = np.zeros(len(ids))
            for t, (sids, mat) in cands.items():
                i = idx.get(t)
                if i is not None and len(mat):
                    arr[i] = mat[:, ci].max()
            layers[f"{label}/{col}"] = arr
        attribution_report(path, cols, gold_refs, label)

    detection_report(ids, lens, y, layers)


if __name__ == "__main__":
    main(sys.argv[1:])
