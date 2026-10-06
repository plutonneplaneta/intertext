"""Сравнение способов убрать длину абзаца из балла: корзины разной мелкости
против прямой регрессии балла на длину.

Зачем вообще. Балл всех трёх слоёв растёт с длиной абзаца механически: чем
больше слов, тем больше попыток совпасть с Библией (корреляция 0,45 у
лексики, 0,44 у синтаксиса). Пока этот рост не убран, порог по баллу
отбирает длинные абзацы, а не цитаты.

Что сравнивается:

  raw                сырой max-балл абзаца
  zbinN              Z-оценка внутри квантильной корзины длины, N корзин
  zbinN_robust       то же, но медиана и MAD вместо среднего и с.к.о.
  zknnK              Z-оценка по K ближайшим по длине абзацам (корзина без границ)
  rankknnK           процентиль балла среди K ближайших по длине (без допущений
                     о форме распределения)
  reg_lin            остаток регрессии балла на log(длины), делённый на общее
                     с.к.о. остатков
  reg_lin_het        то же, но делённый на предсказанный масштаб остатка
                     (регрессия |остаток| на log(длины)) -- поправка на то, что
                     разброс балла у длинных абзацев тоже больше
  reg_quad_het       квадратичная по log(длины) средняя, гетероскедастичный масштаб
  reg_iso_het        изотоническая (монотонная) средняя вместо параметрической

Про фон. Считать фон по не-эталонным абзацам, как делала первая версия
(length_normalize.py), -- оптимистическая ошибка: настоящие цитаты выкидываются
из оценки собственного фона, и их Z-оценка завышается по построению. На новом
тексте так сделать нельзя: эталона там нет. Поэтому здесь по умолчанию
leave-one-out -- фон для абзаца считается по всем остальным абзацам, включая
чужие цитаты. Режим exclude-gold оставлен для сравнения, чтобы видеть размер
этой ошибки.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------- загрузка


def load_lengths(novel_tsv: str) -> tuple[list[str], np.ndarray]:
    ids, lens = [], []
    with open(novel_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(row["verse_id"])
            lens.append(len(row["text"].split()))
    return ids, np.array(lens, dtype=float)


def load_gold_ids(gold_tsv: str, min_match: float = 0.6) -> set[str]:
    out = set()
    with open(gold_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row["novel_verse_id"] and float(row["match_score"]) >= min_match:
                out.add(row["novel_verse_id"])
    return out


def load_max_scores(candidates_tsv: str, ids: list[str]) -> np.ndarray:
    """max балл по каждому абзацу; 0, если метод не предложил ни одного
    кандидата (это не пропуск данных, а «ничего не нашлось»)."""
    idx = {tid: i for i, tid in enumerate(ids)}
    out = np.zeros(len(ids))
    with open(candidates_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            i = idx.get(row["target_id"])
            if i is None:
                continue
            s = float(row["score"])
            if s > out[i]:
                out[i] = s
    return out


# ------------------------------------------------------- нормировки (LOO)


def _loo_mean_std(vals: np.ndarray, group: np.ndarray, in_bg: np.ndarray,
                  n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и с.к.о. фона в своей группе, с исключением самого абзаца
    (если он входит в фон). Через суммы и суммы квадратов -- O(n)."""
    cnt = np.zeros(n_groups)
    s1 = np.zeros(n_groups)
    s2 = np.zeros(n_groups)
    np.add.at(cnt, group[in_bg], 1.0)
    np.add.at(s1, group[in_bg], vals[in_bg])
    np.add.at(s2, group[in_bg], vals[in_bg] ** 2)

    c = cnt[group].copy()
    t1 = s1[group].copy()
    t2 = s2[group].copy()
    # вычитаем сам абзац из его же фона
    c[in_bg] -= 1.0
    t1[in_bg] -= vals[in_bg]
    t2[in_bg] -= vals[in_bg] ** 2

    c = np.maximum(c, 1.0)
    mean = t1 / c
    var = np.maximum(t2 / c - mean ** 2, 0.0)
    std = np.sqrt(var)
    return mean, std


def quantile_bins(lens: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(lens, kind="stable")
    n = len(lens)
    bin_of = np.empty(n, dtype=int)
    for rank, i in enumerate(order):
        bin_of[i] = min(n_bins - 1, rank * n_bins // n)
    return bin_of


def norm_zbin(scores, lens, in_bg, n_bins, robust=False):
    g = quantile_bins(lens, n_bins)
    if not robust:
        mean, std = _loo_mean_std(scores, g, in_bg, n_bins)
        return (scores - mean) / np.maximum(std, 1e-9)
    # робастный вариант: LOO для медианы дорог и почти не меняет её при
    # десятках элементов в корзине -- считаем медиану/MAD по фону корзины
    out = np.zeros_like(scores)
    for b in range(n_bins):
        m = g == b
        bg = scores[m & in_bg]
        if len(bg) < 3:
            bg = scores[m]
        med = np.median(bg)
        mad = np.median(np.abs(bg - med)) * 1.4826
        out[m] = (scores[m] - med) / max(mad, 1e-9)
    return out


def _knn_window(lens: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Для каждого абзаца -- отрезок [lo, hi) в отсортированном по длине
    массиве: k ближайших по длине (окно скользит, границ корзин нет)."""
    order = np.argsort(lens, kind="stable")
    n = len(lens)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    half = k // 2
    lo = np.clip(pos - half, 0, max(n - k, 0))
    hi = np.minimum(lo + k, n)
    return order, np.stack([lo, hi], axis=1)


def norm_zknn(scores, lens, in_bg, k):
    order, win = _knn_window(lens, k)
    s_sorted = scores[order]
    bg_sorted = in_bg[order].astype(float)
    c1 = np.concatenate([[0.0], np.cumsum(bg_sorted)])
    cs = np.concatenate([[0.0], np.cumsum(s_sorted * bg_sorted)])
    cs2 = np.concatenate([[0.0], np.cumsum(s_sorted ** 2 * bg_sorted)])

    pos = np.empty(len(scores), dtype=int)
    pos[order] = np.arange(len(scores))
    lo, hi = win[:, 0], win[:, 1]
    c = c1[hi] - c1[lo]
    t1 = cs[hi] - cs[lo]
    t2 = cs2[hi] - cs2[lo]
    # исключаем сам абзац, если он в окне и входит в фон
    inside = (pos >= lo) & (pos < hi) & in_bg
    c = c - inside
    t1 = t1 - np.where(inside, scores, 0.0)
    t2 = t2 - np.where(inside, scores ** 2, 0.0)
    c = np.maximum(c, 1.0)
    mean = t1 / c
    var = np.maximum(t2 / c - mean ** 2, 0.0)
    return (scores - mean) / np.maximum(np.sqrt(var), 1e-9)


def norm_rankknn(scores, lens, in_bg, k):
    """Доля абзацев окна с баллом ниже данного -- нормировка без допущений
    о форме распределения балла (у лексики оно с тяжёлым правым хвостом,
    и Z-оценка там систематически недооценивает выбросы)."""
    order, win = _knn_window(lens, k)
    n = len(scores)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    s_sorted = scores[order]
    bg_sorted = in_bg[order]
    out = np.empty(n)
    for i in range(n):
        lo, hi = win[i, 0], win[i, 1]
        w = s_sorted[lo:hi]
        wb = bg_sorted[lo:hi]
        vals = w[wb]
        if len(vals) == 0:
            out[i] = 0.5
            continue
        m = len(vals)
        less = np.count_nonzero(vals < scores[i])
        eq = np.count_nonzero(vals == scores[i])
        if lo <= pos[i] < hi and in_bg[i]:
            m -= 1
            eq -= 1
        if m <= 0:
            out[i] = 0.5
        else:
            out[i] = (less + 0.5 * max(eq, 0)) / m
    return out


def _ols_loo_resid(X: np.ndarray, y: np.ndarray, in_bg: np.ndarray) -> np.ndarray:
    """Остатки регрессии y на X, подогнанной по фону; для точек фона --
    leave-one-out остаток через плечо (точная формула, без 3783 подгонок)."""
    Xb = X[in_bg]
    yb = y[in_bg]
    XtX_inv = np.linalg.pinv(Xb.T @ Xb)
    beta = XtX_inv @ (Xb.T @ yb)
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    denom = np.where(in_bg, np.maximum(1.0 - h, 1e-6), 1.0)
    return resid / denom


def _design(lens: np.ndarray, degree: int) -> np.ndarray:
    ll = np.log(np.maximum(lens, 1.0))
    cols = [np.ones_like(ll)]
    for d in range(1, degree + 1):
        cols.append(ll ** d)
    return np.stack(cols, axis=1)


def norm_reg(scores, lens, in_bg, degree=1, hetero=False, iso=False):
    """Прямая регрессия балла на длину: балл минус ожидаемый при этой длине,
    делённый на разброс (общий или тоже зависящий от длины).

    log(длины), а не длина: max из N попыток растёт как log N (обычная
    асимптотика максимума), поэтому линейная по log(длины) средняя -- не
    произвольный выбор формы, а то, что предсказывает теория для max-балла.
    """
    if iso:
        resid = scores - _iso_fit(lens, scores, in_bg)
    else:
        X = _design(lens, degree)
        resid = _ols_loo_resid(X, scores, in_bg)

    if not hetero:
        sd = resid[in_bg].std() or 1e-9
        return resid / sd

    a = np.abs(resid)
    if iso:
        scale = _iso_fit(lens, a, in_bg)
    else:
        scale = _design(lens, degree) @ np.linalg.pinv(
            _design(lens, degree)[in_bg].T @ _design(lens, degree)[in_bg]
        ) @ (_design(lens, degree)[in_bg].T @ a[in_bg])
    floor = np.median(a[in_bg]) * 0.1 + 1e-9
    return resid / np.maximum(scale, floor)


def _iso_fit(lens: np.ndarray, y: np.ndarray, in_bg: np.ndarray) -> np.ndarray:
    """Изотоническая (неубывающая по длине) подгонка среднего -- форма кривой
    не задаётся, только направление: длиннее абзац -> не меньший ожидаемый балл."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(lens[in_bg], y[in_bg])
    return ir.predict(lens)


# ------------------------------------------------------------------ метрики


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUC через ранги (= вероятность, что случайная цитата стоит выше
    случайного нецитатного абзаца); связи учтены средним рангом."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # средний ранг для связок
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ys = y[order]
    cum = np.cumsum(ys)
    prec = cum / np.arange(1, len(ys) + 1)
    n_pos = ys.sum()
    return float((prec * ys).sum() / n_pos) if n_pos else float("nan")


def recall_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    return float(y[order][:k].sum() / y.sum())


def median_rank(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    pos = np.where(y[order] == 1)[0] + 1
    return float(np.median(pos))


def evaluate(name: str, norm: np.ndarray, lens: np.ndarray, y: np.ndarray) -> dict:
    return {
        "norm": name,
        "r_len": float(np.corrcoef(lens, norm)[0, 1]),
        "rho_len": spearman(lens, norm),
        "auc": roc_auc(norm, y),
        "ap": average_precision(norm, y),
        "r@50": recall_at_k(norm, y, 50),
        "r@100": recall_at_k(norm, y, 100),
        "r@200": recall_at_k(norm, y, 200),
        "med_rank": median_rank(norm, y),
    }


# -------------------------------------------------------------------- прогон

NORMALIZERS: list[tuple[str, callable]] = (
    [("raw", lambda s, l, b: s)]
    + [(f"zbin{n}", (lambda n: lambda s, l, b: norm_zbin(s, l, b, n))(n))
       for n in (5, 10, 20, 40, 80, 160, 320)]
    + [(f"zbin{n}_robust", (lambda n: lambda s, l, b: norm_zbin(s, l, b, n, robust=True))(n))
       for n in (10, 40, 160)]
    + [(f"zknn{k}", (lambda k: lambda s, l, b: norm_zknn(s, l, b, k))(k))
       for k in (25, 50, 100, 200, 400)]
    + [(f"rankknn{k}", (lambda k: lambda s, l, b: norm_rankknn(s, l, b, k))(k))
       for k in (50, 100, 200, 400)]
    + [
        ("reg_lin", lambda s, l, b: norm_reg(s, l, b, degree=1)),
        ("reg_lin_het", lambda s, l, b: norm_reg(s, l, b, degree=1, hetero=True)),
        ("reg_quad", lambda s, l, b: norm_reg(s, l, b, degree=2)),
        ("reg_quad_het", lambda s, l, b: norm_reg(s, l, b, degree=2, hetero=True)),
        ("reg_cub_het", lambda s, l, b: norm_reg(s, l, b, degree=3, hetero=True)),
        ("reg_iso", lambda s, l, b: norm_reg(s, l, b, iso=True)),
        ("reg_iso_het", lambda s, l, b: norm_reg(s, l, b, iso=True, hetero=True)),
    ]
)


def run_method(method: str, candidates_tsv: str, ids, lens, gold_ids,
               bg_mode: str, out_rows: list[dict]) -> None:
    scores = load_max_scores(candidates_tsv, ids)
    y = np.array([1 if i in gold_ids else 0 for i in ids])
    in_bg = np.ones(len(ids), dtype=bool) if bg_mode == "loo" else (y == 0)

    print(f"\n=== {method} (фон: {bg_mode}), абзацев {len(ids)}, эталонных {int(y.sum())} ===")
    print(f"{'нормировка':16s} {'r(len)':>7s} {'rho(len)':>8s} {'AUC':>6s} "
          f"{'AP':>6s} {'r@50':>6s} {'r@100':>6s} {'r@200':>6s} {'мед.ранг':>9s}")
    for name, fn in NORMALIZERS:
        norm = fn(scores, lens, in_bg)
        m = evaluate(name, norm, lens, y)
        m["method"] = method
        m["bg_mode"] = bg_mode
        out_rows.append(m)
        print(f"{name:16s} {m['r_len']:7.3f} {m['rho_len']:8.3f} {m['auc']:6.3f} "
              f"{m['ap']:6.3f} {m['r@50']:6.2f} {m['r@100']:6.2f} {m['r@200']:6.2f} "
              f"{m['med_rank']:9.0f}")


def main(argv: list[str]) -> None:
    novel_tsv, gold_tsv, out_tsv = argv[0], argv[1], argv[2]
    specs = argv[3:]  # name=path ...
    ids, lens = load_lengths(novel_tsv)
    gold_ids = load_gold_ids(gold_tsv)

    rows: list[dict] = []
    for spec in specs:
        name, path = spec.split("=", 1)
        for bg_mode in ("loo", "exclude-gold"):
            run_method(name, path, ids, lens, gold_ids, bg_mode, rows)

    cols = ["method", "bg_mode", "norm", "r_len", "rho_len", "auc", "ap",
            "r@50", "r@100", "r@200", "med_rank"]
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols
            ) + "\n")
    print(f"\n-> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1:])
