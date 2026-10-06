"""Переранжирование кандидатов внутри абзаца с поправкой на «хабность» стиха.

Задача атрибуции (какой именно стих), в отличие от задачи отбора абзацев,
от длины абзаца не зависит: абзац один и тот же для всех кандидатов. Зато
она страдает от другого перекоса -- есть стихи, которые оказываются в
кандидатах почти у каждого абзаца романа (длинные повествовательные стихи
с обычной лексикой). Они забивают верх списка у всех абзацев сразу и
именно поэтому сдвигают настоящую находку синтаксического слоя на медианный
ранг 259.

Поправка симметризует сравнение: балл пары сопоставляется не с нулём, а с
тем, сколько этот стих обычно набирает с произвольным абзацем (и наоборот).
Всё считается без меток -- статистика стихов берётся по всему роману, эталон
в неё не входит как эталон.

  raw        исходный балл
  minus_src  балл минус средний балл этого стиха по его лучшим совпадениям
  z_src      (балл - среднее стиха) / с.к.о. стиха
  ratio_src  балл / средний балл стиха
  rank_src   доля абзацев романа, у которых этот стих набрал меньше
  csls       2*балл - (среднее по лучшим у стиха) - (среднее по лучшим у абзаца)
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from evaluate_layers import in_refs, load_gold_refs

TOP_R = 10  # сколько лучших совпадений усредняется в «обычный балл» единицы


def load_pairs(path: str, score_col: str = "score"):
    tgt, src, sc = [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            tgt.append(row["target_id"])
            src.append(row["source_id"])
            sc.append(float(row[score_col]))
    return tgt, src, np.array(sc)


def top_mean(values: list[float], k: int) -> float:
    v = sorted(values, reverse=True)[:k]
    return sum(v) / len(v) if v else 0.0


def main(candidates_tsv: str, gold_tsv: str, score_col: str = "score") -> None:
    tgt, src, sc = load_pairs(candidates_tsv, score_col)
    print(f"пар: {len(sc)}", file=sys.stderr)

    by_src: dict[str, list[float]] = defaultdict(list)
    by_tgt: dict[str, list[float]] = defaultdict(list)
    for t, s, v in zip(tgt, src, sc):
        by_src[s].append(v)
        by_tgt[t].append(v)

    src_mean = {s: float(np.mean(v)) for s, v in by_src.items()}
    src_std = {s: float(np.std(v)) or 1e-9 for s, v in by_src.items()}
    src_top = {s: top_mean(v, TOP_R) for s, v in by_src.items()}
    tgt_top = {t: top_mean(v, TOP_R) for t, v in by_tgt.items()}
    src_sorted = {s: np.sort(np.array(v)) for s, v in by_src.items()}

    variants = {
        "raw": sc,
        "minus_src": np.array([v - src_top[s] for s, v in zip(src, sc)]),
        "z_src": np.array([(v - src_mean[s]) / src_std[s] for s, v in zip(src, sc)]),
        "ratio_src": np.array([v / max(src_mean[s], 1e-9) for s, v in zip(src, sc)]),
        "rank_src": np.array([
            float(np.searchsorted(src_sorted[s], v, side="left")) / len(src_sorted[s])
            for s, v in zip(src, sc)
        ]),
        "csls": np.array([2 * v - src_top[s] - tgt_top[t]
                          for t, s, v in zip(tgt, src, sc)]),
    }

    idx_by_tgt: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tgt):
        idx_by_tgt[t].append(i)

    gold_refs = load_gold_refs(gold_tsv)
    n_checkable = sum(len(g) for g in gold_refs.values())

    print(f"\nатрибуция внутри абзаца, {n_checkable} проверяемых мест эталона")
    print(f"{'вариант':12s} {'найдено':>8s} {'мед.ранг':>9s} {'топ-1':>6s} "
          f"{'топ-5':>6s} {'топ-10':>7s} {'топ-50':>7s} {'топ-200':>8s}")
    for name, vals in variants.items():
        ranks = []
        for tid, groups in gold_refs.items():
            ii = idx_by_tgt.get(tid)
            if not ii:
                continue
            ii = np.array(ii)
            order = ii[np.argsort(-vals[ii], kind="stable")]
            for group in groups:
                for rank, j in enumerate(order, start=1):
                    if in_refs(src[j], group):
                        ranks.append(rank)
                        break
        if not ranks:
            print(f"{name:12s} {'0':>8s}")
            continue
        r = np.array(ranks)
        print(f"{name:12s} {len(r):>3d}/{n_checkable:<4d} {np.median(r):9.0f} "
              f"{np.mean(r<=1):6.2f} {np.mean(r<=5):6.2f} {np.mean(r<=10):7.2f} "
              f"{np.mean(r<=50):7.2f} {np.mean(r<=200):8.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "score")
