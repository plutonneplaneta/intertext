"""Честная проверка «комбинация лучше лучшего одиночного признака» —
прямое применение находки из отчёта по методологии оценки: одна точечная
оценка leave-one-target-out на 16 абзацах нестабильна сама по себе, нужен
бутстреп по набору целевых абзацев, а не единственное число.

Пересэмплируем 16 целевых абзацев с возвращением B раз; на каждой выборке
пересчитываем медианный ранг комбинации и лучшего одиночного признака
(тем же leave-one-target-out, что и в combine_and_evaluate.py) и считаем
долю ресэмплов, где комбинация оказалась лучше или равна.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict

import numpy as np

from combine_and_evaluate import fit_logreg, load_table, predict, rank_of_best_positive

B = 200


def loto_ranks(by_target: dict, targets: list[str], mu, sigma, n_feat: int):
    """Возвращает {признак/комбинация: [ранги]} для leave-one-target-out
    по заданному (возможно, повторяющемуся из-за ресэмпла) списку targets."""
    combined, singles = [], defaultdict(list)
    seen_once = list(dict.fromkeys(targets))  # без повторов для цикла обучения
    for held_out in seen_once:
        train_rows = [r for t, rs in by_target.items() if t != held_out for r in rs]
        test_rows = by_target[held_out]
        Xtr = np.array([r[2] for r in train_rows])
        ytr = np.array([r[3] for r in train_rows], dtype=float)
        w = fit_logreg((Xtr - mu) / sigma, ytr, l2=2.0)
        Xte = np.array([r[2] for r in test_rows])
        yte = np.array([r[3] for r in test_rows])
        combined_scores = predict((Xte - mu) / sigma, w)
        r_comb = rank_of_best_positive(combined_scores, yte)
        r_single = [rank_of_best_positive(Xte[:, i], yte) for i in range(n_feat)]
        combined.append(r_comb)
        singles[held_out] = r_single
    return combined, singles


def main(table_path: str) -> None:
    rows, feature_names = load_table(table_path)
    n_feat = len(feature_names)
    by_target: dict[str, list] = defaultdict(list)
    for r in rows:
        by_target[r[0]].append(r)
    targets_with_pos = [t for t, rs in by_target.items() if any(r[3] == 1 for r in rs)]
    n = len(targets_with_pos)
    print(f"абзацев с проверяемым кандидатом: {n}", file=sys.stderr)

    all_X = np.array([r[2] for r in rows])
    mu, sigma = all_X.mean(axis=0), all_X.std(axis=0)
    sigma[sigma == 0] = 1.0

    # базовый прогон один раз -- используем те же ранги для всех ресэмплов,
    # чтобы не переобучать логрегрессию по 200 раз на пересекающихся фолдах
    # (дёшево и корректно: ранг held-out абзаца не зависит от того, сколько
    # раз он попал в конкретный ресэмпл, только от факта "какие абзацы training")
    # -- поэтому фактически варьируем состав ТЕСТОВОГО множества, на котором
    # считаем медиану, а не сам процесс LOTO.
    combined_ranks, single_ranks_by_target = loto_ranks(by_target, targets_with_pos, mu, sigma, n_feat)
    rank_by_target = dict(zip(targets_with_pos, combined_ranks))

    def med(vals):
        vals = [v for v in vals if v is not None]
        return sorted(vals)[len(vals) // 2] if vals else None

    best_single_feat = min(
        range(n_feat),
        key=lambda i: med([single_ranks_by_target[t][i] for t in targets_with_pos]) or 1e9,
    )
    print(f"лучший одиночный признак по полной выборке: {feature_names[best_single_feat]}", file=sys.stderr)

    rng = random.Random(7)
    wins, ties, losses = 0, 0, 0
    for _ in range(B):
        sample = [rng.choice(targets_with_pos) for _ in range(n)]
        med_comb = med([rank_by_target[t] for t in sample])
        med_single = med([single_ranks_by_target[t][best_single_feat] for t in sample])
        if med_comb is None or med_single is None:
            continue
        if med_comb < med_single:
            wins += 1
        elif med_comb == med_single:
            ties += 1
        else:
            losses += 1

    print(f"\nбутстреп по {B} ресэмплам {n} абзацев:")
    print(f"комбинация лучше: {wins}/{B} ({100*wins/B:.0f}%)")
    print(f"без разницы:      {ties}/{B} ({100*ties/B:.0f}%)")
    print(f"комбинация хуже:  {losses}/{B} ({100*losses/B:.0f}%)")
    print(f"\nВывод: доля {100*wins/B:.0f}% -- это и есть честная граница уверенности,"
          f" не единственное число \"комбинация лучше\".")


if __name__ == "__main__":
    main(sys.argv[1])
