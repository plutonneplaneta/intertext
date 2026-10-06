"""Логистическая регрессия по трём признакам (лексика, эмбеддинги,
синтаксис), оценка leave-one-target-out -- на 46 целевых абзацах и 32
положительных парах обычное train/dev/test было бы шумом одного случайного
разбиения, а не оценкой. Каждый абзац по очереди становится тестовым: модель
обучается на остальных 45, применяется к кандидатам исключённого -- так
переобучение на конкретном абзаце не может завысить его собственный ранг.

Логрегрессия -- градиентный спуск на numpy, L2-регуляризация (существенна:
32 положительных примера на 3 признака переобучаются мгновенно без неё).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def load_table(path: str):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append((
                row["target_id"], row["source_id"],
                float(row["lex"]), float(row["emb"]), float(row["syn"]),
                int(row["label"]),
            ))
    return rows


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, lr: float = 0.5, iters: int = 2000) -> np.ndarray:
    """X уже стандартизирован. Возвращает веса (с учётом свободного члена
    как последнего столбца единиц)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n
        grad[:-1] += l2 * w[:-1] / n  # не штрафуем свободный член
        w -= lr * grad
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def rank_of_best_positive(scores: np.ndarray, labels: np.ndarray) -> int | None:
    order = np.argsort(-scores)
    ranked_labels = labels[order]
    hits = np.where(ranked_labels == 1)[0]
    return int(hits[0]) + 1 if len(hits) else None


def main(table_path: str) -> None:
    rows = load_table(table_path)
    by_target: dict[str, list] = defaultdict(list)
    for r in rows:
        by_target[r[0]].append(r)

    targets_with_pos = [t for t, rs in by_target.items() if any(r[5] == 1 for r in rs)]
    print(f"абзацев всего: {len(by_target)}, с хотя бы одним верным кандидатом: {len(targets_with_pos)}", file=sys.stderr)

    all_X = np.array([[r[2], r[3], r[4]] for r in rows])
    mu, sigma = all_X.mean(axis=0), all_X.std(axis=0)
    sigma[sigma == 0] = 1.0

    results_combined = []
    results_lex = []
    results_emb = []
    results_syn = []

    for held_out in targets_with_pos:
        train_rows = [r for t, rs in by_target.items() if t != held_out for r in rs]
        test_rows = by_target[held_out]

        Xtr = np.array([[r[2], r[3], r[4]] for r in train_rows])
        ytr = np.array([r[5] for r in train_rows], dtype=float)
        Xtr_n = (Xtr - mu) / sigma

        w = fit_logreg(Xtr_n, ytr, l2=2.0)

        Xte = np.array([[r[2], r[3], r[4]] for r in test_rows])
        yte = np.array([r[5] for r in test_rows])
        Xte_n = (Xte - mu) / sigma
        combined_scores = predict(Xte_n, w)

        results_combined.append(rank_of_best_positive(combined_scores, yte))
        results_lex.append(rank_of_best_positive(Xte[:, 0], yte))
        results_emb.append(rank_of_best_positive(Xte[:, 1], yte))
        results_syn.append(rank_of_best_positive(Xte[:, 2], yte))

    def summarize(name, ranks):
        ranks = [r for r in ranks if r is not None]
        top1 = sum(1 for r in ranks if r == 1)
        top5 = sum(1 for r in ranks if r <= 5)
        top10 = sum(1 for r in ranks if r <= 10)
        med = sorted(ranks)[len(ranks) // 2] if ranks else None
        print(f"{name:20s} медиана ранга={med}, топ-1: {top1}/{len(targets_with_pos)}, "
              f"топ-5: {top5}/{len(targets_with_pos)}, топ-10: {top10}/{len(targets_with_pos)}")

    print(f"\nleave-one-target-out, {len(targets_with_pos)} абзацев с проверяемым положительным кандидатом:\n")
    summarize("только лексика", results_lex)
    summarize("только эмбеддинги", results_emb)
    summarize("только синтаксис", results_syn)
    summarize("КОМБИНАЦИЯ (LOO)", results_combined)

    # веса на полных данных -- только для интерпретации, не для оценки
    Xall_n = (all_X - mu) / sigma
    yall = np.array([r[5] for r in rows], dtype=float)
    w_full = fit_logreg(Xall_n, yall, l2=2.0)
    print(f"\nвеса на полном наборе (интерпретация, не оценка): "
          f"лексика={w_full[0]:.2f} эмбеддинги={w_full[1]:.2f} синтаксис={w_full[2]:.2f} своб.член={w_full[3]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
