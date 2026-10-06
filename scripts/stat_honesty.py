"""Статистика поверх рангов, которые fusion_retrieval.py сохраняет в RANKS_OUT.

1. Парный перестановочный тест (знаки разностей) для двух наборов признаков на
   ОДНИХ И ТЕХ ЖЕ местах эталона: H0 -- наборы взаимозаменимы, знак разности
   log-рангов у каждого места равновероятен. Бутстрап в evaluate() даёт
   интервал, этот тест -- p-значение, на малой выборке асимптотике не верят.
2. Полоса чувствительности к неполному эталону. Тихомиров комментирует не всё,
   поэтому часть «неправильных» стихов выше правильного может быть настоящей
   неотмеченной цитатой. Если из k стихов выше правильного (k = 0, 5, 10, 20)
   столько же считать неотмеченными позитивами, ранг правильного улучшается до
   max(1, rank - k). Ширина полосы между k=0 и k=20 -- честная граница, а не
   точка.

bpref здесь НЕ считается намеренно: он определён по размеченным непозитивам, а
в этом проекте их нет -- все неразмеченные пары презюмируются негативами. Без
размеченных негативов bpref вырождается в обычный ранговый показатель и
ничего не добавляет к медиане ранга.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def load(path: str) -> dict[str, dict[int, int]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["label"]] = {int(k): v for k, v in d["ranks"].items() if v is not None}
    return out


def paired_permutation(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Двусторонний p по среднему log-ранга; разность a-b, знаки переворачиваются."""
    d = np.log(a) - np.log(b)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = (signs * d).mean(1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n + 1)
    return obs, p


def band(ranks: np.ndarray) -> None:
    print(f"{'k неотмеченных выше':>20s} {'мед.ранг':>9s} {'топ-10':>7s} {'топ-50':>7s} {'топ-200':>8s}")
    for k in (0, 5, 10, 20):
        r = np.maximum(1, ranks - k)
        print(f"{k:20d} {np.median(r):9.0f} {np.mean(r<=10):7.2f} {np.mean(r<=50):7.2f} {np.mean(r<=200):8.2f}")


def main(*paths: str) -> None:
    sets = {}
    for p in paths:
        sets.update(load(p))
    labels = list(sets)
    base = labels[0]
    print(f"База: «{base}»; мест {len(sets[base])}\n")
    for lab in labels:
        r = np.array(list(sets[lab].values()), float)
        print(f"--- {lab} ---")
        band(r)
        print()
    for lab in labels[1:]:
        keys = sorted(set(sets[lab]) & set(sets[base]))
        a = np.array([sets[lab][k] for k in keys], float)
        b = np.array([sets[base][k] for k in keys], float)
        obs, p = paired_permutation(a, b)
        print(f"«{lab}» против «{base}»: средний log-ранг {obs:+.3f} "
              f"(ранг в e^x раз), перестановочный p = {p:.4f}, мест {len(keys)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
