"""Оценка объединённого извлечения в задаче отбора абзацев: побеждает ли оно
длину, чего не смог ни один слой пилота.

Метрики те же, что в прошлой итерации, чтобы числа были сравнимы: AUC по
выборке, AUC среди абзацев сопоставимой длины (±20%, бутстрап по цитатам),
и логрегрессия «длина + скорер» против «только длина» -- то есть добавляет ли
скорер что-нибудь СВЕРХ длины.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from normalize_experiments import average_precision, recall_at_k, roc_auc

RATIO = 1.2


def load(path: str):
    ids, lens, y, feats, names = [], [], [], [], None
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        names = [c for c in reader.fieldnames
                 if c not in ("target_id", "length", "is_gold")]
        for r in reader:
            ids.append(r["target_id"])
            lens.append(float(r["length"]))
            y.append(int(r["is_gold"]))
            feats.append([float(r[c]) for c in names])
    return ids, np.array(lens), np.array(y), np.array(feats), names


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


def loo(X, y):
    """Полный leave-one-out: 646 абзацев -- считается за секунды, и в отличие
    от прошлой итерации исключается каждый абзац, а не только положительные."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        w = fit_logreg(Xn[m], y[m])
        out[i] = np.hstack([Xn[i], 1.0]) @ w
    return out


def matched(s, lens, pos, neg, ratio=RATIO):
    out = []
    for i in pos:
        m = neg[(lens[neg] >= lens[i] / ratio) & (lens[neg] <= lens[i] * ratio)]
        if len(m) >= 5:
            out.append(float(np.mean((s[i] > s[m]) + 0.5 * (s[i] == s[m]))))
    return np.array(out)


def boot(a, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return a.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_auc_diff(a, b, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        d.append(roc_auc(a[idx], y[idx]) - roc_auc(b[idx], y[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main(table: str) -> None:
    ids, lens, y, X, names = load(table)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    print(f"абзацев {len(ids)}, эталонных {int(y.sum())}, "
          f"нецитатных {len(neg)} (случайная выборка)")
    print(f"длина: эталон медиана {np.median(lens[y==1]):.0f}, "
          f"не-эталон медиана {np.median(lens[y==0]):.0f}\n")

    print(f"{'признак':28s} {'AUC':>6s} {'AP':>6s} {'r@50':>6s} "
          f"{'AUC|равн.длина (95%)':>24s}")

    def line(name, s):
        m, lo, hi = boot(matched(s, lens, pos, neg))
        print(f"{name:28s} {roc_auc(s,y):6.3f} {average_precision(s,y):6.3f} "
              f"{recall_at_k(s,y,50):6.2f}   {m:.3f} ({lo:.3f}..{hi:.3f})")

    line("длина абзаца", lens)
    for k, n in enumerate(names):
        line(n, X[:, k])

    loglen = np.log(np.maximum(lens, 1.0))[:, None]
    base = loo(loglen, y.astype(float))
    print()
    line("логрег: длина", base)
    for k, n in enumerate(names):
        s = loo(np.hstack([loglen, X[:, [k]]]), y.astype(float))
        line(f"логрег: длина + {n}", s)
        lo, hi = boot_auc_diff(s, base, y)
        print(f"{'':28s} прирост над «только длина»: "
              f"{roc_auc(s,y)-roc_auc(base,y):+.3f} (95%: {lo:+.3f}..{hi:+.3f})")
    s = loo(np.hstack([loglen, X]), y.astype(float))
    line("логрег: длина + все шесть", s)
    lo, hi = boot_auc_diff(s, base, y)
    print(f"{'':28s} прирост над «только длина»: "
          f"{roc_auc(s,y)-roc_auc(base,y):+.3f} (95%: {lo:+.3f}..{hi:+.3f})")
    s2 = loo(X, y.astype(float))
    line("логрег: шесть БЕЗ длины", s2)


if __name__ == "__main__":
    main(sys.argv[1])
