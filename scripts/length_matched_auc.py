"""Главная проверка: остаётся ли у слоя сигнал, если сравнивать только
абзацы сопоставимой длины.

Обычная AUC по всему роману отвечает на бесполезный вопрос. Эталонные абзацы
длиннее нецитатных в семь с половиной раз по медиане (154 слова против 20),
поэтому любой балл, хоть как-то растущий с длиной, получает высокую AUC, и
сама длина в этом состязании выигрывает у всех трёх слоёв. Осмысленный
вопрос другой: если взять цитату и нецитатные абзацы ТОЙ ЖЕ длины, поставит
ли балл цитату выше.

Здесь для каждой цитаты берутся нецитатные абзацы, чья длина отличается не
более чем в 1,2 раза, и считается доля пар, где цитата выше. Бутстрап идёт
по цитатам, а не по парам: 46 цитат порождают двадцать тысяч зависимых пар,
и бутстрап по парам дал бы доверительный интервал в десять раз уже
настоящего (0,629..0,641 вместо 0,504..0,681 -- разница между «сигнал есть»
и «граница неотличима от случайного»).

Контрольная строка -- сама длина: внутри полосы ±20% она всё ещё даёт около
0,55, и это та планка, которую слой должен превзойти, а не 0,5.
"""
from __future__ import annotations

import sys

import numpy as np

from normalize_experiments import (average_precision, load_gold_ids,
                                   load_lengths, load_max_scores, norm_reg,
                                   norm_rankknn, norm_zbin, norm_zknn,
                                   recall_at_k, roc_auc)

RATIO = 1.2
N_BOOT = 5000


def matched_auc_per_positive(s, lens, pos, neg, ratio=RATIO):
    out = []
    for i in pos:
        m = neg[(lens[neg] >= lens[i] / ratio) & (lens[neg] <= lens[i] * ratio)]
        if len(m) >= 5:
            out.append(float(np.mean((s[i] > s[m]) + 0.5 * (s[i] == s[m]))))
    return np.array(out)


def boot(a, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    b = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return a.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(novel_tsv: str, gold_tsv: str, *specs: str) -> None:
    ids, lens = load_lengths(novel_tsv)
    gold = load_gold_ids(gold_tsv)
    y = np.array([1 if i in gold else 0 for i in ids])
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    in_bg = np.ones(len(ids), dtype=bool)

    print(f"абзацев {len(ids)}, эталонных {int(y.sum())}")
    print(f"длина эталона (децили 10/25/50/75/90): "
          f"{np.percentile(lens[y==1],[10,25,50,75,90]).astype(int)}")
    print(f"длина не-эталона: {np.percentile(lens[y==0],[10,25,50,75,90]).astype(int)}")

    for spec in specs:
        name, path = spec.split("=", 1)
        s = load_max_scores(path, ids)
        print(f"\n--- {name}: AUC среди абзацев сопоставимой длины (±{int((RATIO-1)*100)}%) ---")
        variants = [
            ("сырой балл", s),
            ("zbin10", norm_zbin(s, lens, in_bg, 10)),
            ("zbin80", norm_zbin(s, lens, in_bg, 80)),
            ("zbin320", norm_zbin(s, lens, in_bg, 320)),
            ("zknn50", norm_zknn(s, lens, in_bg, 50)),
            ("rankknn200", norm_rankknn(s, lens, in_bg, 200)),
            ("reg_lin", norm_reg(s, lens, in_bg, degree=1)),
            ("reg_lin_het", norm_reg(s, lens, in_bg, degree=1, hetero=True)),
            ("reg_iso_het", norm_reg(s, lens, in_bg, iso=True, hetero=True)),
        ]
        for vname, v in variants:
            m, lo, hi = boot(matched_auc_per_positive(v, lens, pos, neg))
            print(f"  {vname:12s} {m:.3f}  (95%: {lo:.3f}..{hi:.3f})")
        print(f"  {'весь роман':12s} AUC={roc_auc(s,y):.3f} AP={average_precision(s,y):.3f} "
              f"r@100={recall_at_k(s,y,100):.2f} r@200={recall_at_k(s,y,200):.2f} "
              f"r(длина)={np.corrcoef(lens,s)[0,1]:.3f}")

    m, lo, hi = boot(matched_auc_per_positive(lens.astype(float), lens, pos, neg))
    print(f"\n(контроль) сама длина внутри полосы: {m:.3f} (95%: {lo:.3f}..{hi:.3f})")
    print(f"(контроль) сама длина по всему роману: AUC={roc_auc(lens,y):.3f} "
          f"AP={average_precision(lens,y):.3f} r@100={recall_at_k(lens,y,100):.2f} "
          f"r@200={recall_at_k(lens,y,200):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
