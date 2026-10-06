# -*- coding: utf-8 -*-
"""Часть 4, З3 (prereg_closure.md): новые детекторы (Д-дубль, Д-доля), шаг 4 и шаг 5 в симуляции. Материал A, основа — эталон v1."""
import sys, os, random, math, collections, numpy as np
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth_defects.py'), encoding='utf8').read().split("KINDS = [")[0]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth_defects.py')
exec(src)
import ext_lib as L
out = ['З3: новые детекторы и шаги 4, 5 в симуляции (A, основа v1, n=%d)' % len(base)]
keyf = lambda r: (r['place'], r['verse_first'], r['verse_last'])
share0 = float(np.mean([len(idxs(r)) > 1 for r in base]))
dup0 = sum(c - 1 for c in collections.Counter(keyf(r) for r in base).values())
out.append(f'чистая основа: доля диапазонов {share0:.3f}; дублей {dup0}')
REPS = 20; rows = []
for kind in ('D4', 'D7'):
    for r_ in (0.05, 0.10, 0.20):
        acc = collections.defaultdict(list)
        for rep in range(REPS):
            rng = random.Random(f'z3-{20261005 + rep}-{kind}-{r_}')
            pick = set(rng.sample(range(len(base)), round(r_ * len(base)))); recs = []; inj = []
            for i, r in enumerate(base):
                if i in pick:
                    new = inject(kind, r, rng); recs += new; inj += [False] * (len(new) - 1) + [True]
                else: recs.append(r); inj.append(False)
            inj = np.array(inj)
            cnt = collections.Counter(keyf(r) for r in recs); fl = np.array([cnt[keyf(r)] > 1 for r in recs])
            acc['dup_recall'].append(fl[inj].mean() if inj.any() else float('nan')); acc['dup_fpr'].append(fl[~inj].mean())
            share = np.mean([len(idxs(r)) > 1 for r in recs if exists(r)])
            acc['share'].append(share); acc['share_flag'].append(float(share - share0 > 0.05))
            lr = {True: [], False: []}
            for r in recs:
                if exists(r): lr[len(idxs(r)) > 1].append(lrank(r))
            acc['gap_range_single'].append(np.mean(lr[True]) - np.mean(lr[False]))
        row = {k: float(np.nanmean(v)) for k, v in acc.items()}; rows.append((kind, r_, row))
out.append(f'\n{"дефект":5s} {"r":>5s} {"Д-дубль полнота":>16s} {"Д-дубль ложн.":>14s} {"доля диап.":>11s} {"Д-доля сработал":>16s} {"шаг4: диап.−одиноч. (log-rank)":>31s}')
for kind, r_, row in rows:
    out.append(f'{kind:5s} {r_:5.2f} {row["dup_recall"]:16.2f} {row["dup_fpr"]:14.3f} {row["share"]:11.3f} {row["share_flag"]:16.2f} {row["gap_range_single"]:31.3f}')
# шаг 5: записей на абзац
byp = collections.defaultdict(list)
for r in base:
    if exists(r): byp[r['place']].append(r)
out.append('\nШаг 5: отношение ширины интервала по абзацам к интервалу по записям при разном числе записей на абзац (BM25 log-rank, A v1)')
rng = random.Random(20261005)
for m in (1, 2, 3, 4):
    ps = [p for p, rs in byp.items() if len(rs) >= m]
    if len(ps) < 15: out.append(f'  m={m}: абзацев с ≥{m} записями {len(ps)} < 15, не считается'); continue
    recs = [r for p in ps for r in rng.sample(byp[p], m)]; lr = np.array([lrank(r) for r in recs]); cl = [r['place'] for r in recs]
    nb = [lr[np.random.RandomState(i).randint(0, len(lr), len(lr))].mean() for i in range(2000)]
    mean, lo, hi = L.cluster_boot_mean(lr, cl, B=2000)
    out.append(f'  m={m}: абзацев {len(ps)}, записей {len(recs)}; отношение ширин {(hi - lo) / (np.percentile(nb, 97.5) - np.percentile(nb, 2.5)):.2f}')
txt = '\n'.join(out); print(txt); open('reports/ext/synth_defects2.txt', 'w', encoding='utf8').write(txt + '\n')
