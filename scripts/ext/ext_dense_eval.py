# -*- coding: utf-8 -*-
"""Часть 4, З1 (prereg_closure.md): плотные скореры на множестве протокола. Использование: ext_dense_eval.py <mat> <каталог dense_*.npy>"""
import sys, os, re, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
mat = sys.argv[1]; DD = sys.argv[2]
pass
import ext_eval as E, ext_lib as L
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); pix = {p: i for i, p in enumerate(pids)}
S = E.load_scores(mat, ['bm25', 'fuse'])
dense = [k for k in ('e5l', 'bgem3', 'sbertru') if os.path.exists(f'{DD}/dense_{mat}_{k}.npy')]
for k in dense: S[k] = np.load(f'{DD}/dense_{mat}_{k}.npy').astype(np.float64)
names = ['bm25', 'fuse'] + dense
_, recs = E.read_gold(E.GOLD[mat].replace('_v0', '_v1')); recs = E.ok_records(recs, 'dense')
_, fb = E.read_gold(f'data/ext/gold/findable_{mat}_v1.tsv'); fm = {(r['note'], r['ref_text']): r['has_bridge'] == '1' for r in fb}
recs = [r for r in recs if fm[(r['note'], r['ref_text'])]]
if E.is_bible(mat): recs = [r for r in recs if len(E.unit_idxs('A', r, units)) == 1]
elif not E.is_d(mat): recs = [r for r in recs if re.search(r'\d+\s*[.:]\s*\d+', r['ref_text'])]
n = len(recs); N = len(units); chance = math.lgamma(N + 1) / N; cl = [r['place'] for r in recs]
K = 1000 if N > 20000 else (100 if N > 1000 else 10)
idx = [E.unit_idxs(mat, r, units) for r in recs]; rows_ = [pix[r['place']] for r in recs]
out = [f'З1 / материал {mat}: множество протокола n={n} записей, {len(set(cl))} абзацев; источник {N} единиц; случайность {chance:.3f}; K={K}']
lr = {k: np.array([math.log(E.rank_tie(S[k][rows_[i]], idx[i])) for i in range(n)]) for k in names}
rng = np.random.RandomState(20261005)
def perm_p(d, B=20000):
    ucl = sorted(set(cl)); D = np.array([d[[i for i, c in enumerate(cl) if c == u]].sum() for u in ucl]); sg = rng.choice([-1., 1.], size=(B, len(ucl)))
    obs = D.sum() / len(d); return obs, (np.sum(np.abs(sg @ D / len(d)) >= abs(obs) - 1e-12) + 1) / (B + 1)
ps = {k: perm_p(lr[k] - lr['bm25']) for k in names[1:]}
fam = [k for k in dense]; order = sorted(fam, key=lambda k: ps[k][1]); hol = {}; run = 0.0
for rank, k in enumerate(order): run = max(run, min(1.0, (len(fam) - rank) * ps[k][1])); hol[k] = run   # Холм только по плотным (prereg З1)
# плацебо
groups = {}
for i, r in enumerate(recs): groups.setdefault(r['place'].split('.')[1], []).append(i)
def plac(k, nperm=2000):
    M = S[k]; cache = {}; means = []
    for _ in range(nperm):
        tot = 0.0
        for g, ii in groups.items():
            perm = rng.permutation(ii)
            for a, c in zip(ii, perm):
                key = (rows_[c], a)
                if key not in cache: cache[key] = math.log(E.rank_tie(M[rows_[c]], idx[a]))
                tot += cache[key]
        means.append(tot / n)
    return np.array(means)
# отбор
pos = {r['place'] for r in E.ok_records(E.read_gold(E.GOLD[mat])[1], 'dense-v0')}
y = np.array([p in pos for p in pids])
def auc(score):
    pi = np.where(y)[0]; ni = np.where(~y)[0]; sa, sb = score[pi][:, None], score[ni][None, :]
    return float((((sa > sb) + 0.5 * (sa == sb)).mean()))
out.append(f'{"скорер":8s} {"ср.log-rank":>11s} {"Δ к BM25":>9s} {"p (Холм)":>9s} {"ранг<=K":>8s} {"реал.−плацебо":>14s} {"хаб, макс. доля":>16s} {"AUC отбора (макс. балл)":>24s}')
for j, k in enumerate(names):
    m = plac(k); real = lr[k].mean(); gap = real - m.mean()
    top = S[k].argmax(axis=1); hub = np.bincount(top, minlength=N).max() / len(pids)
    d = '' if j == 0 else f'{ps[k][0]:+.3f}'; p = '' if j == 0 else (f'{hol[k]:.4f}' if k in hol else f'{ps[k][1]:.4f}*')
    out.append(f'{k:8s} {real:11.3f} {d:>9s} {p:>9s} {np.mean([math.exp(v) <= K for v in lr[k]]):8.3f} {gap:14.3f} {hub:16.3f} {auc(S[k].max(axis=1)):24.3f}')
out.append('* — сырое p (fuse не входит в семейство плотных скореров)')
out.append(f'только длина: AUC {auc(ntok.astype(float)):.3f}')
txt = '\n'.join(out); print(txt); open(f'reports/ext/dense_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
