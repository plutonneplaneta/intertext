#!/usr/bin/env python3
"""Исчерпывающий счёт всех скореров: абзацы поэмы x единицы источника. Кэш .npy в $WORK/out.
Использование: run_scorers.py A|B
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_lib as L, ext_eval as E
mat = sys.argv[1]
WORK = os.environ.get('WORK', '/tmp/intertext-ext') + '/out'
pids, ptexts, nl = E.load_target(mat)
uid, ut = E.source_units(mat)
t0 = time.time()
upos = {u: i for i, u in enumerate(uid)}
if mat == 'E':
    # E: страницы произведений до сотен тысяч слов -- счёт по кускам (<=2000 слов), оценка страницы = максимум по её кускам; fuse после агрегации
    CH = 2000; cid, ctx, own = [], [], []
    for k, (u, t) in enumerate(zip(uid, ut)):
        w = t.split()
        for s_ in range(0, max(len(w), 1), CH): cid.append(f'{u}#{s_ // CH}'); ctx.append(' '.join(w[s_:s_ + CH])); own.append(k)
    own = np.array(own); starts = np.r_[0, np.flatnonzero(np.diff(own)) + 1]
    ixc = L.Index(cid, ctx); raw = ixc.score_raw(ptexts)
    raw = {k: np.maximum.reduceat(v, starts, axis=1) for k, v in raw.items()}
    S = L.Index.finish(raw, None)
    for k, v in S.items():
        if not (v != 0).any(): raise SystemExit(f'скорер {k}: нулевой после агрегации')
else:
    ix = L.Index(uid, ut)
    S = ix.score_all(ptexts, exclude=[upos.get(p, -1) for p in pids])   # D: строфа не сравнивается сама с собой
for k, v in S.items():
    np.save(f'{WORK}/scores_{mat}_{k}.npy', v.astype(np.float64))   # float32 менял ранг при связках (шаг 7)
with open(f'{WORK}/par_{mat}.tsv', 'w', encoding='utf8') as f:
    for p, n, t in zip(pids, nl, ptexts): f.write(f'{p}\t{n}\t{len(L.stems(t))}\n')
with open(f'{WORK}/units_{mat}.tsv', 'w', encoding='utf8') as f:
    for u in uid: f.write(u + '\n')
print(mat, len(pids), 'абзацев x', len(uid), 'единиц;', {k: float((v != 0).mean().round(4)) for k, v in S.items()}, f'{time.time()-t0:.0f}с')
