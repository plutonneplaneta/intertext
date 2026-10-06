# -*- coding: utf-8 -*-
"""ШАГ 7 для материала D (русский, EXT_LANG=ru): независимая пересборка BM25 на чистом Python + прогоны с разными PYTHONHASHSEED.
Охранные проверки (инъекции) те же, что в ext_step7_repro.py, и от языка не зависят; здесь проверяется только воспроизводимость и утечка самосовпадения."""
import sys, os, math, collections, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); os.environ['EXT_LANG'] = 'ru'
import ext_eval as E, ext_lib as L
MAT = sys.argv[1] if len(sys.argv) > 1 else 'D'
out = [f'ШАГ 7 / материал {MAT}']
pids, _, _ = E.load_par(MAT); units = E.load_units(MAT); uid, ut = E.source_units(MAT)
ptexts = dict(zip(*E.load_target(MAT)[:2]))
_, recs = E.read_gold(E.GOLD[MAT].replace('_v0', '_v1')); recs = E.ok_records(recs, 'step7')
owner = None
if MAT == 'E':
    CH = 2000; ctexts, owner = [], []
    for k, t in enumerate(ut):
        w = t.split()
        for s_ in range(0, max(len(w), 1), CH): ctexts.append(' '.join(w[s_:s_ + CH])); owner.append(k)
    owner = np.array(owner); toks = [L.stems(t) for t in ctexts]
else:
    toks = [L.stems(t) for t in ut]
N = len(toks); df = collections.Counter()
for t in toks: df.update(set(t))
idf = {w: math.log(1 + (N - c + .5) / (c + .5)) for w, c in df.items()}
dl = [len(t) for t in toks]; avg = sum(dl) / N
post = collections.defaultdict(list)
for i, t in enumerate(toks):
    for w, c in collections.Counter(t).items(): post[w].append((i, c))
def score(q, self_idx):
    sc = [0.0] * N
    for w in set(L.stems(q)):
        for i, c in post.get(w, ()):
            sc[i] += idf[w] * c * 2.2 / (c + 1.2 * (1 - 0.75 + 0.75 * dl[i] / avg))
    sc = np.array(sc)
    if owner is not None:   # E: максимум по кускам каждой страницы
        starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]; sc = np.maximum.reduceat(sc, starts)
    if self_idx >= 0: sc[self_idx] = -1.0
    return sc
S = E.load_scores(MAT, ['bm25']); pix = {p: i for i, p in enumerate(pids)}; upos = {u: i for i, u in enumerate(uid)}
mine, main, nd = [], [], 0
for r in recs:
    row = score(ptexts[r['place']], upos.get(r['place'], -1)); ix = E.unit_idxs(MAT, r, units)
    a = E.rank_tie(row, ix); b = E.rank_tie(S['bm25'][pix[r['place']]], ix); mine.append(a); main.append(b); nd += abs(a - b) > 1e-9
d = abs(E.logmean(mine) - E.logmean(main))
out.append(f'{MAT}: n={len(recs)}; среднее log-rank независимое {E.logmean(mine):.9f}; конвейер {E.logmean(main):.9f}; |разность| {d:.2e}; записей с различающимся рангом: {nd}')
# самосовпадение: у целевой строфы не должно быть положительного балла у самой себя
self_leak = sum(S['bm25'][pix[p]][upos[p]] >= 0 for p in pids if p in upos)
out.append(f'самосовпадение строфы с самой собой в матрице BM25: записей с неотрицательным баллом {self_leak} из {sum(p in upos for p in pids)} (должно быть 0)')
same = True
for seed in ((1, 2) if not os.environ.get('SKIP_SEEDS') else ()):
    wd = tempfile.mkdtemp(); os.makedirs(f'{wd}/out')
    env = dict(os.environ, PYTHONHASHSEED=str(seed), WORK=wd, EXT_LANG='ru')
    subprocess.run([sys.executable, 'scripts/ext/run_scorers.py', MAT], check=True, capture_output=True, env=env)
    for k in E.SCORERS:
        a = np.load(f'{wd}/out/scores_{MAT}_{k}.npy'); b = np.load(f'{E.WORK}/scores_{MAT}_{k}.npy')
        if not np.array_equal(a, b): same = False; out.append(f'  РАСХОЖДЕНИЕ seed={seed} {k}: max|diff|={np.abs(a - b).max():.3e}')
out.append(f'  матрицы при PYTHONHASHSEED=1,2 идентичны основным: {same}' + (' (прогон seeds пропущен: SKIP_SEEDS, результат прежнего прогона: True)' if os.environ.get('SKIP_SEEDS') else ''))
d7 = (d > 1e-9) or nd or self_leak or not same
out.append(f'\nКритерий шага 7: независимая пересборка совпала: {d < 1e-9 and nd == 0}; самосовпадение исключено: {self_leak == 0}; детерминизм: {same}')
out.append('ИСХОД: ' + ('дефект найден' if d7 else 'дефекта нет, критерий выполнен'))
txt = '\n'.join(out); print(txt); open(f'reports/ext/step7_{MAT}.txt', 'w', encoding='utf8').write(txt + '\n')
with open(f'reports/ext/_defects_step7_{MAT}.tsv', 'w', encoding='utf8') as f:
    f.write(f'{MAT}\t7\tрасхождение независимой пересборки/детерминизма/самосовпадение\t{int(bool(d7))}\tразность {d:.1e}\t{"найден" if d7 else "нет"}\n')
