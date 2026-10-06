#!/usr/bin/env python3
"""ШАГ 6. Контроли. Плацебо (перемешивание «запись -> абзац» внутри книги поэмы), длина источника
(случайная единица той же длины), хабовость (доля абзацев с одним и тем же топ-1), стабильность.
Пункты про LLM (контаминация, судья, повторы при T>0) неприменимы: LLM в методе нет -- исход «неприменим»."""
import sys, os, re, math, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = sys.argv[1]
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); S = E.load_scores(mat)
pix = {p: i for i, p in enumerate(pids)}
_, recs = E.read_gold(E.GOLD[mat].replace('_v0', '_v1'))
_, fb = E.read_gold(f'data/ext/gold/findable_{mat}_v1.tsv'); fmap = {(r['note'], r['ref_text']): r['has_bridge'] == '1' for r in fb}
recs = [r for r in E.ok_records(recs, 'step6') if fmap[(r['note'], r['ref_text'])]]
recs = [r for r in recs if (len(E.unit_idxs('A', r, units)) == 1 if E.is_bible(mat) else (True if E.is_d(mat) else re.search(r'\d+\s*[.:]\s*\d+', r['ref_text'])))]
n = len(recs); N = len(units); chance = math.lgamma(N + 1) / N
out = [f'ШАГ 6 / материал {mat}: множество протокола n={n}; источник {N} единиц; ожидание log-rank при случайности {chance:.3f}']
book_of = np.array([int(p.split('.')[1]) for p in pids])
unit_len = np.array([len(L.stems(t)) for t in E.source_units(mat)[1]])
rng = np.random.RandomState(20261005)
NPERM = 5000
# --- 1. плацебо
idx = [E.unit_idxs(mat, r, units) for r in recs]
rows_b = [pix[r['place']] for r in recs]
groups = {}
for i, r in enumerate(recs): groups.setdefault(int(r['book']), []).append(i)
out.append('\n== 1. Плацебо: запись -> чужой абзац той же книги поэмы (5000 перемешиваний) ==')
out.append(f'{"скорер":8s} {"реально":>8s} {"плацебо (ср.)":>14s} {"[2.5;97.5]":>16s} {"случайность":>12s} {"|плацебо-случ.|":>16s} {"p(реал.<=плацебо)":>18s}')
plac = {}; jrows = []
for k in E.SCORERS:
    M = S[k]
    real = np.mean([math.log(E.rank_tie(M[rows_b[i]], idx[i])) for i in range(n)])
    means = []
    # ранги считаем один раз по (абзац, запись) для 200 перемешиваний -- кэш по паре
    cache = {}
    for t in range(NPERM):
        tot = 0.0
        for b, ii in groups.items():
            perm = rng.permutation(ii)
            for a, c in zip(ii, perm):
                key = (rows_b[c], a)
                if key not in cache: cache[key] = math.log(E.rank_tie(M[rows_b[c]], idx[a]))
                tot += cache[key]
        means.append(tot / n)
    means = np.array(means); plac[k] = means.mean()
    jrows.append((f'{mat}-C-{k}', f'C{mat}', 'актуально', 'да', n, real - means.mean(), math.exp(real - means.mean()), ((means <= real).sum() + 1) / (len(means) + 1), f'реальная привязка против плацебо (перемешивание внутри книги поэмы), {k}, материал {mat}'))
    out.append(f'{k:8s} {real:8.3f} {means.mean():14.3f} [{np.percentile(means,2.5):.3f};{np.percentile(means,97.5):.3f}] {chance:12.3f} {abs(means.mean()-chance):16.3f} {((means<=real).sum()+1)/(len(means)+1):18.4f}')
d_plac = max(abs(v - chance) for v in plac.values()) > 0.10
# --- 2. длина источника: случайная единица той же длины (+-20%)
out.append('\n== 2. Длина источника: правильная единица против случайной единицы той же длины (+-20%), по тому же абзацу ==')
out.append(f'{"скорер":8s} {"правильная":>11s} {"случайная той же длины":>24s} {"разность":>9s}')
null_means = {}
for k in E.SCORERS:
    M = S[k]; vals = []; real = []
    for i in range(n):
        j = idx[i][0]; L0 = unit_len[j]
        cand = np.where((np.abs(unit_len - L0) <= 0.2 * L0) & (np.arange(N) != j))[0]
        if len(cand) == 0: cand = np.array([x for x in range(N) if x != j])
        c = rng.choice(cand, 20)
        vals.append(np.mean([math.log(E.rank_tie(M[rows_b[i]], [x])) for x in c]))
        real.append(math.log(E.rank_tie(M[rows_b[i]], idx[i])))
    null_means[k] = np.mean(vals)
    out.append(f'{k:8s} {np.mean(real):11.3f} {np.mean(vals):24.3f} {np.mean(vals)-np.mean(real):9.3f}')
out.append('разность нулевых средних относительно BM25 (положительная: скорер сам по себе ставит случайную единицу той же длины хуже BM25):')
for k in E.SCORERS[1:]: out.append(f'  {k:7s} {null_means[k]-null_means["bm25"]:+.3f}')
# --- 3. хабы
out.append('\n== 3. Хабовость: доля абзацев поэмы (все %d), у которых топ-1 -- одна и та же единица ==' % len(pids))
hub_found = False
for k in E.SCORERS:
    top = S[k].argmax(axis=1); cnt = np.bincount(top, minlength=N); u = cnt.argmax()
    share = cnt[u] / len(pids)
    if share >= 0.05: hub_found = True
    out.append(f'  {k:7s} топ-1 хаб: {units[u]} у {cnt[u]} абзацев ({share:.1%}); единиц, бывших топ-1 хотя бы раз: {(cnt>0).sum()}')
# --- 4. неприменимо
out.append('\n== 4. Контаминация LLM, жёсткие обманки судьи, повторы при T>0 ==\nИСХОД: критерий неприменим (в методе нет LLM, нет случайности: прогон детерминирован; проверка воспроизводимости -- на шаге 7).')
out.append(f'\nКритерий шага 6: плацебо вне ±0,10 от случайности: {d_plac}; хаб (>=5% абзацев) найден: {hub_found}')
out.append('ИСХОД: ' + ('дефект найден' if (d_plac or hub_found) else 'дефекта нет, критерий выполнен'))
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step6_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
with open(f'reports/ext/_journal_step6_{mat}.tsv', 'w', encoding='utf8') as f:
    for r in jrows: f.write('\t'.join(str(x) if not isinstance(x, float) else f'{x:.6f}' for x in r) + '\n')
with open(f'reports/ext/_defects_step6_{mat}.tsv', 'w', encoding='utf8') as f:
    f.write(f'{mat}\t6\tплацебо вне ±0,10 от случайности и/или хаб-источник\t{int(d_plac)+int(hub_found)}\tплацебо: ' + ', '.join(f'{k} {v:.2f}' for k, v in plac.items()) + f' (случайность {chance:.2f})\t{"найден" if (d_plac or hub_found) else "нет"}\n')
