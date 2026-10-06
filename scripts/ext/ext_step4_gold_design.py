#!/usr/bin/env python3
"""ШАГ 4. Выигрыши, объяснимые устройством эталона.
A: одиночные ссылки (1 стих) против диапазонов (>1 стиха).
B: единица -- книга, диапазонов нет по построению; аналог -- ссылка на конкретное место (указан номер строки)
   против ссылки на книгу целиком («Aeneid book 4», «Odyssey 2»).
Пул кандидатов: исчерпывающий счёт, объединение топ-K не используется (пункт шага неприменим).
Метрики: среднее log-rank по скорерам и разности с BM25 (бутстрап по абзацам)."""
import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = sys.argv[1]; ver = sys.argv[2] if len(sys.argv) > 2 else 'v1'
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); S = E.load_scores(mat)
_, recs = E.read_gold(E.GOLD[mat].replace('_v0', '_' + ver))
recs = E.ok_records(recs, 'step4')
ev = E.eval_records(mat, recs, S, pids, units)
if E.is_bible(mat):
    nverse = np.array([len(E.unit_idxs('A', r, units)) for r in recs])
    grp = {'одиночные (1 стих)': nverse == 1, 'диапазоны (>1 стиха)': nverse > 1}
elif E.is_d(mat):
    if mat == 'E':   # у E нет строф-источников: аналог -- стихи (PoetryCorpus) против прозы (Викитека)
        kind = np.array([r['source_unit'].startswith('Pc') for r in recs]); grp = {'стихи (PoetryCorpus)': kind, 'проза (Викитека)': ~kind}
    else:
        kind = np.array([r['source_unit'].startswith('ON.') for r in recs])
        grp = {'внутритекстовые (строфа Онегина)': kind, 'межавторские (стихотворение)': ~kind}
else:
    line = np.array([bool(re.search(r'\d+\s*[.:]\s*\d+', r['ref_text'])) for r in recs])
    grp = {'с номером строки': line, 'книга целиком': ~line}
cl = [r['place'] for r in recs]
out = [f'ШАГ 4 / материал {mat}, эталон {ver}: записей {len(recs)}']
for g, m in grp.items(): out.append(f'  {g}: n={m.sum()} записей, {len({c for c, x in zip(cl, m) if x})} абзацев')
out.append(f'\n{"скорер":8s} ' + ' '.join(f'{g[:22]:>24s}' for g in ['все'] + list(grp)))
res = {}
for k in E.SCORERS:
    rk = ev[k][0]; row = [E.logmean(rk)] + [E.logmean(rk[m]) if m.sum() >= 5 else float('nan') for m in grp.values()]
    res[k] = row
    out.append(f'{k:8s} ' + ' '.join(f'{v:24.3f}' for v in row))
out.append('\nРазность с BM25 (среднее log-rank; отрицательная -- лучше BM25), 95% бутстрап по абзацам:')
for k in E.SCORERS[1:]:
    parts = []
    for name, m in [('все', np.ones(len(recs), bool))] + list(grp.items()):
        if m.sum() < 5: parts.append(f'{name}: n<5'); continue
        d = np.log(ev[k][0][m]) - np.log(ev['bm25'][0][m])
        mean, lo, hi = L.cluster_boot_mean(d, [c for c, x in zip(cl, m) if x], B=2000)
        parts.append(f'{name}: {mean:+.3f} [{lo:+.3f},{hi:+.3f}]')
    out.append(f'  {k:7s} ' + ' | '.join(parts))
allv = res['bm25'][0]; single = res['bm25'][1]
rel = abs(single - allv) / allv
d4 = rel >= 0.10
out.append(f'\nКритерий шага 4: |BM25 одиночные − все| / все = {rel:.1%} (порог 10%): {d4}')
out.append('ИСХОД: ' + ('дефект найден (оценка зависит от структуры эталона)' if d4 else 'дефекта нет, критерий выполнен') + (' [для B формулировка шага — по аналогу «строка/книга»]' if mat == 'B' else ''))
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step4_{mat}{"" if ver=="v1" else "_"+ver}.txt', 'w', encoding='utf8').write(txt + '\n')
if ver == 'v1':
    with open(f'reports/ext/_defects_step4_{mat}.tsv', 'w', encoding='utf8') as f:
        f.write(f'{mat}\t4\tоценка зависит от структуры эталона (диапазоны / ссылки на книгу)\t{int(grp[list(grp)[1]].sum())}\tBM25 mean log-rank все {allv:.3f} -> {list(grp)[0]} {single:.3f} (отн. {rel:.1%})\t{"найден" if d4 else "нет"}\n')
