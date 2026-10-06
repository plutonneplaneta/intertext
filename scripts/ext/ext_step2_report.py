#!/usr/bin/env python3
"""ШАГ 2: итоговый отчёт. Автопроверки + ручная выборка + поправки + ключевое число до/после."""
import sys, os, math, collections, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E
mat = sys.argv[1]
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); S = E.load_scores(mat, ['bm25'])
v0 = E.GOLD[mat]
v1 = v0.replace('_v0', '_v1')
_, r0 = E.read_gold(v0); _, r1 = E.read_gold(v1); _, au = E.read_gold(f'data/ext/gold/audit_{mat}.tsv')
_, lab = E.read_gold(f'data/ext/gold/step2_manual_labels_{mat}.tsv')
_, cs = E.read_gold(f'data/ext/gold/corrections_{mat}.tsv')
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w
n0 = len(r0); out = [f'ШАГ 2 / материал {mat}', f'записей v0: {n0}; v1: {len(r1)}']
fl = collections.Counter(x for r in au for x in r['flags'].split(',') if x)
out.append('автоматические флаги (v0): ' + (', '.join(f'{k}={v}' for k, v in fl.most_common()) or 'нет'))
hc = [r for r in au if r['cov_head_par'] != '']
ok_head = sum(float(r['cov_head_par']) >= 0.5 for r in hc)
out.append(f'привязка: заголовок примечания найден в абзаце у {ok_head}/{len(hc)} записей ({ok_head/len(hc):.1%}); '
           f'2 флага при ручном просмотре — заголовок «Lines N-M» / орфография имени, не дефект привязки' if mat == 'A' else
           f'привязка: заголовок найден в абзаце у {ok_head}/{len(hc)} ({ok_head/len(hc):.1%})')
if E.is_bible(mat):
    qc = collections.Counter(r['quote'] for r in au if r['quote'])
    out.append(f'сверка цитат (независимый путь): записей с цитатой {sum(qc.values())}; ' + ', '.join(f'{k}={v}' for k, v in qc.items()))
lc = collections.Counter(r['label'] for r in lab)
n = len(lab)
out.append(f'ручная выборка (n={n}, семя 20261005): ' + ', '.join(f'{k}={v}' for k, v in lc.items()))
for k in ('background', 'not_by_text', 'invalid_reference'):
    lo, hi = wilson(lc.get(k, 0), n)
    out.append(f'  доля {k}: {lc.get(k,0)/n:.1%}, 95% Уилсон [{lo:.1%}; {hi:.1%}]')
bad = lc.get('background', 0) + lc.get('not_by_text', 0) + lc.get('invalid_reference', 0)
lo, hi = wilson(bad, n)
out.append(f'  ссылка не на источник строки (background + not_by_text + invalid_reference): {bad/n:.1%} [{lo:.1%}; {hi:.1%}]')
out.append(f'поправки (evidence в corrections_{mat}.tsv): set={sum(c["action"]=="set" for c in cs)} drop={sum(c["action"]=="drop" for c in cs)}; '
           f'записей затронуто {len({(c["note"], c["ref_text"]) for c in cs})} из {n0} ({len({(c["note"], c["ref_text"]) for c in cs})/n0:.1%})')
# ключевое число до/после
def keynum(recs):
    recs = [r for r in recs if r.get('status', 'ok') == 'ok']
    ev = E.eval_records(mat, recs, S, pids, units)['bm25']
    return E.logmean(ev[0]), float(np.median(ev[0])), len(recs)
k0, k1 = keynum(r0), keynum(r1)
out.append(f'ключевое число (BM25, среднее log-rank / медиана / n): v0 {k0[0]:.3f} / {k0[1]:.1f} / {k0[2]}  ->  v1 {k1[0]:.3f} / {k1[1]:.1f} / {k1[2]}')
affected = len({(c["note"], c["ref_text"]) for c in cs if not c['evidence'].startswith('дубль')}); thr = max(3, 0.02 * n0)
ndup = len({(c["note"], c["ref_text"]) for c in cs if c['evidence'].startswith('дубль')})
out.append(f'из них дубли (не дефект достижимости, а завышение числа записей): {ndup}; недостижимые/неверные ссылки и диапазоны: {affected}')
d_rule = (affected >= 3) or (affected / n0 >= 0.02)
d_manual = (bad / n) >= 0.02
out.append(f'критерий шага 2 (>=3 записей или >=2%): недостижимые/неверные {affected} (дубли {ndup} не считаются) -> {d_rule}; ручная выборка {bad/n:.1%} -> {d_manual}')
out.append('ИСХОД: ' + ('дефект найден' if (d_rule or d_manual) else 'дефекта нет, критерий выполнен'))
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step2_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
with open(f'reports/ext/_defects_step2_{mat}.tsv', 'w', encoding='utf8') as f:
    f.write(f'{mat}\t2\tпоправимые дефекты ссылок/диапазонов/дубли (автопроверки)\t{affected}\tmean log-rank BM25 {k0[0]:.3f}->{k1[0]:.3f}\t{"найден" if d_rule else "нет"}\n')
    f.write(f'{mat}\t2\tссылка не на источник строки (ручная выборка, доля)\t{bad}/{n}\tоценка {bad/n:.1%} [{lo:.1%};{hi:.1%}]; не исправляется автоматически\t{"найден" if d_manual else "нет"}\n')
