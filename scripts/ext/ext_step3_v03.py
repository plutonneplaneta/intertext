#!/usr/bin/env python3
"""ШАГ 3 (редакция 0.3: пороги заданы числом, prereg_followup.md П1б). Потолок без порогов и обоснованный знаменатель (findability).
Вход: эталон v1 (после шага 2) -- либо v0 (для проверки порядка шагов), параметр gold=v0|v1.
Пишет reports/ext/step3_<mat>.txt и таблицу data/ext/gold/findable_<mat>_<v>.tsv (запись, мостик).
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = sys.argv[1]; ver = sys.argv[2] if len(sys.argv) > 2 else 'v1'
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat)
S = E.load_scores(mat)
path = E.GOLD[mat].replace('_v0', '_' + ver)
_, recs = E.read_gold(path)
recs = E.ok_records(recs, 'step3')
ev = E.eval_records(mat, recs, S, pids, units)
# --- индекс источника для df и последовательностей
uid, ut = E.source_units(mat); N_ = len(uid)
RARE = max(3, math.ceil(0.01 * N_)); CONTENT = 0.10 * N_
tok = [L.stems(t) for t in ut]; assert uid == units
import collections
df = collections.Counter()
for t in tok: df.update(set(t))
ptexts = dict(zip(*E.load_target(mat)[:2]))
ptok = {p: L.stems(t) for p, t in ptexts.items()}
bridge = []
for r in recs:
    ix = E.unit_idxs(mat, r, units)
    st = set().union(*[set(tok[i]) for i in ix])
    pt = ptok[r['place']]
    rare = sorted(w for w in set(pt) & st if df[w] <= RARE and len(w) >= 4)
    bg_p = {(pt[i], pt[i + 1]) for i in range(len(pt) - 1)}
    bg = set()
    for i in ix:
        t = tok[i]
        for k in range(len(t) - 1):
            if (t[k], t[k + 1]) in bg_p and min(len(t[k]), len(t[k + 1])) >= 4 and df[t[k]] <= CONTENT and df[t[k + 1]] <= CONTENT: bg.add((t[k], t[k + 1]))
    bridge.append((len(rare), len(bg), rare[:3], sorted(bg)[:2]))
n = len(recs)
safe = lambda r: f'{E.logmean(r):.3f}' if len(r) >= 5 else f'не считается (n={len(r)}<5)'
has_rare = np.array([b[0] > 0 for b in bridge]); has_bg = np.array([b[1] > 0 for b in bridge]); has = has_rare | has_bg
out = [f'ШАГ 3 / материал {mat}, эталон {ver}: записей {n}']
K = 1000 if len(units) > 20000 else (100 if len(units) > 1000 else 10)
out.append(f'\n== Потолок без порогов и обучения (источник: {len(units)} единиц) ==')
out.append(f'{"скорер":8s} {"ненулевой балл":>15s} {"ранг<="+str(K):>10s}   (ненулевой балл: общая лемма хоть с чем-то -> потолок вырожден для мешков слов)')
ceil = {}
for k in E.SCORERS:
    rk, nz = ev[k]; ceil[k] = nz.mean()
    out.append(f'{k:8s} {nz.mean():15.3f} {(rk<=K).mean():10.3f}')
out.append(f'\n== Знаменатель: есть ли текстовый мостик (редкая общая основа df<={RARE} или пара содержательных основ подряд) ==')
out.append(f'редкая общая основа: {has_rare.sum()}/{n} ({has_rare.mean():.1%});  пара подряд: {has_bg.sum()}/{n} ({has_bg.mean():.1%});  любой мостик: {has.sum()}/{n} ({has.mean():.1%})')
out.append(f'без мостика: {(~has).sum()} записей ({(~has).mean():.1%})')
out.append(f'\n== Две цифры полноты (BM25, ранг<={K}) ==')
rk = ev['bm25'][0]
out.append(f'относительно всех записей: {(rk<=K).mean():.3f} ({(rk<=K).sum()}/{n});  относительно записей с мостиком: {(rk[has]<=K).mean():.3f} ({(rk[has]<=K).sum()}/{has.sum()})')
out.append(f'среднее log-rank BM25: все {safe(rk)}; с мостиком {safe(rk[has])}; без мостика {safe(rk[~has])}')
for name, mask in (('редкая основа', has_rare), ('пара подряд', has_bg)):
    out.append(f'  по типу мостика «{name}»: n={mask.sum()}, среднее log-rank BM25 {safe(rk[mask])}, ранг<={K}: {(rk[mask]<=K).mean() if mask.sum() else float("nan"):.3f}')
d_nb = (~has).mean() >= 0.05; d_ce = min(ceil.values()) < 0.95
out.append(f'\nКритерий шага 3: без мостика >=5%: {d_nb}; потолок какого-либо скорера <95%: {d_ce} (мин. {min(ceil.items(), key=lambda x:x[1])[0]} {min(ceil.values()):.3f})')
out.append('ИСХОД: ' + ('дефект найден' if (d_nb or d_ce) else 'дефекта нет, критерий выполнен'))
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step3v03_{mat}_{ver}.txt', 'w', encoding='utf8').write(txt + '\n')
outs = [f'data/ext/gold/findable_{mat}_{ver}_v03.tsv'] + ([f'data/ext/gold/findable_{mat}_{ver}.tsv'] if E.is_d(mat) else [])   # D: только определения 0.3
for outp in outs:
    with open(outp, 'w', encoding='utf8') as f:
        f.write('note\tref_text\thas_bridge\trare\tbigram\n')
        for r, b, h in zip(recs, bridge, has):
            f.write(f'{r["note"]}\t{r["ref_text"]}\t{int(h)}\t{",".join(b[2])}\t{" ".join("_".join(x) for x in b[3])}\n')
