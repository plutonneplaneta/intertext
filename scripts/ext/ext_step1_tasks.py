#!/usr/bin/env python3
"""ШАГ 1. Развести отбор и атрибуцию. Наивный эталон v0, все записи, без чисток.
Атрибуция: ранг правильного источника среди всех единиц источника.
Отбор: положительные абзацы (есть запись эталона) против неразмеченных; признак -- максимум балла по единицам.
Пишет reports/ext/step1_<A|B>.txt.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = sys.argv[1]
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat)
S = E.load_scores(mat); _, recs = E.read_gold(E.GOLD[mat])
recs = E.ok_records(recs, 'step1')
ev = E.eval_records(mat, recs, S, pids, units)
out = []
P = out.append
P(f'ШАГ 1 / материал {mat}: записей v0 (status=ok) {len(recs)}, абзацев с записями {len({r["place"] for r in recs})} из {len(pids)}')
P('\n== Атрибуция (задача б): ранг правильного источника среди %d единиц; наивный эталон v0 ==' % len(units))
P(f'{"скорер":8s} {"ср.log-rank":>11s} {"медиана":>8s} {"ненул.":>7s} {"R@10":>6s} {"R@100":>6s}  (случайно: log-rank {math.lgamma(len(units)+1)/len(units):.2f})')
for k in E.SCORERS:
    rk, nz = ev[k]
    P(f'{k:8s} {E.logmean(rk):11.3f} {np.median(rk):8.1f} {nz.mean():7.3f} {(rk<=10).mean():6.3f} {(rk<=100).mean():6.3f}')
# --- отбор
pos_set = {r['place'] for r in recs}
y = np.array([p in pos_set for p in pids])
def auc_pairs(score, y, mask_fn=None):
    pi = np.where(y)[0]; ni = np.where(~y)[0]; num = den = 0.0
    for i in pi:
        for j in ni:
            if mask_fn and not mask_fn(i, j): continue
            den += 1; num += (score[i] > score[j]) + 0.5 * (score[i] == score[j])
    return num / den if den else float('nan'), den
def boot_auc(score, y, mask_fn, B=1000, seed=20261005):
    rng = np.random.RandomState(seed); n = len(y); res = []
    pi = np.where(y)[0]; ni = np.where(~y)[0]
    for _ in range(B):
        a = rng.choice(pi, len(pi)); b = rng.choice(ni, len(ni))
        sa, sb = score[a][:, None], score[b][None, :]
        m = mask_fn(a[:, None], b[None, :]) if mask_fn else np.ones((len(a), len(b)), bool)
        w = m.sum()
        if w == 0: continue
        res.append(((((sa > sb) + 0.5 * (sa == sb)) * m).sum()) / w)
    return np.percentile(res, [2.5, 97.5])
lenmask = lambda i, j: np.abs(ntok[i] - ntok[j]) <= 0.2 * np.maximum(ntok[i], ntok[j])
feats = {'только длина (слова)': ntok.astype(float)}
for k in ['bm25', 'idfcos', 'char4', 'run']:
    feats['макс. ' + k] = S[k].max(axis=1)
P(f'\n== Отбор (задача а): {y.sum()} положительных, {(~y).sum()} неразмеченных абзацев ==')
P(f'{"признак":24s} {"AUC":>6s} {"95% (по абзацам)":>18s} {"AUC ±20% длины":>15s} {"95%":>16s} {"пар":>8s}')
res = {}
for name, f in feats.items():
    if np.ptp(f) == 0: raise SystemExit(f'признак {name} постоянен: тихий отказ')
    a, _ = auc_pairs(f, y)
    ci = boot_auc(f, y, None)
    am, den = auc_pairs(f, y, lenmask)
    cim = boot_auc(f, y, lenmask)
    res[name] = (a, am)
    P(f'{name:24s} {a:6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {am:15.3f} [{cim[0]:.3f},{cim[1]:.3f}] {int(den):8d}')
best = max(v[0] for n, v in res.items() if n != 'только длина (слова)')
bestm = max(v[1] for n, v in res.items() if n != 'только длина (слова)')
len_a, len_m = res['только длина (слова)']
drop = max(res[n][0] - res[n][1] for n in res if n != 'только длина (слова)')
d1 = len_a >= best - 0.03
d2 = drop >= 0.05
P(f'\nКритерий шага 1: длина>=лучший-0,03: {d1} (длина {len_a:.3f}, лучший скорер {best:.3f}); падение AUC при выравнивании длины >=0,05: {d2} (макс. падение {drop:.3f})')
P('ИСХОД: ' + ('дефект найден (длина объясняет отбор / выигрыш скореров от длины)' if (d1 or d2) else 'дефекта нет, критерий выполнен'))
txt = '\n'.join(out)
open(f'reports/ext/step1_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
print(txt)
with open(f'reports/ext/_defects_step1_{mat}.tsv', 'w', encoding='utf8') as f:
    f.write(f'{mat}\t1\tдлина предсказывает эталон / выигрыш от длины\t{int(d1)+int(d2)}\tAUC длины {len_a:.3f}; лучший {best:.3f}; падение {drop:.3f}\t{"найден" if (d1 or d2) else "нет"}\n')
