#!/usr/bin/env python3
"""ШАГ 5. Статистика по объектам (абзацам) + строки для журнала проверок.
Множество протокола: эталон v1 ∩ запись с текстовым мостиком ∩ одиночная ссылка (A: 1 стих; B: ссылка со строкой).
Предзаданное семейство «варианты против BM25» (4 сравнения) на множестве протокола; разведочно -- то же на наивном v0.
Перестановочный тест: перемена знаков разностей на уровне абзаца (20000), двусторонний."""
import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = sys.argv[1]
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); S = E.load_scores(mat)

def load_set(ver, protocol):
    _, recs = E.read_gold(E.GOLD[mat].replace('_v0', '_' + ver))
    recs = E.ok_records(recs, 'step5')
    if protocol:
        _, fb = E.read_gold(f'data/ext/gold/findable_{mat}_{ver}.tsv')
        fmap = {(r['note'], r['ref_text']): r['has_bridge'] == '1' for r in fb}
        recs = [r for r in recs if fmap[(r['note'], r['ref_text'])]]
        if E.is_bible(mat):
            recs = [r for r in recs if len(E.unit_idxs('A', r, units)) == 1]
        elif not E.is_d(mat):
            recs = [r for r in recs if re.search(r'\d+\s*[.:]\s*\d+', r['ref_text'])]
    return recs

def perm_p(d, clusters, n_perm=20000, seed=20261005):
    d = np.asarray(d, float); cl = sorted(set(clusters))
    if len(cl) < 5: raise SystemExit(f'перестановки по {len(cl)} абзацам: недостаточно (<5)')
    D = np.array([d[[i for i, c in enumerate(clusters) if c == k]].sum() for k in cl]); N = len(d)
    obs = D.sum() / N
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(cl)))
    null = signs @ D / N
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)

out = [f'ШАГ 5 / материал {mat}']
rows = []
for tag, ver, protocol, prespec in [('P', 'v1', True, 'да'), ('N', 'v0', False, 'нет')]:
    recs = load_set(ver, protocol)
    ev = E.eval_records(mat, recs, S, pids, units); cl = [r['place'] for r in recs]
    label = 'множество протокола (v1, с мостиком, одиночные)' if protocol else 'наивное множество (v0, все записи)'
    out.append(f'\n== {label}: n={len(recs)} записей, {len(set(cl))} абзацев ==')
    # сравнение интервалов: по записям против по абзацам (BM25)
    lr = np.log(ev['bm25'][0])
    rng = np.random.RandomState(20261005); nb = [lr[rng.randint(0, len(lr), len(lr))].mean() for _ in range(2000)]
    n_lo, n_hi = np.percentile(nb, [2.5, 97.5]); m, c_lo, c_hi = L.cluster_boot_mean(lr, cl, B=2000)
    wr = (c_hi - c_lo) / (n_hi - n_lo)
    out.append(f'BM25 среднее log-rank {m:.3f}; интервал по записям [{n_lo:.3f},{n_hi:.3f}] (ширина {n_hi-n_lo:.3f}); по абзацам [{c_lo:.3f},{c_hi:.3f}] (ширина {c_hi-c_lo:.3f}); отношение ширин {wr:.2f}')
    if protocol:
        d5 = wr >= 1.5; rowd = (len(recs), wr, d5)
    for k in E.SCORERS[1:]:
        d = np.log(ev[k][0]) - lr
        obs, p = perm_p(d, cl)
        rr = float(np.exp(np.mean(np.log(ev[k][0])) - np.mean(lr)))
        out.append(f'  {k:7s} Δ среднее log-rank = {obs:+.3f}  (отношение рангов {rr:.2f})  p(перестановки по абзацам) = {p:.4f}')
        rows.append((f'{mat}-{tag}-{k}', f'{tag}{mat}', 'актуально', prespec, len(recs), obs, rr, p, f'{k} против BM25, материал {mat}, {label}'))
n_p, wr, d5 = rowd
out.append(f'\nКритерий шага 5: отношение ширин интервалов (по абзацам / по записям) >=1,5 на множестве протокола: {d5} ({wr:.2f})')
out.append('Базис сравнений: BM25 (закреплён в предрегистрации; в коде assert). Независимая проверка на другом тексте -- неприменима: обучаемых параметров нет, настройки не подбирались; вариантные сравнения предзаданы.')
out.append('ИСХОД: ' + ('дефект найден (наивный интервал слишком узок)' if d5 else 'дефекта нет, критерий выполнен'))
assert E.SCORERS[0] == 'bm25'
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step5_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
with open(f'reports/ext/_journal_step5_{mat}.tsv', 'w', encoding='utf8') as f:
    for r in rows: f.write('\t'.join(str(x) if not isinstance(x, float) else f'{x:.6f}' for x in r) + '\n')
with open(f'reports/ext/_defects_step5_{mat}.tsv', 'w', encoding='utf8') as f:
    f.write(f'{mat}\t5\tнаивный интервал по записям уже интервала по абзацам\t{wr:.2f}\tотношение ширин {wr:.2f} (порог 1,5); n={n_p}\t{"найден" if d5 else "нет"}\n')
