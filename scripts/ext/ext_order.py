#!/usr/bin/env python3
"""M6. Зависимость результата от порядка шагов-фильтров 2 (поправки эталона), 3 (мостик), 4 (одиночные).
Каждый фильтр действует на текущее СОСТОЯНИЕ записи (v0 до применения поправки, v1 после): мостик и
«одиночность» пересчитаны для обоих состояний. Шесть перестановок; измеряются итоговое множество,
среднее log-rank BM25 и число записей, снятых фильтром на данной позиции (= «дефекты, приписанные шагу»)."""
import sys, os, re, itertools, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
for mat in (sys.argv[1:] or ['A', 'B']):
    pids, _, _ = E.load_par(mat); units = E.load_units(mat); S = E.load_scores(mat, ['bm25']); pix = {p: i for i, p in enumerate(pids)}
    g0 = E.GOLD[mat]; g1 = g0.replace('_v0', '_v1')
    _, r0 = E.read_gold(g0); r0 = E.ok_records(r0, 'order'); _, r1 = E.read_gold(g1)
    key = lambda r: (r['note'], r['ref_text'])
    d0 = {key(r): r for r in r0}; d1 = {key(r): r for r in r1}
    br = {}
    for ver in ('v0', 'v1'):
        _, fb = E.read_gold(f'data/ext/gold/findable_{mat}_{ver}.tsv'); br[ver] = {(r['note'], r['ref_text']): r['has_bridge'] == '1' for r in fb}
    def single(r):
        if E.is_bible(mat): return len(E.unit_idxs('A', r, units)) == 1
        return True if E.is_d(mat) else bool(re.search(r'\d+\s*[.:]\s*\d+', r['ref_text']))
    def lr(r): return float(np.log(E.rank_tie(S['bm25'][pix[r['place']]], E.unit_idxs(mat, r, units))))
    # состояние: (ключ, 'v0'|'v1')
    cache = {}
    def loglev(k, st):
        if (k, st) not in cache: cache[(k, st)] = lr((d0 if st == 'v0' else d1)[k])
        return cache[(k, st)]
    out = [f'M6 / материал {mat}: порядок фильтров 2 (поправки), 3 (мостик), 4 (одиночные); universe = {len(d0)} записей v0 (status=ok)']
    finals = {}
    marg = {2: [], 3: [], 4: []}
    for order in itertools.permutations((2, 3, 4)):
        cur = {k: 'v0' for k in d0}
        line = []
        for step in order:
            before = list(cur)
            m_before = np.mean([loglev(k, cur[k]) for k in cur])
            n_before = len(cur)
            if step == 2:
                cur = {k: 'v1' for k in cur if k in d1}
            elif step == 3:
                cur = {k: s for k, s in cur.items() if br[s].get(k, False)}
            else:
                cur = {k: s for k, s in cur.items() if single((d0 if s == 'v0' else d1)[k])}
            m_after = np.mean([loglev(k, cur[k]) for k in cur])
            line.append(f'{step}: -{n_before-len(cur)} записей, среднее log-rank {m_before:.3f}->{m_after:.3f}')
            marg[step].append((n_before - len(cur), m_after - m_before))
        finals[order] = (set(cur), np.mean([loglev(k, cur[k]) for k in cur]), len(cur))
        out.append(f'порядок {"→".join(map(str, order))}: ' + ' | '.join(line) + f'  => n={len(cur)}, среднее log-rank {finals[order][1]:.4f}')
    ref = finals[(2, 3, 4)]
    same_sets = all(v[0] == ref[0] for v in finals.values())
    out.append(f'\nитоговое множество одно и то же во всех 6 порядках: {same_sets}; число различных итоговых множеств: {len({frozenset(v[0]) for v in finals.values()})}')
    out.append('размах итогового среднего log-rank по порядкам: %.4f; размах n: %d' % (max(v[1] for v in finals.values()) - min(v[1] for v in finals.values()), max(v[2] for v in finals.values()) - min(v[2] for v in finals.values())))
    out.append('приписывание дефектов шагам зависит от позиции (снято записей на шаге: min..max по 6 порядкам; Δ среднего log-rank при применении):')
    for s in (2, 3, 4):
        rm = [a for a, _ in marg[s]]; dm = [b for _, b in marg[s]]
        out.append(f'  шаг {s}: снимает {min(rm)}..{max(rm)} записей; Δ {min(dm):+.3f}..{max(dm):+.3f}')
    txt = '\n'.join(out); print(txt); print()
    open(f'reports/ext/order_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
