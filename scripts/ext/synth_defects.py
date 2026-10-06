# -*- coding: utf-8 -*-
"""П6 (prereg_followup.md): синтетические дефекты и чувствительность шагов протокола на материале A.
Основа -- эталон A v1 (390 записей). Дефекты D1..D7 вносятся в долю r записей (5/10/20%), 20 повторов.
Детекторы без изменений относительно части 1: Д-2а (стиха нет в KJV/диапазон), Д-2б (заголовок в абзаце),
Д-2в (цитата в стихах), Д-3 (нет мостика, правила части 1 для A), Д-2г (ручная выборка 40, оракул), Д-6 (плацебо, r=10%)."""
import sys, os, re, math, collections, random, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
mat = 'A'
pids, nl, ntok = E.load_par(mat); units = E.load_units(mat); pix = {p: i for i, p in enumerate(pids)}
S = E.load_scores(mat, ['bm25'])['bm25']
uid, ut = L.load_kjv('data/ext/source/kjv_verses.tsv'); assert uid == units
upos = {u: i for i, u in enumerate(units)}
utok = [L.stems(t) for t in ut]; ustem = [set(t) for t in utok]
df = collections.Counter()
for t in utok: df.update(set(t))
N = len(units); RARE = 100; CONTENT = 0.10 * N
ptext = dict(zip(*L.load_paragraphs('data/ext/target/pl_lines.tsv')[:2])); ptok = {p: L.stems(t) for p, t in ptext.items()}
notes = {}
for l in open(f'{E.WORK}/pl_notes.tsv', encoding='utf8'):
    b, nid, ln, t = l.rstrip('\n').split('\t', 3); notes[f'{b}:{nid}'] = t
_, base = E.read_gold('data/ext/gold/gold_milton_bible_v1.tsv'); base = E.ok_records(base, 'synth')

def head_of(text):
    m = re.match(r'(.{3,90}?)(?:\.\s|\s{2,}|:\s|$)', text)
    return (m.group(1) if m else text[:60]).strip()
def quotes_of(note):
    qpos = [m.start() for m in re.finditer(r'["“”]', note)]
    return [q for q in (note[a + 1:b] for a, b in zip(qpos[0::2], qpos[1::2])) if len(L.stems(q)) >= 4]
def idxs(r):
    a, b = r['verse_first'], r['verse_last']
    pre, c, v1 = a.rsplit('.', 2); v2 = b.rsplit('.', 1)[1]
    return [upos[f'{pre}.{c}.{v}'] for v in range(int(v1), int(v2) + 1) if f'{pre}.{c}.{v}' in upos]
def exists(r):
    a, b = r['verse_first'], r['verse_last']
    if a not in upos or b not in upos: return False
    return int(b.rsplit('.', 1)[1]) >= int(a.rsplit('.', 1)[1]) and a.rsplit('.', 1)[0] == b.rsplit('.', 1)[0]
def d2a(r): return not exists(r)
def d2b(r):
    hs = [s for s in L.stems(head_of(notes[r['note']])) if len(s) >= 3]
    return bool(hs) and sum(s in set(ptok[r['place']]) for s in hs) / len(hs) < 0.5
def d2c(r):
    qs = quotes_of(notes[r['note']])
    if not qs or not exists(r): return False
    rng = set().union(*[ustem[i] for i in idxs(r)])
    best = max(sum(s in rng for s in L.stems(q)) / len(L.stems(q)) for q in qs)
    if best >= 0.7: return False
    # цитата из Мильтона (находится в абзаце) -- не дефект ссылки
    pt = set(ptok[r['place']])
    return max(sum(s in pt for s in L.stems(q)) / len(L.stems(q)) for q in qs) < 0.7
def d3(r):  # True = НЕТ мостика
    if not exists(r): return True
    ix = idxs(r); st = set().union(*[ustem[i] for i in ix]); pt = ptok[r['place']]
    if any(w in st and df[w] <= RARE for w in set(pt)): return False
    bgp = {(pt[i], pt[i + 1]) for i in range(len(pt) - 1)}
    for i in ix:
        t = utok[i]
        for k in range(len(t) - 1):
            if (t[k], t[k + 1]) in bgp and df[t[k]] <= CONTENT and df[t[k + 1]] <= CONTENT: return False
    return True
DET = {'2a': d2a, '2b': d2b, '2c': d2c, '3': d3}

# самопроверка: Д-3 воспроизводит findable_A_v1
_, fb = E.read_gold('data/ext/gold/findable_A_v1.tsv'); fm = {(r['note'], r['ref_text']): r['has_bridge'] == '1' for r in fb}
bad = sum((not d3(r)) != fm[(r['note'], r['ref_text'])] for r in base)
if bad: raise SystemExit(f'Д-3 в симуляции расходится с шагом 3 у {bad} записей')

rank_cache = {}
def lrank(r):
    key = (r['place'], r['verse_first'], r['verse_last'])
    if key not in rank_cache:
        rank_cache[key] = math.log(E.rank_tie(S[pix[r['place']]], idxs(r)))
    return rank_cache[key]
def mean_lr(recs):
    v = [lrank(r) for r in recs if exists(r)]
    return float(np.mean(v)) if len(v) >= 5 else float('nan')

books_of = collections.defaultdict(list)
for i, u in enumerate(units): books_of[u.split('.')[1]].append(i)
chap_verses = collections.defaultdict(list)
for u in units:
    pre, b, c, v = u.split('.'); chap_verses[(b, c)].append(int(v))
def vid(b, c, v): return f'b.{b}.{c}.{v}'
def mk(r, first, last=None, place=None):
    n = dict(r); n['verse_first'] = first; n['verse_last'] = last or first
    if place: n['place'] = place
    return n
def inject(kind, r, rng):
    _, b, c, v1 = r['verse_first'].split('.'); v1 = int(v1); v2 = int(r['verse_last'].rsplit('.', 1)[1]); c = c
    if kind == 'D1':
        k = rng.choice([1, 2, 3]); return [mk(r, vid(b, c, v1 + k), vid(b, c, v2 + k))]
    if kind == 'D2':
        d = rng.choice([-1, 1]); cc = str(int(c) + d); return [mk(r, vid(b, cc, v1), vid(b, cc, v2))]
    if kind == 'D3':
        i = pix[r['place']]; cand = [j for j in (i - 1, i + 1) if 0 <= j < len(pids) and pids[j].split('.')[1] == pids[i].split('.')[1]]
        return [mk(r, r['verse_first'], r['verse_last'], pids[rng.choice(cand)])]
    if kind == 'D4': return [dict(r), dict(r)]
    if kind == 'D5a':
        j = rng.choice(books_of[b]); return [mk(r, units[j])]
    if kind == 'D5b': return [mk(r, units[rng.randrange(N)])]
    if kind == 'D6':
        vs = [v for v in chap_verses[(b, c)] if abs(v - v1) <= 5 and v != v1] or [v1]
        w = rng.choice(vs); return [mk(r, vid(b, c, w))]
    if kind == 'D7':
        ln = rng.randint(5, 10); mx = max(chap_verses[(b, c)]); lo = max(1, v1 - ln // 2); hi = min(mx, lo + ln - 1); lo = max(1, hi - ln + 1)
        return [mk(r, vid(b, c, lo), vid(b, c, hi))]
KINDS = ['D1', 'D2', 'D3', 'D4', 'D5a', 'D5b', 'D6', 'D7']
# базовые флаги на чистой основе (ложные пометки)
base_flags = {k: np.array([f(r) for r in base]) for k, f in DET.items()}
step2_base = base_flags['2a'] | base_flags['2b'] | base_flags['2c']
out = ['П6: синтетические дефекты, материал A (основа: эталон v1, n=%d)' % len(base),
       'Доля ложных пометок на чистой основе: ' + ', '.join(f'Д-{k} {v.mean():.1%}' for k, v in base_flags.items()) + f'; шаг 2 в целом (2а|2б|2в) {step2_base.mean():.1%}']
b_naive = mean_lr(base)
keep_base = [r for r, a, c in zip(base, base_flags['2a'], base_flags['3']) if not a and not c]
b_prot = mean_lr(keep_base)
out.append(f'Ключевое число на чистой основе: наивно {b_naive:.3f} (n={sum(exists(r) for r in base)}), после Д-2а+Д-3 {b_prot:.3f} (n={len(keep_base)})')
REPS = 20; res = []; tsv = ['defect\tr\tdet_2a\tdet_2b\tdet_2c\tdet_step2\tdet_3\tany_step\tsample40_hit\tbias_naive\tbias_protocol\tshare_ranges']
for kind in KINDS:
    for r_ in (0.05, 0.10, 0.20):
        acc = collections.defaultdict(list)
        for rep in range(REPS):
            rng = random.Random(f'{20261005 + rep}-{kind}-{r_}')
            n_inj = round(r_ * len(base)); pick = set(rng.sample(range(len(base)), n_inj))
            recs, inj = [], []
            for i, r in enumerate(base):
                if i in pick:
                    new = inject(kind, r, rng); recs += new; inj += [True] * len(new)
                    if kind == 'D4': inj[-1] = True; inj[-2] = False   # дубль: «внесённой» считается копия
                else:
                    recs.append(r); inj.append(False)
            inj = np.array(inj)
            fl = {k: np.array([f(r) for r in recs]) for k, f in DET.items()}
            s2 = fl['2a'] | fl['2b'] | fl['2c']
            for k in ('2a', '2b', '2c'): acc['det_' + k].append(fl[k][inj].mean())
            acc['det_step2'].append(s2[inj].mean()); acc['det_3'].append(fl['3'][inj].mean())
            acc['any_step'].append((s2 | fl['3'])[inj].mean())
            acc['sample40_hit'].append(float(inj[rng.sample(range(len(recs)), 40)].any()))
            nv = mean_lr(recs); pv = mean_lr([r for r, a, c in zip(recs, fl['2a'], fl['3']) if not a and not c])
            acc['bias_naive'].append(nv - b_naive); acc['bias_protocol'].append(pv - b_prot)
            acc['share_ranges'].append(np.mean([len(idxs(r)) > 1 for r in recs if exists(r)]))
        row = {k: float(np.nanmean(v)) for k, v in acc.items()}; res.append((kind, r_, row))
        tsv.append(f'{kind}\t{r_}\t' + '\t'.join(f'{row[k]:.3f}' for k in ['det_2a', 'det_2b', 'det_2c', 'det_step2', 'det_3', 'any_step', 'sample40_hit', 'bias_naive', 'bias_protocol', 'share_ranges']))
out.append('\nПолнота детекторов (доля внесённых записей, помеченных) и смещение ключевого числа; средние по 20 повторам.')
out.append('Ложных пометок на чистой основе: Д-3 %.1f%%, шаг 2 %.1f%% -- полнота Д-3 надо читать относительно этой доли.' % (100 * base_flags['3'].mean(), 100 * step2_base.mean()))
out.append(f'{"дефект":6s} {"r":>5s} {"2а":>6s} {"2б":>6s} {"2в":>6s} {"шаг2":>6s} {"Д-3":>6s} {"любой":>6s} {"выб.40":>7s} {"смещ.наивн.":>12s} {"смещ.протокол":>14s} {"доля диап.":>11s}')
for kind, r_, row in res:
    out.append(f'{kind:6s} {r_:5.2f} {row["det_2a"]:6.2f} {row["det_2b"]:6.2f} {row["det_2c"]:6.2f} {row["det_step2"]:6.2f} {row["det_3"]:6.2f} {row["any_step"]:6.2f} {row["sample40_hit"]:7.2f} {row["bias_naive"]:+12.3f} {row["bias_protocol"]:+14.3f} {row["share_ranges"]:11.3f}')
out.append('Прирост Д-3 над ложными пометками (полнота − %.2f): ' % base_flags['3'].mean() + ', '.join(f'{k} {np.mean([row["det_3"] for kk, rr, row in res if kk == k]) - base_flags["3"].mean():+.2f}' for k in KINDS))
out.append(f'(на чистой основе доля диапазонов {np.mean([len(idxs(r)) > 1 for r in base]):.3f})')
# критерии prereg
out.append('\nКритерий (2): при r=10%% остаточное смещение меньше наивного не менее чем в 2 раза?')
for kind, r_, row in res:
    if abs(r_ - 0.10) < 1e-9:
        ok = abs(row['bias_protocol']) * 2 <= abs(row['bias_naive']) if abs(row['bias_naive']) > 0.02 else None
        out.append(f'  {kind}: наивное {row["bias_naive"]:+.3f}, протокол {row["bias_protocol"]:+.3f} -> ' + ('смещение пренебрежимо мало (|наивное|<0,02)' if ok is None else ('да' if ok else 'нет')))
out.append('\nКритерий (3): шаги, нечувствительные во всей симуляции (полнота <10% для всех типов):')
for name, key in [('Д-2а', 'det_2a'), ('Д-2б', 'det_2b'), ('Д-2в', 'det_2c'), ('Д-3', 'det_3')]:
    mx = max(row[key] for _, _, row in res)
    out.append(f'  {name}: максимальная полнота {mx:.2f} -> ' + ('нечувствителен' if mx < 0.10 else 'чувствителен'))

# ---- Д-6: плацебо (r=10%, 5 повторов, 300 перемешиваний внутри книги поэмы), BM25
def placebo_gap(recs, rng, nperm=300):
    recs = [r for r in recs if exists(r)]
    real = np.mean([lrank(r) for r in recs]); groups = collections.defaultdict(list)
    for i, r in enumerate(recs): groups[r['place'].split('.')[1]].append(i)
    means = []
    for _ in range(nperm):
        tot = 0.0
        for g, ii in groups.items():
            perm = ii[:]; rng.shuffle(perm)
            for a, c in zip(ii, perm):
                key = (recs[c]['place'], recs[a]['verse_first'], recs[a]['verse_last'])
                if key not in rank_cache: rank_cache[key] = math.log(E.rank_tie(S[pix[recs[c]['place']]], idxs(recs[a])))
                tot += rank_cache[key]
        means.append(tot / len(recs))
    return real - float(np.mean(means))
out.append('\nД-6: разность «реальное − плацебо» (BM25, среднее log-rank; r=10%, 5 повторов, 300 перемешиваний)')
g0 = placebo_gap(base, random.Random(1))
out.append(f'  чистая основа: {g0:+.3f}')
for kind in KINDS:
    gs = []
    for rep in range(5):
        rng = random.Random(f'plac-{20261005 + rep}-{kind}')
        pick = set(rng.sample(range(len(base)), round(0.10 * len(base)))); recs = []
        for i, r in enumerate(base): recs += inject(kind, r, rng) if i in pick else [r]
        gs.append(placebo_gap(recs, rng))
    out.append(f'  {kind}: {np.mean(gs):+.3f} (сдвиг к нулю {np.mean(gs) - g0:+.3f}; размах по повторам {min(gs):+.3f}..{max(gs):+.3f})')
txt = '\n'.join(out); print(txt)
open('reports/ext/synth_defects.txt', 'w', encoding='utf8').write(txt + '\n')
open('reports/ext/synth_defects.tsv', 'w', encoding='utf8').write('\n'.join(tsv) + '\n')
