# -*- coding: utf-8 -*-
"""Анализ LLM-судьи (материал D): AUC золото против обманок A/B/C, проверка цитат, стабильность, free recall.
Использование: llm_judge_analyze.py <каталог с items.jsonl, results.jsonl, recall_results.jsonl, free_recall.jsonl> <отчёт>"""
import sys, os, json, re, collections, numpy as np
D, REP = sys.argv[1], sys.argv[2]
MAT = sys.argv[3] if len(sys.argv) > 3 else 'D'
def norm(s): return re.sub(r'[^а-яa-z0-9]+', '', s.lower().replace('ё', 'е'))
def parse(raw):
    m = re.search(r'\{.*\}', raw, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        sc = re.search(r'"score"\s*:\s*(\d)', raw); return {'score': int(sc.group(1)), 'quote_a': '', 'quote_b': ''} if sc else None
items = {}
for l in open(f'{D}/items.jsonl', encoding='utf8'):
    it = json.loads(l); items[it['id']] = it
res = {}
for l in open(f'{D}/results.jsonl', encoding='utf8'):
    r = json.loads(l); res[r['id']] = r['raw']
out = [f'LLM-судья, материал {MAT}: ответов {len(res)} из {len(items)}']
stat = collections.Counter(); score = {}; valid = {}
for i, raw in res.items():
    it = items[i]; p = parse(raw)
    if p is None or raw.startswith('ERROR'): stat['нет разбора/ошибка'] += 1; score[i] = 0; valid[i] = False; continue
    try: s = int(p.get('score', 0))
    except Exception: s = 0
    s = max(0, min(3, s)); qa, qb = str(p.get('quote_a', '')), str(p.get('quote_b', ''))
    ok = True
    if s >= 1:
        if not qa or not qb or norm(qa) not in norm(it['stanza']) or norm(qb) not in norm(it['unit_window']): ok = False
    if not ok: stat['цитата не найдена в тексте (балл обнулён)'] += 1
    score[i] = s if ok else 0; valid[i] = ok
    stat['разобрано'] += 1
out.append('разбор: ' + ', '.join(f'{k} {v}' for k, v in stat.items()))
main = {i: it for i, it in items.items() if it['rep'] == 0}
def per_rec(sc):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, it in main.items():
        if i in sc: d[it['rec']][it['kind']].append(sc[i])
    return d
def auc_stats(sc, kind, B=5000, seed=20261005):
    d = per_rec(sc); recs = [r for r in d if d[r]['gold'] and d[r][kind]]
    v = np.array([np.mean([(d[r]['gold'][0] > x) + 0.5 * (d[r]['gold'][0] == x) for x in d[r][kind]]) for r in recs])
    rng = np.random.RandomState(seed); bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(B)]
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)
out.append('\n== Основной прогон (T=0), балл с обнулением недостоверных цитат ==')
out.append(f'{"сравнение":22s} {"AUC":>6s} {"95% (по записям)":>20s} {"n":>4s}')
aucs = {}
for kind, name in (('A', 'золото против A (топ fuse)'), ('B', 'золото против B (тот же автор/глава)'), ('C', 'золото против C (случайные)')):
    a, lo, hi, n = auc_stats(score, kind); aucs[kind] = (a, lo, hi); out.append(f'{name:22s} {a:6.3f} [{lo:.3f},{hi:.3f}] {n:4d}')
# без обнуления (сырой балл)
raw_sc = {}
for i, raw in res.items():
    p = parse(raw); 
    try: raw_sc[i] = max(0, min(3, int(p.get('score', 0)))) if p else 0
    except Exception: raw_sc[i] = 0
out.append('сырой балл (без проверки цитат): ' + ', '.join(f'{k} {auc_stats(raw_sc, k)[0]:.3f}' for k in 'ABC'))
out.append('\nдоля баллов >=2 и средний балл по видам пар:')
for kind in ('gold', 'A', 'B', 'C'):
    v = [score[i] for i, it in main.items() if it['kind'] == kind and i in score]
    out.append(f'  {kind:5s} n={len(v):3d}  доля >=2: {np.mean([x >= 2 for x in v]):.3f}  средний балл {np.mean(v):.2f}  достоверных цитат у баллов >=1: {np.mean([valid[i] for i, it in main.items() if it["kind"] == kind and i in valid and raw_sc.get(i, 0) >= 1] or [float("nan")]):.2f}')
# стабильность
HAS_REC = os.path.exists(f'{D}/recall_results.jsonl')
rep = collections.defaultdict(list)
for i, it in items.items():
    if it['rep'] > 0 and i in score: rep[it['id'].rsplit('|rep', 1)[0]].append(score[i] >= 2)
flips = [len(set(v)) > 1 for v in rep.values() if len(v) == 3] or [float('nan')]
vs0 = [any(x != (score[b] >= 2) for x in v) for b, v in rep.items() if b in score and len(v) == 3] or [float('nan')]
out.append(f'\n== Стабильность (3 повтора при T=0,7, {len(flips)} пар: золото и A на 40 записях) ==')
out.append(f'доля пар, где вердикт (балл >=2) не одинаков во всех трёх повторах: {np.mean(flips):.3f}; доля пар, где хотя бы один повтор расходится с прогоном T=0: {np.mean(vs0):.3f}')
if HAS_REC:
    # free recall
    rec = []
    for l in open(f'{D}/recall_results.jsonl', encoding='utf8'):
        r = json.loads(l); p = parse(r['raw']); rec.append((r['id'], p.get('author', '') if p else ''))
    gold = {json.loads(l)['id']: json.loads(l)['gold_author'] for l in open(f'{D}/free_recall.jsonl', encoding='utf8')}
    def sur(a): return norm(a.split(':')[0].replace('(Онегин)', '').split()[-1]) if a.strip() else ''
    hit = [sur(gold[i]) != '' and sur(gold[i])[:5] in norm(a) for i, a in rec]
    from collections import Counter
    _c = Counter(sur(g)[:5] for g in gold.values()); _top = _c.most_common(1)[0][0]; base = _c[_top] / len(gold)
    out.append(f'\n== Free recall (только строфа, без источника): автор золотого источника назван верно у {np.sum(hit)}/{len(hit)} = {np.mean(hit):.3f}; доля ответов самого частого автора ({_top}): {np.mean([(_top in norm(a)) for _, a in rec]):.3f}; доля записей, где золото — Пушкин/Онегин: {base:.3f} (базис «всегда самый частый автор»)')
    nonp = [(h) for (i, a), h in zip(rec, hit) if sur(gold[i])[:5] != _top]
    out.append(f'  по записям, где золото — не самый частый автор: попаданий {np.sum(nonp)}/{len(nonp)}')

a, lo, hi = aucs['A']
out.append(f'\nПравило судьи: признак, если AUC(золото против A) >= 0,65 и нижняя граница > 0,5: {"признак" if a >= 0.65 and lo > 0.5 else "только фильтр"} (AUC {a:.3f}, нижняя граница {lo:.3f})')
txt = '\n'.join(out); print(txt); open(REP, 'w', encoding='utf8').write(txt + '\n')
