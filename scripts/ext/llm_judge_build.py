# -*- coding: utf-8 -*-
"""LLM-судья (шаг 6, пункты про LLM) для материала D: строит набор пар (prereg_material_D.md + поправка D-2).
Записи: эталон v1 (все, n~49; множество протокола слишком мало, поправка D-2). Пары: золото, 3 обманки A (топ fuse, не золото, не сама строфа),
2 обманки B (тот же автор / та же глава), 2 случайные C. Окно источника: 220 слов с максимальным пересечением лемм со строфой (одинаково для всех).
Выход: items.jsonl, free_recall.jsonl, key.tsv."""
import sys, os, json, random, re, collections
sys.path.insert(0, os.path.dirname(__file__)); os.environ['EXT_LANG'] = 'ru'
import numpy as np, ext_eval as E, ext_lib as L
OUT = sys.argv[1]
MAT = sys.argv[2] if len(sys.argv) > 2 else 'D'
os.makedirs(OUT, exist_ok=True)
pids, ptexts, _ = E.load_target(MAT); ptext = dict(zip(pids, ptexts)); pix = {p: i for i, p in enumerate(pids)}
uid, ut = E.source_units(MAT); utext = dict(zip(uid, ut)); upos = {u: i for i, u in enumerate(uid)}
rows = [l.rstrip('\n').split('\t', 3) for l in open({'D': 'data/ext/source/ru_units.tsv', 'E': 'data/ext/source/ru_units_e.tsv'}[MAT], encoding='utf8').read().split('\n')[1:] if l]
meta = {r[0]: (r[1], r[2]) for r in rows}   # kind, title
S = E.load_scores(MAT, ['fuse'])['fuse']
_, recs = E.read_gold(E.GOLD[MAT].replace('_v0', '_v1'))
rng = random.Random(20261005)
SYS_E = ('Ты — филолог. Оцени, перекликается ли текст А (отрывок романа Ильфа и Петрова «Двенадцать стульев») с текстом Б (отрывок из произведения другого автора). '
         'Шкала: 0 — связи нет (общие слова и общая тема не считаются); 1 — слабое тематическое сходство; 2 — заметная перекличка: общий образ, оборот или мотив, который можно показать цитатами; '
         '3 — прямая цитата, пародия, парафраз или явное заимствование. Если оценка 1 и выше, приведи ТОЧНЫЕ цитаты из обоих текстов, подтверждающие оценку. '
         'Ответ — строго JSON: {"score": 0-3, "quote_a": "...", "quote_b": "..."}')
RECALL_E = 'Ниже отрывок из романа Ильфа и Петрова «Двенадцать стульев». Назови автора и произведение, к которым, по-твоему, отсылает этот отрывок (пародия, цитата, перекличка). Ответ — строго JSON: {"author": "...", "work": "..."}\n\n'
def author(u):
    k, t = meta[u]
    return 'Пушкин (Онегин)' if k == 'stanza' else t.split(':')[0]
def window(text, stanza, size=220):
    w = text.split(); sl = set(L.stems(stanza))
    if len(w) <= size: return text
    best = (-1, 0)
    for i in range(0, len(w) - size + 1, 30):
        sc = len(sl & set(L.stems(' '.join(w[i:i + size]))))
        if sc > best[0]: best = (sc, i)
    return ' '.join(w[best[1]:best[1] + size])
by_auth = collections.defaultdict(list)
for u in uid: by_auth[author(u)].append(u)
items = []; key = []; recall = []
for r in recs:
    place, gold = r['place'], r['source_unit']; row = S[pix[place]]
    order = [uid[i] for i in np.argsort(-row)]
    A = [u for u in order if u != gold and u != place][:3]
    pool = [u for u in by_auth[author(gold)] if u not in (gold, place) and u not in A]
    if meta[gold][0] == 'stanza':   # та же глава
        ch = gold.split('.')[1]; pool = [u for u in pool if u.split('.')[1] == ch] or pool
    B = rng.sample(pool, min(2, len(pool)))
    same_kind = [u for u in uid if meta[u][0] == meta[gold][0] and u not in [gold, place] + A + B]
    C = rng.sample(same_kind, 2)
    for kind, us in (('gold', [gold]), ('A', A), ('B', B), ('C', C)):
        for u in us:
            items.append({'id': f'{r["note"]}|{r["ref_text"]}|{kind}|{u}', 'rec': f'{r["note"]}|{r["ref_text"]}', 'kind': kind, 'unit': u,
                          'stanza': ptext[place], 'unit_window': window(utext[u], ptext[place]), 'rep': 0, 'temp': 0.0, **({'system': SYS_E} if MAT == 'E' else {})})
    recall.append({'id': f'{r["note"]}|{r["ref_text"]}', 'stanza': ptext[place], 'gold_author': author(gold), **({'prompt': RECALL_E} if MAT == 'E' else {})})
# повторы T=0.7 на 40 записях (золото + A), 3 раза
rep_recs = sorted({it['rec'] for it in items}); rng.shuffle(rep_recs); rep_recs = set(rep_recs[:40])
extra = []
for it in items:
    if it['rec'] in rep_recs and it['kind'] in ('gold', 'A'):
        for k in (1, 2, 3):
            d = dict(it); d['rep'] = k; d['temp'] = 0.7; d['id'] = it['id'] + f'|rep{k}'; extra.append(d)
with open(f'{OUT}/items.jsonl', 'w', encoding='utf8') as f:
    for it in items + extra: f.write(json.dumps(it, ensure_ascii=False) + '\n')
with open(f'{OUT}/free_recall.jsonl', 'w', encoding='utf8') as f:
    for it in recall: f.write(json.dumps(it, ensure_ascii=False) + '\n')
print(len(recs), 'записей;', len(items), 'пар основного прогона;', len(extra), 'повторов;', len(recall), 'free recall;', collections.Counter(i['kind'] for i in items))
