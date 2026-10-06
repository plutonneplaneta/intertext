# -*- coding: utf-8 -*-
"""LLM-судья на материале C (prereg_closure.md, З2): золото, 3 обманки A (топ fuse), 2 B (та же книга Библии), 2 C (случайные). Все записи C v1."""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(__file__)); os.environ['EXT_LANG'] = 'en'
import numpy as np, ext_eval as E, ext_lib as L
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
pids, ptexts, _ = E.load_target('C'); ptext = dict(zip(pids, ptexts)); pix = {p: i for i, p in enumerate(pids)}
uid, ut = E.source_units('C'); utext = dict(zip(uid, ut)); upos = {u: i for i, u in enumerate(uid)}
S = E.load_scores('C', ['fuse'])['fuse']
_, recs = E.read_gold('data/ext/gold/gold_bunyan_v1.tsv'); recs = E.ok_records(recs, 'llm-c')
SYS = ('You are a literary scholar. Decide whether Text A (a passage from John Bunyan\'s "The Pilgrim\'s Progress") quotes, paraphrases or clearly alludes to Text B (a Bible passage, KJV). '
       'Scale: 0 - no connection (shared common words and a shared general theme do not count); 1 - weak thematic similarity; 2 - a noticeable echo: a shared image, phrase or motif that can be shown with quotations; '
       '3 - direct quotation, close paraphrase or explicit borrowing. If the score is 1 or higher, give EXACT quotations from both texts that support the score. '
       'Answer strictly as JSON: {"score": 0-3, "quote_a": "...", "quote_b": "..."}')
def window(text, ref, size=200):
    w = text.split(); sl = set(L.stems(ref))
    if len(w) <= size: return text
    best = (-1, 0)
    for i in range(0, len(w) - size + 1, 25):
        sc = len(sl & set(L.stems(' '.join(w[i:i + size]))))
        if sc > best[0]: best = (sc, i)
    return ' '.join(w[best[1]:best[1] + size])
rng = random.Random(20261005); items = []; books = collections.defaultdict(list)
for i, u in enumerate(uid): books[u.split('.')[1]].append(i)
for r in recs:
    place = r['place']; ix = E.unit_idxs('C', r, uid); gold_text = ' '.join(utext[uid[i]] for i in ix); row = S[pix[place]]
    order = np.argsort(-row); A = [int(i) for i in order if int(i) not in ix][:3]
    book = uid[ix[0]].split('.')[1]; poolB = [i for i in books[book] if i not in ix and i not in A]
    B = rng.sample(poolB, min(2, len(poolB))); Cc = rng.sample([i for i in range(len(uid)) if i not in ix + A + B], 2)
    stan = window(ptext[place], gold_text)
    def add(kind, text, uidx):
        items.append({'id': f'{r["note"]}|{r["ref_text"]}|{kind}|{uidx}', 'rec': f'{r["note"]}|{r["ref_text"]}', 'kind': kind, 'unit': str(uidx), 'stanza': stan,
                      'unit_window': text, 'rep': 0, 'temp': 0.0, 'system': SYS})
    add('gold', gold_text, 'gold')
    for k, lst in (('A', A), ('B', B), ('C', Cc)):
        for i in lst: add(k, utext[uid[i]], i)
with open(f'{OUT}/items.jsonl', 'w', encoding='utf8') as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + '\n')
print(len(recs), 'записей', len(items), 'пар', collections.Counter(i['kind'] for i in items))
