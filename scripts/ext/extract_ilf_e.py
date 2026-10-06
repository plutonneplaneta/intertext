# -*- coding: utf-8 -*-
"""Материал E (prereg_material_E.md): блоки романа, источники, наивный эталон v0 по комментарию Щеглова.
Вход: ilf_novel.jsonl, shch_notes.tsv (extract_shcheglov.py), каталог ws/*.jsonl (fetch_ws_authors.py), PoetryCorpus all.xml.
Выход: ilf_lines.tsv, ru_units_e.tsv, gold_ilf_v0.tsv + отчёт разбора."""
import sys, os, re, io, json, glob, html, collections
sys.path.insert(0, os.path.dirname(__file__)); os.environ['EXT_LANG'] = 'ru'
import ext_lib as L
NOVEL, NOTES, WS, XML, OUT_T, OUT_S, OUT_G = sys.argv[1:8]
# ---- блоки романа
blocks = []   # (chapter, text)
for l in io.open(NOVEL, encoding='utf8'):
    c = json.loads(l); cur = []; n = 0
    for para in [p.strip() for p in re.split(r'\n+', c['text']) if p.strip()]:
        cur.append(para); n += len(para.split())
        if n >= 100: blocks.append((c['chapter'], ' '.join(cur))); cur = []; n = 0
    if cur:
        if blocks and blocks[-1][0] == c['chapter'] and n < 40: blocks[-1] = (c['chapter'], blocks[-1][1] + ' ' + ' '.join(cur))
        else: blocks.append((c['chapter'], ' '.join(cur)))
bid = []; cnt = collections.Counter()
for c, t in blocks: cnt[c] += 1; bid.append('IF.%d.p%03d' % (c, cnt[c]))
with io.open(OUT_T, 'w', encoding='utf8') as f:
    for (c, t), i in zip(blocks, bid): f.write(f'{c}\t{int(i.split("p")[1])}\t{int(i.split("p")[1])}\t{" ".join(t.split())}\n')
blem = [set(L.stems(t)) for _, t in blocks]
novel_lem = [L.stems(t) for _, t in blocks]
novel_big = set(); allnov = []
for x in novel_lem: allnov += x
novel_big = set(zip(allnov, allnov[1:])); novel_set = set(allnov)
# ---- источники
AUTH = {'Chekhov': 'Чехов', 'Gogol': 'Гоголь', 'Bulgakov': 'Булгаков', 'Averchenko': 'Аверченко', 'Bunin': 'Бунин', 'Teffi': 'Тэффи', 'Zoshchenko': 'Зощенко',
        'Dostoevsky': 'Достоевск', 'Kuprin': 'Куприн', 'Gorky': 'Горьк', 'Leskov': 'Лесков', 'Gilyarovsky': 'Гиляровск'}
ABBR = {'Gogol': r'Ревизор|Мертв\. ?душ|Мёртв\. ?душ|Шинель|Нос\b', 'Bulgakov': r'Собачье сердце|Маст\. и Марг|Мастер и Маргарита|Роковые яйца|Дьяволиада|Белая гвардия',
        'Chekhov': r'Скрипка Ротшильда|Вишнёвый сад|Вишневый сад|Три сестры|Дама с собачкой'}
units = []   # (id, author, title, text)
for fn in sorted(glob.glob(f'{WS}/*.jsonl')):
    a = os.path.basename(fn)[:-6]
    for i, l in enumerate(io.open(fn, encoding='utf8'), 1):
        d = json.loads(l); units.append((f'W{a[:3]}{i:05d}', a, d['title'], d['text']))
xml = io.open(XML, encoding='utf8').read()
PO = {'Владимир Маяковский': 'Маяковск', 'Михаил Лермонтов': 'Лермонтов', 'Николай Некрасов': 'Некрасов', 'Александр Блок': 'Блок', 'Сергей Есенин': 'Есенин', 'Александр Пушкин': 'Пушкин', 'Федор Тютчев': 'Тютчев', 'Афанасий Фет': 'Фет', 'Иван Крылов': 'Крылов'}
for it in re.findall(r'<item>(.*?)</item>', xml, re.S):
    a = re.search(r'<author>(.*?)</author>', it); tx = re.search(r'<text>(.*?)</text>', it, re.S); nm = re.search(r'<name>(.*?)</name>', it, re.S); d = re.search(r'<date_from>(.*?)</date_from>', it)
    if a and tx and d and a.group(1) in PO:
        try: y = int(d.group(1))
        except Exception: continue
        if y <= 1930: units.append((f'Pc{len(units):05d}', a.group(1), (nm.group(1) if nm else '')[:40], html.unescape(tx.group(1))))
AUTH.update({a: s for a, s in PO.items()})
RU = {'Chekhov': 'Антон Чехов', 'Gogol': 'Николай Гоголь', 'Bulgakov': 'Михаил Булгаков', 'Averchenko': 'Аркадий Аверченко', 'Bunin': 'Иван Бунин', 'Teffi': 'Надежда Тэффи', 'Zoshchenko': 'Михаил Зощенко',
      'Dostoevsky': 'Федор Достоевский', 'Kuprin': 'Александр Куприн', 'Gorky': 'Максим Горький', 'Leskov': 'Николай Лесков', 'Gilyarovsky': 'Владимир Гиляровский'}
with io.open(OUT_S, 'w', encoding='utf8') as f:
    f.write('unit_id\tkind\ttitle\ttext\n')
    for u in units: f.write(f'{u[0]}\tunit\t{RU.get(u[1], u[1])}: {u[2].replace(chr(9), " ")}\t{" ".join(u[3].split())}\n')
ulem = [L.stems(u[3]) for u in units]; uset = [set(x) for x in ulem]; ubig = [set(zip(x, x[1:])) for x in ulem]
by_auth = collections.defaultdict(list)
for i, u in enumerate(units): by_auth[u[1]].append(i)
# ---- заметки и якоря
notes = [l.rstrip('\n').split('\t', 3) for l in io.open(NOTES, encoding='utf8')]
gold = []; stat = collections.Counter(); ptr = 0
def run_len(a, b):
    pos = {}
    for j, y in enumerate(b): pos.setdefault(y, []).append(j)
    prev = {}; best = 0
    for x in a:
        cur = {}
        for j in pos.get(x, []): cur[j] = prev.get(j - 1, 0) + 1; best = max(best, cur[j])
        prev = cur
    return best
for ni, (chap, num, head, body) in enumerate(notes):
    hl = [w for w in L.stems(head) if len(w) >= 3]
    best = (0.0, -1)
    if len(hl) >= 3:
        covs = [sum(w in blem[i] for w in hl) / len(hl) for i in range(len(blocks))]
        top = max(covs)
        if top >= 0.6:
            cands = [i for i, c in enumerate(covs) if c >= top - 0.05]
            i = min(cands, key=lambda j: (abs(j - ptr) if j >= ptr - 3 else 10 ** 6 + abs(j - ptr)))   # ближайший к текущей позиции, вперёд
            best = (top, i)
    if best[0] < 0.6: stat['заметка без якоря'] += 1; continue
    ptr = best[1]; place = bid[ptr]
    names = [a for a, s in AUTH.items() if re.search(s, body) or (a in ABBR and re.search(ABBR[a], body))]
    cand = [i for a in names for i in by_auth.get(a, [])]
    qs = [q for q in re.findall(r'«([^»]{25,500})»', body) if len(L.stems(q)) >= 4]
    for qi, q in enumerate(qs):
        ql = L.stems(q)
        if len(ql) < 4: continue
        qs_ = set(ql); qb = set(zip(ql, ql[1:]))
        if sum(w in novel_set for w in qs_) / len(qs_) >= 0.8 and qb & novel_big: stat['цитата из романа (самоссылка)'] += 1; continue
        if not cand: stat['в заметке нет автора из корпуса'] += 1; continue
        hits = [(sum(w in uset[i] for w in qs_) / len(qs_), len(qb & ubig[i]), i) for i in cand]
        hits = [h for h in hits if h[0] >= 0.8 and h[1]]
        if not hits: stat['цитата не найдена у названных авторов'] += 1; continue
        hits.sort(key=lambda x: (-x[0], -x[1]))
        for cv, bg, i in hits[:3]:
            stat['запись'] += 1
            gold.append((f'{chap}:{num}:q{qi}', place, int(place.split('.')[1]), int(place.split('p')[1]), f'{RU.get(units[i][1], units[i][1])}: {units[i][2][:40]} [{units[i][0]}]', units[i][0], units[i][0], 'ok', head[:140], q[:100]))
with io.open(OUT_G, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tsource_unit\tverse_last\tstatus\tnote_head\tquote\n')
    for r in gold: f.write('\t'.join(map(str, r)) + '\n')
print(f'блоков {len(blocks)}; заметок {len(notes)}; единиц-источников {len(units)} ({len(by_auth)} авторов); записей {len(gold)}; блоков с записями {len({g[1] for g in gold})}')
print(dict(stat)); print({a: len(v) for a, v in by_auth.items()})
