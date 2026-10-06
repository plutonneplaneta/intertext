# -*- coding: utf-8 -*-
"""Материал D: «Онегин» по строфам, корпус стихотворений-источников, наивный эталон v0 по Лотману (prereg_material_D.md).
Вход: $LOT (c?-?.html), $ON (ch?.wiki), PoetryCorpus all.xml. Выход: on_lines.tsv, ru_poems.tsv, gold_onegin_v0.tsv, отчёт."""
import sys, os, re, io, glob, html, collections
sys.path.insert(0, os.path.dirname(__file__))
os.environ['EXT_LANG'] = 'ru'
import ext_lib as L
LOT, ON, XML, OUT_T, OUT_S, OUT_G = sys.argv[1:7]
RULE = int(os.environ.get('D_RULE', '1'))
ROM = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
def rom(s):
    v = 0
    for i, c in enumerate(s):
        x = ROM[c]; v += -x if i + 1 < len(s) and ROM[s[i + 1]] > x else x
    return v
# ---- Онегин
stanzas = []   # (chapter, number, text)
for c in range(1, 9):
    w = io.open(f'{ON}/ch{c}.wiki', encoding='utf8').read()
    w = w.split('=== [[../Примечания Пушкина')[0]
    parts = re.split(r'\{\{poem-section\|\{\{roman\|(\d+)\}\}\.\}\}', w)
    for i in range(1, len(parts), 2):
        t = parts[i + 1]
        t = re.sub(r'<ref[^>]*>.*?</ref>', '', t, flags=re.S); t = re.sub(r'<ref[^>]*/>', '', t)
        t = re.sub(r'\{\{[^{}]*\}\}', '', t); t = re.sub(r'\{\{[^{}]*\}\}', '', t)
        t = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', t); t = re.sub(r"<[^>]+>|'''?|\}\}|#tag:poem\|", '', t)
        t = re.sub(r'\n\s*\n+', '\n', t).strip()
        if len(t.split()) >= 20: stanzas.append((c, int(parts[i]), t))
with io.open(OUT_T, 'w', encoding='utf8') as f:
    k = collections.Counter()
    for c, n, t in stanzas:
        k[c] += 1
        f.write(f'{c}\t{n}\t{n}\t' + ' '.join(t.split('\n')) + '\n')
sid = {(c, n): 'ON.%d.p%03d' % (c, i) for c in range(1, 9) for i, (cc, n, _) in enumerate([x for x in stanzas if x[0] == c], 1)}
ontext = ' '.join(t for _, _, t in stanzas); on_lem = L.stems(ontext)
# ---- Лотман
def lot_text(f):
    b = open(f, 'rb').read()
    for enc in ('cp1251', 'utf-8'):
        try: t = b.decode(enc)
        except: continue
        if 'Пушкин' in t: break
    t = re.sub(r'<br\s*/?>', '\n', t); t = re.sub(r'</p>|<p[^>]*>|</h\d>|<h\d[^>]*>|</div>|<div[^>]*>|</li>|<li[^>]*>', '\n', t)
    t = re.sub(r'<script.*?</script>', ' ', t, flags=re.S); t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(re.sub(r'[ \t\xa0]+', ' ', t))
    return [l.strip() for l in t.split('\n') if l.strip()]
notes = []   # (chapter, stanza, head, text)
for c in range(1, 9):
    cur_st = None; cur = None
    for p in range(1, 5):
        for l in lot_text(f'{LOT}/c{c}-{p}.html'):
            m = re.match(r'^Примечания ([IVXL]+)$', l)
            if m: cur_st = rom(m.group(1)); cur = None; continue
            m = re.match(r'^(?:([IVXL]+), )?(\d+(?:\s*[—–-]\s*\d+)?)\s+—\s+(.*)$', l)
            if m and cur_st is not None:
                st = rom(m.group(1)) if m.group(1) else cur_st
                cur = [c, st, m.group(3)[:200], m.group(3)]; notes.append(cur); cur_st = st; continue
            if cur is not None and not l.startswith(('Назад', 'Примечания', 'Пушкин А.С.', 'Лотман.')):
                cur[3] += '\n' + l
# ---- корпус источников
xml = io.open(XML, encoding='utf8').read()
items = re.findall(r'<item>(.*?)</item>', xml, re.S)
allnotes = ' '.join(n[3] for n in notes)
poems = []
auth_years = collections.defaultdict(list)
for it in items:
    a = re.search(r'<author>(.*?)</author>', it); tx = re.search(r'<text>(.*?)</text>', it, re.S); nm = re.search(r'<name>(.*?)</name>', it, re.S); d = re.search(r'<date_from>(.*?)</date_from>', it)
    if not (a and tx and d): continue
    try: y = int(d.group(1))
    except: continue
    if y <= 1837: poems.append((a.group(1), nm.group(1) if nm else '', y, html.unescape(tx.group(1))))
auth_cnt = collections.Counter(p[0] for p in poems)
def surname(a): return a.split()[-1]
chosen = {a for a in auth_cnt if len(surname(a)) >= 5 and len(re.findall(surname(a)[:max(5, len(surname(a)) - 2)], allnotes)) >= 3}
poems = [p for p in poems if p[0] in chosen and 'Онегин' not in p[1]]
poems = sorted(poems, key=lambda p: (p[0], p[2], p[1]))
# З4 (prereg_closure.md): дополнительные источники -- пушкинские поэмы и драмы из Викитеки (D_EXTRA=каталог с *.wiki)
EXTRA = os.environ.get('D_EXTRA')
if EXTRA:
    for fn in sorted(glob.glob(f'{EXTRA}/*.wiki')):
        w = io.open(fn, encoding='utf8').read()
        if len(w) < 3000: print('пропущен (короткий):', os.path.basename(fn), len(w)); continue
        w = re.sub(r'<ref[^>]*>.*?</ref>', '', w, flags=re.S); w = re.sub(r'\{\{[^{}]*\}\}', '', w); w = re.sub(r'\{\{[^{}]*\}\}', '', w)
        w = re.sub(r'\[\[(?:Категория|Category)[^\]]*\]\]', '', w); w = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', w); w = re.sub(r"<[^>]+>|'''?|__[A-Z]+__", '', w)
        poems.append(('Александр Пушкин', os.path.basename(fn)[:-5].replace('_Пушкин_', '').replace('_', ' '), 1830, w))
with io.open(OUT_S, 'w', encoding='utf8') as f:
    f.write('unit_id\tauthor\ttitle\ttext\n')
    for i, (a, nm, y, tx) in enumerate(poems, 1): f.write(f'P{i:05d}\t{a}\t{nm.replace(chr(9), " ").replace(chr(10), " ")}\t{" ".join(tx.split())}\n')
# ---- единицы источника: стихотворения + строфы «Онегина» (правило 3, поправка D-1)
units = [(f'P{i+1:05d}', 'poem', f'{p[0]}: {p[1][:40]}', p[3]) for i, p in enumerate(poems)]
units += [(sid[(c, n)], 'stanza', f'Онегин {c}.{n}', t) for c, n, t in stanzas]
ulem = [L.stems(u[3]) for u in units]; uset = [set(x) for x in ulem]; ubig = [set(zip(x, x[1:])) for x in ulem]
upos = {u[0]: i for i, u in enumerate(units)}
with io.open(OUT_S, 'w', encoding='utf8') as f:
    f.write('unit_id\tkind\ttitle\ttext\n')
    for u in units: f.write(f'{u[0]}\t{u[1]}\t{u[2].replace(chr(9), " ")}\t{" ".join(u[3].split())}\n')
def covers(qs_, qb, i): return sum(w in uset[i] for w in qs_) / len(qs_), len(qb & ubig[i])
gold = []; stat = collections.Counter()
for ni, (c, st, head_, text) in enumerate(notes):
    if (c, st) not in sid: stat['заметка к строфе вне текста'] += 1; continue
    me = upos[sid[(c, st)]]
    flat = ' '.join(text.split('\n'))
    qs = [q for q in re.findall(r'«([^»]{15,400})»', flat) if len(L.stems(q)) >= 4]
    lines = text.split('\n'); blk = []
    for ln in lines + ['']:
        if 4 <= len(ln) <= 75 and not re.search(r'\.\s+[А-ЯA-Z]', ln[:60]): blk.append(ln)
        else:
            if len(blk) >= 2: qs.append(' '.join(blk))
            blk = []
    for qi, q in enumerate(qs):
        ql = L.stems(q)
        if len(ql) < 4: stat['цитата без русских слов'] += 1; continue
        qs_ = set(ql); qb = set(zip(ql, ql[1:]))
        cv, bg = covers(qs_, qb, me)
        if cv >= 0.8 and bg: stat['цитата найдена в строфе самой заметки'] += 1; continue
        hits = []
        for i in range(len(units)):
            if i == me: continue
            cv, bg = covers(qs_, qb, i)
            if cv >= 0.8 and bg: hits.append((cv, bg, i))
        if not hits: stat['цитата не найдена'] += 1; continue
        hits.sort(key=lambda x: (-x[0], -x[1]))
        for cv, bg, i in hits[:3]:
            stat['запись: ' + units[i][1]] += 1
            gold.append((f'{c}:{st}:n{ni}.q{qi}', sid[(c, st)], c, st, units[i][2], units[i][0], units[i][0], 'ok', head_[:140], q[:100]))
with io.open(OUT_G, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tsource_unit\tverse_last\tstatus\tnote_head\tquote\n')
    for r in gold: f.write('\t'.join(map(str, r)) + '\n')
print(f'строф {len(stanzas)}; заметок {len(notes)}; единиц-источников {len(units)} ({len(poems)} стихотворений у {len(chosen)} авторов + {len(stanzas)} строф); записей {len(gold)}; строф с записями {len({g[1] for g in gold})}')
print(dict(stat))
