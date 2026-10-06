# -*- coding: utf-8 -*-
"""ШАГ 2 для материала C: поправки наивного эталона v0 (prereg_material_C.md, правило 4 -> шаг 2 протокола).
Находит (а) записи v0 со статусом != ok, (б) скобки в тексте, содержащие ссылки, но не давшие записей (правило «через ;» их пропустило).
Для каждой такой скобки пересобирает записи терпимым разбором (разбиение по границам названий книг) и пишет:
  corrections_C.tsv -- drop плохих записей v0 с evidence; additions_C.tsv -- новые записи с evidence.
Ссылки на целую главу (без стиха) в записи не превращаются: записываются как drop/пропуск с причиной."""
import sys, os, re, io, collections
sys.path.insert(0, os.path.dirname(__file__))
import ext_bible as B
SRC, KJV, GOLD, OUT_C, OUT_A = sys.argv[1:6]
ORIG_ALIAS = dict(B.ALIAS)   # таблица v0 (для воспроизведения статусов v0)
B.ALIAS.update({'1thes': '1TH', '2thes': '2TH'})
for k, c in {'thes': None, 'exo': 'EXO', 'pro': 'PRO', 'micah': 'MIC', 'eze': 'EZE', 'kings': None, 'chron': None, 'song': 'SON'}.items():
    if c: B.ALIAS[k] = c
t = io.open(SRC, encoding='utf8').read()
a = t.index('*** START OF'); a = t.index('\n', a); b = t.index('*** END OF'); body = t[a:b]
m = None
for mm in re.finditer(r"THE PILGRIM'S PROGRESS\s+In the Similitude of a Dream", body): m = mm
body = re.sub(r'\{\d+\}\s*', '', body[m.end():])
verses = {l.split('\t')[0] for l in io.open(KJV, encoding='utf8').read().split('\n')[1:] if l}
SINGLE = {'JUD', 'PHM', '2JO', '3JO', 'OBA'}   # книги из одной главы: «Jude 14» = Иуд 1:14
paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
H = ['note', 'place', 'book', 'line', 'ref_text', 'verse_first', 'verse_last', 'status', 'note_head']
gold = [dict(zip(H, l.rstrip('\n').split('\t'))) for l in io.open(GOLD, encoding='utf8').read().split('\n')[1:] if l]
by_place = collections.defaultdict(list)
for r in gold: by_place[r['place']].append(r)
SEG = re.compile(r'(?<=[,;] )(?=(?:[123] ?)?[A-Z][a-z]+\.? ?\d)|(?<=[,;])(?=(?:[123] ?)?[A-Z][a-z]+\.? ?\d)')
def tolerant(inner):
    inner = re.sub(r'\s+', ' ', inner).strip().rstrip('.').replace('Song of Solomon', 'Song')
    res = []
    for seg in SEG.split(inner):
        seg = seg.strip(' ,;')
        if not seg: continue
        mm = re.match(r'^((?:[123]\s?)?[A-Z][a-z]+\.?(?:_?of_?[A-Z][a-z]+)?|Song of Solomon)\.?\s*(.*)$', seg)
        if not mm: return None
        bt = mm.group(1).replace(' of ', '_of_'); code = B.norm_book(bt)
        rest = mm.group(2)
        if code is None:
            res.append((None, bt, None, None, None, 'NO_BOOK')); continue
        # терпимый разбор: «Eccl. 1; 2:11,17»-- перенос главы; одна глава: Jude 14
        items = [x.strip() for x in rest.replace(';', ',').split(',') if x.strip()]
        chap = 1 if code in SINGLE else None
        for x in items:
            q = re.match(r'^(?:(\d+)\s*[:.]\s*)?(\d+)(?:\s*[-–]\s*(\d+))?$', x)
            if not q: res.append((code, bt, None, None, None, 'UNPARSED')); continue
            if q.group(1): chap = int(q.group(1)); v1 = int(q.group(2)); v2 = int(q.group(3) or v1)
            elif code in SINGLE: v1 = int(q.group(2)); v2 = int(q.group(3) or v1)
            elif chap is None or (':' not in x and '.' not in x and len(items) == 1 and rest.count(':') == 0):
                res.append((code, bt, int(q.group(2)), None, None, 'CHAPTER_ONLY')); chap = int(q.group(2)); continue
            else:
                v1 = int(q.group(2)); v2 = int(q.group(3) or v1)
            res.append((code, bt, chap, v1, max(v1, v2), 'ok'))
    return res
drops, adds, report = [], [], collections.Counter()
for k, p in enumerate(paras, 1):
    place = 'PP.1.p%03d' % k
    brs = list(re.finditer(r'\[([^\[\]]{2,300})\]', p))
    for bi, mm in enumerate(brs):
        inner = mm.group(1)
        if not re.search(r'(?:[123]\s?)?[A-Z][a-z]+\.?\s*\d', inner): continue
        head = ' '.join(re.sub(r'\[[^\]]*\]', ' ', p[:mm.start()]).split()[-15:])
        NEW_ALIAS = dict(B.ALIAS); B.ALIAS.clear(); B.ALIAS.update(ORIG_ALIAS)
        parsed0 = B.parse_bracket(inner)
        B.ALIAS.clear(); B.ALIAS.update(NEW_ALIAS)
        bad0 = parsed0 is None or any(x[5] != 'ok' for x in parsed0)
        if not bad0: continue
        new = tolerant(inner)
        report['скобок с дефектом v0'] += 1
        report['причина: не разобрана правилом «;»' if parsed0 is None else 'причина: статус NO_BOOK/NO_VERSE'] += 1
        # drop записей v0 этой скобки с плохим статусом
        for r in by_place[place]:
            if r['status'] != 'ok' and r['note_head'] == head:
                drops.append((r['note'], r['ref_text'], 'drop', '*', '', '', f'запись v0 со статусом {r["status"]} в скобке [{inner[:70]}]; пересобрана терпимым разбором'))
        for j, x in enumerate(new or [], 1):
            code, bt, ch, v1, v2, st = x
            if st == 'CHAPTER_ONLY': report['ссылка на главу без стиха: пропущена'] += 1; continue
            if st != 'ok': report['не разобрано терпимо: ' + st] += 1; continue
            f, l = f'b.{code}.{ch}.{v1}', f'b.{code}.{ch}.{v2}'
            if f not in verses: report['нет стиха в KJV'] += 1; continue
            if l not in verses: l = f
            # если такая запись уже есть в v0 (status ok) -- не дублируем
            if any(r['status'] == 'ok' and r['place'] == place and r['verse_first'] == f and r['verse_last'] == l for r in by_place[place]): continue
            adds.append((f'1:{place}.t{bi}.{j}', place, 1, k, f'{bt} {ch}:{v1}' + (f'-{v2}' if v2 != v1 else ''), f, l, 'ok', head,
                         f'терпимый разбор скобки [{inner[:70]}]'))
with io.open(OUT_C, 'w', encoding='utf8') as f:
    f.write('note\tref_text\taction\tfield\told\tnew\tevidence\n')
    seen = set()
    for r in drops:
        if (r[0], r[1]) in seen: continue
        seen.add((r[0], r[1])); f.write('\t'.join(r) + '\n')
with io.open(OUT_A, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tverse_first\tverse_last\tstatus\tnote_head\tevidence\n')
    for r in adds: f.write('\t'.join(map(str, r)) + '\n')
print(dict(report), 'drop', len(seen), 'add', len(adds))
