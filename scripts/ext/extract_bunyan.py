# -*- coding: utf-8 -*-
"""Баньян, «Путь паломника» (Gutenberg #131): целевой текст без ссылочных скобок + наивный эталон v0.
Правила -- prereg_material_C.md. Вход: pg131.txt, kjv_verses.tsv. Выход: pp_lines.tsv, gold_bunyan_v0.tsv, отчёт разбора."""
import sys, os, re, io, collections
sys.path.insert(0, os.path.dirname(__file__))
import ext_bible as B
SRC, KJV, OUT_T, OUT_G = sys.argv[1:5]
t = io.open(SRC, encoding='utf8').read()
a = t.index('*** START OF'); a = t.index('\n', a); b = t.index('*** END OF')
body = t[a:b]
m = None
for mm in re.finditer(r"THE PILGRIM'S PROGRESS\s+In the Similitude of a Dream", body):
    m = mm
if m is None: raise SystemExit('нет маркера начала сновидения')
body = body[m.end():]
body = re.sub(r'\{\d+\}\s*', '', body)
verses = {l.split('\t')[0] for l in io.open(KJV, encoding='utf8').read().split('\n')[1:] if l}
paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
rows = []; gold = []; nonref = []; stats = collections.Counter()
for k, p in enumerate(paras, 1):
    place = 'PP.1.p%03d' % k   # непрерывная нумерация абзацев (одна часть)
    pid = 'PP.1.p%03d' % k
    text = p
    refs = []
    for mm in re.finditer(r'\[([^\[\]]{2,300})\]', p):
        parsed = B.parse_bracket(mm.group(1))
        if parsed is None:
            nonref.append(mm.group(1)[:60]); continue
        head = ' '.join(re.sub(r'\[[^\]]*\]', ' ', p[:mm.start()]).split()[-15:])
        for (book, btxt, ch, v1, v2, st) in parsed:
            refs.append((mm.group(0), book, btxt, ch, v1, v2, st, head))
    clean = re.sub(r'\[([^\[\]]{2,300})\]', lambda mm: ' ' if B.parse_bracket(mm.group(1)) is not None else mm.group(0), p)
    clean = re.sub(r'\s+', ' ', clean).strip()
    rows.append((pid, clean))
    for j, (raw, book, btxt, ch, v1, v2, st, head) in enumerate(refs, 1):
        if st == 'ok':
            f, l = f'b.{book}.{ch}.{v1}', f'b.{book}.{ch}.{v2}'
            if f not in verses: st = 'NO_SUCH_VERSE'
            elif l not in verses: l = f
        else:
            f = l = ''
        stats[st] += 1
        gold.append((f'1:{pid}.{j}', pid, 1, k, f'{btxt} {ch}:{v1}' + (f'-{v2}' if v2 != v1 else ''), f, l, st, head))
# тест на утечку (правило 3)
leak = sum(1 for _, c in rows if any(B.parse_bracket(x) is not None for x in re.findall(r'\[([^\[\]]{2,300})\]', c)))
if leak: raise SystemExit(f'утечка: {leak} абзацев сохранили ссылочные скобки')
with io.open(OUT_T, 'w', encoding='utf8') as f:
    for pid, c in rows: f.write(f'1\t{int(pid.split("p")[1])}\t{int(pid.split("p")[1])}\t{c}\n')
with io.open(OUT_G, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tverse_first\tverse_last\tstatus\tnote_head\n')
    for r in gold: f.write('\t'.join(map(str, r)) + '\n')
print(f'абзацев {len(rows)}; абзацев с ссылками {len({g[1] for g in gold})}; записей {len(gold)}; статусы {dict(stats)}; скобок не-ссылок {len(nonref)}')
print('примеры не-ссылок:', nonref[:8])
