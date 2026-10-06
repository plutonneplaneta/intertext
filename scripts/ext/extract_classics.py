#!/usr/bin/env python3
"""Классические источники для «Потерянного рая» (Gutenberg, общественное достояние):
Вергилий, «Энеида» (Драйден); Гомер, «Илиада» и «Одиссея» (Поуп); Овидий, «Метаморфозы» (Райли, проза).
Единица -- книга (песнь). Номер книги соответствует оригиналу (Aeneid.6 = Aen. VI).
Вход: каталог с aen.txt ili.txt ody.txt met.txt met2.txt; выход TSV unit_id \t text.
Проверка: число найденных книг == 12/24/24/15, иначе ошибка.
"""
import sys, re

D, OUT = sys.argv[1], sys.argv[2]
CLEAN = len(sys.argv) > 3 and sys.argv[3] == 'clean'   # правила очистки: prereg_followup.md, П1а
ROM = {}
def roman(n):
    v = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]; s = ''
    for k, r in v:
        while n >= k: s += r; n -= k
    return s
for i in range(1, 25): ROM[roman(i)] = i
ORD = {w: i for i, w in enumerate('FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH ELEVENTH TWELFTH THIRTEENTH FOURTEENTH FIFTEENTH'.split(), 1)}

def body(fn):
    t = open(f'{D}/{fn}', encoding='utf8').read()
    a = t.index('*** START OF'); a = t.index('\n', a)
    b = t.index('*** END OF')
    t = t[a:b]
    if CLEAN and fn == 'ili.txt':
        c = t.rindex('\nCONCLUDING NOTE.')   # последнее вхождение: первое -- оглавление
        t = t[:c]
    if CLEAN and fn.startswith('met'):
        t = re.sub(r'\[Footnote \d+:.*?\](?=\n\s*\n)', ' ', t, flags=re.S)
        t = re.sub(r'\[\d+\]', ' ', t)
    return t.split('\n')

def split_books(lines, pat, conv, expect, name):
    pos = {}
    for i, l in enumerate(lines):
        m = pat.match(l)
        if m:
            k = conv(m.group(1))
            if k: pos[k] = i          # последнее вхождение: первое -- оглавление
    if sorted(pos) != list(range(1, expect + 1)):
        raise SystemExit(f'{name}: найдены книги {sorted(pos)}, ожидалось 1..{expect}')
    ks = sorted(pos); out = {}
    for j, k in enumerate(ks):
        end = pos[ks[j + 1]] if j + 1 < len(ks) else len(lines)
        out[k] = re.sub(r'\s+', ' ', ' '.join(lines[pos[k] + 1:end])).strip()
    return out

rom_pat = re.compile(r'^\s*BOOK\s+([IVXL]+)\.?\s*$')
res = []
for fn, name, exp in [('aen.txt', 'Aeneid', 12), ('ili.txt', 'Iliad', 24), ('ody.txt', 'Odyssey', 24)]:
    bk = split_books(body(fn), rom_pat, lambda s: ROM.get(s), exp, name)
    res += [(f'{name}.{k}', t) for k, t in bk.items()]
met = {}
met.update(split_books(body('met.txt'), re.compile(r'^BOOK THE ([A-Z]+)\.?\s*$'), lambda s: ORD.get(s) if ORD.get(s, 99) <= 7 else None, 7, 'Met1-7'))
m2 = split_books(body('met2.txt'), re.compile(r'^BOOK THE ([A-Z]+)\.?\s*$'), lambda s: ORD.get(s) - 7 if ORD.get(s, 0) > 7 else None, 8, 'Met8-15')
met.update({k + 7: t for k, t in m2.items()})
res += [(f'Metamorphoses.{k}', t) for k, t in sorted(met.items())]
with open(OUT, 'w', encoding='utf8') as f:
    f.write('unit_id\ttext\n')
    for u, t in res: f.write(f'{u}\t{t}\n')
print(len(res), 'книг;', sum(len(t.split()) for _, t in res), 'слов')
for u, t in res[::9]: print(u, len(t.split()), t[:70])
