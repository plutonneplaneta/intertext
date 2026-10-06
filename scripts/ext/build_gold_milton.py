#!/usr/bin/env python3
"""Наивный эталон «Потерянный рай» -> KJV и -> классики из примечаний Milton Reading Room.

НАИВНЫЙ -- сознательно: каждое примечание с ссылкой вида «Книга глава:стих» даёт запись,
без проверки, что ссылка указывает на источник строки, а не на фон. Дефекты ищет шаг 2
протокола (audit_ext_gold.py), а не этот скрипт.

Вход : pl_lines.tsv, pl_notes.tsv (extract_milton.py), kjv_verses.tsv
Выход: gold_milton_bible_v0.tsv, gold_milton_classics_v0.tsv
Единица цели: абзац стиха (par). Единица источника: стих KJV / книга классика.
"""
import sys, re, collections

LINES, NOTES, KJV, OUT_B, OUT_C = sys.argv[1:6]

BOOKS = {
 'GEN': 'Genesis Gen', 'EXO': 'Exodus Exod Ex', 'LEV': 'Leviticus Lev', 'NUM': 'Numbers Num',
 'DEU': 'Deuteronomy Deut', 'JOS': 'Joshua Josh', 'JDG': 'Judges Judg', 'RUT': 'Ruth',
 '1SA': '1Samuel 1Sam', '2SA': '2Samuel 2Sam', '1KI': '1Kings', '2KI': '2Kings',
 '1CH': '1Chronicles 1Chron', '2CH': '2Chronicles 2Chron', 'EZR': 'Ezra', 'NEH': 'Nehemiah Neh',
 'EST': 'Esther', 'JOB': 'Job', 'PSA': 'Psalm Psalms Ps', 'PRO': 'Proverbs Prov',
 'ECC': 'Ecclesiastes Eccl', 'SON': 'Song', 'ISA': 'Isaiah Isa', 'JER': 'Jeremiah Jer',
 'LAM': 'Lamentations Lam', 'EZE': 'Ezekiel Ezek', 'DAN': 'Daniel Dan', 'HOS': 'Hosea',
 'JOE': 'Joel', 'AMO': 'Amos', 'OBA': 'Obadiah', 'JON': 'Jonah', 'MIC': 'Micah', 'NAH': 'Nahum',
 'HAB': 'Habakkuk', 'ZEP': 'Zephaniah', 'HAG': 'Haggai', 'ZEC': 'Zechariah Zech', 'MAL': 'Malachi Mal',
 'MAT': 'Matthew Matt', 'MAR': 'Mark', 'LUK': 'Luke', 'JOH': 'John', 'ACT': 'Acts', 'ROM': 'Romans Rom',
 '1CO': '1Corinthians 1Cor', '2CO': '2Corinthians 2Cor', 'GAL': 'Galatians Gal', 'EPH': 'Ephesians Eph',
 'PHI': 'Philippians Phil', 'COL': 'Colossians Col', '1TH': '1Thessalonians 1Thess',
 '2TH': '2Thessalonians 2Thess', '1TI': '1Timothy 1Tim', '2TI': '2Timothy 2Tim', 'TIT': 'Titus',
 'PHM': 'Philemon', 'HEB': 'Hebrews Heb', 'JAM': 'James', '1PE': '1Peter 1Pet', '2PE': '2Peter 2Pet',
 '1JO': '1John', '2JO': '2John', '3JO': '3John', 'JUD': 'Jude', 'REV': 'Revelation Revelations Rev',
}
ALIAS = {}
for code, names in BOOKS.items():
    for n in names.split():
        ALIAS[n.lower()] = code
names_re = '|'.join(sorted({re.escape(re.sub(r'^(\d)', r'\1 ?', n)) if n[0].isdigit() else re.escape(n)
                            for n in (x for v in BOOKS.values() for x in v.split())}, key=len, reverse=True))
# "1 Cor" записан в BOOKS как 1Cor -> допускаем необязательный пробел
BIB = re.compile(r'(?<![A-Za-z])((?:[123] ?)?(?:%s))\.?\s+(\d{1,3})\s*[:.]\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?' %
                 '|'.join(sorted({re.escape(x) for v in BOOKS.values() for x in v.split() if not x[0].isdigit()}, key=len, reverse=True)))

def norm_book(s):
    s = s.replace(' ', '').lower()
    return ALIAS.get(s)

verses = {}
for l in open(KJV, encoding='utf8').read().split('\n')[1:]:
    if l:
        vid, t = l.split('\t', 1); verses[vid] = t

# --- целевые абзацы: плотная нумерация
par_ids = {}
line_par = {}
for l in open(LINES, encoding='utf8'):
    b, ln, par, t = l.rstrip('\n').split('\t', 3)
    key = (int(b), int(par))
    par_ids.setdefault(key, len(par_ids) + 1)
    line_par[(int(b), int(ln))] = key
pid = lambda b, ln: 'PL.%d.p%03d' % (b, sum(1 for k in par_ids if k[0] == b and par_ids[k] <= par_ids[line_par[(b, ln)]]) )

rows_b, rows_c = [], []
seen = set()
for l in open(NOTES, encoding='utf8'):
    b, nid, ln, text = l.rstrip('\n').split('\t', 3)
    b, ln = int(b), int(ln)
    if ln == 0 or (b, ln) not in line_par:   # примечания к Argument/введению -- без строки
        continue
    place = pid(b, ln)
    head = text[:140].replace('\t', ' ')
    for m in BIB.finditer(text):
        code = norm_book(m.group(1))
        if not code: continue
        ch, v1 = int(m.group(2)), int(m.group(3)); v2 = int(m.group(4)) if m.group(4) else v1
        if v2 < v1: v2 = v1
        first, last = f'b.{code}.{ch}.{v1}', f'b.{code}.{ch}.{v2}'
        if first not in verses:
            rows_b.append((f'{b}:{nid}', place, b, ln, f'{m.group(0)}', first, first, 'NO_SUCH_VERSE', head)); continue
        if last not in verses: last = first
        key = (b, nid, first, last)
        if key in seen: continue
        seen.add(key)
        rows_b.append((f'{b}:{nid}', place, b, ln, m.group(0), first, last, 'ok', head))
with open(OUT_B, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tverse_first\tverse_last\tstatus\tnote_head\n')
    for r in rows_b: f.write('\t'.join(map(str, r)) + '\n')

# --- классики: ссылка на произведение + номер книги (песни); строка -- справочно
CL = re.compile(r"(Aeneid|Iliad|Odyssey|Metamorphoses)\s+(?:book\s+)?(\d{1,2})(?:\s*[.:]\s*(\d{1,4})(?:\s*[-–]\s*\d{1,4})?)?")
MAXB = {'Aeneid': 12, 'Iliad': 24, 'Odyssey': 24, 'Metamorphoses': 15}
seen = set()
for l in open(NOTES, encoding='utf8'):
    b, nid, ln, text = l.rstrip('\n').split('\t', 3)
    b, ln = int(b), int(ln)
    if ln == 0 or (b, ln) not in line_par: continue
    place = pid(b, ln)
    for m in CL.finditer(text):
        w, k = m.group(1), int(m.group(2))
        if k > MAXB[w] or k < 1: continue
        key = (b, nid, w, k)
        if key in seen: continue
        seen.add(key)
        rows_c.append((f'{b}:{nid}', place, b, ln, m.group(0), f'{w}.{k}', 'ok', text[:140]))
with open(OUT_C, 'w', encoding='utf8') as f:
    f.write('note\tplace\tbook\tline\tref_text\tsource_unit\tstatus\tnote_head\n')
    for r in rows_c: f.write('\t'.join(map(str, r)) + '\n')
print('Библия:', len(rows_b), 'записей,', len({r[1] for r in rows_b}), 'абзацев,', sum(r[7] != 'ok' for r in rows_b), 'без стиха в KJV')
print('Классики:', len(rows_c), 'записей,', len({r[1] for r in rows_c}), 'абзацев')
