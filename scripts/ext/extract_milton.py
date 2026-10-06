#!/usr/bin/env python3
"""Milton Reading Room (Dartmouth): текст «Потерянного рая» + примечания.

Читает сырые страницы text.shtml и annotation.js (скачиваются fetch_ext.sh в $WORK/milton),
пишет:
  $OUT/pl_lines.tsv   book \t line \t par \t text               (текст, по строкам поэмы)
  $OUT/pl_notes.tsv   book \t note_id \t line \t note_text (примечание, привязанное к ближайшей
                                                          метке строки [ N ] после якоря)
Привязка: якорь <a id=... class=annotBtn> лежит перед ближайшей меткой строки вида
`<span class="line" id="lineN">`; строка якоря = первая метка N после него, минус 4
(метки стоят через 5 строк, якорь в пределах пятёрки).
Пройденные проверки: число якорей == число записей в annotList (иначе ошибка).
"""
import sys, os, re, json, html

WORK, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

def strip(h):
    h = re.sub(r'<br\s*/?>', '\n', h)
    h = re.sub(r'<[^>]+>', '', h)
    return html.unescape(h)

lines_out, notes_out = [], []
for b in range(1, 13):
    page = open(f'{WORK}/b{b}.html', encoding='utf8', errors='replace').read()
    js = open(f'{WORK}/a{b}.js', encoding='utf8', errors='replace').read()
    # annotList -- не валидный JSON (\\' и посторонний HTML в конце): разбираем записью "id":"текст"
    js = js[js.index('{'):]
    d = {}
    for mm in re.finditer(r'"([A-Za-z0-9_\-]+)"\s*:\s*"(.*?)"\s*(?=,\s*"[A-Za-z0-9_\-]+"\s*:|\}\s*(?:;|$))', js, re.S):
        v = mm.group(2)
        v = re.sub(r'\\(.)', lambda x: {'n': ' ', 't': ' '}.get(x.group(1), x.group(1)), v)
        d[mm.group(1)] = v
    n_decl = len(re.findall(r'(?:^|,)\s*"[A-Za-z0-9_\-]+"\s*:\s*"', js, re.M))
    if n_decl != len(d):
        print(f'book {b}: объявлено ключей ~{n_decl}, разобрано {len(d)}', file=sys.stderr)
    # --- текст поэмы: от первой строки стиха до конца; метки строк разбивают поток
    # поэма начинается с первого абзаца, содержащего <br> (Argument набран без <br>)
    start = next(pm.start() for pm in re.finditer(r'<p[^>]*>.*?</p>', page, re.S) if '<br' in pm.group(0))
    text = page[start:]
    anchors = [mm.group(1) for mm in re.finditer(r'<a id="([^"]+)" class="annotBtn"', text)]
    ids_in_page = set(anchors)
    pre_ids = {mm.group(1) for mm in re.finditer(r'<a id="([^"]+)" class="annotBtn"', page[:start])}
    missing = set(d) - ids_in_page - pre_ids
    if missing:
        print(f'book {b}: {len(missing)} примечаний без якоря на странице', sorted(missing)[:5], file=sys.stderr)
    # строки: метки номеров убираем, якоря заменяем маркерами \x01id\x02, режем по <br>
    seg = re.sub(r'<span class="line"[^>]*>\s*\[\s*\d+\s*\]\s*</span>', '', text)
    seg = seg.replace('\n', ' ')
    seg = re.sub(r'<a id="([^"]+)" class="annotBtn"[^>]*>', lambda m: '\x01' + m.group(1) + '\x02', seg)
    seg = re.sub(r'</p>|<p[^>]*>', '<br/>\x03<br/>', seg)
    plain = re.sub(r'[ ]{2,}', ' ', strip(seg))
    raw_ls = [l.strip() for l in plain.split('\n') if l.strip()]
    ls, par_of, par = [], [], 0
    for l in raw_ls:
        if l == '\x03':
            par += 1
            continue
        ls.append(l.lstrip('\x03')); par_of.append(par)
    end = next((i for i, l in enumerate(ls) if re.match(r'(The End of the|THE END)', l)), None)
    if end is None:
        raise SystemExit(f'book {b}: нет строки "The End of": {[x[:60] for x in ls[-3:]]}')
    ls, par_of = ls[:end], par_of[:end]
    anchor_line = {}
    for i, l in enumerate(ls, 1):
        for aid in re.findall('\x01([^\x02]+)\x02', l):
            anchor_line.setdefault(aid, i)
        lines_out.append((b, i, par_of[i - 1], re.sub('\x01[^\x02]*\x02', '', l).replace('\t', ' ')))
    for aid in anchors:
        if aid in d and aid in anchor_line:
            notes_out.append((b, aid, anchor_line[aid], d[aid]))
    for aid in pre_ids:
        if aid in d:
            notes_out.append((b, aid, 0, d[aid]))
    notes_out[:] = [(x[0], x[1], x[2], strip(x[3]).replace('\t', ' ').replace('\n', ' ')) if i >= 0 else x
                    for i, x in enumerate(notes_out)]

with open(f'{OUT}/pl_lines.tsv', 'w', encoding='utf8') as f:
    for r in lines_out: f.write('\t'.join(map(str, r)) + '\n')
with open(f'{OUT}/pl_notes.tsv', 'w', encoding='utf8') as f:
    for r in notes_out: f.write('\t'.join(map(str, r)) + '\n')
print(len(lines_out), 'строк;', len(notes_out), 'примечаний')
