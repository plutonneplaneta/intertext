# -*- coding: utf-8 -*-
"""Щеглов, комментарий к «Двенадцати стульям» (OCR archive.org) -> заметки: глава, номер, лемма (цитата романа), текст.
Выход: shch_notes.tsv (chapter, num, head, text). Проверки: число глав в комментарии 41 (иначе ошибка с перечнем), номера заметок идут подряд."""
import sys, re, io
SRC, OUT = sys.argv[1:3]
t = io.open(SRC, encoding='utf8', errors='replace').read()
t = re.sub(r'-[ \t]*\n\s*', '', t)   # перенос слова: OCR оставляет пробел между «-» и концом строки
t = re.sub(r'[ \t]+', ' ', t)
i0 = t.index('ПОЯСНЕНИЯ К КОММЕНТАРИЯМ')   # в этом издании роман идёт первым, комментарии -- после
t = t[i0:]
t = re.sub(r'\n\s*[—-]\s*\d{2,3}\s*[—-]\s*\n', '\n', t)            # колонцифры «— 432 —»
t = re.sub(r'\n\s*(Ю\. ?К\. ?Щеглов|Комментарии|И\. ?Ильф, ?Е\. ?Петров)\s*\n', '\n', t)
flat = re.sub(r'\s+', ' ', t)
# заметки: последовательные номера внутри главы; сброс на «1.» при last>=2 открывает новую главу (заголовков глав в OCR комментариев нет)
cand = [(m.start(), int(m.group(1))) for m in re.finditer(r'(?<=[\s.»])(\d{1,2})\. (?=[А-ЯЁ«"\[(—])', flat)]
pos = []; last = 0; chap = 0
for p_, n in cand:
    if last > 0 and last < n <= last + 3:          # допускаем пропуск до 2 номеров (OCR теряет цифры)
        pos.append((p_, chap, n)); last = n
    elif n <= 2 and (last == 0 or last >= 3):       # сброс нумерации -- новая глава
        chap += 1; pos.append((p_, chap, n)); last = n
notes = []
for k, (p_, c, num) in enumerate(pos):
    q = pos[k + 1][0] if k + 1 < len(pos) else len(flat)
    body = re.sub(r'^\d{1,2}\. ', '', flat[p_:q].strip())
    m = re.search(r'[.!?…»]?\s*[—–-]\s', body)
    head = body[:m.start() + 1] if m and m.start() < 400 else body[:150]
    notes.append((c, num, head.strip(), body))
with io.open(OUT, 'w', encoding='utf8') as f:
    for c, n, h, b in notes: f.write(f'{c}\t{n}\t{h}\t{b}\n')
import collections
cnt = collections.Counter(n[0] for n in notes)
print('заметок', len(notes), 'в', len(cnt), 'главах; по главам:', [cnt[c] for c in sorted(cnt)])
if len(cnt) != 41: print('ВНИМАНИЕ: глав', len(cnt), 'вместо 41 -- сверить разбиение')
