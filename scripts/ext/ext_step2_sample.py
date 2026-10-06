#!/usr/bin/env python3
"""ШАГ 2г. Выборка 40 записей для ручной разметки (семя 20261005). Печатает записи для чтения;
метки хранятся отдельно в data/ext/gold/step2_manual_labels_<mat>.tsv (source/background/not_by_text)."""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E
mat = sys.argv[1]
h, recs = E.read_gold(E.GOLD[mat])
notes = E.load_notes(mat, recs)
ids = sorted({(r['note'], r.get('ref_text')) for r in recs})
random.Random(20261005).shuffle(ids)
pick = sorted(ids[:40])
for i, (n, ref) in enumerate(pick, 1):
    r = next(x for x in recs if x['note'] == n and x.get('ref_text') == ref)
    src = r.get('verse_first', r.get('source_unit'))
    print(f'#{i} {n} | ref={ref} -> {src}\n   {notes[n][:420]}\n')
