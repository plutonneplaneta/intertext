# -*- coding: utf-8 -*-
"""Экспорт цель/источник для плотных скореров (сервер). Каталог: targets_<mat>.tsv, units_<mat>.tsv (id \t text)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
for mat in (sys.argv[2:] or ['A', 'B2', 'C', 'D']):
    ids, texts, _ = E.load_target(mat)
    with open(f'{OUT}/targets_{mat}.tsv', 'w', encoding='utf8') as f:
        for i, t in zip(ids, texts): f.write(f'{i}\t{" ".join(t.split())}\n')
    uid, ut = E.source_units(mat)
    with open(f'{OUT}/units_{mat}.tsv', 'w', encoding='utf8') as f:
        for i, t in zip(uid, ut): f.write(f'{i}\t{" ".join(t.split())}\n')
print('экспорт готов', OUT)
