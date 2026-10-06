#!/usr/bin/env python3
"""Применение поправок эталона (как scripts/apply_gold_corrections.py): каждая поправка -- строка с колонкой
evidence; старое значение сверяется, расхождение = ошибка. Использование: in.tsv corrections.tsv out.tsv [additions.tsv]
additions.tsv: полные новые записи с колонкой evidence (материал C)."""
import sys
src, corr, out = sys.argv[1:4]
add = sys.argv[4] if len(sys.argv) > 4 else None
def rd(p):
    with open(p, encoding='utf8') as f:
        h = f.readline().rstrip('\n').split('\t')
        return h, [dict(zip(h, l.rstrip('\n').split('\t'))) for l in f if l.strip()]
h, recs = rd(src); _, cs = rd(corr)
n_set = n_drop = 0
for c in cs:
    if not c['evidence'].strip():
        raise SystemExit(f'поправка без evidence: {c}')
    hit = [r for r in recs if r['note'] == c['note'] and r['ref_text'] == c['ref_text']]
    if len(hit) != 1:
        raise SystemExit(f'поправка {c["note"]} / {c["ref_text"]}: найдено {len(hit)} записей, ожидалась ровно одна')
    r = hit[0]
    if c['action'] == 'drop':
        recs.remove(r); n_drop += 1
    elif c['action'] == 'set':
        if r[c['field']] != c['old']:
            raise SystemExit(f'поправка {c["note"]}: поле {c["field"]} = {r[c["field"]]!r}, ожидалось {c["old"]!r}')
        r[c['field']] = c['new']; n_set += 1
    else:
        raise SystemExit(f'неизвестное действие {c["action"]}')
n_add = 0
if add:
    ah, ar = rd(add)
    for r in ar:
        if not r['evidence'].strip():
            raise SystemExit(f'добавление без evidence: {r["note"]}')
        if any(x['note'] == r['note'] and x['ref_text'] == r['ref_text'] for x in recs):
            raise SystemExit(f'добавление дублирует запись: {r["note"]} / {r["ref_text"]}')
        recs.append({k: r[k] for k in h}); n_add += 1
with open(out, 'w', encoding='utf8') as f:
    f.write('\t'.join(h) + '\n')
    for r in recs: f.write('\t'.join(r[k] for k in h) + '\n')
print(f'поправок set={n_set} drop={n_drop} add={n_add}; записей {len(recs)}')
