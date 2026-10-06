#!/usr/bin/env python3
"""Сборка единого журнала проверок (формат reports/test_journal.tsv) и журнала дефектов по шагам.
Холм -- внутри семейства (колонка family), Бонферрони -- на ВСЕ строки журнала. Решение по предзаданным строкам:
«подтверждено» если p по Холму < 0,05 и Δ < 0 (лучше базы), иначе «не подтверждено»."""
import glob, csv, sys
rows = []
for f in sorted(glob.glob('reports/ext/_journal_step*_*.tsv')):
    for l in open(f, encoding='utf8'):
        r = l.rstrip('\n').split('\t')
        rows.append(dict(id=r[0], family=r[1], status=r[2], prespecified=r[3], n=r[4], delta=float(r[5]), ratio=float(r[6]), p=float(r[7]), note=r[8]))
if not rows: raise SystemExit('журнал пуст: шаги 5-6 не запускались')
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0; m = len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj
fam = {}
for r in rows: fam.setdefault(r['family'], []).append(r)
for k, rs in fam.items():
    for r, a in zip(rs, holm([x['p'] for x in rs])): r['holm'] = a
M = len(rows)
for r in rows: r['bonf'] = min(1.0, r['p'] * M)
def verdict(r):
    if r['prespecified'] != 'да': return 'разведочное'
    if r['family'].startswith('C'): return 'реальная привязка лучше плацебо' if (r['holm'] < 0.05 and r['delta'] < 0) else 'не отличимо от плацебо'
    return 'подтверждено' if (r['holm'] < 0.05 and r['delta'] < 0) else ('не подтверждено (хуже базы)' if (r['holm'] < 0.05 and r['delta'] > 0) else 'не подтверждено')
with open('reports/ext/test_journal.tsv', 'w', encoding='utf8') as f:
    f.write('id\tfamily\tstatus\tprespecified\tn\tdelta_logrank\trank_ratio\tp\tp_holm_family\tp_bonferroni_all\tdecision\tnote\n')
    for r in rows:
        f.write(f"{r['id']}\t{r['family']}\t{r['status']}\t{r['prespecified']}\t{r['n']}\t{r['delta']:+.3f}\t{r['ratio']:.2f}\t{r['p']:.4f}\t{r['holm']:.4f}\t{r['bonf']:.4f}\t{verdict(r)}\t{r['note']}\n")
print(f'журнал: {M} строк, {len(fam)} семейств')
with open('reports/ext/defect_log.tsv', 'w', encoding='utf8') as f:
    f.write('материал\tшаг\tкласс дефекта\tчисло/мера\tэффект на ключевое число\tисход\n')
    for p in sorted(glob.glob('reports/ext/_defects_step*.tsv')):
        f.write(open(p, encoding='utf8').read())
print(open('reports/ext/defect_log.tsv', encoding='utf8').read())
