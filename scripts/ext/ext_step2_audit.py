#!/usr/bin/env python3
"""ШАГ 2. Проверка эталона до модели. Не использует ни одного скорера.
Проверки (prereg §4, шаг 2): а) существование/диапазон; б) привязка: заголовок примечания в абзаце;
в) цитата в кавычках находится в указанном стихе (A); г) ручная выборка 40 записей (отдельный файл);
д) нумерация; плюс дубли. Результат: таблица флагов data/ext/gold/audit_<mat>.tsv и отчёт step2_<mat>.txt.
"""
import sys, os, re, collections, random
sys.path.insert(0, os.path.dirname(__file__))
import ext_lib as L, ext_eval as E
mat = sys.argv[1]
WORK = E.WORK
h, recs = E.read_gold(E.GOLD[mat])
pids, _, _ = E.load_par(mat); units = E.load_units(mat)
# полный текст примечаний (локально, не в git)
notes = E.load_notes(mat, recs)
# абзацы и строки
par_text = dict(zip(*E.load_target(mat)[:2]))
lines = {}
for l in open(E.TARGET[mat][0], encoding='utf8'):
    b, ln, par, t = l.rstrip('\n').split('\t', 3); lines[(int(b), int(ln))] = t
if E.is_bible(mat):
    uid, ut = L.load_kjv('data/ext/source/kjv_verses.tsv'); vtext = dict(zip(uid, ut))
    vstem = {u: set(L.stems(t)) for u, t in vtext.items()}
if E.is_d(mat):
    uid, ut = E.source_units(mat); dtext = dict(zip(uid, ut))


def longest_run(a, b):
    """Длина самой длинной общей непрерывной цепочки лемм (независимый путь к покрытию множества)."""
    best = 0; prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y: cur[j] = prev[j - 1] + 1; best = max(best, cur[j])
        prev = cur
    return best

def head_of(text):
    m = re.match(r'(.{3,90}?)(?:\.\s|\s{2,}|:\s|$)', text)
    return (m.group(1) if m else text[:60]).strip()

flags = collections.defaultdict(list)
rows = []
for r in recs:
    f = []
    note = notes.get(r['note'], '')
    if not note: f.append('NOTE_NOT_FOUND')
    head = head_of(note)
    hs = [s for s in L.stems(head) if len(s) >= 3]
    ptoks = set(L.stems(par_text[r['place']]))
    win = ' '.join(lines.get((int(r['book']), k), '') for k in range(int(r['line']) - 5, int(r['line']) + 6))
    wtoks = set(L.stems(win))
    cov_p = sum(s in ptoks for s in hs) / len(hs) if hs else None
    cov_w = sum(s in wtoks for s in hs) / len(hs) if hs else None
    if hs and cov_p < 0.5: f.append('HEAD_NOT_IN_PARAGRAPH')
    if r.get('status', 'ok') != 'ok': f.append('NO_SUCH_VERSE')
    if E.is_bible(mat) and r['status'] == 'ok':
        m = re.search(r'(\d+)\s*[:.]\s*(\d+)\s*[-–]\s*(\d+)', r['ref_text'])
        if m and int(m.group(3)) < int(m.group(2)):
            f.append('RANGE_SHORTHAND' if int(m.group(3)) < 10 or len(m.group(3)) < len(m.group(2)) else 'RANGE_BACKWARDS')
        # другая версия / не KJV рядом со ссылкой
        i = note.find(r['ref_text'])
        ctx = note[max(0, i - 60): i + len(r['ref_text']) + 40] if i >= 0 else ''
        if re.search(r'Geneva|Vulgate|Septuagint|Tyndale|Milton\'s (own )?(translation|paraphrase)|Douay|Hebrew text', ctx):
            f.append('OTHER_VERSION')
        # цитаты
        qpos = [m.start() for m in re.finditer(r'["“”]', note)]
        quotes = [note[a + 1:b] for a, b in zip(qpos[0::2], qpos[1::2])]   # кавычки парами подряд
        quotes = [q for q in quotes if len(L.stems(q)) >= 4]
        verdict = None
        if quotes:
            idxs = E.unit_idxs('A', r, units)
            rng = set().union(*[vstem[units[i]] for i in idxs])
            ptk = ptoks
            best = None
            for q in quotes:
                qs = [s for s in L.stems(q)]
                c_v = sum(s in rng for s in qs) / len(qs)
                c_p = sum(s in ptk for s in qs) / len(qs)
                if best is None or c_v > best[0]: best = (c_v, c_p, q)
            c_v, c_p, q = best
            if c_v >= 0.7: verdict = 'QUOTE_IN_VERSE'
            elif c_p >= 0.7: verdict = 'QUOTE_IS_MILTON'
            else:
                verdict = 'QUOTE_NEITHER'; f.append('QUOTE_NOT_IN_CITED_VERSE')
        r['quote'] = verdict or ''
    if E.is_d(mat):
        ql = L.stems(r.get('quote', ''))
        if len(ql) >= 4:
            run = longest_run(ql, L.stems(dtext[r['source_unit']]))
            r['quote_run'] = f'{run}/{len(ql)}'
            if run < min(3, len(ql)): f.append('QUOTE_NO_VERBATIM_RUN')
    r['cov_head_par'] = '' if cov_p is None else f'{cov_p:.2f}'
    r['flags'] = ','.join(f)
    rows.append(r)
# дубли
cnt = collections.Counter((r['place'], r.get('verse_first', r.get('source_unit')), r.get('verse_last', '')) for r in rows)
dups = sum(c - 1 for c in cnt.values() if c > 1)
cols = list(h) + ['cov_head_par', 'quote', 'quote_run', 'flags']
with open(f'data/ext/gold/audit_{mat}.tsv', 'w', encoding='utf8') as f:
    f.write('\t'.join(cols) + '\n')
    for r in rows: f.write('\t'.join(r.get(c, '') for c in cols) + '\n')
n = len(rows)
fc = collections.Counter(x for r in rows for x in r['flags'].split(',') if x)
rec_flag = sum(1 for r in rows if r['flags'])
out = [f'ШАГ 2 / материал {mat}: записей {n}', f'записей с флагом: {rec_flag} ({rec_flag/n:.1%})', f'дублей (место, источник, диапазон): {dups}']
for k, v in fc.most_common(): out.append(f'  {k}: {v} ({v/n:.1%})')
if E.is_bible(mat):
    qc = collections.Counter(r.get('quote') for r in rows if r.get('quote'))
    out.append(f'цитата в примечании: {sum(qc.values())} записей; {dict(qc)}')
txt = '\n'.join(out); print(txt)
open(f'reports/ext/step2_auto_{mat}.txt', 'w', encoding='utf8').write(txt + '\n')
