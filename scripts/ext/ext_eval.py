"""Загрузка эталона и счёт рангов по кэшу скореров. Общий для шагов 1-7."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
WORK = os.environ.get('WORK', '/tmp/intertext-ext') + '/out'
SCORERS = ['bm25', 'idfcos', 'char4', 'run', 'fuse']
GOLD = {'A': 'data/ext/gold/gold_milton_bible_v0.tsv', 'B': 'data/ext/gold/gold_milton_classics_v0.tsv',
        'B2': 'data/ext/gold/gold_milton_classics_v0.tsv',
        'C': 'data/ext/gold/gold_bunyan_v0.tsv', 'D': 'data/ext/gold/gold_onegin_v0.tsv', 'D2': 'data/ext/gold/gold_onegin2_v0.tsv', 'E': 'data/ext/gold/gold_ilf_v0.tsv'}   # B2 = B с очищенным источником (prereg_followup.md, П1а)


TARGET = {'A': ('data/ext/target/pl_lines.tsv', 'PL'), 'B': ('data/ext/target/pl_lines.tsv', 'PL'),
          'B2': ('data/ext/target/pl_lines.tsv', 'PL'), 'C': ('data/ext/target/pp_lines.tsv', 'PP'),
          'D': ('data/ext/target/on_lines.tsv', 'ON'), 'D2': ('data/ext/target/on_lines.tsv', 'ON'), 'E': ('data/ext/target/ilf_lines.tsv', 'IF')}


def is_d(mat):
    return mat in ('D', 'D2', 'E')


def is_bible(mat):
    return mat in ('A', 'C')


def load_target(mat):
    import ext_lib
    return ext_lib.load_paragraphs(*TARGET[mat])


def load_notes(mat, recs=None):
    """Тексты примечаний (A, B, B2: Милтон, локально); для C у автора нет примечаний -- текст = 15 слов перед скобкой."""
    if mat in ('C', 'D', 'D2', 'E'):
        return {r['note']: r['note_head'] for r in recs}
    d = {}
    for l in open(f'{WORK}/pl_notes.tsv', encoding='utf8'):
        b, nid, ln, t = l.rstrip('\n').split('\t', 3); d[f'{b}:{nid}'] = t
    return d


def source_units(mat):
    """(ids, texts) единиц источника материала."""
    import ext_lib
    if is_bible(mat):
        return ext_lib.load_kjv('data/ext/source/kjv_verses.tsv')
    if is_d(mat):
        rows = [l.rstrip('\n').split('\t', 3) for l in open({'D': 'data/ext/source/ru_units.tsv', 'D2': 'data/ext/source/ru_units2.tsv', 'E': 'data/ext/source/ru_units_e.tsv'}[mat], encoding='utf8').read().split('\n')[1:] if l]
        return [r[0] for r in rows], [r[3] for r in rows]
    return ext_lib.load_classics(classics_path(mat))


def classics_path(mat):
    return f'{WORK}/classics_books_clean.tsv' if mat == 'B2' else f'{WORK}/classics_books.tsv'


def load_par(mat):
    ids, nl, ntok = [], [], []
    for l in open(f'{WORK}/par_{mat}.tsv', encoding='utf8'):
        a, b, c = l.rstrip('\n').split('\t'); ids.append(a); nl.append(int(b)); ntok.append(int(c))
    return ids, np.array(nl), np.array(ntok)


def load_units(mat):
    return [l.rstrip('\n') for l in open(f'{WORK}/units_{mat}.tsv', encoding='utf8')]


def load_scores(mat, names=SCORERS):
    return {k: np.load(f'{WORK}/scores_{mat}_{k}.npy').astype(np.float64) for k in names}


def read_gold(path):
    with open(path, encoding='utf8') as f:
        h = f.readline().rstrip('\n').split('\t')
        return h, [dict(zip(h, l.rstrip('\n').split('\t'))) for l in f if l.strip()]


def unit_idxs(mat, rec, units):
    pos = {u: i for i, u in enumerate(units)}
    if mat in ('B', 'B2', 'D', 'D2', 'E'):
        return [pos[rec['source_unit']]]
    f, l = rec['verse_first'], rec['verse_last']
    pre, c1, v1 = f.rsplit('.', 2); _, c2, v2 = l.rsplit('.', 2)
    if c1 != c2:
        l = f
        v2 = v1
    out = [pos[f'{pre}.{c1}.{v}'] for v in range(int(v1), int(v2) + 1) if f'{pre}.{c1}.{v}' in pos]
    if not out:
        raise SystemExit(f'запись {rec.get("note")}: ни одного стиха диапазона {f}..{l} в источнике')
    return out


def rank_tie(row, idxs):
    best = None
    for i in idxs:
        v = row[i]
        r = (row > v).sum() + ((row == v).sum() + 1) / 2.0
        best = r if best is None or r < best else best
    return best


def eval_records(mat, recs, S, pids, units):
    """-> dict scorer -> (ranks, nonzero) для записей recs (список dict)."""
    pix = {p: i for i, p in enumerate(pids)}
    out = {}
    for k, M in S.items():
        rk, nz = [], []
        for r in recs:
            row = M[pix[r['place']]]
            ix = unit_idxs(mat, r, units)
            rk.append(rank_tie(row, ix)); nz.append(bool((row[ix] > 0).any()))
        out[k] = (np.array(rk), np.array(nz))
    return out


def logmean(r):
    r = np.asarray(r, float)
    if len(r) < 5:
        raise SystemExit(f'выборка из {len(r)} записей: отказ считать среднее log-rank (< 5)')
    return float(np.log(r).mean())


def ok_records(recs, where=''):
    """Записи со status == ok. Отброшенные -- не молча: число печатается в stderr (тихий отказ шага 7)."""
    keep = [r for r in recs if r.get('status', 'ok') == 'ok']
    if len(keep) != len(recs):
        print(f'[{where}] отброшено {len(recs) - len(keep)} записей со status != ok: ' +
              ', '.join(r['note'] for r in recs if r.get('status', 'ok') != 'ok'), file=sys.stderr)
    return keep
