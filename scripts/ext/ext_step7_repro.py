#!/usr/bin/env python3
"""ШАГ 7. Тихие отказы и воспроизводимость.
1) Инъекции неисправностей: охранные проверки обязаны сработать (SystemExit), иначе -- дефект.
2) Независимая пересборка ключевых чисел: BM25 на чистом Python (словари, float64, ранг сортировкой),
   без scipy и без кэша; сверка со средним log-rank из основного конвейера на множестве протокола.
3) Второй прогон с другим PYTHONHASHSEED сверен отдельно (побитово, см. отчёт)."""
import sys, os, re, math, collections, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ext_eval as E, ext_lib as L
out = ['ШАГ 7: тихие отказы и воспроизводимость']
# ---------- 1. инъекции
def expect_fail(name, fn):
    try:
        fn()
    except SystemExit as e:
        out.append(f'  [сработало] {name}: {str(e)[:90]}'); return True
    except Exception as e:
        out.append(f'  [упало иначе] {name}: {type(e).__name__}: {str(e)[:70]}'); return None
    out.append(f'  [НЕ СРАБОТАЛО] {name}: молчаливый проход'); return False
res = []
res.append(expect_fail('пустой источник', lambda: L.Index(['u1'], [''])))
res.append(expect_fail('скорер, нулевой во всех строках (абзацы без общих слов)', lambda: L.Index(['u1', 'u2'], ['alpha beta gamma delta', 'omega sigma']).score_all(['zzzz qqqq xxxx'])))
res.append(expect_fail('среднее log-rank по 3 записям', lambda: E.logmean([5, 10, 20])))
res.append(expect_fail('бутстрап по 3 абзацам', lambda: L.cluster_boot_mean([1., 2., 3.], ['a', 'b', 'c'])))
res.append(expect_fail('пустой TSV', lambda: L.load_tsv(os.devnull)))
tmp = tempfile.mkdtemp()
open(f'{tmp}/g.tsv', 'w').write('note\tref_text\tstatus\tverse_first\n1:x\tGen 1:1\tok\tb.GEN.1.1\n')
open(f'{tmp}/c.tsv', 'w').write('note\tref_text\taction\tfield\told\tnew\tevidence\n1:x\tGen 1:1\tset\tverse_first\tb.GEN.1.2\tb.GEN.1.3\tпричина\n')
def script_fails(*args):
    r = subprocess.run([sys.executable, 'scripts/ext/ext_apply_corrections.py', *args], capture_output=True)
    if r.returncode == 0: return
    raise SystemExit(r.stderr.decode().strip().splitlines()[-1])
res.append(expect_fail('поправка со старым значением, не совпадающим с эталоном', lambda: script_fails(f'{tmp}/g.tsv', f'{tmp}/c.tsv', f'{tmp}/o.tsv')))
open(f'{tmp}/c2.tsv', 'w').write('note\tref_text\taction\tfield\told\tnew\tevidence\n1:x\tGen 1:1\tdrop\t*\t\t\t\n')
res.append(expect_fail('поправка без evidence', lambda: script_fails(f'{tmp}/g.tsv', f'{tmp}/c2.tsv', f'{tmp}/o.tsv')))
units = ['b.GEN.1.1', 'b.GEN.1.2']
res.append(expect_fail('диапазон стихов, которого нет в источнике', lambda: E.unit_idxs('A', {'note': 'x', 'verse_first': 'b.GEN.9.1', 'verse_last': 'b.GEN.9.2'}, units)))
# ---------- 1б. запись status != ok: не молча
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    E.ok_records([{'note': 'z', 'status': 'NO_SUCH_VERSE'}], 'test')
out.append(f'  [{"сработало" if "отброшено" in buf.getvalue() else "НЕ СРАБОТАЛО"}] отбрасывание записи со status != ok сообщается: {buf.getvalue().strip()[:70]}')
res.append('отброшено' in buf.getvalue())
fails = sum(1 for r in res if r is not True)
# ---------- 2. независимая пересборка (чистый Python)
def bm25_independent(unit_texts, par_texts):
    toks = [L.stems(t) for t in unit_texts]
    N = len(toks); df = collections.Counter()
    for t in toks: df.update(set(t))
    idf = {w: math.log(1 + (N - c + .5) / (c + .5)) for w, c in df.items()}
    dl = [len(t) for t in toks]; avg = sum(dl) / N
    post = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for w, c in collections.Counter(t).items(): post[w].append((i, c))
    def scores(q):
        sc = [0.0] * N
        for w in set(L.stems(q)):
            if w not in post: continue
            for i, c in post[w]:
                sc[i] += idf[w] * c * 2.2 / (c + 1.2 * (1 - 0.75 + 0.75 * dl[i] / avg))
        return sc
    return scores
out.append('\n== Независимая пересборка BM25 (чистый Python, float64) против кэша конвейера (.npy) ==')
all_ok = True
for mat in 'ABC':
    pids, _, _ = E.load_par(mat); units_m = E.load_units(mat)
    uid, ut = (L.load_kjv('data/ext/source/kjv_verses.tsv') if E.is_bible(mat) else L.load_classics(f'{E.WORK}/classics_books.tsv'))
    ptexts = dict(zip(*E.load_target(mat)[:2]))
    _, recs = E.read_gold(E.GOLD[mat].replace('_v0', '_v1')); recs = E.ok_records(recs, 'step7')
    sc = bm25_independent(ut, None)
    S = E.load_scores(mat, ['bm25']); pix = {p: i for i, p in enumerate(pids)}
    mine, main, ndiff = [], [], 0
    cache = {}
    for r in recs:
        if r['place'] not in cache: cache[r['place']] = np.array(sc(ptexts[r['place']]))
        row = cache[r['place']]; ix = E.unit_idxs(mat, r, units_m)
        a = E.rank_tie(row, ix); b = E.rank_tie(S['bm25'][pix[r['place']]], ix)
        mine.append(a); main.append(b); ndiff += abs(a - b) > 1e-9
    d = abs(E.logmean(mine) - E.logmean(main))
    out.append(f'{mat}: n={len(recs)}; среднее log-rank независимое {E.logmean(mine):.9f}; конвейер {E.logmean(main):.9f}; |разность| {d:.2e}; записей с различающимся рангом: {ndiff}')
    all_ok &= (d < 1e-9 and ndiff == 0)

out.append('\n== Второй и третий прогон run_scorers.py (PYTHONHASHSEED=1 и 2, чистый каталог), матрицы сопоставлены побитово с основными ==')
same = True
for seed in (1, 2):
    wd = tempfile.mkdtemp(); os.makedirs(f'{wd}/out')
    subprocess.run(['cp', f'{E.WORK}/classics_books.tsv', f'{wd}/out/'], check=True)
    for mat in 'ABC':
        subprocess.run([sys.executable, 'scripts/ext/run_scorers.py', mat], check=True, capture_output=True,
                       env=dict(os.environ, PYTHONHASHSEED=str(seed), WORK=wd))
        for k in E.SCORERS:
            a = np.load(f'{wd}/out/scores_{mat}_{k}.npy'); b = np.load(f'{E.WORK}/scores_{mat}_{k}.npy')
            if not np.array_equal(a, b):
                same = False; out.append(f'  РАСХОЖДЕНИЕ seed={seed} {mat} {k}: max|diff|={np.abs(a - b).max():.3e}')
out.append(f'  все 30 матриц идентичны основным: {same}')
out.append('')
out.append('== Дефекты, найденные на шаге 7 (все исправлены; см. deviations.md) ==')
out.append('1. Кэш скореров хранился в float32: связки по float32 отличались от float64, у 1 записи из 390 (A) менялся ранг')
out.append('   (среднее log-rank BM25 6,539818 против 6,539828, разность 1,0e-05). Найдено независимой пересборкой на чистом')
out.append('   Python. Кэш переведён в float64, шаги 1-6 пересчитаны, независимая пересборка совпала (разность 0).')
out.append('2. Записи со status != ok (1 запись в A) молча отбрасывались в шагах 1, 3, 4, 5, 6. Теперь число и')
out.append('   идентификаторы отброшенных записей печатаются (E.ok_records).')
out.append('3. После перевода кэша в float64 второй прогон с другим PYTHONHASHSEED дал расхождение char4 порядка 1e-16/1e-15')
out.append('   (порядок столбцов и суммирования зависел от итерации по set). В float32 оно скрывалось округлением. Порядок')
out.append('   зафиксирован сортировкой (ext_lib.Index), матрицы пересчитаны; расхождений нет, если строка выше говорит True.')
out.append(f'\nКритерий шага 7 в финальном состоянии: охранных проверок сработало {len(res) - fails}/{len(res)}; '
           f'независимая пересборка совпала: {all_ok}; прогоны с разными PYTHONHASHSEED идентичны: {same}')
out.append('ИСХОД: дефект найден (три, исправлены); ' + ('после исправления критерий выполнен' if (all_ok and same) else 'ПОСЛЕ ИСПРАВЛЕНИЯ КРИТЕРИЙ НЕ ВЫПОЛНЕН'))
txt = '\n'.join(out); print(txt)
open('reports/ext/step7.txt', 'w', encoding='utf8').write(txt + '\n')
with open('reports/ext/_defects_step7_AB.tsv', 'w', encoding='utf8') as f:
    f.write('A\t7\tкэш float32 менял ранг при связках (независимая пересборка)\t1\t1 запись из 390; mean log-rank 6.539818 -> 6.539828\tнайден\n')
    f.write('A\t7\tрасхождение char4 между прогонами с разным PYTHONHASHSEED (порядок суммирования по set)\t1\tmax|diff| 6e-15; в float32 скрывалось\tнайден\n')
    f.write('A\t7\tмолчаливое отбрасывание записей со status != ok\t1\t1 запись в шагах 1,3,4,5,6\tнайден\n')
    f.write('B\t7\tкэш float32 / отбрасывание status\t0\tв B проявлений нет\tнет\n')
