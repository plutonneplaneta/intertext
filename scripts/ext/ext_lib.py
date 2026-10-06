"""Общая библиотека для проверки протокола на новых материалах (английский).

Скореры написаны заново и намеренно простые; ничего из основного проекта (веса, пороги, список
скореров) не переносится. Все скореры исчерпывающие: балл каждой единицы источника для каждого
целевого абзаца, без порогов и без шортлистов. Тихие отказы запрещены: пустые входы и нулевые
столбцы -- ошибка (SystemExit), а не нулевой признак.
"""
import re, math, zlib, collections, os, functools
import numpy as np
import scipy.sparse as sp

MODERN = {'thir': 'their', 'hee': 'he', 'mee': 'me', 'wee': 'we', 'shee': 'she', 'bee': 'be',
          'onely': 'only', 'sev': 'seven', 'heav': 'heaven'}
SUFFIXES = ['ational', 'ations', 'ation', 'ingly', 'edly', 'ness', 'ment', 'eth', 'est', 'ing',
            'ied', 'ies', 'ed', 'es', 'ly', 's']


LANG = os.environ.get('EXT_LANG', 'en')   # 'ru': pymorphy3, леммы (материал D)
_MORPH = None


@functools.lru_cache(maxsize=None)
def _lemma_ru(w):
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH.parse(w)[0].normal_form


def norm_words(text):
    if LANG == 'ru':
        return re.findall(r'[а-яё]{2,}', text.lower().replace('ё', 'е'))
    t = text.lower().replace('’', "'").replace('‘', "'")
    t = re.sub(r"(\w)'d\b", r"\1ed", t)
    t = re.sub(r"(\w)'n\b", r"\1en", t)
    t = re.sub(r"(\w)'r\b", r"\1er", t)
    t = re.sub(r"(\w)'st\b", r"\1est", t)
    t = re.sub(r"(\w)'s\b", r"\1", t)
    t = t.replace("'", '')
    return re.findall(r'[a-z]{2,}', t)


def stem(w):
    if LANG == 'ru':
        return _lemma_ru(w).replace('ё', 'е')
    w = MODERN.get(w, w)
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def stems(text):
    return [stem(w) for w in norm_words(text)]


def load_tsv(path, ncols=None):
    rows = []
    with open(path, encoding='utf8') as f:
        header = f.readline().rstrip('\n').split('\t')
        for l in f:
            l = l.rstrip('\n')
            if l:
                rows.append(l.split('\t') if ncols is None else l.split('\t', ncols - 1))
    if not rows:
        raise SystemExit(f'{path}: пустой вход')
    return header, rows


def load_kjv(path):
    _, rows = load_tsv(path, 2)
    return [r[0] for r in rows], [r[1] for r in rows]


def load_paragraphs(path, prefix='PL'):
    """data/ext/target/*_lines.tsv -> (ids, texts, nlines). id как в эталоне: <prefix>.<книга>.p<NNN>."""
    ids, texts, nl = [], [], []
    cur = None
    order = {}
    with open(path, encoding='utf8') as f:
        for l in f:
            b, ln, par, t = l.rstrip('\n').split('\t', 3)
            key = (int(b), int(par))
            if key not in order:
                order[key] = 1 + sum(1 for k in order if k[0] == int(b))
                ids.append('%s.%d.p%03d' % (prefix, int(b), order[key])); texts.append([]); nl.append(0)
            texts[-1].append(t); nl[-1] += 1
    return ids, [' '.join(t) for t in texts], nl


def load_classics(path):
    _, rows = load_tsv(path, 2)
    return [r[0] for r in rows], [r[1] for r in rows]


class Index:
    """Индекс источника: стеммы единиц, df, idf, матрицы."""
    def __init__(self, unit_ids, unit_texts):
        self.ids = list(unit_ids)
        self.pos = {u: i for i, u in enumerate(self.ids)}
        self.tok = [stems(t) for t in unit_texts]
        if not any(self.tok):
            raise SystemExit('Index: пустой источник')
        self.N = len(self.ids)
        df = collections.Counter()
        for t in self.tok: df.update(set(t))
        self.df = df
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        self.idf = {w: math.log(1 + (self.N - c + .5) / (c + .5)) for w, c in df.items()}
        self.lens = np.array([len(t) for t in self.tok], dtype=np.float64)
        self.avgdl = self.lens.mean()
        self._build()

    def _build(self):
        rows, cols, vals = [], [], []
        for i, t in enumerate(self.tok):
            for w, c in collections.Counter(t).items():
                rows.append(i); cols.append(self.vocab[w]); vals.append(c)
        self.tf = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, len(self.vocab)), dtype=np.float64)
        self.idfv = np.array([self.idf[w] for w in sorted(self.vocab, key=self.vocab.get)])
        # bm25-веса документов
        k1, b = 1.2, 0.75
        tf = self.tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * self.lens[tf.row] / self.avgdl)
        w = tf.data * (k1 + 1) / denom
        self.bm25_doc = sp.csr_matrix((w, (tf.row, tf.col)), shape=self.tf.shape)
        # idf-косинус: бинарный tf * idf, L2
        bin_ = self.tf.copy(); bin_.data[:] = 1
        m = bin_.multiply(self.idfv).tocsr()
        n = np.sqrt(m.multiply(m).sum(axis=1)).A1; n[n == 0] = 1
        self.cos_doc = sp.diags(1 / n) @ m
        # символьные 4-граммы
        self.c4 = {}
        c4docs = [self._char4(t) for t in self.tok]
        for g in sorted(set().union(*c4docs)):   # sorted: порядок столбцов не должен зависеть от PYTHONHASHSEED
            self.c4[g] = len(self.c4)
        r, c, v = [], [], []
        cdf = collections.Counter()
        for i, d in enumerate(c4docs):
            for g in sorted(d):
                r.append(i); c.append(self.c4[g]); v.append(1.0); cdf[g] += 1
        m4 = sp.csr_matrix((v, (r, c)), shape=(self.N, len(self.c4)))
        self.c4idf = np.zeros(len(self.c4))
        for g, i in self.c4.items(): self.c4idf[i] = math.log(1 + self.N / cdf[g])
        m4 = m4.multiply(self.c4idf).tocsr()
        n = np.sqrt(m4.multiply(m4).sum(axis=1)).A1; n[n == 0] = 1
        self.c4_doc = sp.diags(1 / n) @ m4
        # инвертированный индекс биграмм (поз.) для скорера «серия»
        self.bi = collections.defaultdict(list)
        for i, t in enumerate(self.tok):
            for p in range(len(t) - 1):
                self.bi[(t[p], t[p + 1])].append((i, p))

    @staticmethod
    def _char4(toks):
        s = ' '.join(toks)
        return {s[i:i + 4] for i in range(len(s) - 3)}

    def query_vecs(self, texts):
        """Бинарные запросы по основам абзацев."""
        toks = [stems(t) for t in texts]
        r, c = [], []
        for i, t in enumerate(toks):
            for w in sorted(set(t)):
                if w in self.vocab: r.append(i); c.append(self.vocab[w])
        q = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(texts), len(self.vocab)))
        return toks, q

    def score_all(self, texts, exclude=None):
        """exclude: список индексов единиц (или -1) -- единица, совпадающая с самой целью, не участвует в ранжировании."""
        out = self.score_raw(texts)
        return self.finish(out, exclude)

    def score_raw(self, texts):
        toks, q = self.query_vecs(texts)
        out = {}
        # BM25: sum_w idf(w) * bm25_doc(w, d) для w в запросе
        out['bm25'] = (q.multiply(self.idfv).tocsr() @ self.bm25_doc.T).toarray()
        qn = q.multiply(self.idfv).tocsr()
        nrm = np.sqrt(qn.multiply(qn).sum(axis=1)).A1; nrm[nrm == 0] = 1
        out['idfcos'] = ((sp.diags(1 / nrm) @ qn) @ self.cos_doc.T).toarray()
        q4 = []
        r, c, v = [], [], []
        for i, t in enumerate(toks):
            for g in sorted(self._char4(t)):
                if g in self.c4: r.append(i); c.append(self.c4[g]); v.append(self.c4idf[self.c4[g]])
        m = sp.csr_matrix((v, (r, c)), shape=(len(texts), len(self.c4)))
        n = np.sqrt(m.multiply(m).sum(axis=1)).A1; n[n == 0] = 1
        out['char4'] = ((sp.diags(1 / n) @ m) @ self.c4_doc.T).toarray()
        out['run'] = self._run(toks)
        return out

    @staticmethod
    def finish(out, exclude=None):
        if exclude is not None:
            for k in ('bm25', 'idfcos', 'char4', 'run'):
                for qi, ui in enumerate(exclude):
                    if ui >= 0: out[k][qi, ui] = -1.0
        out['fuse'] = np.mean([pct_rows(out[k]) for k in ('bm25', 'idfcos', 'char4')], axis=0)
        for k, v in out.items():
            if not np.isfinite(v).all():
                raise SystemExit(f'скорер {k}: нечисловые значения')
            if not (v != 0).any():
                raise SystemExit(f'скорер {k}: нулевой во всех строках (тихий отказ запрещён)')
        return out

    def _run(self, toks):
        """«Серия»: для (абзац, единица) -- максимальная по цепочкам сумма idf слов самой длинной
        цепочки подряд идущих общих пар основ (>= 2 слов). 0, если общих пар нет."""
        S = np.zeros((len(toks), self.N))
        for qi, t in enumerate(toks):
            pb = {(t[p], t[p + 1]) for p in range(len(t) - 1)}
            hits = collections.defaultdict(set)
            for bg in pb:
                for (u, p) in self.bi.get(bg, ()):
                    hits[u].add(p)
            for u, ps in hits.items():
                toku = self.tok[u]; best = 0.0
                for p in sorted(ps):
                    if p - 1 in ps: continue          # начало цепочки
                    e = p
                    while e + 1 in ps: e += 1
                    wsum = sum(self.idf[toku[k]] for k in range(p, e + 2))
                    best = max(best, wsum)
                S[qi, u] = best
        return S


def pct_rows(M):
    """Процентиль внутри строки со средним рангом при связках (0..1)."""
    from scipy.stats import rankdata
    return np.apply_along_axis(lambda r: (rankdata(r, 'average') - 1) / (len(r) - 1), 1, M)


def tie_rank(row, idxs):
    """Ранг (1 = лучший) лучшей из единиц idxs; ранг при связках -- средний среди равных."""
    best = None
    for i in idxs:
        v = row[i]
        gt = (row > v).sum(); eq = (row == v).sum()
        r = gt + (eq + 1) / 2.0
        if best is None or r < best: best = r
    return best


def cluster_boot_mean(vals, clusters, B=5000, seed=20261005):
    """Бутстрап среднего по кластерам (абзацам): ресемплируются кластеры целиком."""
    vals = np.asarray(vals, float)
    cl = sorted(set(clusters))
    if len(cl) < 5:
        raise SystemExit(f'бутстрап по {len(cl)} кластерам: недостаточно (< 5)')
    idx = {c: np.where(np.array(clusters) == c)[0] for c in cl}
    rng = np.random.RandomState(seed)
    out = np.empty(B)
    for b in range(B):
        pick = rng.randint(0, len(cl), len(cl))
        out[b] = np.concatenate([vals[idx[cl[i]]] for i in pick]).mean()
    return vals.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)
