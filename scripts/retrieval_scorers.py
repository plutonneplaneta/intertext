"""Банк скореров «абзац (или предложение) против всех 31 102 стихов».

Вынесено из oracle_diagnostics.py, чтобы диагностика и обучаемое объединение
считали ровно одно и то же. Шесть скореров, из них три первых в конвейере
пилота отсутствовали:

  bm25        BM25 по леммам-униграммам -- порядок слов не важен
  tfidf_cos   косинус idf-взвешенных мешков лемм
  char4       косинус по символьным 4-граммам: морфология и
              церковнославянские формы без словаря
  ngram       лучшая точная n-грамма лемм (лексический слой пилота)
  edges       общие рёбра дерева разбора (синтаксический слой пилота)
  emb         статические векторы со SIF-взвешиванием (семантический слой)

Разделение на «точные» и «плотные» здесь не косметическое. Точные (ngram,
edges) дают ненулевой балл правильному стиху лишь в 11% и 34% случаев --
это их потолок полноты, никакой порог его не поднимет. Плотные (bm25,
tfidf_cos, char4, emb) видят правильный стих почти всегда, но ставят его в
среднем на тысячный ранг. Поэтому расти можно только за счёт плотных, и
только улучшая их ранжирование.
"""
from __future__ import annotations

import json
import math
import zlib

from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

BM25_K1, BM25_B = 1.5, 0.75
CHAR_N = 4
CHAR_BUCKETS = 1 << 20
SCORERS = ("bm25", "tfidf_cos", "char4", "ngram", "edges", "emb",
           "tfidf_rare", "skipbigram", "soft_edges", "dep_pair2", "runmatch")
SPARSE_SCORERS = ("ngram", "edges", "skipbigram", "soft_edges", "dep_pair2",
                  "runmatch")

# Три последних добавлены по результату ceiling_scan.py, а не по догадке:
#   tfidf_rare   мешок только редких лемм (df <= 2% стихов). Полнота ниже (0,75
#                против 0,98), но точность заметно выше: медиана ранга 530
#                против 920, а доля мест в топ-200 -- лучшая среди всех
#                вариантов, 0,28. Общий пласт частых слов у Достоевского и
#                Синодального перевода велик сам по себе и только шумит.
#   skipbigram   неупорядоченные пары лемм в окне 4. Полнота всего 0,30, зато
#                медиана ранга 54 из 31 102 -- самый точный вариант из всех.
#                Ловит формулу с переставленными словами, которую n-грамма
#                теряет, а мешок слов не отличает от общей темы.
#   soft_edges   пары лемм из дерева разбора БЕЗ типа связи. Строго лучше
#                edges: полнота 0,41 против 0,34 при том же ранге -- требование
#                совпадения ещё и типа связи отсекало и не добавляло точности.
#   dep_pair2    пары содержательных лемм, соединённых путём в дереве разбора
#                длины <= 2. По измерению потолка -- самый точный вариант из
#                всех: медиана ранга 39 из 31 102 при полноте 0,33, тогда как у
#                оконного skipbigram при той же полноте 47. Путь по дереву --
#                лучшее определение «связанных слов», чем близость в тексте.
#                Требует аннотации с топологией (nlp_annotate.py, поле heads).
#   runmatch     длина самого длинного дословного совпадения лемм подряд, БЕЗ
#                требования редкости. Добавлен по разбору недостижимых мест:
#                восемь мест эталона -- дословные цитаты, составленные целиком
#                из частых слов («Се человек», «Приидет в тот день», «но
#                приидите и вы», «Поди на перекрёсток»), и find_candidates.py
#                выбрасывает их ТРИЖДЫ: MIN_N=3 не пускает двусловные, индекс
#                требует редкой леммы в n-грамме, а MIN_SCORE=6.0 по сумме idf
#                частым словам недостижим. При этом три слова подряд из
#                Синодального перевода в русской прозе -- сильная улика сама по
#                себе, независимо от частотности этих слов.
RARE_DF_RATIO = 0.02
SKIP_WINDOW = 4
DEP_PAIR_DIST = 2
RUN_MIN_N, RUN_MAX_N = 2, 8


def load_ann(path: str):
    ids, lemmas, poss, edges, raw = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["id"])
            lemmas.append(r["lemmas"])
            poss.append(r.get("pos", ["X"] * len(r["lemmas"])))
            edges.append([tuple(e) for e in r["edges"]])
            raw.append(r.get("raw", ""))
    return ids, lemmas, poss, edges, raw


def _token_matrix(docs, vocab):
    indptr, indices, data = [0], [], []
    for lem in docs:
        c = Counter(l for l in lem if l in vocab)
        for l, n in c.items():
            indices.append(vocab[l])
            data.append(float(n))
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr),
                            shape=(len(docs), len(vocab)), dtype=np.float32)


def _char_matrix(texts, n=CHAR_N, buckets=CHAR_BUCKETS):
    indptr, indices, data = [0], [], []
    for t in texts:
        s = " " + " ".join(t.lower().split()) + " "
        # zlib.crc32, а не встроенный hash(): hash() для строк рандомизирован
        # при каждом запуске интерпретатора (PYTHONHASHSEED), и символьные
        # признаки получались невоспроизводимыми между прогонами
        c = Counter(zlib.crc32(s[i:i + n].encode("utf-8")) % buckets
                    for i in range(len(s) - n + 1))
        for h, k in c.items():
            indices.append(h)
            data.append(float(k))
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr),
                             shape=(len(texts), buckets), dtype=np.float32)


def _ngram_hash(parts) -> int:
    return zlib.crc32("\x00".join(parts).encode("utf-8"))


def _skipbigrams(lem: list[str], window: int) -> list[tuple[str, str]]:
    out = []
    for i in range(len(lem)):
        for j in range(i + 1, min(i + window, len(lem))):
            a, b = lem[i], lem[j]
            out.append((a, b) if a <= b else (b, a))
    return out


def _soft_edges(edges) -> list[tuple[str, str]]:
    return [(h, c) if h <= c else (c, h) for h, _, c in edges]


def _idf_from(m):
    n_docs = m.shape[0]
    df = np.asarray((m > 0).sum(axis=0)).ravel().astype(np.float64)
    return np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)


def _l2_rows(m):
    nrm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    return sparse.diags((1.0 / nrm).astype(np.float32)) @ m


class ScorerBank:
    """Готовит представления Библии один раз; потом score(...) отвечает
    вектором длины «число стихов» на любую единицу цели."""

    def __init__(self, bible_ann: str, extra_lemma_corpora: list[list[list[str]]],
                 bible_graph_ann: str | None = None):
        self.s_ids, s_lem, s_pos, s_edges, s_raw = load_ann(bible_ann)
        self.s_lem = s_lem
        n_src = len(self.s_ids)

        seen = {l for lem in s_lem for l in lem}
        for corpus in extra_lemma_corpora:
            seen |= {l for lem in corpus for l in lem}
        self.vocab = {l: i for i, l in enumerate(seen)}

        S_tok = _token_matrix(s_lem, self.vocab)
        self.idf_tok = _idf_from(S_tok)
        self.idf_plain = {l: float(self.idf_tok[i]) for l, i in self.vocab.items()}

        dl = np.asarray(S_tok.sum(axis=1)).ravel()
        avgdl = dl.mean() or 1.0
        coo = S_tok.tocoo(copy=True)
        denom = coo.data + BM25_K1 * (1 - BM25_B + BM25_B * dl[coo.row] / avgdl)
        coo.data = (coo.data * (BM25_K1 + 1) / denom * self.idf_tok[coo.col]).astype(np.float32)
        self.S_bm25_T = coo.tocsr().T.tocsr()

        self.S_tfidf_T = _l2_rows(S_tok @ sparse.diags(self.idf_tok)).T.tocsr()

        S_char = _char_matrix(s_raw)
        self.idf_char = _idf_from(S_char)
        self.S_char_T = _l2_rows(S_char @ sparse.diags(self.idf_char)).T.tocsr()

        self.src_index: dict[tuple, list[int]] = defaultdict(list)
        for j, lem in enumerate(s_lem):
            for n in range(3, 7):
                for i in range(len(lem) - n + 1):
                    self.src_index[tuple(lem[i:i + n])].append(j)

        # индекс для runmatch: все n-граммы 2..8 без отбора по редкости.
        # Хранится хеш, а не тюпл: записей около трёх миллионов, и словарь
        # тюплов строк съел бы гигабайты (тот же приём, что в
        # lexical_candidates_v2.py)
        self.run_index: dict[int, list[int]] = defaultdict(list)
        for j, lem in enumerate(s_lem):
            for n in range(RUN_MIN_N, RUN_MAX_N + 1):
                for i in range(len(lem) - n + 1):
                    self.run_index[_ngram_hash(lem[i:i + n])].append(j)

        edge_df: Counter = Counter()
        for e in s_edges:
            edge_df.update(set(e))
        self.n_src = n_src
        self.evocab = {e: i for i, e in enumerate(edge_df)}
        self.eidf = np.array([math.log(n_src / edge_df[e]) for e in self.evocab],
                             dtype=np.float32)
        cnt = [len(set(e)) for e in s_edges]
        self.S_edge = sparse.csr_matrix(
            ([1.0] * sum(cnt),
             [self.evocab[x] for e in s_edges for x in set(e)],
             np.cumsum([0] + cnt)),
            shape=(n_src, len(self.evocab)), dtype=np.float32)

        # --- редкие леммы: тот же порог, что у find_candidates.py
        df_lemma: Counter = Counter()
        for lem in s_lem:
            df_lemma.update(set(lem))
        self.rare = {l for l, c in df_lemma.items() if c / n_src <= RARE_DF_RATIO}
        rare_docs = [[l for l in lem if l in self.rare] for lem in s_lem]
        self.rare_vocab = {l: i for i, l in enumerate(self.rare)}
        S_rare = _token_matrix(rare_docs, self.rare_vocab)
        self.idf_rare = _idf_from(S_rare)
        self.S_rare_T = _l2_rows(S_rare @ sparse.diags(self.idf_rare)).T.tocsr()

        # --- неупорядоченные пары лемм в окне
        skip_docs = [_skipbigrams(lem, SKIP_WINDOW) for lem in s_lem]
        self.skip_vocab = {x: i for i, x in enumerate({x for d in skip_docs for x in d})}
        S_skip = _token_matrix(skip_docs, self.skip_vocab)
        self.idf_skip = _idf_from(S_skip)
        self.S_skip_T = _l2_rows(S_skip @ sparse.diags(self.idf_skip)).T.tocsr()

        # --- пары лемм из разбора без типа связи
        soft_docs = [_soft_edges(e) for e in s_edges]
        self.soft_vocab = {x: i for i, x in enumerate({x for d in soft_docs for x in d})}
        S_soft = _token_matrix(soft_docs, self.soft_vocab)
        self.idf_soft = _idf_from(S_soft)
        self.S_soft_T = _l2_rows(S_soft @ sparse.diags(self.idf_soft)).T.tocsr()

        # --- пары лемм, связанные путём в дереве разбора
        self.dep_vocab: dict = {}
        self.S_dep_T = None
        self.idf_dep = None
        if bible_graph_ann:
            from graph_features import dep_pairs, load_graph_ann
            g_ids, g_lem, g_pos, g_heads, _, _ = load_graph_ann(bible_graph_ann)
            assert g_ids == self.s_ids, "порядок стихов в графовой аннотации не совпал"
            docs = [dep_pairs(l, p, h, DEP_PAIR_DIST)
                    for l, p, h in zip(g_lem, g_pos, g_heads)]
            self.dep_vocab = {x: i for i, x in enumerate({x for d in docs for x in d})}
            S_dep = _token_matrix(docs, self.dep_vocab)
            self.idf_dep = _idf_from(S_dep)
            self.S_dep_T = _l2_rows(S_dep @ sparse.diags(self.idf_dep)).T.tocsr()

        self.s_pos, self.s_raw = s_pos, s_raw
        self._vec_ready = False

    def prepare_vectors(self, target_corpora: list[tuple[list, list, list]]):
        """Векторы Библии и цели: первая главная компонента вычитается по
        объединению, иначе она у двух корпусов разная и сравнение портится."""
        import spacy

        from semantic_static import build, drop_first_pc, normalize
        nlp = spacy.load("ru_core_news_lg", disable=["ner", "parser", "tagger"])
        freq: Counter = Counter()
        for lst in self.s_lem:
            freq.update(lst)
        for ids, lem, pos in target_corpora:
            for lst in lem:
                freq.update(lst)
        tot = sum(freq.values())

        self.S_vec = build(nlp, self.s_ids, self.s_lem, self.s_pos, freq, tot)
        self.T_vecs = [build(nlp, ids, lem, pos, freq, tot)
                       for ids, lem, pos in target_corpora]
        drop_first_pc([self.S_vec] + self.T_vecs)
        self.S_vec = normalize(self.S_vec)
        self.T_vecs = [normalize(v) for v in self.T_vecs]
        self._vec_ready = True

    def score(self, lem: list[str], raw: str, edges: list[tuple],
              vec: np.ndarray | None,
              dep_doc: list[tuple] | None = None) -> dict[str, np.ndarray]:
        q_tok = _token_matrix([lem], self.vocab)
        q_tfidf = _l2_rows(q_tok @ sparse.diags(self.idf_tok))
        q_char = _l2_rows(_char_matrix([raw]) @ sparse.diags(self.idf_char))

        ng = np.zeros(self.n_src, dtype=np.float32)
        for n in range(3, 7):
            for i in range(len(lem) - n + 1):
                hits = self.src_index.get(tuple(lem[i:i + n]))
                if not hits:
                    continue
                sc = sum(self.idf_plain.get(l, 0.0) for l in lem[i:i + n])
                for j in hits:
                    if sc > ng[j]:
                        ng[j] = sc

        erow = np.zeros(len(self.evocab), dtype=np.float32)
        for x in set(edges):
            k = self.evocab.get(x)
            if k is not None:
                erow[k] = self.eidf[k]

        out = {
            "bm25": np.asarray(((q_tok > 0).astype(np.float32) @ self.S_bm25_T).todense()).ravel(),
            "tfidf_cos": np.asarray((q_tfidf @ self.S_tfidf_T).todense()).ravel(),
            "char4": np.asarray((q_char @ self.S_char_T).todense()).ravel(),
            "ngram": ng,
            "edges": self.S_edge @ erow,
        }
        out["emb"] = (self.S_vec @ vec) if vec is not None else np.zeros(self.n_src, np.float32)

        def cos_against(docs, vocab, idf, ST):
            q = _l2_rows(_token_matrix([docs], vocab) @ sparse.diags(idf))
            return np.asarray((q @ ST).todense()).ravel()

        out["tfidf_rare"] = cos_against([l for l in lem if l in self.rare],
                                        self.rare_vocab, self.idf_rare, self.S_rare_T)
        out["skipbigram"] = cos_against(_skipbigrams(lem, SKIP_WINDOW),
                                        self.skip_vocab, self.idf_skip, self.S_skip_T)
        out["soft_edges"] = cos_against(_soft_edges(edges),
                                        self.soft_vocab, self.idf_soft, self.S_soft_T)
        out["dep_pair2"] = (cos_against(dep_doc, self.dep_vocab, self.idf_dep,
                                        self.S_dep_T)
                            if (self.S_dep_T is not None and dep_doc is not None)
                            else np.zeros(self.n_src, np.float32))

        # длина самого длинного дословного совпадения: идём от коротких к
        # длинным и переписываем максимум, так что в итоге у каждого стиха
        # стоит длина наибольшего совпавшего отрезка
        run = np.zeros(self.n_src, dtype=np.float32)
        for n in range(RUN_MIN_N, RUN_MAX_N + 1):
            if len(lem) < n:
                break
            for i in range(len(lem) - n + 1):
                hits = self.run_index.get(_ngram_hash(lem[i:i + n]))
                if hits:
                    run[hits] = np.maximum(run[hits], float(n))
        out["runmatch"] = run
        return out
