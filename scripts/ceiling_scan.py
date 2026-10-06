"""Сканирование потолка полноты по многим вариантам признака сразу.

Урок предыдущей итерации: у признака есть предел полноты, и он измеряется до
всякой настройки. У n-грамм лемм он 11%, у рёбер разбора 34% -- пилот полтора
шага настраивал пороги внутри этих границ. Здесь та же мерка прикладывается к
вариантам, которых в конвейере нет, ЧТОБЫ РЕШИТЬ, что вообще стоит строить.

Мерится два числа на вариант:
  «виден»   доля мест эталона, где правильный стих получил ненулевой балл --
            предел полноты, выше которого вариант не поднимется никакой
            настройкой;
  ранг      где он стоит среди 31 102 стихов, когда виден -- сколько работы
            остаётся ранжированию.

Плюс кривая «размер шортлиста -> достижимость» для объединений: сколько
кандидатов на абзац надо отдать эксперту, чтобы правильный стих там был.

Всё считается партиями (матрица предложений на матрицу стихов), а не по одной
единице: поштучно те же шестнадцать вариантов считались бы часами.

Варианты (кроме шести уже известных):
  tfidf_rare      только редкие леммы (df <= 2% стихов): частый пласт слов,
                  общий у Достоевского и Синодального перевода, исключён
  tfidf_content   без служебных частей речи
  prefix5         пятибуквенные начала содержательных лемм -- грубое
                  стемминг-приближение, устойчивое к архаичным окончаниям
                  («возглаголють» и «глаголать» делят «возгл»/«глаго»)
  bigram          мешок биграмм лемм с весом по редкости: двусловные формулы,
                  которые MIN_N=3 у пилота теряет
  skipbigram      неупорядоченные пары лемм в окне 4 -- порядок не важен,
                  но связь остаётся локальной
  soft_edges      пары лемм из дерева разбора БЕЗ типа связи: пилот требовал
                  совпадения и связи тоже
  window3         косинус против окна из трёх соседних стихов: цитата часто
                  приходится на стык
  char4_bm25      BM25 по символьным 4-граммам вместо косинуса
  names           только имена собственные с весом по редкости
  emb_maxpool     максимум по измерениям векторов токенов вместо среднего
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from oracle_diagnostics import load_gold, verse_matches
from retrieval_scorers import (BM25_B, BM25_K1, CHAR_BUCKETS, CHAR_N, _idf_from,
                               _l2_rows, load_ann)

FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT"}
RARE_DF_RATIO = 0.02
CHUNK = 256


# ------------------------------------------------- построители представлений


def _bag(docs: list[list], vocab: dict) -> sparse.csr_matrix:
    indptr, indices, data = [0], [], []
    for d in docs:
        c = Counter(x for x in d if x in vocab)
        for x, n in c.items():
            indices.append(vocab[x])
            data.append(float(n))
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr),
                            shape=(len(docs), len(vocab)), dtype=np.float32)


def _vocab_of(*corpora) -> dict:
    seen = set()
    for c in corpora:
        for d in c:
            seen.update(d)
    return {x: i for i, x in enumerate(seen)}


def lemmas_content(lem, pos):
    return [l for l, p in zip(lem, pos) if p not in FUNCTION_POS]


def prefix5(lem, pos):
    return [l[:5] for l in lemmas_content(lem, pos) if len(l) >= 3]


def bigrams(lem):
    return [(lem[i], lem[i + 1]) for i in range(len(lem) - 1)]


def ngrams36(lem):
    return [tuple(lem[i:i + n]) for n in range(3, 7)
            for i in range(len(lem) - n + 1)]


def skipbigrams(lem, window=4):
    out = []
    for i in range(len(lem)):
        for j in range(i + 1, min(i + window, len(lem))):
            a, b = lem[i], lem[j]
            out.append((a, b) if a <= b else (b, a))
    return out


def soft_edges(edges):
    return [(h, c) if h <= c else (c, h) for h, _, c in edges]


def propn(lem, pos):
    return [l for l, p in zip(lem, pos) if p == "PROPN"]


def char_bag(texts, n=CHAR_N, buckets=CHAR_BUCKETS):
    docs = []
    for t in texts:
        s = " " + " ".join(t.lower().split()) + " "
        docs.append([zlib.crc32(s[i:i + n].encode("utf-8")) % buckets
                     for i in range(len(s) - n + 1)])
    return docs


def bm25_rows(m: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    dl = np.asarray(m.sum(axis=1)).ravel()
    avgdl = dl.mean() or 1.0
    coo = m.tocoo(copy=True)
    denom = coo.data + BM25_K1 * (1 - BM25_B + BM25_B * dl[coo.row] / avgdl)
    coo.data = (coo.data * (BM25_K1 + 1) / denom * idf[coo.col]).astype(np.float32)
    return coo.tocsr()


def window_rows(m: sparse.csr_matrix, ids: list[str], radius: int = 1) -> sparse.csr_matrix:
    """Каждая строка -- сумма строк соседних стихов той же главы."""
    key = [".".join(i.split(".")[:3]) for i in ids]
    rows_i, rows_j = [], []
    for j in range(len(ids)):
        for d in range(-radius, radius + 1):
            k = j + d
            if 0 <= k < len(ids) and key[k] == key[j]:
                rows_i.append(j)
                rows_j.append(k)
    A = sparse.csr_matrix((np.ones(len(rows_i), dtype=np.float32), (rows_i, rows_j)),
                          shape=(len(ids), len(ids)))
    return (A @ m).tocsr()


def emb_maxpool(nlp, lemmas, poss) -> np.ndarray:
    dim = nlp.vocab.vectors.shape[1]
    out = np.zeros((len(lemmas), dim), dtype=np.float32)
    for i, (lem, pos) in enumerate(zip(lemmas, poss)):
        vecs = [nlp.vocab[l].vector for l, p in zip(lem, pos)
                if p not in FUNCTION_POS and nlp.vocab[l].has_vector]
        if vecs:
            out[i] = np.max(np.stack(vecs), axis=0)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


# ------------------------------------------------------------------- прогон


def add_graph_variants(variants, pair, bible_graph_ann, sent_graph_ann,
                       s_ids, q_ids, s_lem, s_pos, q_lem, q_pos):
    """Признаки, использующие топологию дерева разбора, а не отдельное ребро.
    Требуют аннотации с индексами (nlp_annotate.py, поля heads/rels/adp_of):
    по тройкам лемм путь между двумя словами восстановить нельзя."""
    from graph_features import (build_cooc_neighbours, collapsed_edges,
                                cooc_expand, dep_pairs, dep_path2,
                                load_graph_ann)

    gs_ids, gs_lem, gs_pos, gs_heads, gs_rels, gs_adp = load_graph_ann(bible_graph_ann)
    gq_ids, gq_lem, gq_pos, gq_heads, gq_rels, gq_adp = load_graph_ann(sent_graph_ann)
    # порядок строк в графовой аннотации тот же, что в обычной (тот же вход),
    # но проверим -- иначе признаки поедут относительно эталона
    assert gs_ids == s_ids, "порядок стихов в графовой аннотации не совпал"
    q_index = {q: i for i, q in enumerate(gq_ids)}
    sel = [q_index[q] for q in q_ids]
    gq_lem = [gq_lem[i] for i in sel]
    gq_pos = [gq_pos[i] for i in sel]
    gq_heads = [gq_heads[i] for i in sel]
    gq_rels = [gq_rels[i] for i in sel]
    gq_adp = [gq_adp[i] for i in sel]

    for dist in (2, 3):
        variants[f"dep_pair{dist}"] = pair(
            [dep_pairs(l, p, h, dist) for l, p, h in zip(gs_lem, gs_pos, gs_heads)],
            [dep_pairs(l, p, h, dist) for l, p, h in zip(gq_lem, gq_pos, gq_heads)])
    variants["dep_path2"] = pair(
        [dep_path2(l, p, h) for l, p, h in zip(gs_lem, gs_pos, gs_heads)],
        [dep_path2(l, p, h) for l, p, h in zip(gq_lem, gq_pos, gq_heads)])
    variants["collapsed"] = pair(
        [collapsed_edges(l, p, h, r, a) for l, p, h, r, a
         in zip(gs_lem, gs_pos, gs_heads, gs_rels, gs_adp)],
        [collapsed_edges(l, p, h, r, a) for l, p, h, r, a
         in zip(gq_lem, gq_pos, gq_heads, gq_rels, gq_adp)])

    neigh = build_cooc_neighbours(gs_lem, gs_pos)
    variants["cooc_expand"] = pair(
        [cooc_expand(l, p, {}) for l, p in zip(gs_lem, gs_pos)],
        [cooc_expand(l, p, neigh) for l, p in zip(gq_lem, gq_pos)])
    print(f"граф совместной встречаемости: {len(neigh)} лемм с соседями",
          file=sys.stderr)


def main(bible_ann: str, novel_sent_ann: str, gold_tsv: str, out_tsv: str,
         bible_graph_ann: str | None = None,
         sent_graph_ann: str | None = None) -> None:
    s_ids, s_lem, s_pos, s_edges, s_raw = load_ann(bible_ann)
    q_ids_all, q_lem_all, q_pos_all, q_edges_all, q_raw_all = load_ann(novel_sent_ann)
    gold = load_gold(gold_tsv)
    gold_pars = sorted({g[0] for g in gold})
    keep = [i for i, q in enumerate(q_ids_all) if q.split("#", 1)[0] in set(gold_pars)]
    q_ids = [q_ids_all[i] for i in keep]
    q_lem = [q_lem_all[i] for i in keep]
    q_pos = [q_pos_all[i] for i in keep]
    q_edges = [q_edges_all[i] for i in keep]
    q_raw = [q_raw_all[i] for i in keep]
    par_of_sent = np.array([gold_pars.index(q.split("#", 1)[0]) for q in q_ids])
    n_src, n_par = len(s_ids), len(gold_pars)
    print(f"стихов {n_src}, эталонных абзацев {n_par}, их предложений {len(q_ids)}",
          file=sys.stderr)

    correct = [{j for j, sid in enumerate(s_ids) if verse_matches(sid, g[1])}
               for g in gold]
    par_of_rec = [gold_pars.index(g[0]) for g in gold]

    # --- определения вариантов: (имя, док-представление стихов, запросов)
    def pair(sdocs, qdocs, mode="cos"):
        vocab = _vocab_of(sdocs, qdocs)
        S, Q = _bag(sdocs, vocab), _bag(qdocs, vocab)
        idf = _idf_from(S)
        if mode == "cos":
            return _l2_rows(S @ sparse.diags(idf)), _l2_rows(Q @ sparse.diags(idf))
        if mode == "bm25":
            return bm25_rows(S, idf), (Q > 0).astype(np.float32)
        raise ValueError(mode)

    s_content = [lemmas_content(l, p) for l, p in zip(s_lem, s_pos)]
    q_content = [lemmas_content(l, p) for l, p in zip(q_lem, q_pos)]

    df_lemma = Counter()
    for l in s_lem:
        df_lemma.update(set(l))
    rare = {l for l, c in df_lemma.items() if c / n_src <= RARE_DF_RATIO}

    variants: dict[str, tuple] = {}
    variants["tfidf_cos"] = pair(s_lem, q_lem)
    variants["bm25"] = pair(s_lem, q_lem, "bm25")
    variants["tfidf_rare"] = pair([[x for x in d if x in rare] for d in s_lem],
                                  [[x for x in d if x in rare] for d in q_lem])
    variants["tfidf_content"] = pair(s_content, q_content)
    variants["prefix5"] = pair([prefix5(l, p) for l, p in zip(s_lem, s_pos)],
                               [prefix5(l, p) for l, p in zip(q_lem, q_pos)])
    variants["bigram"] = pair([bigrams(l) for l in s_lem], [bigrams(l) for l in q_lem])
    # мешок n-грамм 3..6 -- та же достижимость, что у «лучшей точной n-граммы»
    # пилота (балл ненулевой тогда же), нужен, чтобы строка «пилот» в кривой
    # пула считалась по всем трём его слоям, а не по двум
    variants["ngram_bag"] = pair([ngrams36(l) for l in s_lem],
                                 [ngrams36(l) for l in q_lem])
    variants["skipbigram"] = pair([skipbigrams(l) for l in s_content],
                                  [skipbigrams(l) for l in q_content])
    variants["edges"] = pair([[tuple(e) for e in d] for d in s_edges],
                             [[tuple(e) for e in d] for d in q_edges])
    variants["soft_edges"] = pair([soft_edges(d) for d in s_edges],
                                  [soft_edges(d) for d in q_edges])
    variants["names"] = pair([propn(l, p) for l, p in zip(s_lem, s_pos)],
                             [propn(l, p) for l, p in zip(q_lem, q_pos)])
    variants["char4"] = pair(char_bag(s_raw), char_bag(q_raw))
    variants["char4_bm25"] = pair(char_bag(s_raw), char_bag(q_raw), "bm25")

    # окно из трёх стихов: считается на непронормированной матрице, потом норма
    vocab_w = _vocab_of(s_lem, q_lem)
    S_raw_bag = _bag(s_lem, vocab_w)
    idf_w = _idf_from(S_raw_bag)
    S_win = window_rows(S_raw_bag @ sparse.diags(idf_w), s_ids, radius=1)
    variants["window3"] = (_l2_rows(S_win),
                           _l2_rows(_bag(q_lem, vocab_w) @ sparse.diags(idf_w)))

    import spacy
    from semantic_static import build, drop_first_pc, normalize
    nlp = spacy.load("ru_core_news_lg", disable=["ner", "parser", "tagger"])
    freq = Counter()
    for l in s_lem:
        freq.update(l)
    for l in q_lem:
        freq.update(l)
    tot = sum(freq.values())
    S_v = build(nlp, s_ids, s_lem, s_pos, freq, tot)
    Q_v = build(nlp, q_ids, q_lem, q_pos, freq, tot)
    drop_first_pc([S_v, Q_v])
    variants["emb"] = (sparse.csr_matrix(normalize(S_v)), sparse.csr_matrix(normalize(Q_v)))
    variants["emb_maxpool"] = (sparse.csr_matrix(emb_maxpool(nlp, s_lem, s_pos)),
                               sparse.csr_matrix(emb_maxpool(nlp, q_lem, q_pos)))

    if bible_graph_ann and sent_graph_ann:
        add_graph_variants(variants, pair, bible_graph_ann, sent_graph_ann,
                           s_ids, q_ids, s_lem, s_pos, q_lem, q_pos)

    # --- счёт: максимум по предложениям абзаца
    par_scores: dict[str, np.ndarray] = {}
    for name, (S, Q) in variants.items():
        agg = np.zeros((n_par, n_src), dtype=np.float32)
        ST = S.T.tocsr()
        for a in range(0, Q.shape[0], CHUNK):
            block = np.asarray((Q[a:a + CHUNK] @ ST).todense(), dtype=np.float32)
            for r in range(block.shape[0]):
                p = par_of_sent[a + r]
                np.maximum(agg[p], block[r], out=agg[p])
        par_scores[name] = agg
        print(f"{name}: готово", file=sys.stderr, flush=True)

    # --- отчёт
    names_sorted = list(variants)
    rows = []
    for k, g in enumerate(gold):
        if not correct[k]:
            continue
        rec = {"rec": k, "par": g[0]}
        for name in names_sorted:
            sc = par_scores[name][par_of_rec[k]]
            best = sc[list(correct[k])].max()
            greater = int((sc > best).sum())
            equal = int((sc == best).sum())
            rec[f"{name}_rank"] = greater + (equal + 1) / 2.0
            rec[f"{name}_hit"] = int(best > 0)
        rows.append(rec)

    print(f"\nпотолок полноты по {len(rows)} местам эталона, {n_src} стихов, "
          f"единица -- предложение\n")
    print(f"{'вариант':16s} {'виден':>6s} {'медиана':>8s} {'кв.25':>7s} {'кв.75':>7s} "
          f"{'топ-50':>7s} {'топ-200':>8s} {'топ-1000':>9s}")
    summary = []
    for name in names_sorted:
        hit = np.array([r[f"{name}_hit"] for r in rows], dtype=bool)
        v = np.array([r[f"{name}_rank"] for r in rows], dtype=float)[hit]
        if len(v) == 0:
            print(f"{name:16s} {0.0:6.2f}")
            continue
        summary.append((name, hit.mean(), np.median(v),
                        np.mean(v <= 200) * hit.mean()))
        print(f"{name:16s} {hit.mean():6.2f} {np.median(v):8.0f} "
              f"{np.percentile(v,25):7.0f} {np.percentile(v,75):7.0f} "
              f"{np.mean(v<=50)*hit.mean():7.2f} {np.mean(v<=200)*hit.mean():8.2f} "
              f"{np.mean(v<=1000)*hit.mean():9.2f}")

    # --- кривая «размер шортлиста -> достижимость» для объединений
    sets = {
        "пилот: n-граммы + рёбра + векторы": ["ngram_bag", "edges", "emb"],
        "шесть из третьей итерации": ["bm25", "tfidf_cos", "char4",
                                      "ngram_bag", "edges", "emb"],
        "девять (плюс три найденных здесь)": ["bm25", "tfidf_cos", "char4",
                                              "ngram_bag", "edges", "emb",
                                              "tfidf_rare", "skipbigram", "soft_edges"],
        "все варианты": names_sorted,
        "девять + графовые": [n for n in ["bm25", "tfidf_cos", "char4", "ngram_bag",
                                          "edges", "emb", "tfidf_rare", "skipbigram",
                                          "soft_edges", "dep_pair3", "collapsed"]
                              if n in variants],
        "лучшая пятёрка по топ-200": [n for n, _, _, _ in
                                      sorted(summary, key=lambda x: -x[3])[:5]],
    }
    print("\nдостижимость правильного стиха при шортлисте из топ-K на скорер:")
    print(f"{'набор':32s} " + " ".join(f"K={k:<6d}" for k in (50, 100, 300, 1000, 3000)))
    for label, members in sets.items():
        line = []
        for K in (50, 100, 300, 1000, 3000):
            tops = {}
            for name in members:
                sc = par_scores[name]
                tops[name] = np.argpartition(-sc, min(K, n_src - 1), axis=1)[:, :K]
            ok = 0
            for k, g in enumerate(gold):
                if not correct[k]:
                    continue
                pool = set()
                for name in members:
                    pool.update(tops[name][par_of_rec[k]].tolist())
                ok += int(bool(pool & correct[k]))
            line.append(ok / len(rows))
        print(f"{label:32s} " + " ".join(f"{v:<8.2f}" for v in line)
              + f"  (<= {len(members)}K кандидатов)")

    cols = ["rec", "par"] + [f"{n}_{s}" for n in names_sorted for s in ("hit", "rank")]
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
