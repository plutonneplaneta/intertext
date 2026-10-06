"""Диагностика: где стоит правильный стих, если не отсекать ничего.

Все три слоя оценивались по полноте «нашёл или не нашёл» -- то есть по тому,
попал ли правильный стих в список кандидатов после порогов и индексации.
37,5% у объединения слоёв значит, что 40 из 64 документированных мест не
находит ни один слой. Но из этой цифры не видно, где потеря: порог отсёк
слабое совпадение, индекс до него не дошёл, или сигнала нет вовсе.

Здесь каждый скорер считается ИСЧЕРПЫВАЮЩЕ: 31 102 стиха против абзаца, без
порогов, без индекса по редким леммам, без топ-K. Ранг правильного стиха в
полном списке -- это потолок слоя. Если ранг 3, а слой его не нашёл -- виноват
порог. Если ранг 20 000 -- сигнала в этом слое нет, и порог тут ни при чём.

Скореры (первые три в конвейере отсутствуют вовсе, и это, похоже, главная
дыра -- существующая лексика требует трёх подряд идущих лемм):

  bm25        BM25 по леммам-униграммам: порядок слов не важен, вес по
              редкости, длина стиха учтена нормировкой
  tfidf_cos   косинус idf-взвешенных мешков лемм
  char4       косинус по символьным 4-граммам: ловит морфологию и
              церковнославянские формы без словаря («руце» и «рука»
              делят «рук»)
  ngram       лучшая точная n-грамма лемм (текущая лексика, но без MIN_SCORE
              и без индекса)
  edges       общие рёбра разбора (текущий синтаксис, без порога)
  emb         статические векторы (текущая семантика)

Каждый считается и по абзацу, и по максимуму среди предложений абзаца.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import zlib

from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

BM25_K1, BM25_B = 1.5, 0.75
CHAR_N = 4
CHAR_BUCKETS = 1 << 20


def load_ann(path: str):
    ids, lemmas, edges, raw = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["id"])
            lemmas.append(r["lemmas"])
            edges.append([tuple(e) for e in r["edges"]])
            raw.append(r.get("raw", ""))
    return ids, lemmas, edges, raw


def load_gold(path: str, min_match: float = 0.6):
    """[(novel_id, [(книга, глава, от, до), ...]), ...] -- по одной записи на
    примечание Тихомирова, а не на абзац: у одного абзаца их бывает несколько."""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row["novel_verse_id"] or float(row["match_score"]) < min_match:
                continue
            group = []
            for ref in row["bible_refs"].split(";"):
                if not ref:
                    continue
                book, chap, verses = ref.split(".", 2)
                if "-" in verses:
                    vf, vt = map(int, verses.split("-"))
                else:
                    vf = vt = int(verses)
                group.append((book, int(chap), vf, vt))
            if group:
                out.append((row["novel_verse_id"], group, row["fragment"]))
    return out


def verse_matches(verse_id: str, group) -> bool:
    p = verse_id.split(".")
    if len(p) != 4:
        return False
    _, book, chap, verse = p
    try:
        chap, verse = int(chap), int(verse)
    except ValueError:
        return False
    return any(book == b and chap == c and vf <= verse <= vt for b, c, vf, vt in group)


# ------------------------------------------------------- построение матриц


def token_matrix(docs: list[list[str]], vocab: dict[str, int]) -> sparse.csr_matrix:
    indptr, indices, data = [0], [], []
    for lem in docs:
        c = Counter(l for l in lem if l in vocab)
        for l, n in c.items():
            indices.append(vocab[l])
            data.append(float(n))
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr),
                            shape=(len(docs), len(vocab)), dtype=np.float32)


def char_matrix(texts: list[str], n: int = CHAR_N,
                buckets: int = CHAR_BUCKETS) -> sparse.csr_matrix:
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


def idf_from(m: sparse.csr_matrix) -> np.ndarray:
    n_docs = m.shape[0]
    df = np.asarray((m > 0).sum(axis=0)).ravel().astype(np.float64)
    return np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)


def l2_rows(m: sparse.csr_matrix) -> sparse.csr_matrix:
    nrm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    return sparse.diags((1.0 / nrm).astype(np.float32)) @ m


def bm25_doc_matrix(m: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    """Стихи в BM25-представлении: вес термина в стихе уже включает idf и
    нормировку на длину стиха, так что оценка запроса -- обычное произведение."""
    dl = np.asarray(m.sum(axis=1)).ravel()
    avgdl = dl.mean() or 1.0
    out = m.tocoo(copy=True)
    denom = out.data + BM25_K1 * (1 - BM25_B + BM25_B * dl[out.row] / avgdl)
    out.data = (out.data * (BM25_K1 + 1) / denom * idf[out.col]).astype(np.float32)
    return out.tocsr()


def edge_matrices(src_edges, tgt_edges):
    df: Counter = Counter()
    for e in src_edges:
        df.update(set(e))
    for e in tgt_edges:
        df.update(set(e))
    n_docs = len(src_edges) + len(tgt_edges)
    vocab = {e: i for i, e in enumerate(df)}
    idf = np.array([math.log(n_docs / df[e]) for e in vocab], dtype=np.float32)

    def build(edge_lists):
        indptr, indices, data = [0], [], []
        for e in edge_lists:
            for x in set(e):
                indices.append(vocab[x])
                data.append(1.0)
            indptr.append(len(indices))
        return sparse.csr_matrix((data, indices, indptr),
                                 shape=(len(edge_lists), len(vocab)), dtype=np.float32)

    S, T = build(src_edges), build(tgt_edges)
    W = sparse.diags(idf)
    return (T @ W @ S.T).tocsr() if False else (T @ W, S.T)


def best_ngram_scores(tgt_lemmas: list[str], src_lemmas: list[list[str]],
                      idf: dict[str, float], src_index, min_n=3, max_n=6) -> np.ndarray:
    """Лучшая точная n-грамма на каждый стих -- без MIN_SCORE и без требования
    редкой леммы в n-грамме (то есть потолок текущего лексического слоя)."""
    out = np.zeros(len(src_lemmas), dtype=np.float32)
    for n in range(min_n, max_n + 1):
        for i in range(len(tgt_lemmas) - n + 1):
            ng = tuple(tgt_lemmas[i:i + n])
            hits = src_index.get(ng)
            if not hits:
                continue
            sc = sum(idf.get(l, 0.0) for l in ng)
            for j in hits:
                if sc > out[j]:
                    out[j] = sc
    return out


def main(bible_ann: str, novel_ann: str, novel_sent_ann: str, gold_tsv: str,
         out_tsv: str) -> None:
    s_ids, s_lem, s_edges, s_raw = load_ann(bible_ann)
    t_ids, t_lem, t_edges, t_raw = load_ann(novel_ann)
    q_ids, q_lem, q_edges, q_raw = load_ann(novel_sent_ann)
    print(f"стихов {len(s_ids)}, абзацев {len(t_ids)}, предложений {len(q_ids)}",
          file=sys.stderr)

    gold = load_gold(gold_tsv)
    gold_pars = sorted({g[0] for g in gold})
    t_pos = {t: i for i, t in enumerate(t_ids)}
    sent_of_par = defaultdict(list)
    for i, q in enumerate(q_ids):
        sent_of_par[q.split("#", 1)[0]].append(i)

    # правильные стихи для каждой записи эталона
    correct = []
    for par, group, frag in gold:
        idx = [j for j, sid in enumerate(s_ids) if verse_matches(sid, group)]
        correct.append(idx)
    print(f"записей эталона {len(gold)}, "
          f"без найденных стихов в корпусе: {sum(1 for c in correct if not c)}",
          file=sys.stderr)

    vocab = {l: i for i, l in enumerate(
        {l for lem in s_lem for l in lem} | {l for lem in t_lem for l in lem})}
    S_tok = token_matrix(s_lem, vocab)
    idf_tok = idf_from(S_tok)
    S_bm25 = bm25_doc_matrix(S_tok, idf_tok)
    S_tfidf = l2_rows(S_tok @ sparse.diags(idf_tok))

    S_char = char_matrix(s_raw)
    idf_char = idf_from(S_char)
    S_char_n = l2_rows(S_char @ sparse.diags(idf_char))

    # транспонированные копии считаются один раз: внутри цикла по 64 местам
    # эталона .T.tocsc() на матрице 31102 x 38000 стоил дороже самих оценок
    S_bm25_T = S_bm25.T.tocsr()
    S_tfidf_T = S_tfidf.T.tocsr()
    S_char_T = S_char_n.T.tocsr()

    # индекс точных n-грамм источника (для потолка лексического слоя)
    src_index: dict[tuple, list[int]] = defaultdict(list)
    for j, lem in enumerate(s_lem):
        for n in range(3, 7):
            for i in range(len(lem) - n + 1):
                src_index[tuple(lem[i:i + n])].append(j)
    idf_plain = {l: float(idf_tok[vocab[l]]) for l in vocab}

    # рёбра
    edge_df: Counter = Counter()
    for e in s_edges:
        edge_df.update(set(e))
    for e in t_edges:
        edge_df.update(set(e))
    n_docs_e = len(s_edges) + len(t_edges)
    evocab = {e: i for i, e in enumerate(edge_df)}
    eidf = np.array([math.log(n_docs_e / edge_df[e]) for e in evocab], dtype=np.float32)

    def edge_row(edges):
        v = np.zeros(len(evocab), dtype=np.float32)
        for x in set(edges):
            k = evocab.get(x)
            if k is not None:
                v[k] = eidf[k]
        return v

    S_edge = sparse.csr_matrix(
        ([1.0] * sum(len(set(e)) for e in s_edges),
         [evocab[x] for e in s_edges for x in set(e)],
         np.cumsum([0] + [len(set(e)) for e in s_edges])),
        shape=(len(s_edges), len(evocab)), dtype=np.float32)

    # семантика: статические векторы
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import spacy
    from semantic_static import SIF_A, SKIP_POS, build, drop_first_pc, normalize
    nlp = spacy.load("ru_core_news_lg", disable=["ner", "parser", "tagger"])
    freq: Counter = Counter()
    for lst in s_lem:
        freq.update(lst)
    for lst in t_lem:
        freq.update(lst)
    tot = sum(freq.values())

    def poss_of(path):
        return [json.loads(l)["pos"] for l in open(path, encoding="utf-8") if l.strip()]
    S_vec = build(nlp, s_ids, s_lem, poss_of(bible_ann), freq, tot)
    T_vec = build(nlp, t_ids, t_lem, poss_of(novel_ann), freq, tot)
    Q_vec = build(nlp, q_ids, q_lem, poss_of(novel_sent_ann), freq, tot)
    drop_first_pc([S_vec, T_vec, Q_vec])
    S_vec, T_vec, Q_vec = normalize(S_vec), normalize(T_vec), normalize(Q_vec)

    SCORERS = ["bm25", "tfidf_cos", "char4", "ngram", "edges", "emb"]
    rows = []

    def scores_for(lem, raw, edges, vec) -> dict[str, np.ndarray]:
        q_tok = token_matrix([lem], vocab)
        q_tfidf = l2_rows(q_tok @ sparse.diags(idf_tok))
        q_char = l2_rows(char_matrix([raw]) @ sparse.diags(idf_char))
        return {
            "bm25": np.asarray(((q_tok > 0).astype(np.float32) @ S_bm25_T).todense()).ravel(),
            "tfidf_cos": np.asarray((q_tfidf @ S_tfidf_T).todense()).ravel(),
            "char4": np.asarray((q_char @ S_char_T).todense()).ravel(),
            "ngram": best_ngram_scores(lem, s_lem, idf_plain, src_index),
            "edges": S_edge @ edge_row(edges),
            "emb": S_vec @ vec,
        }

    for k, (par, group, frag) in enumerate(gold):
        if not correct[k]:
            continue
        ti = t_pos[par]
        par_sc = scores_for(t_lem[ti], t_raw[ti], t_edges[ti], T_vec[ti])

        sent_sc = {name: np.zeros(len(s_ids), dtype=np.float32) for name in SCORERS}
        for qi in sent_of_par.get(par, []):
            sc = scores_for(q_lem[qi], q_raw[qi], q_edges[qi], Q_vec[qi])
            for name in SCORERS:
                np.maximum(sent_sc[name], sc[name], out=sent_sc[name])

        rec = {"gold_id": k, "par": par, "fragment": frag[:40],
               "n_correct_verses": len(correct[k])}
        for name in SCORERS:
            for unit, sc in (("par", par_sc[name]), ("sent", sent_sc[name])):
                best = sc[correct[k]].max()
                # средний ранг среди связок. У разреженных скореров (n-граммы,
                # рёбра) почти все стихи получают ровно 0, и «число строго
                # больших плюс один» дало бы нулевому баллу ранг 50 при 31 тысяче
                # нулей -- то есть выдало бы «сигнала нет» за «ранг 50».
                greater = int((sc > best).sum())
                equal = int((sc == best).sum())
                rec[f"{name}_{unit}"] = greater + (equal + 1) / 2.0
                rec[f"{name}_{unit}_hit"] = int(best > 0)
        rows.append(rec)
        if (k + 1) % 10 == 0:
            print(f"{k+1}/{len(gold)}", file=sys.stderr, flush=True)

    cols = (["gold_id", "par", "fragment", "n_correct_verses"]
            + [f"{n}_{u}{suf}" for n in SCORERS for u in ("par", "sent")
               for suf in ("", "_hit")])
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"\nранг правильного стиха среди {len(s_ids)} без порогов и индекса, "
          f"{len(rows)} мест эталона\n")
    print(f"{'скорер':14s} {'виден':>6s} {'медиана':>8s} {'кв.25':>7s} {'кв.75':>7s} "
          f"{'топ-10':>7s} {'топ-50':>7s} {'топ-200':>8s} {'топ-1000':>9s}")
    print("(«виден» -- у скольких мест правильный стих получил ненулевой балл;\n"
          " ранги ниже считаны только по этим местам, иначе они ни о чём)")
    for name in SCORERS:
        for unit in ("par", "sent"):
            hit = np.array([r[f"{name}_{unit}_hit"] for r in rows], dtype=bool)
            v = np.array([r[f"{name}_{unit}"] for r in rows], dtype=float)[hit]
            label = f"{name}/{'абзац' if unit == 'par' else 'предл'}"
            if len(v) == 0:
                print(f"{label:14s} {0.0:6.2f}")
                continue
            print(f"{label:14s} {hit.mean():6.2f} {np.median(v):8.0f} "
                  f"{np.percentile(v,25):7.0f} {np.percentile(v,75):7.0f} "
                  f"{np.mean(v<=10)*hit.mean():7.2f} {np.mean(v<=50)*hit.mean():7.2f} "
                  f"{np.mean(v<=200)*hit.mean():8.2f} {np.mean(v<=1000)*hit.mean():9.2f}")
    print("\nтоп-k -- доля ВСЕХ мест эталона (не только видимых), чтобы числа\n"
          "сравнивались между скорерами напрямую")
    print(f"\n-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
