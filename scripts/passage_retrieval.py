"""Пункт 2 из списка «мерить следующим»: пассаж как единица извлечения.

Зачем. Эталон Тихомирова в 28 случаях из 64 указывает диапазон стихов, то есть
единица цитирования у него -- не стих, а пассаж. Проверенный ранее window3 это
НЕ тот тест: он оценивал стихи содержимым окна, а извлекал по-прежнему стихи.
Здесь документом становится сам пассаж: окно из k подряд идущих стихов одной
главы.

Как сравнивать честно. Ранги при разном размере окна несравнимы напрямую:
пассажей меньше, чем стихов, поэтому ранг механически лучше. Поэтому мерится
цена для человека -- сколько РАЗЛИЧНЫХ СТИХОВ придётся прочесть, чтобы в
шортлисте оказалось правильное место. Это единственная величина, в которой
стих и пассаж сопоставимы.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from oracle_diagnostics import load_gold, verse_matches
from retrieval_scorers import (_idf_from, _l2_rows, _soft_edges, _token_matrix,
                               load_ann)

WINDOWS = (1, 2, 3, 5)
RARE_DF_RATIO = 0.02
FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT"}
CHUNK = 256
COST_POINTS = (50, 100, 200, 500, 1000, 2000, 5000)


def build_passages(s_ids: list[str], k: int) -> list[list[int]]:
    """Окна из k подряд идущих стихов одной главы (шаг 1)."""
    chap = [".".join(i.split(".")[:3]) for i in s_ids]
    out = []
    for j in range(len(s_ids)):
        members = [j]
        for d in range(1, k):
            m = j + d
            if m < len(s_ids) and chap[m] == chap[j]:
                members.append(m)
        if len(members) == k or k == 1:
            out.append(members)
    return out


def char4_docs(texts: list[str]) -> list[list[int]]:
    import zlib
    docs = []
    for t in texts:
        s = " " + " ".join(t.lower().split()) + " "
        docs.append([zlib.crc32(s[i:i + 4].encode("utf-8")) % (1 << 20)
                     for i in range(len(s) - 3)])
    return docs


def cosine_pair(sdocs, qdocs):
    vocab: dict = {}
    for d in sdocs:
        for x in d:
            vocab.setdefault(x, len(vocab))
    for d in qdocs:
        for x in d:
            vocab.setdefault(x, len(vocab))
    S = _token_matrix(sdocs, vocab)
    Q = _token_matrix(qdocs, vocab)
    idf = _idf_from(S)
    W = sparse.diags(idf)
    return _l2_rows(S @ W), _l2_rows(Q @ W)


def main(bible_ann: str, novel_sent_ann: str, gold_tsv: str,
         bible_graph_ann: str | None = None,
         sent_graph_ann: str | None = None) -> None:
    s_ids, s_lem, s_pos, s_edges, s_raw = load_ann(bible_ann)
    q_ids_all, q_lem_all, q_pos_all, q_edges_all, q_raw_all = load_ann(novel_sent_ann)
    gold = load_gold(gold_tsv)
    gold_pars = sorted({g[0] for g in gold})
    keep = [i for i, q in enumerate(q_ids_all) if q.split("#", 1)[0] in set(gold_pars)]
    q_lem = [q_lem_all[i] for i in keep]
    q_pos = [q_pos_all[i] for i in keep]
    q_edges = [q_edges_all[i] for i in keep]
    q_raw = [q_raw_all[i] for i in keep]
    par_of_sent = np.array([gold_pars.index(q_ids_all[i].split("#", 1)[0]) for i in keep])
    n_par = len(gold_pars)
    print(f"стихов {len(s_ids)}, эталонных абзацев {n_par}, предложений {len(q_lem)}",
          file=sys.stderr)

    correct_verses = [{j for j, sid in enumerate(s_ids) if verse_matches(sid, g[1])}
                      for g in gold]
    par_of_rec = [gold_pars.index(g[0]) for g in gold]

    df = Counter()
    for l in s_lem:
        df.update(set(l))
    rare = {l for l, c in df.items() if c / len(s_ids) <= RARE_DF_RATIO}

    print(f"\n{'скорер':12s} {'окно':>5s} {'пассажей':>9s} " +
          " ".join(f"{c:>6d}" for c in COST_POINTS))
    print(f"{'':12s} {'':5s} {'':9s} "
          + "  доля мест эталона при таком числе прочитанных стихов")

    for k in WINDOWS:
        passages = build_passages(s_ids, k)
        pmembers = passages
        p_lem = [[l for j in m for l in s_lem[j]] for m in pmembers]
        p_raw = [" ".join(s_raw[j] for j in m) for m in pmembers]
        p_edges = [[e for j in m for e in s_edges[j]] for m in pmembers]
        # пассаж правильный, если он пересекается с диапазоном эталона
        correct_p = [{pi for pi, m in enumerate(pmembers) if set(m) & cv}
                     for cv in correct_verses]
        n_verses_in = np.array([len(m) for m in pmembers])

        scorers = {
            "tfidf_rare": cosine_pair([[x for x in d if x in rare] for d in p_lem],
                                      [[x for x in d if x in rare] for d in q_lem]),
            "char4": cosine_pair(char4_docs(p_raw), char4_docs(q_raw)),
            "soft_edges": cosine_pair([_soft_edges(d) for d in p_edges],
                                      [_soft_edges(d) for d in q_edges]),
        }

        for name, (S, Q) in scorers.items():
            ST = S.T.tocsr()
            agg = np.zeros((n_par, S.shape[0]), dtype=np.float32)
            for a in range(0, Q.shape[0], CHUNK):
                block = np.asarray((Q[a:a + CHUNK] @ ST).todense(), dtype=np.float32)
                for r in range(block.shape[0]):
                    p = par_of_sent[a + r]
                    np.maximum(agg[p], block[r], out=agg[p])

            # для каждого места: сколько различных стихов надо прочесть, идя по
            # списку пассажей сверху, пока не встретится правильный
            costs = []
            for ri, g in enumerate(gold):
                if not correct_p[ri]:
                    costs.append(np.inf)
                    continue
                sc = agg[par_of_rec[ri]]
                order = np.argsort(-sc, kind="stable")
                seen: set[int] = set()
                cost = np.inf
                for pi in order:
                    seen.update(pmembers[pi])
                    if pi in correct_p[ri]:
                        cost = len(seen)
                        break
                costs.append(cost)
            costs = np.array(costs, dtype=float)
            line = [float(np.mean(costs <= c)) for c in COST_POINTS]
            print(f"{name:12s} {k:5d} {len(pmembers):9d} "
                  + " ".join(f"{v:6.2f}" for v in line))
        print()


if __name__ == "__main__":
    main(*sys.argv[1:6])
