"""Пункт 4 из списка «мерить следующим»: чистый тест стягивания предлогов.

В graph_features.py вариант collapsed заодно выбрасывал служебные части речи с
обоих концов ребра, поэтому 0,12 против 0,34 у обычных рёбер сравнивало два
изменения сразу, и вывода о стягивании сделать было нельзя.

Здесь все три варианта считаются на ОДНОМ И ТОМ ЖЕ наборе рёбер -- том, что у
пилота (всё, кроме punct/det/case, без петель), -- и различаются только ключом:

  plain      (лемма головы, тип связи, лемма зависимого)      -- как у пилота
  collapsed  (лемма головы, тип связи + предлог, лемма)       -- предлог втянут
  soft       (лемма головы, лемма зависимого), без порядка    -- тип связи убран

Ожидание от стягивания: предлог у Достоевского и в Синодальном переводе часто
разный при том же отношении, поэтому втягивание предлога в метку должно СУЖАТЬ
признак. Если так -- это довод против, а не за.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
from scipy import sparse

from graph_features import load_graph_ann
from oracle_diagnostics import load_gold, verse_matches
from retrieval_scorers import _idf_from, _l2_rows, _token_matrix

SKIP_DEP = {"punct", "det", "case"}
CHUNK = 256


def edge_keys(lemmas, heads, rels, adps, mode: str):
    out = []
    for child, h in enumerate(heads):
        if not (0 <= h < len(lemmas)) or h == child:
            continue
        if rels[child] in SKIP_DEP:
            continue
        a, b = lemmas[h], lemmas[child]
        if a == b:
            continue
        if mode == "plain":
            out.append((a, rels[child], b))
        elif mode == "collapsed":
            adp = adps[child] if child < len(adps) else ""
            out.append((a, f"{rels[child]}:{adp}" if adp else rels[child], b))
        elif mode == "soft":
            out.append((a, b) if a <= b else (b, a))
    return out


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
    W = sparse.diags(_idf_from(S))
    return _l2_rows(S @ W), _l2_rows(Q @ W)


def main(bible_graph_ann: str, sent_graph_ann: str, gold_tsv: str) -> None:
    s_ids, s_lem, s_pos, s_heads, s_rels, s_adp = load_graph_ann(bible_graph_ann)
    q_ids, q_lem, q_pos, q_heads, q_rels, q_adp = load_graph_ann(sent_graph_ann)
    gold = load_gold(gold_tsv)
    gold_pars = sorted({g[0] for g in gold})
    keep = [i for i, q in enumerate(q_ids) if q.split("#", 1)[0] in set(gold_pars)]
    par_of_sent = np.array([gold_pars.index(q_ids[i].split("#", 1)[0]) for i in keep])
    n_par = len(gold_pars)
    correct = [{j for j, sid in enumerate(s_ids) if verse_matches(sid, g[1])}
               for g in gold]
    par_of_rec = [gold_pars.index(g[0]) for g in gold]
    print(f"стихов {len(s_ids)}, абзацев {n_par}, предложений {len(keep)}",
          file=sys.stderr)

    print(f"\n{'ключ ребра':12s} {'рёбер в словаре':>16s} {'виден':>6s} "
          f"{'медиана':>8s} {'топ-50':>7s} {'топ-200':>8s} {'топ-1000':>9s}")
    for mode in ("plain", "collapsed", "soft"):
        sdocs = [edge_keys(s_lem[i], s_heads[i], s_rels[i], s_adp[i], mode)
                 for i in range(len(s_ids))]
        qdocs = [edge_keys(q_lem[i], q_heads[i], q_rels[i], q_adp[i], mode)
                 for i in keep]
        n_keys = len({x for d in sdocs for x in d})
        S, Q = cosine_pair(sdocs, qdocs)
        ST = S.T.tocsr()
        agg = np.zeros((n_par, len(s_ids)), dtype=np.float32)
        for a in range(0, Q.shape[0], CHUNK):
            block = np.asarray((Q[a:a + CHUNK] @ ST).todense(), dtype=np.float32)
            for r in range(block.shape[0]):
                np.maximum(agg[par_of_sent[a + r]], block[r],
                           out=agg[par_of_sent[a + r]])
        hits, ranks = [], []
        for ri, g in enumerate(gold):
            if not correct[ri]:
                continue
            sc = agg[par_of_rec[ri]]
            best = sc[list(correct[ri])].max()
            hits.append(int(best > 0))
            if best > 0:
                greater = int((sc > best).sum())
                equal = int((sc == best).sum())
                ranks.append(greater + (equal + 1) / 2.0)
        h = float(np.mean(hits))
        r = np.array(ranks, dtype=float)
        print(f"{mode:12s} {n_keys:16d} {h:6.2f} {np.median(r):8.0f} "
              f"{np.mean(r<=50)*h:7.2f} {np.mean(r<=200)*h:8.2f} "
              f"{np.mean(r<=1000)*h:9.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
