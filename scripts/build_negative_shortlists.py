"""Шортлисты кандидатов для НЕэталонных абзацев -- под отрицательный контроль
LLM-обогащения (scripts/llm_enrich.py, режим negative).

Доля цитатных абзацев в романе 1,1%, поэтому вопрос «как часто модель уверенно
называет источник там, где его нет» решает судьбу всей затеи: при полноте хоть
90% метод бесполезен, если ложная тревога на каждом пятом абзаце. Считать эту
долю на десятке абзацев нельзя, нужны сотни -- отсюда отдельный сборщик.

Формат вывода совпадает с fusion_retrieval.py build, чтобы llm_enrich.py читал
обе таблицы одинаково. Метка везде 0: правильного ответа тут по определению нет.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from oracle_diagnostics import load_gold
from retrieval_scorers import SCORERS, ScorerBank, load_ann

N_SAMPLE = 400
TOP_K = 300
SEED = 0


def main(bible_ann: str, novel_ann: str, sent_ann: str, gold_tsv: str,
         out_tsv: str, bible_graph_ann: str | None = None,
         sent_graph_ann: str | None = None) -> None:
    t_ids, t_lem, t_pos, t_edges, t_raw = load_ann(novel_ann)
    q_ids, q_lem, q_pos, q_edges, q_raw = load_ann(sent_ann)
    bank = ScorerBank(bible_ann, [t_lem, q_lem], bible_graph_ann=bible_graph_ann)
    bank.prepare_vectors([(t_ids, t_lem, t_pos), (q_ids, q_lem, q_pos)])
    Q_vec = bank.T_vecs[1]

    dep_docs = {}
    if bible_graph_ann and sent_graph_ann:
        from graph_features import dep_pairs, load_graph_ann
        from retrieval_scorers import DEP_PAIR_DIST
        g_ids, g_lem, g_pos, g_heads, _, _ = load_graph_ann(sent_graph_ann)
        pos_of = {q: i for i, q in enumerate(g_ids)}
        for i, q in enumerate(q_ids):
            k = pos_of.get(q)
            if k is not None:
                dep_docs[i] = dep_pairs(g_lem[k], g_pos[k], g_heads[k], DEP_PAIR_DIST)

    gold_pars = {g[0] for g in load_gold(gold_tsv)}
    rng = np.random.default_rng(SEED)
    pool = [t for t in t_ids if t not in gold_pars]
    sample = sorted(rng.choice(pool, size=min(N_SAMPLE, len(pool)),
                               replace=False).tolist())
    print(f"нецитатных абзацев в выборке: {len(sample)}", file=sys.stderr)

    sent_of = defaultdict(list)
    for i, q in enumerate(q_ids):
        sent_of[q.split("#", 1)[0]].append(i)

    n_src = len(bank.s_ids)
    names = [f"{n}_raw" for n in SCORERS] + [f"{n}_pct" for n in SCORERS]
    n_rows = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\trec_id\tin_natural_pool\t"
                + "\t".join(names) + "\tlabel\n")
        for k, par in enumerate(sample):
            agg = {n: np.zeros(n_src, dtype=np.float32) for n in SCORERS}
            for qi in sent_of.get(par, []):
                sc = bank.score(q_lem[qi], q_raw[qi], q_edges[qi], Q_vec[qi],
                                dep_doc=dep_docs.get(qi))
                for n in SCORERS:
                    np.maximum(agg[n], sc[n], out=agg[n])
            pct = {}
            for n in SCORERS:
                order = np.argsort(agg[n], kind="stable")
                r = np.empty(n_src, dtype=np.float32)
                r[order] = np.arange(n_src, dtype=np.float32) / n_src
                pct[n] = r
            pool_idx: set[int] = set()
            for n in SCORERS:
                kk = min(TOP_K, n_src)
                pool_idx.update(np.argpartition(agg[n], -kk)[-kk:].tolist())
            for j in sorted(pool_idx):
                feats = [agg[n][j] for n in SCORERS] + [pct[n][j] for n in SCORERS]
                f.write(f"{par}\t{bank.s_ids[j]}\t-1\t0\t"
                        + "\t".join(f"{v:.6f}" for v in feats) + "\t0\n")
                n_rows += 1
            if (k + 1) % 25 == 0:
                print(f"{k+1}/{len(sample)}, строк {n_rows}", file=sys.stderr, flush=True)
    print(f"строк {n_rows} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:8])
