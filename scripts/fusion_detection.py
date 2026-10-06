"""Проверка: меняет ли объединённое извлечение задачу ОТБОРА абзацев.

В прошлой итерации ни один слой не добавлял к длине абзаца ничего измеримого
(прирост AUC от -0,004 до +0,006, нуль внутри интервала). Но там слои были
те, у которых потолок полноты 11% и 34%: слой, который вообще не видит
правильный стих у двух третей мест, и не мог ничего добавить.

Теперь скореры видят правильный стих почти всегда, и вопрос стоит заново.
Признак абзаца -- максимум балла скорера по всем 31 102 стихам (и по всем
предложениям абзаца). Соперник -- по-прежнему длина.

Считается на всех эталонных абзацах и случайной выборке нецитатных: AUC от
доли положительных не зависит, поэтому выборка отрицательных даёт
несмещённую оценку и экономит час счёта.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from oracle_diagnostics import load_gold
from retrieval_scorers import SCORERS, ScorerBank, load_ann

N_NEG_SAMPLE = 600
SEED = 0


def main(bible_ann: str, novel_ann: str, novel_sent_ann: str, gold_tsv: str,
         out_tsv: str) -> None:
    t_ids, t_lem, t_pos, t_edges, t_raw = load_ann(novel_ann)
    q_ids, q_lem, q_pos, q_edges, q_raw = load_ann(novel_sent_ann)
    bank = ScorerBank(bible_ann, [t_lem, q_lem])
    bank.prepare_vectors([(t_ids, t_lem, t_pos), (q_ids, q_lem, q_pos)])
    Q_vec = bank.T_vecs[1]

    gold_pars = {g[0] for g in load_gold(gold_tsv)}
    lengths = {tid: len(raw.split()) for tid, raw in zip(t_ids, t_raw)}

    rng = np.random.default_rng(SEED)
    negs = [t for t in t_ids if t not in gold_pars]
    sample = sorted(gold_pars) + sorted(
        rng.choice(negs, size=min(N_NEG_SAMPLE, len(negs)), replace=False).tolist())
    print(f"абзацев в выборке: {len(sample)} "
          f"(эталонных {len(gold_pars)}, нецитатных {len(sample)-len(gold_pars)})",
          file=sys.stderr)

    sent_of_par = defaultdict(list)
    for i, q in enumerate(q_ids):
        sent_of_par[q.split("#", 1)[0]].append(i)

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tlength\tis_gold\t"
                + "\t".join(f"{n}_max" for n in SCORERS) + "\n")
        for k, par in enumerate(sample):
            best = {n: 0.0 for n in SCORERS}
            for qi in sent_of_par.get(par, []):
                sc = bank.score(q_lem[qi], q_raw[qi], q_edges[qi], Q_vec[qi])
                for n in SCORERS:
                    v = float(sc[n].max())
                    if v > best[n]:
                        best[n] = v
            f.write(f"{par}\t{lengths[par]}\t{int(par in gold_pars)}\t"
                    + "\t".join(f"{best[n]:.6f}" for n in SCORERS) + "\n")
            if (k + 1) % 50 == 0:
                print(f"{k+1}/{len(sample)}", file=sys.stderr, flush=True)
    print(f"-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
