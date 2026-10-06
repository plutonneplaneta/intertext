"""Переводит пары (абзац, стих) в пары (предложение, стих).

Для каждой пары берётся предложение абзаца, ближе всего к стиху по векторам E5
(как и сам конвейер берёт максимум по предложениям). Так LLM видит не весь абзац,
где цитата -- одна фраза среди десятка, а то место, из-за которого пара вообще
оказалась кандидатом. Выбор по E5, а не по эталонному фрагменту: фрагмент
известен только для золотых пар, и подсказка работала бы только в их пользу.

Вход: пары (target_id, source_id, decoy_type), npz Библии и предложений.
Выход: те же пары с target_id = id предложения.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def main(pairs_tsv: str, bible_npz: str, sent_npz: str, out_tsv: str) -> None:
    b = np.load(bible_npz, allow_pickle=True)
    q = np.load(sent_npz, allow_pickle=True)
    b_pos = {s: i for i, s in enumerate(b["ids"])}
    S, Q = b["embeddings"], q["embeddings"]
    sents = defaultdict(list)
    for i, sid in enumerate(q["ids"]):
        sents[sid.split("#", 1)[0]].append(i)

    out, skipped = [], 0
    with open(pairs_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            idx = sents.get(r["target_id"])
            if not idx or r["source_id"] not in b_pos:
                skipped += 1
                continue
            sims = Q[idx] @ S[b_pos[r["source_id"]]]
            best = idx[int(np.argmax(sims))]
            out.append((q["ids"][best], r["source_id"], r["decoy_type"]))
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tdecoy_type\n")
        for t, s, ty in out:
            f.write(f"{t}\t{s}\t{ty}\n")
    print(f"пар {len(out)}, без предложений {skipped} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
