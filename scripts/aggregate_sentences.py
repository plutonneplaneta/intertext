"""Свод предложенческих кандидатов обратно к абзацу: балл пары
(абзац, стих) = лучший балл среди предложений абзаца.

Так балл перестаёт быть суммой по всему абзацу и становится «насколько
похоже лучшее предложение», то есть перестаёт расти с длиной абзаца
линейно. Остаточная зависимость -- максимум из N предложений, она
логарифмическая, а не линейная.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


def main(in_tsv: str, out_tsv: str, score_col: str = "score") -> None:
    best: dict[tuple[str, str], tuple[float, str]] = {}
    with open(in_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            tid = row["target_id"].split("#", 1)[0]
            key = (tid, row["source_id"])
            s = float(row[score_col])
            prev = best.get(key)
            if prev is None or s > prev[0]:
                best[key] = (s, row["target_id"])

    n_sent_per_par: dict[str, set] = defaultdict(set)
    for (tid, _), (_, sent_id) in best.items():
        n_sent_per_par[tid].add(sent_id)

    with open(out_tsv, "w", encoding="utf-8") as out:
        out.write("score\tn\tngram\ttarget_id\tsource_id\tbest_sentence_id\n")
        for (tid, sid), (s, sent_id) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            out.write(f"{s:.4f}\t\t\t{tid}\t{sid}\t{sent_id}\n")
    print(f"пар (абзац, стих): {len(best)} -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         sys.argv[3] if len(sys.argv) > 3 else "score")
