"""Признаки абзаца для обнаружения: максимум по пулу кандидатов.

Из таблицы пула (fusion_retrieval.py build с эталоном-пустышкой) берутся максимумы сырых
баллов скореров по всем парам абзаца (в пул входят топ-K каждого скорера, поэтому максимум
по пулу равен максимуму по всем 31 102 стихам) и, если есть, признаки surprisal: максимум
delta на токен предложения, среднее трёх лучших и максимум суммарной delta.

Выход -- таблица для eval_fusion_detection.py: target_id, length, is_gold, признаки.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

SCORER_COLS = ["e5_raw", "tfidf_rare_raw", "dep_pair2_raw", "runmatch_raw", "ngram_raw", "skipbigram_raw",
               "bm25_raw", "char4_raw"]


def main(meta_tsv: str, pool_tsv: str, out_tsv: str, sur_tsv: str = "") -> None:
    meta = {}
    with open(meta_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            meta[r["target_id"]] = (r["length"], r["is_gold"])
    best = defaultdict(lambda: {c: 0.0 for c in SCORER_COLS})
    with open(pool_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            b = best[r["target_id"]]
            for c in SCORER_COLS:
                v = float(r[c])
                if v > b[c]:
                    b[c] = v
    sur = defaultdict(list)
    if sur_tsv:
        seen = set()
        with open(sur_tsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                k = (r["target_id"], r["source_id"])
                if k not in seen:
                    seen.add(k)
                    sur[r["target_id"]].append((float(r["sur_gain_raw"]), float(r["sur_delta_raw"])))
    cols = [c.replace("_raw", "_max") for c in SCORER_COLS]
    extra = ["sur_gain_max", "sur_gain_top3", "sur_delta_max"] if sur_tsv else []
    with open(out_tsv, "w", encoding="utf-8") as g:
        g.write("target_id\tlength\tis_gold\t" + "\t".join(cols + extra) + "\n")
        for p, (ln, ig) in meta.items():
            row = [f"{best[p][c]:.6f}" for c in SCORER_COLS]
            if sur_tsv:
                gains = sorted((x[0] for x in sur[p]), reverse=True) or [0.0]
                dmax = max((x[1] for x in sur[p]), default=0.0)
                row += [f"{gains[0]:.6f}", f"{sum(gains[:3]) / len(gains[:3]):.6f}", f"{dmax:.6f}"]
            g.write(f"{p}\t{ln}\t{ig}\t" + "\t".join(row) + "\n")
    print(f"-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5]) if len(sys.argv) > 4 else main(*sys.argv[1:4])
