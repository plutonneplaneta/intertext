"""Левое присоединение колонок из TSV с (target_id, source_id) к таблице пула.

Не трогает build_features: таблица пула остаётся такой, какой её строит
fusion_retrieval.py, новые признаки добавляются отдельным шагом. Пары без
значения получают 0.
"""
from __future__ import annotations

import csv
import sys


def main(table: str, out: str, *extras: str) -> None:
    add: dict[tuple[str, str], dict[str, str]] = {}
    cols: list[str] = []
    for path in extras:
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            names = [c for c in rd.fieldnames if c not in ("target_id", "source_id")]
            cols += [c for c in names if c not in cols]
            for r in rd:
                add.setdefault((r["target_id"], r["source_id"]), {}).update({c: r[c] for c in names})
    with open(table, encoding="utf-8", newline="") as f, open(out, "w", encoding="utf-8") as g:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        fields = list(rd.fieldnames)
        # label остаётся последней колонкой
        head = [c for c in fields if c != "label"] + cols + ["label"]
        g.write("\t".join(head) + "\n")
        n = miss = 0
        for r in rd:
            extra = add.get((r["target_id"], r["source_id"]), {})
            miss += 0 if extra else 1
            row = {**r, **{c: extra.get(c, "0") for c in cols}}
            g.write("\t".join(row[c] for c in head) + "\n")
            n += 1
    print(f"строк {n}, без значения {miss}, добавлено колонок {len(cols)} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
