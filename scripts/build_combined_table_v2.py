"""Широкая таблица признаков для комбинированной модели атрибуции.

Отличия от build_combined_table.py: (1) признаков не три, а сколько задано,
включая поправку на хабность стиха и предложенческие варианты; (2) статистика
стиха («сколько этот стих обычно набирает») считается по ВСЕМУ роману, а не по
эталонным абзацам -- иначе она была бы оценена по 46 абзацам и оказалась бы
шумом, а заодно подглядывала бы в разметку.

Вызов: build_combined_table_v2.py gold.tsv out.tsv префикс:кол1,кол2[:хаб]=файл ...
Суффикс «:хаб» добавляет к первой колонке файла производные minus_src и
rank_src.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from build_combined_table import is_positive, load_gold

TOP_R = 10


def load_file(path: str, cols: list[str]):
    """(target_id, source_id) -> вектор значений; плюс значения по стихам."""
    pairs: dict[tuple[str, str], list[float]] = {}
    by_src: dict[str, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            key = (row["target_id"].split("#", 1)[0], row["source_id"])
            vals = [float(row[c]) for c in cols]
            prev = pairs.get(key)
            if prev is None or vals[0] > prev[0]:
                pairs[key] = vals
            by_src[row["source_id"]].append(vals[0])
    return pairs, by_src


def main(argv: list[str]) -> None:
    gold_path, out_path = argv[0], argv[1]
    specs = argv[2:]

    gold = load_gold(gold_path)
    targets = set(gold)
    print(f"целевых абзацев: {len(targets)}", file=sys.stderr)

    names: list[str] = []
    tables: list[dict] = []
    for spec in specs:
        head, path = spec.split("=", 1)
        parts = head.split(":")
        prefix, cols = parts[0], parts[1].split(",")
        hub = len(parts) > 2 and parts[2] == "хаб"
        pairs, by_src = load_file(path, cols)
        src_top = {s: float(np.mean(sorted(v, reverse=True)[:TOP_R]))
                   for s, v in by_src.items()}
        src_sorted = {s: np.sort(np.array(v)) for s, v in by_src.items()}

        col_names = [f"{prefix}_{c}" for c in cols]
        if hub:
            col_names += [f"{prefix}_minus_src", f"{prefix}_rank_src"]
        names += col_names

        tab: dict[tuple[str, str], list[float]] = {}
        for (t, s), vals in pairs.items():
            if t not in targets:
                continue
            v = list(vals)
            if hub:
                v.append(vals[0] - src_top.get(s, 0.0))
                arr = src_sorted.get(s)
                v.append(float(np.searchsorted(arr, vals[0], side="left")) / len(arr)
                         if arr is not None and len(arr) else 0.0)
            tab[(t, s)] = v
        tables.append(tab)
        print(f"{prefix}: колонок {len(col_names)}, пар по эталону {len(tab)}",
              file=sys.stderr)

    all_keys = set()
    for tab in tables:
        all_keys |= set(tab)

    n_pos = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\t" + "\t".join(names) + "\tlabel\n")
        for (t, s) in sorted(all_keys):
            row: list[float] = []
            for tab, spec in zip(tables, specs):
                head = spec.split("=", 1)[0].split(":")
                width = len(head[1].split(",")) + (2 if len(head) > 2 else 0)
                row += tab.get((t, s), [0.0] * width)
            label = 1 if is_positive(s, gold[t]) else 0
            n_pos += label
            f.write(f"{t}\t{s}\t" + "\t".join(f"{v:.6f}" for v in row) + f"\t{label}\n")

    n_t = len({t for (t, s) in all_keys})
    print(f"строк {len(all_keys)}, положительных {n_pos}, абзацев {n_t} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
