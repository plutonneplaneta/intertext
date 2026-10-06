"""Механизм: помогает ли surprisal там, где нет лексического мостика, или только где он есть.

Для каждого места эталона считается тип мостика между абзацем и правильными стихами (как в
gold_findability.py): «редкая лемма» (есть общая редкая лемма), «только подряд» (нет общей
редкой, но есть совпадение >= 2 лемм подряд), «нет моста». Затем по каждому типу -- медиана и
средний log-ранг для выбранных наборов признаков из файлов рангов.

Если surprisal -- сглаженное лексическое пересечение, его выигрыш сосредоточится в «редкой
лемме» и пропадёт в «нет моста»; если он ловит смысл, выигрыш останется и там.

Использование:
  mechanism_by_bridge.py <bible_ann> <novel_ann> <gold.tsv> <ranks_file>:<метка> [...]
"""
from __future__ import annotations

import math
import sys
from collections import Counter

import numpy as np

from gold_findability import load_ann, load_seq
from oracle_diagnostics import load_gold, verse_matches

RARE_DF_RATIO = 0.02


def longest_run(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def categories(bible_ann: str, novel_ann: str, gold_tsv: str) -> dict[int, str]:
    s_lem, t_lem = load_ann(bible_ann), load_ann(novel_ann)
    s_seq, t_seq = load_seq(bible_ann), load_seq(novel_ann)
    n = len(s_lem)
    df = Counter()
    for lem in s_lem.values():
        df.update(set(lem))
    rare = {l for l, c in df.items() if c / n <= RARE_DF_RATIO}
    out = {}
    for k, (par, group, _frag) in enumerate(load_gold(gold_tsv)):
        pl = set(t_lem.get(par, []))
        pseq = t_seq.get(par, [])
        best_rare, best_run = 0, 0
        for vid, vl in s_lem.items():
            if not verse_matches(vid, group):
                continue
            best_rare = max(best_rare, len((pl & set(vl)) & rare))
            best_run = max(best_run, longest_run(pseq, s_seq.get(vid, [])))
        out[k] = "редкая лемма" if best_rare >= 1 else ("только подряд" if best_run >= 2 else "нет моста")
    return out


def load_ranks(spec: str):
    import json
    path, label = spec.rsplit(":", 1)
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d["label"] == label:
            return label, {int(k): v for k, v in d["ranks"].items() if v}
    raise SystemExit(f"нет набора «{label}» в {path}")


def main(bible_ann: str, novel_ann: str, gold_tsv: str, *specs: str) -> None:
    cat = categories(bible_ann, novel_ann, gold_tsv)
    sets = [load_ranks(s) for s in specs]
    names = ["редкая лемма", "только подряд", "нет моста"]
    print(f"{'тип мостика':16s} {'мест':>5s} | " + " | ".join(f"{lbl[:26]:>26s}" for lbl, _ in sets))
    print(f"{'':16s} {'':>5s} | " + " | ".join(f"{'мед.ранг  ср.log':>26s}" for _ in sets))
    for c in names:
        keys = [k for k, v in cat.items() if v == c and all(k in r for _, r in sets)]
        if not keys:
            continue
        cells = []
        for _, r in sets:
            v = np.array([r[k] for k in keys], float)
            cells.append(f"{np.median(v):9.0f}  {np.mean(np.log(v)):8.2f}")
        print(f"{c:16s} {len(keys):5d} | " + " | ".join(f"{x:>26s}" for x in cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
