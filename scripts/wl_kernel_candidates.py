"""Weisfeiler-Lehman переразметка поверх рёбер разбора — четвёртый сигнал,
предложенный в исследовании LLM/графов: не считать совпадающие рёбра по
одному (то, что уже показало высокий recall и почти нулевую различающую
способность в syntax_candidates.py), а хешировать окрестность узла —
лемму вместе с тем, через какие связи и с какими соседями она встречается.
Совпадение такой окрестности — куда более специфичный сигнал, чем
совпадение одного ребра, потому что требует согласия сразу по нескольку
связям вокруг одной леммы, а не по одной паре.

Ограничение (сознательно принятое, не скрытое): рёбра в *_deps.jsonl
хранят леммы, а не идентификаторы токенов, поэтому граф здесь — граф по
леммам предложения (одноимённые слова в разных ролях сливаются в один
узел), а не честное дерево токенов. Для коротких библейских стихов и
абзацев это редко меняет картину, но эффект не проверен отдельно.

Схема разметки:
  h=0: лемма сама по себе.
  h=1: лемма + отсортированный набор (тип_связи, лемма_соседа) по всем
       инцидентным рёбрам — то есть "эта лемма, а вокруг неё вот что".
Обе разметки идут в один "мешок" меток на единицу текста; редкие метки
индексируются и сопоставляются так же, как рёбра в syntax_candidates.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

MIN_SCORE = 3.0
RARITY_DF_RATIO = 0.02


def short_hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=6).hexdigest()


@dataclass
class Unit:
    id: str
    labels: frozenset[str]
    raw: str


def wl_labels(edges: list[list[str]], depths: set[str]) -> frozenset[str]:
    """Метки для графа, собранного по леммам из рёбер разбора.

    depths управляет тем, что идёт в общий "мешок": {"h0"} — голое
    совпадение редкой леммы (эквивалент мешка слов), {"h1"} — только
    совпадение окрестности (лемма + с чем и как она связана), {"h0","h1"}
    — обе вместе (исходный, слишком щедрый вариант)."""
    neighbors: dict[str, list[tuple[str, str]]] = defaultdict(list)
    nodes: set[str] = set()
    for head, dep, child in edges:
        neighbors[head].append((dep, child))
        neighbors[child].append((f"{dep}^-1", head))  # обратное направление своим тегом
        nodes.add(head)
        nodes.add(child)

    labels: set[str] = set()
    for n in nodes:
        if "h0" in depths:
            labels.add(f"h0:{n}")
        if "h1" in depths:
            ctx = tuple(sorted(neighbors[n]))
            h1 = short_hash(n + "|" + "|".join(f"{d}:{c}" for d, c in ctx))
            labels.add(f"h1:{n}:{h1}")
    return frozenset(labels)


def load_jsonl(path: str, depths: set[str]) -> list[Unit]:
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = wl_labels(rec["edges"], depths)
            units.append(Unit(rec["id"], labels, rec.get("raw", "")))
    return units


def main(source_jsonl: str, target_jsonl: str, out_tsv: str, depths_arg: str = "h0,h1") -> None:
    depths = set(depths_arg.split(","))
    source = load_jsonl(source_jsonl, depths)
    target = load_jsonl(target_jsonl, depths)
    print(f"источник: {len(source)} единиц; цель: {len(target)} единиц", file=sys.stderr)

    n_docs = len(source) + len(target)
    df: Counter[str] = Counter()
    for u in source:
        df.update(u.labels)
    for u in target:
        df.update(u.labels)
    idf = {lbl: math.log(n_docs / c) for lbl, c in df.items()}
    rare = {lbl for lbl, c in df.items() if c / n_docs <= RARITY_DF_RATIO}
    print(f"меток всего: {len(df)}, редких: {len(rare)}", file=sys.stderr)

    index: dict[str, list[Unit]] = defaultdict(list)
    for u in source:
        for lbl in u.labels:
            if lbl in rare:
                index[lbl].append(u)

    best_by_pair: dict[tuple[str, str], float] = {}
    for tu in target:
        scored: dict[str, float] = defaultdict(float)
        for lbl in u_labels_rare(tu.labels, rare):
            w = idf.get(lbl, 0.0)
            for su in index.get(lbl, []):
                scored[su.id] += w
        for sid, score in scored.items():
            if score < MIN_SCORE:
                continue
            key = (tu.id, sid)
            if key not in best_by_pair or score > best_by_pair[key]:
                best_by_pair[key] = score

    ranked = sorted(best_by_pair.items(), key=lambda kv: -kv[1])
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("score\tn\tngram\ttarget_id\tsource_id\n")
        for (tid, sid), score in ranked:
            f.write(f"{score:.2f}\t\t\t{tid}\t{sid}\n")
    print(f"кандидатов: {len(ranked)} -> {out_tsv}")


def u_labels_rare(labels: frozenset[str], rare: set[str]) -> frozenset[str]:
    return labels & rare


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
