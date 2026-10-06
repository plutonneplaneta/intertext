"""Кандидаты по синтаксическому совпадению: общее ребро дерева разбора
(управляющее -- зависимое, лемма -- тип связи -- лемма), вес по редкости
ребра, порядок слов не важен. Ловит то, что лексический n-грамм теряет
из-за перестановки слов при пересказе.

Только точное совпадение ребра (лемма+связь+лемма) -- первая, самая простая
версия синтаксического слоя; мягкое совпадение (та же пара лемм, другая
связь) оставлено как возможное расширение, не включено, чтобы не раздувать
индекс общими парами и не размывать сигнал на первом прогоне.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

MIN_SCORE = 3.0
# как в find_candidates.py: индексируем только редкие рёбра -- частые
# (типа "сказать-Бог" в Библии) ничего не различают и только замедляют поиск
RARITY_DF_RATIO = 0.02


@dataclass
class Unit:
    id: str
    edges: list[tuple[str, str, str]]
    raw: str


def load_jsonl(path: str) -> list[Unit]:
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            edges = [tuple(e) for e in rec["edges"]]
            units.append(Unit(rec["id"], edges, rec.get("raw", "")))
    return units


def main(source_jsonl: str, target_jsonl: str, out_tsv: str) -> None:
    source = load_jsonl(source_jsonl)
    target = load_jsonl(target_jsonl)
    print(f"источник: {len(source)} единиц; цель: {len(target)} единиц", file=sys.stderr)

    all_edge_sets = [set(u.edges) for u in source] + [set(u.edges) for u in target]
    n_docs = len(all_edge_sets)
    df: Counter[tuple[str, str, str]] = Counter()
    for edges in all_edge_sets:
        df.update(edges)
    idf = {e: math.log(n_docs / c) for e, c in df.items()}
    rare_edges = {e for e, c in df.items() if c / n_docs <= RARITY_DF_RATIO}
    print(f"рёбер всего: {len(df)}, редких: {len(rare_edges)}", file=sys.stderr)

    edge_index: dict[tuple[str, str, str], list[Unit]] = defaultdict(list)
    for u in source:
        for e in set(u.edges):
            if e in rare_edges:
                edge_index[e].append(u)

    best_by_pair: dict[tuple[str, str], float] = {}
    for tu in target:
        scored: dict[str, float] = defaultdict(float)
        for e in set(tu.edges):
            hits = edge_index.get(e)
            if not hits:
                continue
            w = idf.get(e, 0.0)
            for su in hits:
                scored[su.id] += w

        for sid, score in scored.items():
            if score < MIN_SCORE:
                continue
            key = (tu.id, sid)
            if key not in best_by_pair or score > best_by_pair[key]:
                best_by_pair[key] = score

    # без текста в каждой строке -- при сотнях кандидатов на абзац (обычная
    # картина для этого метода) полный текст в каждой строке даёт файл на
    # гигабайты; id хватает для оценки полноты, текст ищется по id при надобности
    ranked = sorted(best_by_pair.items(), key=lambda kv: -kv[1])
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("score\tn\tngram\ttarget_id\tsource_id\n")
        for (tid, sid), score in ranked:
            f.write(f"{score:.2f}\t\t\t{tid}\t{sid}\n")

    print(f"кандидатов: {len(ranked)} -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
