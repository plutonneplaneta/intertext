"""Поиск кандидатов заимствования: точное совпадение леммной n-граммы,
взвешенное по редкости, а не по частоте (см. обсуждение порядка PMI в
другом проекте автора — там частое подавалось как сигнал, здесь сигнал даёт редкое).

Вход — два JSONL от lemmatize.py: источник (Библия, по стихам) и цель
(роман, по абзацам). Строим общий словарь частот по объединённому корпусу
(IDF по единицам — стих или абзац), индексируем n-граммы длины 3..6 из
источника по редким леммам, ищем точные совпадения n-грамм в цели.

Порог редкости и длины — параметры, не встроенные константы: это первая,
самая простая версия метода (лексическое совпадение, порядок слов сохраняется
— полный аналог поверхностных n-грамм, но по леммам, а не словоформам).
Синтаксический сигнал (порядок неважен) и семантический (эмбеддинги) —
следующие слои, здесь их нет.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

MIN_N, MAX_N = 3, 6
# лемма считается "редкой" и годной как якорь индексации, если встречается
# не более чем в этой доле единиц объединённого корпуса
RARITY_DF_RATIO = 0.02
# минимальный суммарный вес редкости n-граммы, чтобы считаться кандидатом
MIN_SCORE = 6.0


@dataclass
class Unit:
    id: str
    lemmas: list[str]
    raw: str


def load_jsonl(path: str) -> list[Unit]:
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            units.append(Unit(rec["id"], rec["lemmas"], rec.get("raw", "")))
    return units


def build_idf(all_units: list[list[str]]) -> dict[str, float]:
    n_docs = len(all_units)
    df: Counter[str] = Counter()
    for lemmas in all_units:
        df.update(set(lemmas))
    idf = {lemma: math.log(n_docs / count) for lemma, count in df.items()}
    return idf, df, n_docs


def ngrams(lemmas: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(lemmas[i:i + n]) for i in range(len(lemmas) - n + 1)]


def main(source_jsonl: str, target_jsonl: str, out_tsv: str) -> None:
    source = load_jsonl(source_jsonl)
    target = load_jsonl(target_jsonl)
    print(f"источник: {len(source)} единиц; цель: {len(target)} единиц", file=sys.stderr)

    all_lemma_lists = [u.lemmas for u in source] + [u.lemmas for u in target]
    idf, df, n_docs = build_idf(all_lemma_lists)
    rare_lemmas = {l for l, c in df.items() if c / n_docs <= RARITY_DF_RATIO}
    print(f"словарь: {len(idf)} лемм, редких (порог {RARITY_DF_RATIO}): {len(rare_lemmas)}", file=sys.stderr)

    # индекс: n-грамма источника -> список (unit_id, raw) источника, только
    # если хотя бы одна лемма n-граммы редкая — иначе индекс не даёт признака
    index: dict[tuple[str, ...], list[Unit]] = defaultdict(list)
    for u in source:
        for n in range(MIN_N, MAX_N + 1):
            for ng in ngrams(u.lemmas, n):
                if any(l in rare_lemmas for l in ng):
                    index[ng].append(u)

    print(f"n-грамм в индексе: {len(index)}", file=sys.stderr)

    best_by_pair: dict[tuple[str, str], tuple[float, tuple[str, ...], int]] = {}
    for tu in target:
        for n in range(MAX_N, MIN_N - 1, -1):  # длинные сначала — сильнее сигнал
            for ng in ngrams(tu.lemmas, n):
                hits = index.get(ng)
                if not hits:
                    continue
                score = sum(idf.get(l, 0.0) for l in ng)
                if score < MIN_SCORE:
                    continue
                for su in hits:
                    key = (tu.id, su.id)
                    prev = best_by_pair.get(key)
                    if prev is None or score > prev[0]:
                        best_by_pair[key] = (score, ng, n)

    ranked = sorted(best_by_pair.items(), key=lambda kv: -kv[1][0])
    src_by_id = {u.id: u for u in source}
    tgt_by_id = {u.id: u for u in target}
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("score\tn\tngram\ttarget_id\tsource_id\ttarget_text\tsource_text\n")
        for (tid, sid), (score, ng, n) in ranked:
            f.write(
                f"{score:.2f}\t{n}\t{' '.join(ng)}\t{tid}\t{sid}\t"
                f"{tgt_by_id[tid].raw}\t{src_by_id[sid].raw}\n"
            )
    print(f"кандидатов: {len(ranked)} -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
