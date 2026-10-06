"""Таблица признаков для комбинированной оценки: по каждому целевому абзацу
эталона -- объединение кандидатов трёх методов (лексика, эмбеддинги,
синтаксис) с их баллами (0, если метод эту пару вообще не предложил) и
меткой (1, если source_id попадает в диапазон стиха, документированного
Тихомировым для этого абзаца).

Заведомо небольшая выборка (64 целевых абзаца, из них у 24 вообще есть
хоть один верный кандидат хоть у одного метода) -- обычное разбиение на
train/dev/test её обесценило бы шумом одного случайного разбиения.
Оценка -- leave-one-target-out (см. combine_and_evaluate.py).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load_gold(path: str) -> dict[str, list[tuple[str, int, int, int]]]:
    """novel_verse_id -> [(book, chapter, verse_from, verse_to), ...]"""
    by_target: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if not row["novel_verse_id"] or float(row["match_score"]) < 0.6:
            continue
        for ref in row["bible_refs"].split(";"):
            if not ref:
                continue
            book, chap, verses = ref.split(".", 2)
            if "-" in verses:
                vf, vt = map(int, verses.split("-"))
            else:
                vf = vt = int(verses)
            by_target[row["novel_verse_id"]].append((book, int(chap), vf, vt))
    return by_target


def load_scores(path: str, targets: set[str]) -> dict[str, dict[str, float]]:
    """target_id -> {source_id: score}, только для нужных target_id."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["target_id"] not in targets:
            continue
        s = float(row["score"])
        prev = out[row["target_id"]].get(row["source_id"])
        if prev is None or s > prev:
            out[row["target_id"]][row["source_id"]] = s
    return out


def is_positive(source_id: str, refs: list[tuple[str, int, int, int]]) -> bool:
    parts = source_id.split(".")
    if len(parts) != 4:
        return False
    _, book, chap, verse = parts
    chap, verse = int(chap), int(verse)
    return any(book == b and chap == c and vf <= verse <= vt for b, c, vf, vt in refs)


def main(gold_path: str, lex_path: str, emb_path: str, syn_path: str, out_path: str) -> None:
    gold = load_gold(gold_path)
    targets = set(gold.keys())
    print(f"целевых абзацев в эталоне: {len(targets)}", file=sys.stderr)

    lex = load_scores(lex_path, targets)
    emb = load_scores(emb_path, targets)
    syn = load_scores(syn_path, targets)

    rows = []
    for tid, refs in gold.items():
        source_ids = set(lex.get(tid, {})) | set(emb.get(tid, {})) | set(syn.get(tid, {}))
        for sid in source_ids:
            label = 1 if is_positive(sid, refs) else 0
            rows.append((
                tid, sid,
                lex.get(tid, {}).get(sid, 0.0),
                emb.get(tid, {}).get(sid, 0.0),
                syn.get(tid, {}).get(sid, 0.0),
                label,
            ))

    n_pos = sum(r[-1] for r in rows)
    n_targets_with_pos = len({r[0] for r in rows if r[-1] == 1})
    print(f"строк (пар): {len(rows)}, положительных: {n_pos}, "
          f"абзацев хотя бы с одним верным кандидатом: {n_targets_with_pos}/{len(targets)}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tlex\temb\tsyn\tlabel\n")
        for tid, sid, l, e, s, lab in rows:
            f.write(f"{tid}\t{sid}\t{l:.4f}\t{e:.4f}\t{s:.4f}\t{lab}\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
