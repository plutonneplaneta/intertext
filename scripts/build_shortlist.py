"""Шорт-лист для LLM-верификации: топ-K кандидатов на эталонный абзац от
каждого метода, объединение без дублей. Отдельно подмешивает N заведомо
неверных пар (случайный стих не из диапазона, документированного для
этого абзаца) — контроль на контаминацию по методике отчёта: если LLM
увереннее на золотых парах, чем на не менее правдоподобных фиктивных,
это подозрение на запоминание конкретных связей, а не текстовый анализ.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

TOP_K = 20


def load_gold_targets_and_refs(path: str) -> dict[str, list[tuple[str, int, int, int]]]:
    by_target = defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if not row["novel_verse_id"] or float(row["match_score"]) < 0.6:
            continue
        for ref in row["bible_refs"].split(";"):
            if not ref:
                continue
            book, chap, verses = ref.split(".", 2)
            vf, vt = (map(int, verses.split("-")) if "-" in verses else (int(verses), int(verses)))
            by_target[row["novel_verse_id"]].append((book, int(chap), vf, vt))
    return by_target


def is_positive(source_id: str, refs: list[tuple[str, int, int, int]]) -> bool:
    parts = source_id.split(".")
    if len(parts) != 4:
        return False
    _, book, chap, verse = parts
    chap, verse = int(chap), int(verse)
    return any(book == b and chap == c and vf <= verse <= vt for b, c, vf, vt in refs)


def top_k_per_target(path: str, targets: set[str], k: int) -> dict[str, list[str]]:
    rows = defaultdict(list)
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["target_id"] in targets:
            rows[row["target_id"]].append((float(row["score"]), row["source_id"]))
    return {t: [sid for _, sid in sorted(v, reverse=True)[:k]] for t, v in rows.items()}


def main(gold_path: str, out_path: str, all_bible_ids_path: str, *cand_paths: str) -> None:
    gold = load_gold_targets_and_refs(gold_path)
    targets = set(gold.keys())
    print(f"целевых абзацев: {len(targets)}", file=sys.stderr)

    all_bible_ids = [row["verse_id"] for row in
                     csv.DictReader(open(all_bible_ids_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)]

    union_by_target: dict[str, set[str]] = defaultdict(set)
    for path in cand_paths:
        top = top_k_per_target(path, targets, TOP_K)
        for t, sids in top.items():
            union_by_target[t].update(sids)

    rng = random.Random(11)
    pairs = []
    n_pos = 0
    for t in targets:
        for sid in union_by_target.get(t, []):
            pairs.append((t, sid))
            if is_positive(sid, gold[t]):
                n_pos += 1
        # один заведомо фиктивный кандидат на абзац -- контроль контаминации
        while True:
            fake_sid = rng.choice(all_bible_ids)
            if not is_positive(fake_sid, gold[t]):
                pairs.append((t, fake_sid))
                break

    print(f"пар всего: {len(pairs)}, из них положительных (по эталону): {n_pos}", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\n")
        for t, sid in pairs:
            f.write(f"{t}\t{sid}\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *sys.argv[4:])
