"""Первая настоящая проверка RQ1: находит ли метод (find_candidates.py)
места, документированные Тихомировым, — по полноте (recall), без всякой
подгонки порога на этом же наборе (это pilot dev, не test — цифра
предварительная, честная настройка будет отдельно на разбиении dev/test)."""
from __future__ import annotations

import csv
import sys


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_candidates(path: str) -> dict[str, list[tuple[str, float]]]:
    by_target: dict[str, list[tuple[str, float]]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            by_target.setdefault(row["target_id"], []).append(
                (row["source_id"], float(row["score"]))
            )
    return by_target


def verse_in_range(verse_id: str, book_code: str, chapter: int, v_from: int, v_to: int) -> bool:
    # verse_id формат "b.MAR.9.25"
    parts = verse_id.split(".")
    if len(parts) != 4:
        return False
    _, book, chap, verse = parts
    return book == book_code and int(chap) == chapter and v_from <= int(verse) <= v_to


def main(gold_path: str, candidates_path: str) -> None:
    gold = load_gold(gold_path)
    candidates = load_candidates(candidates_path)

    n_with_match = 0
    n_checkable = 0
    ranks = []
    misses = []
    for g in gold:
        if not g["novel_verse_id"] or float(g["match_score"]) < 0.6:
            continue
        refs = [r for r in g["bible_refs"].split(";") if r]
        if not refs:
            continue
        n_checkable += 1
        cands = sorted(candidates.get(g["novel_verse_id"], []), key=lambda x: -x[1])
        found_rank = None
        for rank, (src_id, score) in enumerate(cands, start=1):
            for ref in refs:
                book, chap, verses = ref.split(".", 2)
                if "-" in verses:
                    v_from, v_to = map(int, verses.split("-"))
                else:
                    v_from = v_to = int(verses)
                if verse_in_range(src_id, book, int(chap), v_from, v_to):
                    found_rank = rank
                    break
            if found_rank:
                break
        if found_rank:
            n_with_match += 1
            ranks.append(found_rank)
        else:
            misses.append((g["novel_verse_id"], g["bible_refs"], len(cands)))

    print(f"проверяемых (эталон с уверенным совпадением и ссылкой): {n_checkable}")
    print(f"найдено методом среди кандидатов вообще: {n_with_match} ({100*n_with_match/n_checkable:.0f}%)")
    if ranks:
        ranks.sort()
        print(f"ранг находки (место в списке кандидатов для этого абзаца): медиана {ranks[len(ranks)//2]}, "
              f"на первом месте: {sum(1 for r in ranks if r==1)}/{len(ranks)}")
    print(f"\nпримеры пропущенных методом ({len(misses)}):")
    for tid, refs, n_cands in misses[:15]:
        print(f"  {tid} refs=[{refs}] (кандидатов у этого абзаца всего: {n_cands})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
