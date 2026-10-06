"""Анализ результатов LLM-верификации: ранжирование внутри шорт-листа и
контроль на контаминацию (сравнение балла на золотых парах против
заведомо фиктивных -- вторая колонка build_shortlist.py, один "fake_"
кандидат на абзац, гарантированно вне диапазона стиха)."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


def load_gold_refs(path: str) -> dict[str, list[tuple[str, int, int, int]]]:
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


def main(gold_path: str, verify_path: str) -> None:
    gold = load_gold_refs(gold_path)

    by_target: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with open(verify_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row["llm_score"]:
                continue
            by_target[row["target_id"]].append((row["source_id"], int(row["llm_score"])))

    ranks = []
    hits_by_score_level: dict[int, int] = defaultdict(int)
    n_targets_with_pos = 0
    for tid, cands in by_target.items():
        refs = gold.get(tid, [])
        pos_scores = [s for sid, s in cands if is_positive(sid, refs)]
        if not pos_scores:
            continue
        n_targets_with_pos += 1
        best_pos_score = max(pos_scores)
        # ранг = 1 + число кандидатов со строго большим баллом (средний ранг
        # при связках: сколько кандидатов делят тот же балл, что и лучший позитив)
        higher = sum(1 for _, s in cands if s > best_pos_score)
        tied = sum(1 for _, s in cands if s == best_pos_score)
        rank = higher + (tied + 1) / 2
        ranks.append(rank)
        hits_by_score_level[best_pos_score] += 1

    ranks.sort()
    n = len(ranks)
    print(f"абзацев с положительным кандидатом в шорт-листе: {n_targets_with_pos}")
    print(f"медиана ранга лучшей верной пары (с учётом связок): {ranks[n//2]:.1f}" if n else "нет данных")
    print(f"топ-1: {sum(1 for r in ranks if r <= 1)}/{n}, топ-3: {sum(1 for r in ranks if r <= 3)}/{n}, "
          f"топ-5: {sum(1 for r in ranks if r <= 5)}/{n}")
    print(f"распределение лучшего балла на верной паре: {dict(sorted(hits_by_score_level.items()))}")

    # контроль контаминации: балл на фиктивных ("fake") кандидатах против
    # балла на настоящих золотых -- fake_sid определяется тем, что он НЕ
    # входит в положительный диапазон ни для одного абзаца по построению
    print("\n--- контроль контаминации ---")
    gold_scores, other_scores = [], []
    for tid, cands in by_target.items():
        refs = gold.get(tid, [])
        for sid, s in cands:
            if is_positive(sid, refs):
                gold_scores.append(s)
            else:
                other_scores.append(s)
    def stats(name, vals):
        if not vals:
            print(f"{name}: нет данных")
            return
        print(f"{name}: n={len(vals)}, средний={sum(vals)/len(vals):.2f}, "
              f"доля score>=2: {sum(1 for v in vals if v>=2)/len(vals):.2f}")
    stats("золотые пары (документированы Тихомировым)", gold_scores)
    stats("прочие кандидаты в шорт-листе (включая фиктивные)", other_scores)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
