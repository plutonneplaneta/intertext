"""Жёсткие обманки для проверки LLM-судьи на контаминацию.

Случайный фиктивный стих проверяет мало: его отличает от золотого любой судья.
Здесь обманки подобраны так, чтобы быть правдоподобными:

  A  самые похожие по мнению самого конвейера неверные стихи из пула абзаца
     (средний процентиль по всем скорерам) -- проверка «рассуждает по тексту или
     повторяет мнение конвейера»;
  B  стихи той же главы, что и процитированный, но в стороне от цитируемого
     диапазона -- тот же контекст, другая мысль;
  C  стихи, процитированные Тихомировым для ДРУГОГО абзаца и попавшие в пул
     этого -- проверка на запомненные ассоциации, а не на текст;
  R  один случайный стих (для сравнимости с прежними числами);
  G  золотые стихи (до GOLD_PER_TARGET на абзац).

Вход: таблица пула fusion_retrieval.py (колонки *_pct), эталон, стихи Библии.
Единица -- пара (абзац, стих), как и везде в проекте.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

from oracle_diagnostics import load_gold, verse_matches

A_PER_TARGET = 3
B_PER_TARGET = 2
C_PER_TARGET = 2
GOLD_PER_TARGET = 4


def main(table_tsv: str, gold_tsv: str, bible_tsv: str, out_tsv: str) -> None:
    gold = load_gold(gold_tsv)
    groups_of_par = defaultdict(list)
    for par, group, _ in gold:
        groups_of_par[par].append(group)

    def positive(par: str, sid: str) -> bool:
        return any(verse_matches(sid, g) for g in groups_of_par[par])

    with open(bible_tsv, encoding="utf-8", newline="") as f:
        all_ids = [r["verse_id"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    id_set = set(all_ids)

    score: dict[tuple[str, str], float] = {}
    with open(table_tsv, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        pct = [c for c in rd.fieldnames if c.endswith("_pct")]
        for row in rd:
            score[(row["target_id"], row["source_id"])] = sum(float(row[c]) for c in pct) / len(pct)

    pool = defaultdict(list)
    for (par, sid), v in score.items():
        pool[par].append((v, sid))

    cited_anywhere: dict[str, set[str]] = defaultdict(set)
    for par, group, _ in gold:
        for sid in all_ids:
            if verse_matches(sid, group):
                cited_anywhere[par].add(sid)

    rng = random.Random(11)
    rows = []
    for par in sorted(groups_of_par):
        gold_ids = [sid for sid in all_ids if positive(par, sid)]
        for sid in gold_ids[:GOLD_PER_TARGET]:
            rows.append((par, sid, "G"))

        wrong = sorted(((v, s) for v, s in pool[par] if not positive(par, s)), reverse=True)
        for _, sid in wrong[:A_PER_TARGET]:
            rows.append((par, sid, "A"))

        near = []
        for sid in gold_ids:
            _, book, chap, verse = sid.split(".")
            for d in (4, 5, 6, 7, 8, -4, -5, -6, -7, -8):
                cand = f"b.{book}.{chap}.{int(verse) + d}"
                if cand in id_set and not positive(par, cand):
                    near.append(cand)
        near = list(dict.fromkeys(near))
        rng.shuffle(near)
        for sid in near[:B_PER_TARGET]:
            rows.append((par, sid, "B"))

        elsewhere = set().union(*[v for p, v in cited_anywhere.items() if p != par])
        cands = sorted(((v, s) for v, s in pool[par]
                        if s in elsewhere and not positive(par, s)), reverse=True)
        for _, sid in cands[:C_PER_TARGET]:
            rows.append((par, sid, "C"))

        while True:
            sid = rng.choice(all_ids)
            if not positive(par, sid):
                rows.append((par, sid, "R"))
                break

    seen, uniq = set(), []
    for r in rows:
        if (r[0], r[1]) not in seen:
            seen.add((r[0], r[1]))
            uniq.append(r)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tdecoy_type\n")
        for par, sid, t in uniq:
            f.write(f"{par}\t{sid}\t{t}\n")
    cnt = defaultdict(int)
    for *_, t in uniq:
        cnt[t] += 1
    print(f"пар {len(uniq)}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())) + f" -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
