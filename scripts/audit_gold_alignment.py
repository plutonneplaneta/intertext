"""Проверка привязки эталона к абзацам романа.

Зачем. build_gold.py искал абзац по цитируемому фрагменту в тексте, снятом с
ilibrary.ru (`crime_and_punishment.tsv`), а весь конвейер давно работает на
`cp_fb2.tsv` -- другой экстракции того же романа. Если разбиение на абзацы
разошлось, номера в эталоне указывают не туда, и эти места недостижимы ни одним
методом -- ровно как было с нумерацией Псалтири.

Проверяется двумя мерами:

  покрытие   доля фрагмента, покрытая совпадающими блоками (та же мера, что у
             build_gold.py -- чтобы числа были сравнимы с колонкой match_score)
  ряд        самый длинный совпадающий ОТРЕЗОК ПОДРЯД идущих слов, делённый на
             длину фрагмента. Мера строгая: покрытие набирается и из отдельных
             частых слов, а ряд из пяти слов подряд случайно не возникает

Дальше сравнивается, тот ли абзац назначен: лучший по покрытию внутри той же
части и главы, и лучший по всему роману.
"""
from __future__ import annotations

import csv
import re
import sys
from difflib import SequenceMatcher

WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")


def norm_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def containment(frag: list[str], text: list[str]) -> float:
    if not frag:
        return 0.0
    sm = SequenceMatcher(None, frag, text, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(frag)


def longest_run(frag: list[str], text: list[str]) -> int:
    if not frag or not text:
        return 0
    sm = SequenceMatcher(None, frag, text, autojunk=False)
    m = sm.find_longest_match(0, len(frag), 0, len(text))
    return m.size


def main(gold_tsv: str, novel_tsv: str, out_tsv: str) -> None:
    novel = []
    with open(novel_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            novel.append((row["verse_id"], norm_words(row["text"])))
    by_id = dict(novel)
    print(f"абзацев в романе: {len(novel)}", file=sys.stderr)

    with open(gold_tsv, encoding="utf-8", newline="") as f:
        gold = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    rows = []
    for g in gold:
        vid = g["novel_verse_id"]
        if not vid:
            continue
        frag = norm_words(g["fragment"])
        assigned = by_id.get(vid, [])
        cov_a = containment(frag, assigned)
        run_a = longest_run(frag, assigned)

        part, chap = vid.split(".")[1], vid.split(".")[2]
        best_ch, best_ch_id = 0.0, ""
        best_all, best_all_id = 0.0, ""
        for i, words in novel:
            c = containment(frag, words)
            if c > best_all:
                best_all, best_all_id = c, i
            p, ch = i.split(".")[1], i.split(".")[2]
            if p == part and ch == chap and c > best_ch:
                best_ch, best_ch_id = c, i
        rows.append({
            "novel_verse_id": vid,
            "match_score": g.get("match_score", ""),
            "n_frag_words": len(frag),
            "cov_assigned": round(cov_a, 3),
            "run_assigned": run_a,
            "run_share": round(run_a / len(frag), 3) if frag else 0.0,
            "best_in_chapter": best_ch_id,
            "cov_best_in_chapter": round(best_ch, 3),
            "best_in_novel": best_all_id,
            "cov_best_in_novel": round(best_all, 3),
            "is_argmax_chapter": int(best_ch_id == vid),
            "is_argmax_novel": int(best_all_id == vid),
            "fragment": g["fragment"][:60].replace("\t", " "),
        })

    n = len(rows)
    print(f"\nзаписей эталона с привязкой: {n}")
    print(f"назначенный абзац -- лучший в своей главе: "
          f"{sum(r['is_argmax_chapter'] for r in rows)}/{n}")
    print(f"назначенный абзац -- лучший по всему роману: "
          f"{sum(r['is_argmax_novel'] for r in rows)}/{n}")
    for thr in (0.9, 0.7, 0.5):
        print(f"покрытие фрагмента назначенным абзацем >= {thr}: "
              f"{sum(1 for r in rows if r['cov_assigned'] >= thr)}/{n}")
    for thr in (0.5, 0.3):
        print(f"самый длинный ряд подряд идущих слов >= {thr} от фрагмента: "
              f"{sum(1 for r in rows if r['run_share'] >= thr)}/{n}")
    print(f"ряд из >= 4 слов подряд: "
          f"{sum(1 for r in rows if r['run_assigned'] >= 4)}/{n}")

    bad = [r for r in rows if not r["is_argmax_chapter"] or r["run_assigned"] < 4]
    print(f"\nподозрительных записей: {len(bad)}")
    for r in sorted(bad, key=lambda x: x["run_assigned"])[:15]:
        print(f"  {r['novel_verse_id']:16s} match={r['match_score']:>4s} "
              f"покр={r['cov_assigned']:.2f} ряд={r['run_assigned']:>2d}/"
              f"{r['n_frag_words']:<3d} лучший в главе={r['best_in_chapter']:16s} "
              f"({r['cov_best_in_chapter']:.2f})  «{r['fragment'][:38]}»")

    cols = list(rows[0].keys())
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
