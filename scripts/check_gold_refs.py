"""Сверка ссылок эталона с цитатами внутри самих комментариев Тихомирова.

Три дефекта эталона подряд (нумерация Псалтири, привязка к абзацам, и вот это)
нашлись не в модели, а в данных, поэтому стоит проверить и ссылки. Тихомиров
часто называет место прямо в тексте примечания -- «(Лк. 17: 30)», «Пс. 142: 8»,
-- и это независимый источник: если разобранная ссылка расходится с тем, что
сказано в комментарии, одна из двух неверна.

Пример, с которого проверка началась: у места «Господи!» (CP.P1.V.58) в
bible_refs стоит Пс 142:3, а комментарий говорит «Очень близкий парафраз 8-го
стиха 142-го псалма ... „Господи! <...> Укажи мне путь, по которому идти“».
Стих 8, а не 3.

Сверяются только книга и глава с номером стиха; диапазоны в комментарии
учитываются. Расхождение -- не приговор: Тихомиров иногда ссылается в тексте на
параллельное место, а в bible_refs стоит основное. Поэтому вывод -- список для
глазами, а не автоматическая правка.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict

RU_BOOKS = {
    "быт": "GEN", "исх": "EXO", "лев": "LEV", "числ": "NUM", "втор": "DEU",
    "нав": "JOS", "суд": "JDG", "руф": "RUT", "пс": "PSA", "притч": "PRO",
    "еккл": "ECC", "песн": "SON", "ис": "ISA", "иер": "JER", "плач": "LAM",
    "иез": "EZE", "дан": "DAN", "ос": "HOS", "иоил": "JOE", "ам": "AMO",
    "авд": "OBA", "иона": "JON", "мих": "MIC", "наум": "NAH", "авв": "HAB",
    "соф": "ZEP", "агг": "HAG", "зах": "ZEC", "мал": "MAL", "иов": "JOB",
    "мф": "MAT", "мк": "MAR", "лк": "LUK", "ин": "JOH", "деян": "ACT",
    "рим": "ROM", "гал": "GAL", "еф": "EPH", "флп": "PHI", "кол": "COL",
    "тит": "TIT", "евр": "HEB", "иак": "JAM", "иуд": "JUD", "откр": "REV",
}
# «Лк. 17: 30», «Пс. 142: 8-9», «Мф. 5: 3»
CITE = re.compile(
    r"\b([А-Яа-я]{2,6})\.?\s*(\d{1,3})\s*[:,]\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?")


def parse_cites(text: str) -> set[tuple[str, int, int]]:
    out = set()
    for m in CITE.finditer(text or ""):
        book = RU_BOOKS.get(m.group(1).lower().rstrip("."))
        if not book:
            continue
        ch = int(m.group(2))
        v1 = int(m.group(3))
        v2 = int(m.group(4)) if m.group(4) else v1
        for v in range(v1, min(v2, v1 + 20) + 1):
            out.add((book, ch, v))
    return out


def parse_refs(field: str) -> set[tuple[str, int, int]]:
    out = set()
    for ref in field.split(";"):
        if not ref:
            continue
        parts = ref.split(".", 2)
        if len(parts) != 3:
            continue
        book, ch, verses = parts
        if "-" in verses:
            a, b = verses.split("-")
        else:
            a = b = verses
        try:
            for v in range(int(a), int(b) + 1):
                out.add((book, int(ch), v))
        except ValueError:
            continue
    return out


def main(gold_tsv: str, ref_col: str = "bible_refs_synodal") -> None:
    with open(gold_tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    n_with_cite = n_agree = n_disagree = 0
    disagree = []
    for r in rows:
        cites = parse_cites(r.get("commentary_snippet", ""))
        if not cites:
            continue
        n_with_cite += 1
        refs = parse_refs(r.get(ref_col) or r.get("bible_refs", ""))
        if cites & refs:
            n_agree += 1
        else:
            n_disagree += 1
            # совпадает ли хотя бы глава -- тогда расходится только номер стиха
            same_chapter = {(b, c) for b, c, _ in cites} & {(b, c) for b, c, _ in refs}
            disagree.append((r, cites, refs, bool(same_chapter)))

    print(f"записей эталона: {len(rows)}")
    print(f"из них с явной ссылкой внутри комментария: {n_with_cite}")
    print(f"  ссылка совпала с bible_refs: {n_agree}")
    print(f"  разошлась: {n_disagree}\n")
    for r, cites, refs, same_ch in sorted(disagree, key=lambda x: not x[3]):
        tag = "та же глава, другой стих" if same_ch else "другая глава или книга"
        c = ";".join(f"{b}.{ch}.{v}" for b, ch, v in sorted(cites))[:56]
        f_ = ";".join(f"{b}.{ch}.{v}" for b, ch, v in sorted(refs))[:56]
        print(f"  {r['novel_verse_id']:16s} [{tag}]")
        print(f"      в комментарии: {c}")
        print(f"      в bible_refs:  {f_}")
        print(f"      «{r['fragment'][:56]}»")


if __name__ == "__main__":
    main(*sys.argv[1:3])
