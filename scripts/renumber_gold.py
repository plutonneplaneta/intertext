"""Перевод ссылок эталона из синодальной нумерации в нумерацию корпуса.

НАЙДЕННЫЙ БАГ. Корпус `bible_verses.tsv` (из XML bible-corpus) пронумерован по
масоретскому счёту, как KJV: Псалтирь -- 150 глав и 2461 стих. Тихомиров же
ссылается по синодальному счёту, который следует Септуагинте и на большей части
Псалтири отстаёт на одну главу. Поэтому эталонный `PSA.21.7-9` («покивание
глав», поругание) указывал в корпусе на «ибо царь уповает на Господа» -- другой
псалом. Шесть мест эталона из 64 были недостижимы ни одним методом по
построению и всё это время стояли в знаменателе каждой цифры полноты.

Исправляется без внешних таблиц: корпус сам несёт синодальный номер в начале
текста стиха, например

    b.PSA.22.7  ->  "(21:8) Все, видящие меня, ругаются надо мною..."

Отсюда строится точное соответствие (книга, синод. глава, синод. стих) ->
verse_id корпуса. В одном стихе корпуса скобочных номеров бывает несколько
(b.PSA.22.1 покрывает синодальные 21:1 и 21:2), поэтому берутся все.

Правило приоритета. Если у книги скобочная нумерация больше чем у половины
стихов (это Псалтирь), она считается основной: ссылка сначала ищется по карте.
Иначе основной считается прямая нумерация, а карта -- дополнительным
псевдонимом. Так книги, где расходятся единичные стихи (Иов, Числа, Даниил),
не ломаются.

Выход -- gold_standard_v5.tsv того же формата: всё дальше по конвейеру
работает без изменений.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict

ALT = re.compile(r"\((\d+)[:\-](\d+)[a-zа-я]?\)")
ALT_MAJOR_SHARE = 0.5


def build_alt_map(bible_tsv: str):
    """(книга, глава, стих) в чужой нумерации -> список verse_id корпуса."""
    alt: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    total = Counter()
    with_alt = Counter()
    with open(bible_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            vid = row["verse_id"]
            book = vid.split(".")[1]
            total[book] += 1
            found = ALT.findall(row["text"])
            if not found:
                continue
            with_alt[book] += 1
            for ch, v in found:
                alt[(book, int(ch), int(v))].append(vid)
    alt_books = {b for b in total
                 if with_alt[b] / total[b] > ALT_MAJOR_SHARE}
    return alt, alt_books, total, with_alt


def direct_exists(direct: set[str], book: str, ch: int, v: int) -> bool:
    return f"b.{book}.{ch}.{v}" in direct


def resolve(book: str, ch: int, v: int, alt, alt_books, direct) -> list[str]:
    """Ссылка Тихомирова -> verse_id корпуса."""
    key = (book, ch, v)
    if book in alt_books:
        hit = alt.get(key)
        if hit:
            return hit
        # в карте нет (например, псалом, где нумерация совпадает) -- прямой путь
        return [f"b.{book}.{ch}.{v}"] if direct_exists(direct, book, ch, v) else []
    if direct_exists(direct, book, ch, v):
        return [f"b.{book}.{ch}.{v}"]
    return alt.get(key, [])


def compress(ids: list[str]) -> list[str]:
    """Список verse_id -> ссылки формата BOOK.CH.V или BOOK.CH.V1-V2."""
    parsed = sorted({(p[1], int(p[2]), int(p[3]))
                     for p in (i.split(".") for i in ids)})
    out = []
    i = 0
    while i < len(parsed):
        b, c, v = parsed[i]
        j = i
        while (j + 1 < len(parsed) and parsed[j + 1][0] == b
               and parsed[j + 1][1] == c and parsed[j + 1][2] == parsed[j][2] + 1):
            j += 1
        out.append(f"{b}.{c}.{v}" if j == i else f"{b}.{c}.{v}-{parsed[j][2]}")
        i = j + 1
    return out


def main(gold_tsv: str, bible_tsv: str, out_tsv: str) -> None:
    alt, alt_books, total, with_alt = build_alt_map(bible_tsv)
    direct = set()
    with open(bible_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            direct.add(row["verse_id"])
    print(f"книги с чужой нумерацией как основной: {sorted(alt_books)}", file=sys.stderr)
    print(f"записей в карте: {len(alt)}", file=sys.stderr)

    n_changed = n_refs = n_lost = 0
    with open(gold_tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        fields = list(rows[0].keys()) if rows else []

    for row in rows:
        new_refs = []
        for ref in row["bible_refs"].split(";"):
            if not ref:
                continue
            n_refs += 1
            book, chap, verses = ref.split(".", 2)
            if "-" in verses:
                vf, vt = map(int, verses.split("-"))
            else:
                vf = vt = int(verses)
            ids: list[str] = []
            for v in range(vf, vt + 1):
                ids.extend(resolve(book, int(chap), v, alt, alt_books, direct))
            if not ids:
                n_lost += 1
                new_refs.append(ref)
                continue
            mapped = compress(ids)
            if mapped != [ref]:
                n_changed += 1
            new_refs.extend(mapped)
        row["bible_refs_synodal"] = row["bible_refs"]
        row["bible_refs"] = ";".join(dict.fromkeys(new_refs))

    out_fields = fields + ["bible_refs_synodal"]
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(out_fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")).replace("\t", " ")
                              for k in out_fields) + "\n")
    print(f"ссылок всего {n_refs}, переписано {n_changed}, "
          f"не разрешилось {n_lost} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
