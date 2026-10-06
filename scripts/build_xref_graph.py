"""Пункт 1: граф перекрёстных ссылок Библии из внешнего справочника.

Источник -- openbible.info cross-references (CC-BY), производная Treasury of
Scripture Knowledge и подобных: около 340 тысяч связей стих-стих с числом
голосов. Сам файл в репозиторий не кладётся (лицензия требует указания
авторства, а файл большой и регенерируется одной командой):

    curl -sSL -o cross_references.zip https://a.openbible.info/data/cross-references.zip

Зачем. В graph_rerank.py рёбра «параллельное место» выводились из лексического
сходства внутри Евангелий -- их нашлось 1326, и они по определению покрывают
только то, что лексически похоже. Настоящий справочник даёт на два порядка
больше и, главное, связи Ветхого и Нового Заветов, которые лексически невидимы:
«покивание глав» у Марка и в 21-м псалме общих слов почти не имеют.

Две тонкости сопоставления с корпусом:

  обозначения книг  справочник пишет Gen/Ps/Matt/1Cor, корпус -- GEN/PSA/MAT/1CO.
                    Сопоставление однозначное, обе стороны по 66 книг.
  нумерация         справочник в масоретском счёте, как и корпус (см.
                    renumber_gold.py), поэтому номера берутся напрямую. Это
                    проверяется: доля ссылок, попавших в существующие стихи,
                    печатается -- при неверной нумерации она просела бы.
"""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from collections import defaultdict

BOOK_MAP = {
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM", "Deut": "DEU",
    "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT", "1Sam": "1SA", "2Sam": "2SA",
    "1Kgs": "1KI", "2Kgs": "2KI", "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR",
    "Neh": "NEH", "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SON", "Isa": "ISA", "Jer": "JER", "Lam": "LAM",
    "Ezek": "EZE", "Dan": "DAN", "Hos": "HOS", "Joel": "JOE", "Amos": "AMO",
    "Obad": "OBA", "Jonah": "JON", "Mic": "MIC", "Nah": "NAH", "Hab": "HAB",
    "Zeph": "ZEP", "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL",
    "Matt": "MAT", "Mark": "MAR", "Luke": "LUK", "John": "JOH", "Acts": "ACT",
    "Rom": "ROM", "1Cor": "1CO", "2Cor": "2CO", "Gal": "GAL", "Eph": "EPH",
    "Phil": "PHI", "Col": "COL", "1Thess": "1TH", "2Thess": "2TH",
    "1Tim": "1TI", "2Tim": "2TI", "Titus": "TIT", "Phlm": "PHM", "Heb": "HEB",
    "Jas": "JAM", "1Pet": "1PE", "2Pet": "2PE", "1John": "1JO", "2John": "2JO",
    "3John": "3JO", "Jude": "JUD", "Rev": "REV",
}
REF = re.compile(r"^([1-3]?[A-Za-z]+)\.(\d+)\.(\d+)$")


def parse_side(side: str) -> list[tuple[str, int, int]]:
    """«Rom.1.19-Rom.1.20» -> все стихи диапазона; «Gen.1.1» -> один."""
    parts = side.split("-")
    ends = []
    for p in parts:
        m = REF.match(p.strip())
        if not m:
            return []
        book = BOOK_MAP.get(m.group(1))
        if not book:
            return []
        ends.append((book, int(m.group(2)), int(m.group(3))))
    if len(ends) == 1:
        return ends
    (b1, c1, v1), (b2, c2, v2) = ends[0], ends[-1]
    if b1 != b2 or c1 != c2 or v2 < v1:
        return [ends[0], ends[-1]]
    return [(b1, c1, v) for v in range(v1, v2 + 1)]


def main(zip_path: str, bible_tsv: str, out_tsv: str, min_votes: str = "0") -> None:
    min_votes_i = int(min_votes)
    known = set()
    with open(bible_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            known.add(row["verse_id"])

    edges: dict[tuple[str, str], int] = defaultdict(int)
    n_lines = n_bad_book = n_missing = 0
    z = zipfile.ZipFile(zip_path)
    name = z.namelist()[0]
    with z.open(name) as f:
        f.readline()
        for line in f:
            cols = line.decode("utf-8").rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            n_lines += 1
            votes = 0
            if len(cols) > 2 and cols[2].strip().lstrip("-").isdigit():
                votes = int(cols[2])
            if votes < min_votes_i:
                continue
            left = parse_side(cols[0])
            right = parse_side(cols[1])
            if not left or not right:
                n_bad_book += 1
                continue
            for lb, lc, lv in left:
                a = f"b.{lb}.{lc}.{lv}"
                if a not in known:
                    n_missing += 1
                    continue
                for rb, rc, rv in right:
                    b = f"b.{rb}.{rc}.{rv}"
                    if b not in known:
                        n_missing += 1
                        continue
                    if a == b:
                        continue
                    key = (a, b) if a < b else (b, a)
                    edges[key] = max(edges[key], votes)

    print(f"строк в справочнике: {n_lines}", file=sys.stderr)
    print(f"нераспознанных обозначений книг: {n_bad_book}", file=sys.stderr)
    print(f"ссылок на стихи, которых нет в корпусе: {n_missing} "
          f"(при неверной нумерации это число было бы огромным)", file=sys.stderr)
    print(f"рёбер (без направления, дубликаты слиты): {len(edges)}", file=sys.stderr)

    # сколько рёбер соединяют разные Заветы -- именно они лексически невидимы
    ot = set(list(BOOK_MAP.values())[:39])
    cross_testament = sum(1 for (a, b) in edges
                          if (a.split(".")[1] in ot) != (b.split(".")[1] in ot))
    print(f"из них между Заветами: {cross_testament}", file=sys.stderr)

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("verse_a\tverse_b\tvotes\n")
        for (a, b), v in sorted(edges.items()):
            f.write(f"{a}\t{b}\t{v}\n")
    print(f"-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
