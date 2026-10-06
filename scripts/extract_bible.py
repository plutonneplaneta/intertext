"""Разбор Russian.xml (bible-corpus, CES-разметка) в построчный TSV: id стиха, текст.

Формат id — b.KNIGA.ГЛАВА.СТИХ (английские трёхбуквенные коды книг), сохраняем
как есть — это и есть точная адресация для эталона и для генератора кандидатов.
"""
from __future__ import annotations

import csv
import html
import re
import sys
from pathlib import Path

SEG_RE = re.compile(r'<seg id="([^"]+)" type="verse">(.*?)</seg>', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def main(xml_path: str, out_tsv: str) -> None:
    raw = Path(xml_path).read_text(encoding="utf-8")
    rows = []
    for m in SEG_RE.finditer(raw):
        verse_id, text = m.group(1), m.group(2)
        text = TAG_RE.sub("", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            rows.append((verse_id, text))
    # без csv.writer: поля не содержат ни табов, ни переводов строк (уже
    # схлопнуты \s+ выше), а вот кавычки " встречаются часто — csv-модуль
    # на запись с QUOTE_NONE и кавычкой внутри поля требует escapechar,
    # а на чтение по умолчанию (без QUOTE_NONE) интерпретирует " как начало
    # многострочного поля и склеивает соседние стихи. Простая построчная
    # запись/чтение табуляцией снимает вопрос целиком.
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("verse_id\ttext\n")
        for verse_id, text in rows:
            f.write(f"{verse_id}\t{text}\n")
    books = sorted({r[0].split(".")[1] for r in rows})
    print(f"стихов: {len(rows)}, книг: {len(books)}")
    print("книги:", ", ".join(books))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
