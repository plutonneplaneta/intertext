"""Извлечение чистого текста романа из глав ilibrary.ru (кодировка windows-1251).

Каждая страница содержит <div id="text" class="t hya">, внутри — необязательный
<h2>Часть N</h2>, обязательный <h3>римская цифра</h3> (номер главы в части) и
абзацы в тегах <z>...</z>. Сохраняем структуру: часть/глава как метки, абзацы —
построчно, чтобы потом можно было сослаться на точное место цитаты.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"<[^>]+>")
Z_RE = re.compile(r"<z>(.*?)</z>", re.S)
H2_RE = re.compile(r"<h2>(.*?)</h2>", re.S)
H3_RE = re.compile(r"<h3>(.*?)</h3>", re.S)
TEXT_DIV_RE = re.compile(r'<div id="text" class="t hya">(.*?)<div id="thdr"', re.S)
# на первой странице между text-div и thdr не всегда есть контент по такой границе;
# берём весь блок до </body> с запасом и режем по последующим служебным div вручную
BODY_CUT_RE = re.compile(r'<div id="bnav"', re.S)


def clean_fragment(raw: str) -> str:
    raw = raw.replace("<o></o>", "")
    raw = TAG_RE.sub("", raw)
    raw = html.unescape(raw)
    raw = raw.replace("\xa0", " ")
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def extract_chapter(html_text: str) -> tuple[str | None, str | None, list[str]]:
    part = None
    m = H2_RE.search(html_text)
    if m:
        part = clean_fragment(m.group(1))
    chapter = None
    m = H3_RE.search(html_text)
    if m:
        chapter = clean_fragment(m.group(1))
    paragraphs = [clean_fragment(p) for p in Z_RE.findall(html_text)]
    paragraphs = [p for p in paragraphs if p]
    return part, chapter, paragraphs


PART_ABBR = {
    "Часть первая": "P1", "Часть вторая": "P2", "Часть третья": "P3",
    "Часть четвертая": "P4", "Часть пятая": "P5", "Часть шестая": "P6",
    "Эпилог": "EP",
}


def main(chapters_dir: str, out_txt: str, out_tsv: str | None = None) -> None:
    files = sorted(
        Path(chapters_dir).glob("p*.html"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()),
    )
    out_lines: list[str] = []
    tsv_rows: list[tuple[str, str]] = []
    current_part = None
    total_paragraphs = 0
    for f in files:
        raw = f.read_bytes().decode("windows-1251", errors="replace")
        part, chapter, paragraphs = extract_chapter(raw)
        if not paragraphs:
            print(f"ПУСТО: {f.name}", file=sys.stderr)
            continue
        if part and part != current_part:
            out_lines.append(f"\n=== {part} ===\n")
            current_part = part
        if chapter:
            out_lines.append(f"\n--- Глава {chapter} ---\n")
        out_lines.extend(paragraphs)
        total_paragraphs += len(paragraphs)
        part_abbr = PART_ABBR.get(current_part, "X")
        for i, p in enumerate(paragraphs, start=1):
            para_id = f"CP.{part_abbr}.{chapter or '?'}.{i}"
            tsv_rows.append((para_id, p))
    Path(out_txt).write_text("\n".join(out_lines), encoding="utf-8")
    if out_tsv:
        with open(out_tsv, "w", encoding="utf-8", newline="") as f:
            f.write("verse_id\ttext\n")
            for pid, text in tsv_rows:
                # некоторые абзацы — вложенные стихотворные вставки с <br>,
                # после снятия тегов остаются внутренние переводы строк;
                # схлопываем в пробелы, чтобы строка TSV оставалась одной строкой
                flat = " ".join(text.split("\n")).strip()
                flat = re.sub(r"\s+", " ", flat)
                if flat:
                    f.write(f"{pid}\t{flat}\n")
    words = sum(len(line.split()) for line in out_lines)
    print(f"файлов: {len(files)}, абзацев: {total_paragraphs}, слов приблизительно: {words}")


if __name__ == "__main__":
    out_tsv = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], out_tsv)
