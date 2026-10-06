"""Извлечение текста романа из FB2 (чистый и полный источник — взамен
постраничного скрейпинга ilibrary.ru, который резался на 16 КБ при частых
запросах). Структура: <body><section><title>Часть N</title><p>...</p>...
Главы внутри части не отдельные <section>, а помечены одиночным <p> вида
"I. ", "II. " и т. д. — как в бумажном издании.
"""
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET

NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
CHAPTER_MARKER_RE = re.compile(r"^\s*([IVXLC]{1,6})\.?\s*$")
PART_TITLE_RE = re.compile(r"Часть\s*(\d+)")
PART_ABBR = {"1": "P1", "2": "P2", "3": "P3", "4": "P4", "5": "P5", "6": "P6"}


def main(fb2_path: str, out_txt: str, out_tsv: str) -> None:
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    body = root.find("fb:body", NS)

    out_lines: list[str] = []
    tsv_rows: list[tuple[str, str]] = []

    for section in body.findall("fb:section", NS):
        title_el = section.find("fb:title", NS)
        title_text = "".join(title_el.itertext()).strip() if title_el is not None else ""
        m = PART_TITLE_RE.search(title_text)
        part_abbr = PART_ABBR.get(m.group(1)) if m else ("EP" if "Эпилог" in title_text else "X")

        out_lines.append(f"\n=== {title_text} ===\n")
        current_chapter = None
        para_idx = 0

        for p in section.findall("fb:p", NS):
            text = "".join(p.itertext())
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            cm = CHAPTER_MARKER_RE.match(text)
            if cm:
                current_chapter = cm.group(1)
                out_lines.append(f"\n--- Глава {current_chapter} ---\n")
                para_idx = 0
                continue
            para_idx += 1
            out_lines.append(text)
            chapter_tag = current_chapter or "0"
            para_id = f"CP.{part_abbr}.{chapter_tag}.{para_idx}"
            tsv_rows.append((para_id, text))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("verse_id\ttext\n")
        for pid, text in tsv_rows:
            f.write(f"{pid}\t{text}\n")

    words = sum(len(t.split()) for t in out_lines)
    print(f"абзацев: {len(tsv_rows)}, слов приблизительно: {words}")
    parts_seen = sorted({row[0].split(".")[1] for row in tsv_rows})
    print("части:", parts_seen)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
