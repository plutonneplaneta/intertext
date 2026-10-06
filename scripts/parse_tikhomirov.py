"""Разбор книги-комментария Тихомирова в структурированный эталон.

Формат примечаний: "Стр. NN[-MM].(k) <цитируемый фрагмент романа> — <комментарий>"
Часть и глава отслеживаются по running-заголовкам "ЧАСТЬ ..." / "ГЛАВА ..."
и по одиночным римским номерам на своей строке (внутритекстовые маркеры глав).
Библейские ссылки в комментарии — по стандартным русским сокращениям книг.
"""
from __future__ import annotations

import csv
import json
import re
import sys

import fitz

PART_WORDS = {
    "ПЕРВАЯ": "P1", "ВТОРАЯ": "P2", "ТРЕТЬЯ": "P3",
    "ЧЕТВЕРТАЯ": "P4", "ПЯТАЯ": "P5", "ШЕСТАЯ": "P6",
}
ROMAN_CHAPTER_RE = re.compile(r"^\s*([IVXLC]{1,6})\s*$")
PART_HEADER_RE = re.compile(r"ЧАСТЬ\s+(ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ|ПЯТАЯ|ШЕСТАЯ)")
GLAVA_HEADER_RE = re.compile(r"^\s*ГЛАВА\s+([IVXLC]{1,6})\s*$")
EPILOGUE_RE = re.compile(r"ЭПИЛОГ")
NOTE_START_RE = re.compile(r"Стр\.\s*(\d+(?:[–-]\d+)?)\.(?:\((\d+)\))?\s*")

# сокращение книги (русское, синодальное) -> трёхбуквенный код как в моём bible_verses.tsv
BOOK_ABBR = {
    "Быт": "GEN", "Исх": "EXO", "Лев": "LEV", "Чис": "NUM", "Втор": "DEU",
    "Иис.Нав": "JOS", "Нав": "JOS", "Суд": "JDG", "Руф": "RUT",
    "1Цар": "1SA", "2Цар": "2SA", "3Цар": "1KI", "4Цар": "2KI",
    "1Пар": "1CH", "2Пар": "2CH", "Езд": "EZR", "Неем": "NEH", "Есф": "EST",
    "Иов": "JOB", "Пс": "PSA", "Притч": "PRO", "Еккл": "ECC", "Песн": "SON",
    "Ис": "ISA", "Иер": "JER", "Плач": "LAM", "Иез": "EZE", "Дан": "DAN",
    "Ос": "HOS", "Иоил": "JOE", "Ам": "AMO", "Авд": "OBA", "Иона": "JON",
    "Мих": "MIC", "Наум": "NAH", "Авв": "HAB", "Соф": "ZEP", "Агг": "HAG",
    "Зах": "ZEC", "Мал": "MAL",
    "Мф": "MAT", "Мк": "MAR", "Лк": "LUK", "Ин": "JOH", "Деян": "ACT",
    "Рим": "ROM", "1Кор": "1CO", "2Кор": "2CO", "Гал": "GAL", "Еф": "EPH",
    "Флп": "PHI", "Кол": "COL", "1Фес": "1TH", "2Фес": "2TH",
    "1Тим": "1TI", "2Тим": "2TI", "Тит": "TIT", "Флм": "PHM", "Евр": "HEB",
    "Иак": "JAM", "1Пет": "1PE", "2Пет": "2PE",
    "1Ин": "1JO", "2Ин": "2JO", "3Ин": "3JO", "Иуд": "JUD", "Откр": "REV",
}
# в порядке убывания длины ключа, чтобы "1Кор" не терялось на фоне "Кор" и т.п.
BOOK_KEYS_SORTED = sorted(BOOK_ABBR.keys(), key=len, reverse=True)
BIBLE_REF_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BOOK_KEYS_SORTED) + r")\.?\s*"
    r"(\d{1,3})\s*[:\.]\s*(\d{1,3})(?:[–\-](\d{1,3}))?"
)


DEHYPHEN_RE = re.compile(r"([а-яё])-\n([а-яё])")


def extract_full_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    text = "\n".join(parts)
    # PDF переносит слова по слогам на границе строки печатного издания;
    # "разнолич-\nного" должно читаться как одно слово при сопоставлении
    text = DEHYPHEN_RE.sub(r"\1\2", text)
    return text


def find_bible_refs(text: str) -> list[dict]:
    refs = []
    for m in BIBLE_REF_RE.finditer(text):
        book_ru, chap, v1, v2 = m.groups()
        code = BOOK_ABBR.get(book_ru)
        if not code:
            continue
        refs.append({
            "book_ru": book_ru, "book_code": code,
            "chapter": int(chap), "verse_from": int(v1),
            "verse_to": int(v2) if v2 else int(v1),
            "matched": m.group(0),
        })
    return refs


def segment_notes(full_text: str) -> list[dict]:
    lines = full_text.split("\n")
    current_part = None
    current_chapter = None
    # первый проход: линейно идём по строкам, отмечаем часть/главу,
    # склеиваем текст в один поток с метками места на каждой строке
    tagged_lines = []
    for line in lines:
        if EPILOGUE_RE.search(line):
            if current_part != "EP":
                current_chapter = None
            current_part = "EP"
        else:
            m = PART_HEADER_RE.search(line)
            if m:
                new_part = PART_WORDS[m.group(1)]
                if new_part != current_part:
                    current_chapter = None
                current_part = new_part
        # два независимых сигнала главы: повторяющийся заголовок "ГЛАВА N"
        # и одиночный римский номер на своей строке (внутритекстовый маркер,
        # точнее по месту, но реже встречается) — берём любой, что сработал
        m3 = GLAVA_HEADER_RE.match(line)
        if m3:
            current_chapter = m3.group(1)
        else:
            m2 = ROMAN_CHAPTER_RE.match(line)
            if m2:
                current_chapter = m2.group(1)
        tagged_lines.append((current_part, current_chapter, line))

    joined = "\n".join(l for _, _, l in tagged_lines)
    # позиции начала каждой строки в joined, чтобы потом найти part/chapter по индексу совпадения
    offsets = []
    pos = 0
    for _, _, l in tagged_lines:
        offsets.append(pos)
        pos += len(l) + 1

    def part_chapter_at(idx: int) -> tuple[str | None, str | None]:
        # находим последнюю строку, чей offset <= idx
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return tagged_lines[lo][0], tagged_lines[lo][1]

    starts = list(NOTE_START_RE.finditer(joined))
    notes = []
    for i, m in enumerate(starts):
        start = m.start()
        end = starts[i + 1].start() if i + 1 < len(starts) else min(start + 4000, len(joined))
        chunk = joined[start:end]
        part, chapter = part_chapter_at(start)
        page_ref, note_no = m.group(1), m.group(2)
        body = chunk[m.end() - start:]
        refs = find_bible_refs(body)
        notes.append({
            "part": part, "chapter": chapter,
            "page_ref": page_ref, "note_no": note_no,
            "text": body[:2000],
            "bible_refs": refs,
        })
    return notes


def main(pdf_path: str, out_jsonl: str) -> None:
    full_text = extract_full_text(pdf_path)
    notes = segment_notes(full_text)
    with_refs = [n for n in notes if n["bible_refs"]]
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for n in notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"всего примечаний: {len(notes)}, с библейской ссылкой: {len(with_refs)}")
    parts_seen = sorted({n["part"] for n in notes if n["part"]})
    print("части, встреченные в разметке:", parts_seen)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
