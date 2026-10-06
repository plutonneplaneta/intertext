"""Сборка эталона: примечания Тихомирова с библейской ссылкой -> место в
моём тексте романа (verse_id из crime_and_punishment.tsv).

Цитируемый фрагмент отделяется от комментария первым " — " (длинное тире
с пробелами по обе стороны — формат книги-комментария). Дальше ищем абзац
в нужной части/главе моего текста, содержащий наибольшую долю значимых
слов фрагмента подряд (устойчиво к разночтениям многоточия/пунктуации
между изданием Тихомирова и текстом с ilibrary.ru).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from difflib import SequenceMatcher

FRAGMENT_SPLIT_RE = re.compile(r"\s+—\s+")
WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")


def load_novel(tsv_path: str) -> list[dict]:
    rows = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append(row)
    return rows


def part_chapter_of(verse_id: str) -> tuple[str, str]:
    # CP.P1.I.7 -> ("P1", "I")
    parts = verse_id.split(".")
    return parts[1], parts[2]


def containment_score(frag_norm: str, text_norm: str) -> float:
    """Доля фрагмента, покрытая совпадающими блоками внутри text_norm —
    не штрафует за то, что candidate длиннее фрагмента (обычный случай:
    цитата — несколько слов внутри абзаца в несколько предложений)."""
    if not frag_norm:
        return 0.0
    sm = SequenceMatcher(None, frag_norm, text_norm, autojunk=False)
    covered = sum(block.size for block in sm.get_matching_blocks())
    return covered / len(frag_norm)


def best_match(fragment: str, candidates: list[dict]) -> tuple[dict | None, float]:
    frag_norm = " ".join(WORD_RE.findall(fragment.lower()))
    if not frag_norm:
        return None, 0.0
    best, best_score = None, 0.0
    for c in candidates:
        text_norm = " ".join(WORD_RE.findall(c["text"].lower()))
        score = containment_score(frag_norm, text_norm)
        if score > best_score:
            best_score, best = score, c
    return best, best_score


def main(notes_jsonl: str, novel_tsv: str, out_tsv: str) -> None:
    notes = [json.loads(l) for l in open(notes_jsonl, encoding="utf-8")]
    with_refs = [n for n in notes if n["bible_refs"]]
    novel = load_novel(novel_tsv)
    by_part_chapter: dict[tuple[str, str], list[dict]] = {}
    for row in novel:
        pc = part_chapter_of(row["verse_id"])
        by_part_chapter.setdefault(pc, []).append(row)

    out_rows = []
    unmatched = 0
    for n in with_refs:
        pc = (n["part"], n["chapter"])
        candidates = by_part_chapter.get(pc, [])
        fragment = FRAGMENT_SPLIT_RE.split(n["text"], maxsplit=1)[0]
        fragment = fragment.strip()
        match, score = best_match(fragment, candidates) if candidates else (None, 0.0)
        if not match or score < 0.6:
            unmatched += 1
        refs_str = ";".join(
            f"{r['book_code']}.{r['chapter']}.{r['verse_from']}"
            + (f"-{r['verse_to']}" if r["verse_to"] != r["verse_from"] else "")
            for r in n["bible_refs"]
        )
        out_rows.append({
            "novel_verse_id": match["verse_id"] if match else "",
            "match_score": f"{score:.2f}",
            "part": n["part"], "chapter": n["chapter"],
            "page_ref": n["page_ref"], "note_no": n["note_no"] or "",
            "fragment": fragment,
            "bible_refs": refs_str,
            "commentary_snippet": n["text"][len(fragment):][:300].replace("\n", " ").strip(),
        })

    # построчно, без csv.writer — те же основания, что в extract_bible.py:
    # поля содержат кавычки ", а не табы/переводы строк, и csv-модуль на
    # чтении по умолчанию склеивает строки по непарной кавычке
    fieldnames = list(out_rows[0].keys())
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in out_rows:
            values = [str(row[k]).replace("\t", " ").replace("\n", " ") for k in fieldnames]
            f.write("\t".join(values) + "\n")

    print(f"строк эталона: {len(out_rows)}, без уверенного совпадения (score<0.3): {unmatched}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
