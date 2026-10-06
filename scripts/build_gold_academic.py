"""Второй эталон: библейские примечания академического ПСС (комментаторы не Тихомиров).

Независимый комментатор и независимый текст: комментарии на rvb.ru к «Преступлению и
наказанию», «Идиоту», «Бесам», «Подростку» и «Братьям Карамазовым». Примечания оформлены
«С. 16. «Се человек!» -- слова Пилата ... (гл. 19, ст. 5)». Берутся только записи, где
названа книга и указан стих или диапазон стихов; ссылки на главу целиком и на диапазоны
глав отброшены (иначе «правильным» станет любой стих главы).

Книга ссылки -- ближайшее упоминание слева в той же записи («от Марка, гл. 5, ст. 25»,
«Откровение Иоанна, гл. 13; гл. 17, ст. 3-17»); «Стихи 38-40» без книги наследуют
последнюю полную ссылку.

Полный текст комментария в репозиторий не кладётся (издание под правами): скачивается
curl-ом, в эталон идут короткий фрагмент (до 45 знаков), ссылка и страница издания.

Привязка к абзацу -- по покрытию фрагмента словами абзаца; среди равных кандидатов
берётся ближайший к ожидаемому месту по номеру страницы издания (страницы идут по
роману равномерно в первом приближении). Часть и глава в комментарии не указаны.

Выход совместим с oracle_diagnostics.load_gold: novel_verse_id, match_score, fragment,
bible_refs (+ page, note).

Использование: build_gold_academic.py <comment.htm> <novel.tsv> <out.tsv> [префикс_id_абзацев]
"""
from __future__ import annotations

import csv
import re
import sys

WORD_RE = re.compile(r"[а-яё]+")

BOOKS = [
    (r"(?:Евангел\w*\s+)?от\s+Матфея|Мф", "MAT"), (r"(?:Евангел\w*\s+)?от\s+Марка|Мк", "MAR"),
    (r"(?:Евангел\w*\s+)?от\s+Луки|Лк", "LUK"), (r"(?:Евангел\w*\s+)?от\s+Иоанна|Ин\.", "JOH"),
    (r"Откровени\w*(?:\s+св\w*\.?)?(?:\s+Иоанна(?:\s+Богослова)?)?|Апокалипсис\w*", "REV"),
    (r"Деяни\w+\s+апостолов", "ACT"), (r"Бытие|Бытия", "GEN"), (r"Исход\w*", "EXO"),
    (r"книг\w+\s+Иова|Иова", "JOB"), (r"Псалом|Псалтир\w+|Пс\.", "PSA"),
    (r"Исаии|Исаия", "ISA"), (r"Иеремии", "JER"), (r"Иезекииля", "EZE"),
    (r"Притч\w+", "PRO"), (r"Екклесиаст\w*", "ECC"), (r"Послани\w+\s+к\s+Римлянам", "ROM"),
    (r"Послани\w+\s+к\s+Евреям", "HEB"),
]
BOOK_RE = re.compile("|".join(f"(?P<b{i}>{p})" for i, (p, _) in enumerate(BOOKS)))
REF_RE = re.compile(
    r"гл\.\s*(?P<ch>\d+)\s*[,.;]?\s*(?:ст(?:\.|ихи|иха)?\s*(?P<v1>\d+)(?:\s*[—–-]\s*(?P<v2>\d+))?)?"
    r"(?P<more>\s*[—–-]\s*\d+)?"
)
REL_RE = re.compile(r"Стих(?:и|а)?\s+(?P<v1>\d+)(?:\s*[—–-]\s*(?P<v2>\d+))?")
NOTE_SPLIT = re.compile(r"(?=С\.\s*\d+(?:\s*[—–-]\s*\d+)?\.?\s)")


def norm_words(t: str) -> list[str]:
    return WORD_RE.findall(t.lower().replace("ё", "е"))


def book_at(m: re.Match) -> str:
    for i, (_, code) in enumerate(BOOKS):
        if m.group(f"b{i}"):
            return code
    return ""


def parse_refs(comment: str, last: tuple | None):
    """Ссылки записи [(код, глава, стих1, стих2)] и последняя полная ссылка."""
    books = [(m.start(), book_at(m)) for m in BOOK_RE.finditer(comment)]
    refs = []
    for r in REF_RE.finditer(comment):
        if r.group("more") or not r.group("v1"):
            continue
        prior = [c for pos, c in books if pos < r.start()]
        if not prior:
            continue
        v1, v2 = int(r.group("v1")), r.group("v2")
        refs.append((prior[-1], int(r.group("ch")), v1, int(v2) if v2 and int(v2) > v1 else None))
    if refs:
        last = (refs[-1][0], refs[-1][1])
    elif last:
        rel = REL_RE.match(comment.strip())
        if rel:
            v1, v2 = int(rel.group("v1")), rel.group("v2")
            refs.append((last[0], last[1], v1, int(v2) if v2 and int(v2) > v1 else None))
    return refs, last


def fmt(ref) -> str:
    code, ch, v1, v2 = ref
    return f"{code}.{ch}.{v1}" + (f"-{v2}" if v2 else "")


def main(html_path: str, novel_tsv: str, out_tsv: str, id_prefix: str = "") -> None:
    raw = open(html_path, "rb").read().decode("windows-1251", errors="replace")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))
    first = re.search(r"С\.\s*\d+", text)
    notes = NOTE_SPLIT.split(text[first.start():] if first else text)

    pars = []
    with open(novel_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["verse_id"].startswith(id_prefix):
                w = norm_words(r["text"])
                pars.append((r["verse_id"], w, set(w)))

    parsed, pages = [], []
    for n in notes:
        m = re.match(r"С\.\s*(\d+)(?:\s*[—–-]\s*\d+)?\.?\s*(.*)", n)
        if m:
            pages.append(int(m.group(1)))
            parsed.append((m.group(1), m.group(2)))
    pmin, pmax = min(pages), max(pages)

    out, last = [], None
    for page, body in parsed:
        frag, sep, comment = body.partition(" — ")
        if not sep:
            continue
        refs, last = parse_refs(comment, last)
        if not refs:
            continue
        head = re.split(r"\s*~\s*", frag.strip("«»\"„“' ."))[0]
        words = norm_words(head)
        if len(words) < 2:
            continue
        covs = [sum(w in ws for w in words) / len(words) for _, _, ws in pars]
        top = max(covs)
        expected = (int(page) - pmin) / max(1, pmax - pmin) * len(pars)
        cands = [i for i, c in enumerate(covs) if c >= top - 0.02]
        pick = min(cands, key=lambda i: abs(i - expected))
        out.append((pars[pick][0], covs[pick], frag.strip()[:45], ";".join(dict.fromkeys(map(fmt, refs))),
                    page, comment.strip()[:60]))

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("novel_verse_id\tmatch_score\tfragment\tbible_refs\tpage\tnote\n")
        for r in out:
            f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2].replace(chr(9), ' ')}\t{r[3]}\t{r[4]}\t{r[5].replace(chr(9), ' ')}\n")
    ok = sum(1 for r in out if r[1] >= 0.6)
    print(f"записей с точной библейской ссылкой: {len(out)}, привязано (>=0,6): {ok} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
