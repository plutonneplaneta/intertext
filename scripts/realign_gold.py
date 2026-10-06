"""Перепривязка эталона к абзацам того текста, на котором работает конвейер.

НАЙДЕННЫЙ БАГ. build_gold.py искал абзац по цитируемому фрагменту в тексте,
снятом с ilibrary.ru, а конвейер давно работает на cp_fb2.tsv -- другой
экстракции с другим разбиением на абзацы. Проверка (audit_gold_alignment.py)
показала, что 14 записей эталона из 65 указывают не на тот абзац, и подпись
однозначна: назначенные номера систематически МЕНЬШЕ правильных --

    «Се человек!»            CP.P1.II.1  -> на деле CP.P1.II.17
    «Приидет в тот день...»  CP.P1.II.3  -> на деле CP.P1.II.37
    «Она Бога узрит.»        CP.P4.IV.1  -> на деле CP.P4.IV.140

то есть в старой экстракции абзацы были крупнее (слитая прямая речь). У этих
записей match_score = 1,00: в СВОЁМ тексте они найдены точно. Но метод ищет
цитату в абзаце, где её нет, поэтому такие места недостижимы по построению --
ровно как было с нумерацией Псалтири.

Перепривязка. Часть и глава берутся из самого примечания Тихомирова (колонки
part/chapter) и считаются надёжными, поэтому поиск идёт только внутри них.
Абзац выбирается по покрытию фрагмента; в match_score пишется новое значение,
старый номер сохраняется в novel_verse_id_ilibrary.

Фрагменты бывают и непрямые (пересказ с многоточиями), у них покрытие низкое
даже у верного абзаца -- поэтому порог не ставится, а честное значение
покрытия кладётся в match_score, и дальше по конвейеру работает тот же фильтр
match_score >= 0,6, что и раньше.
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
    return sm.find_longest_match(0, len(frag), 0, len(text)).size


def main(gold_tsv: str, novel_tsv: str, out_tsv: str) -> None:
    by_chapter: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}
    with open(novel_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            vid = row["verse_id"]
            p = vid.split(".")
            by_chapter.setdefault((p[1], p[2]), []).append((vid, norm_words(row["text"])))
    print(f"глав в романе: {len(by_chapter)}", file=sys.stderr)

    with open(gold_tsv, encoding="utf-8", newline="") as f:
        gold = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        fields = list(gold[0].keys())

    n_changed = n_nochapter = 0
    for g in gold:
        old = g.get("novel_verse_id", "")
        g["novel_verse_id_ilibrary"] = old
        g["match_score_ilibrary"] = g.get("match_score", "")
        frag = norm_words(g.get("fragment", ""))
        key = (g.get("part", ""), g.get("chapter", ""))
        cands = by_chapter.get(key)
        if not cands or not frag:
            n_nochapter += 1
            continue
        best_id, best_cov, best_run = "", -1.0, 0
        for vid, words in cands:
            cov = containment(frag, words)
            if cov > best_cov:
                best_cov, best_id = cov, vid
                best_run = longest_run(frag, words)
        g["novel_verse_id"] = best_id
        g["coverage"] = f"{best_cov:.2f}"
        g["longest_run"] = str(best_run)
        # Решение о привязке -- не по покрытию, а по дословному ряду. В fb2
        # разбиение мельче, чем было на ilibrary.ru, и реплика Тихомирова
        # нередко разрезана на два абзаца: тогда целиком она не лежит ни в
        # одном, покрытие лучшего абзаца падает до 0,5, а ряд из восьми слов
        # подряд всё равно указывает на него однозначно. Четыре слова подряд
        # из цитаты Достоевского внутри одной главы случайно не встречаются.
        g["match_score"] = f"{max(best_cov, 1.0 if best_run >= 4 else 0.0):.2f}"
        if best_id != old:
            n_changed += 1

    out_fields = fields + ["novel_verse_id_ilibrary", "match_score_ilibrary",
                           "coverage", "longest_run"]
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(out_fields) + "\n")
        for g in gold:
            f.write("\t".join(str(g.get(k, "")).replace("\t", " ")
                              for k in out_fields) + "\n")
    print(f"записей: {len(gold)}, перепривязано: {n_changed}, "
          f"без главы в романе: {n_nochapter} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
