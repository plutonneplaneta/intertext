"""Точечные поправки эталона, каждая с обоснованием.

Три дефекта эталона в этом проекте правились скриптами, потому что были
систематическими (нумерация Псалтири, разбиение на абзацы). Остаются единичные,
где ошибку видно только из текста самого комментария. Править их прямо в
gold_standard_v4.tsv нельзя -- он воспроизводится из PDF Тихомирова и является
входными данными; поэтому поправки лежат отдельным файлом с колонкой evidence,
и любую можно проверить, не поверив на слово.

Ключ поправки -- страница книги и номер примечания (page_ref + note_no):
novel_verse_id для ключа не годится, он сам бывает предметом поправки.
"""
from __future__ import annotations

import csv
import sys


def main(gold_tsv: str, corrections_tsv: str, out_tsv: str) -> None:
    with open(corrections_tsv, encoding="utf-8", newline="") as f:
        corr = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    with open(gold_tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        fields = list(rows[0].keys())

    applied = skipped = 0
    for c in corr:
        hit = False
        for r in rows:
            if (r.get("page_ref") != c["page_ref"]
                    or r.get("note_no") != c["note_no"]):
                continue
            if r.get(c["field"]) != c["old_value"]:
                print(f"ПРОПУЩЕНО: {c['field']} у примечания {c['page_ref']}/"
                      f"{c['note_no']} сейчас «{r.get(c['field'])}», а поправка "
                      f"ждала «{c['old_value']}» -- данные изменились",
                      file=sys.stderr)
                skipped += 1
                hit = True
                break
            r[c["field"]] = c["new_value"]
            applied += 1
            hit = True
            break
        if not hit:
            print(f"ПРОПУЩЕНО: примечание {c['page_ref']}/{c['note_no']} не найдено",
                  file=sys.stderr)
            skipped += 1

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")).replace("\t", " ")
                              for k in fields) + "\n")
    print(f"поправок применено {applied}, пропущено {skipped} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
