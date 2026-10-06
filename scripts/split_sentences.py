"""Роман по предложениям, а не по абзацам: TSV с id вида CP.P1.II.17#3.

Зачем. Абзац Достоевского -- от 5 до тысячи слов, и это ломает всё разом:
балл любого слоя растёт с длиной механически, а внутри длинного абзаца
цитата тонет среди посторонних предложений (особенно у синтаксического
слоя, где балл -- сумма по всем рёбрам абзаца, и рёбра исповеди
Мармеладова складываются с рёбрами описания лестницы). Предложение --
единица сопоставимого объёма и естественная единица цитирования.

Границы предложений берутся из разбора spaCy (nlp_annotate.py, поле sents),
а не из регулярного выражения по точке: сокращения и прямая речь у
Достоевского на каждой странице.
"""
from __future__ import annotations

import json
import sys


def main(ann_jsonl: str, out_tsv: str, min_words: int = 3) -> None:
    n_units = n_sents = 0
    with open(ann_jsonl, encoding="utf-8") as f, \
            open(out_tsv, "w", encoding="utf-8") as out:
        out.write("verse_id\ttext\n")
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_units += 1
            for k, sent in enumerate(rec["sents"]):
                s = sent.replace("\t", " ").replace("\n", " ").strip()
                if len(s.split()) < min_words:
                    continue
                out.write(f"{rec['id']}#{k}\t{s}\n")
                n_sents += 1
    print(f"абзацев {n_units} -> предложений {n_sents} (от {min_words} слов) -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 3)
