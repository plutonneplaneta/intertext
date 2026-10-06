"""Синтаксические связи (управляющее -- зависимое), независимо от порядка
слов -- слой, дополняющий лексические n-граммы (которым порядок нужен) и
эмбеддинги (которые усредняют весь абзац). Единица сопоставления -- ребро
дерева разбора: (лемма вершины, тип связи, лемма зависимого). Хранится
леммой, а не словоформой -- иначе разбор ru_core_news_lg на архаичных формах
(см. находку по Мармеладову) обесценивает и эту часть тоже.

Как и в lemmatize.py: spaCy на CPU (GPU занята задачей основного проекта).
"""
from __future__ import annotations

import csv
import json
import sys
import time

import spacy

# связи, которые почти всегда шумят (интерпункция, детерминаторы и т.п. --
# не несут содержательного отношения между леммами)
SKIP_DEP = {"punct", "det", "case"}


def load_csl_dict(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            mapping[row["form"].lower()] = row["modern_lemma"]
    return mapping


def main(
    in_tsv: str,
    out_jsonl: str,
    id_col: str = "verse_id",
    csl_dict_path: str | None = None,
) -> None:
    csl_dict = load_csl_dict(csl_dict_path)
    if csl_dict:
        print(f"словарь архаизмов: {len(csl_dict)} форм", file=sys.stderr)

    nlp = spacy.load("ru_core_news_lg", disable=["ner"])

    rows = []
    with open(in_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append((row[id_col], row["text"]))

    def lemma_of(tok) -> str:
        return csl_dict.get(tok.text.lower(), tok.lemma_.lower())

    t0 = time.time()
    texts = [r[1] for r in rows]
    ids = [r[0] for r in rows]
    n_written = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
            edges = []
            for tok in doc:
                if tok.dep_ == "ROOT" or tok.dep_ in SKIP_DEP:
                    continue
                if tok.is_space or tok.is_punct or tok.head.is_punct:
                    continue
                head_l, child_l = lemma_of(tok.head), lemma_of(tok)
                if head_l == child_l:
                    continue
                edges.append([head_l, tok.dep_, child_l])
            out.write(json.dumps(
                {"id": ids[i], "edges": edges, "raw": texts[i]}, ensure_ascii=False,
            ) + "\n")
            n_written += 1
            if n_written % 2000 == 0:
                print(f"{n_written}/{len(rows)} за {time.time()-t0:.0f}с", file=sys.stderr, flush=True)

    print(f"готово: {n_written} единиц за {time.time()-t0:.0f}с -> {out_jsonl}")


if __name__ == "__main__":
    in_tsv, out_jsonl = sys.argv[1], sys.argv[2]
    id_col = sys.argv[3] if len(sys.argv) > 3 else "verse_id"
    csl_path = sys.argv[4] if len(sys.argv) > 4 else None
    main(in_tsv, out_jsonl, id_col=id_col, csl_dict_path=csl_path)
