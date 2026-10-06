"""Лемматизация корпуса для поиска заимствований.

Вход — TSV с колонками id, text (по строке на единицу: стих или абзац).
Выход — JSONL: {"id": ..., "lemmas": [...], "pos": [...], "raw": "..."}.

Работает на глобальном интерпретаторе сервера (spaCy 3.8.3 + ru_core_news_lg),
отдельного окружения не создаём — переиспользуем то, что уже установлено для
основного проекта, только на чтение.
"""
from __future__ import annotations

import csv
import json
import sys
import time

import spacy

# Служебные части речи — не несут признака заимствования сами по себе,
# но не исключаются из леммы полностью (для синтаксической части понадобятся);
# здесь только помечаются, отсечение — на этапе индексации.
FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "PRON", "DET", "AUX"}


def load_csl_dict(path: str | None) -> dict[str, str]:
    """Словарь нормализации церковнославянских/архаичных форм к современной
    лемме — нужен потому, что ru_core_news_lg обучен на современном русском
    и архаичную форму лемматизирует как саму себя ("дщерь" -> "дщерь"),
    из-за чего лексическое сопоставление с современным Синодальным
    переводом структурно невозможно именно там, где библейских цитат
    больше всего (см. находку по исповеди Мармеладова)."""
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
    text_col: str = "text",
    id_col: str = "verse_id",
    csl_dict_path: str | None = None,
) -> None:
    csl_dict = load_csl_dict(csl_dict_path)
    if csl_dict:
        print(f"словарь архаизмов: {len(csl_dict)} форм", file=sys.stderr)
    nlp = spacy.load("ru_core_news_lg", disable=["ner"])
    rows = []
    with open(in_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            rows.append((row[id_col], row[text_col]))

    t0 = time.time()
    n_written = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        # nlp.pipe для скорости, порциями, чтобы видеть прогресс в журнале
        texts = [r[1] for r in rows]
        ids = [r[0] for r in rows]
        for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
            toks = [tok for tok in doc if not tok.is_space and not tok.is_punct]
            lemmas = [
                csl_dict.get(tok.text.lower(), tok.lemma_.lower())
                for tok in toks
            ]
            pos = [tok.pos_ for tok in toks]
            out.write(json.dumps({"id": ids[i], "lemmas": lemmas, "pos": pos, "raw": texts[i]}, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 2000 == 0:
                elapsed = time.time() - t0
                print(f"{n_written}/{len(rows)} за {elapsed:.0f}с", file=sys.stderr, flush=True)
    print(f"готово: {n_written} единиц за {time.time() - t0:.0f}с -> {out_jsonl}")


if __name__ == "__main__":
    in_tsv, out_jsonl = sys.argv[1], sys.argv[2]
    id_col = sys.argv[3] if len(sys.argv) > 3 else "verse_id"
    csl_path = sys.argv[4] if len(sys.argv) > 4 else None
    main(in_tsv, out_jsonl, id_col=id_col, csl_dict_path=csl_path)
