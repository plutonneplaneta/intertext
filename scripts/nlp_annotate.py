"""Один проход spaCy вместо трёх: леммы, части речи, рёбра разбора и границы
предложений в одном JSONL.

lemmatize.py и parse_deps.py гоняли ru_core_news_lg по одному и тому же
корпусу дважды, а парсер -- самая дорогая часть конвейера. Формат вывода
надмножество обоих: find_candidates.py читает "lemmas", syntax_candidates.py
читает "edges", и оба работают с этим файлом без изменений. Добавлены
"sents" -- предложения абзаца: они нужны семантическому слою (косинус по
целому абзацу усредняет цитату с окружающей прозой, по предложению -- нет).
"""
from __future__ import annotations

import csv
import json
import sys
import time

import spacy

SKIP_DEP = {"punct", "det", "case"}


def load_csl_dict(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            mapping[row["form"].lower()] = row["modern_lemma"]
    return mapping


def main(in_tsv: str, out_jsonl: str, id_col: str = "verse_id",
         csl_dict_path: str | None = None, n_process: int = 1) -> None:
    csl = load_csl_dict(csl_dict_path)
    if csl:
        print(f"словарь архаизмов: {len(csl)} форм", file=sys.stderr)

    nlp = spacy.load("ru_core_news_lg", disable=["ner"])

    rows = []
    with open(in_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append((row[id_col], row["text"]))
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    def lemma_of(tok) -> str:
        return csl.get(tok.text.lower(), tok.lemma_.lower())

    t0 = time.time()
    with open(out_jsonl, "w", encoding="utf-8") as out:
        pipe = nlp.pipe(texts, batch_size=32, n_process=n_process)
        for i, doc in enumerate(pipe):
            sent_index = {}
            for si, sent in enumerate(doc.sents):
                for t in sent:
                    sent_index[t.i] = si

            toks = [t for t in doc if not t.is_space and not t.is_punct]
            lemmas = [lemma_of(t) for t in toks]
            pos = [t.pos_ for t in toks]
            # номер предложения для каждой леммы -- чтобы семантический слой
            # мог считать вектор предложения, а не всего абзаца
            sent_of = [sent_index.get(t.i, 0) for t in toks]

            # топология разбора в индексах отфильтрованного списка лемм:
            # тройки (голова, связь, зависимое) по леммам её теряют -- леммы
            # повторяются, и восстановить путь между двумя словами по ним
            # нельзя. Для графовых признаков (пары слов, соединённых путём
            # длины <= k, стянутые предложные рёбра) нужны именно индексы.
            pos_of = {t.i: k for k, t in enumerate(toks)}
            heads = [pos_of.get(t.head.i, -1) for t in toks]
            rels = [t.dep_ for t in toks]
            # лемма предлога, висящего на слове -- для стягивания рёбер
            adp_of = [""] * len(toks)
            for t in doc:
                if t.pos_ == "ADP" and t.head.i in pos_of:
                    adp_of[pos_of[t.head.i]] = lemma_of(t)

            edges = []
            for tok in doc:
                if tok.dep_ == "ROOT" or tok.dep_ in SKIP_DEP:
                    continue
                if tok.is_space or tok.is_punct or tok.head.is_punct:
                    continue
                h, c = lemma_of(tok.head), lemma_of(tok)
                if h == c:
                    continue
                edges.append([h, tok.dep_, c])

            sents = [s.text.strip() for s in doc.sents if s.text.strip()]

            out.write(json.dumps({
                "id": ids[i], "lemmas": lemmas, "pos": pos,
                "edges": edges, "sents": sents, "sent_of": sent_of,
                "heads": heads, "rels": rels, "adp_of": adp_of,
                "raw": texts[i],
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                print(f"{i+1}/{len(rows)} за {time.time()-t0:.0f}с", file=sys.stderr, flush=True)

    print(f"готово: {len(rows)} единиц за {time.time()-t0:.0f}с -> {out_jsonl}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "verse_id",
         a[3] if len(a) > 3 and a[3] != "-" else None,
         int(a[4]) if len(a) > 4 else 1)
