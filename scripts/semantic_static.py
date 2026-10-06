"""Семантический слой на статических векторах ru_core_news_lg.

ВАЖНО про сопоставимость. В пилоте этот слой считался на multilingual-e5-base.
В текущем окружении huggingface.co закрыт политикой сети, поэтому здесь --
взвешенное среднее статических векторов spaCy (300 измерений, 500 тыс. форм),
а не обученный кодировщик предложений. Это заведомо более слабый семантический
слой, и его абсолютные числа с пилотными не сравниваются. Сравнивать можно
только устройство: тот же слой при разных способах агрегации и нормировки.

Взвешивание SIF (a / (a + p(слово))): частое слово получает малый вес, так что
«и сказал» не перетягивает вектор на себя. Затем из всех векторов вычитается
первая главная компонента -- она у усреднённых векторов почти всегда кодирует
«общий текст на русском» и одинакова у всех единиц, то есть только добавляет
всем сходства, не различая ничего.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
import spacy

SIF_A = 1e-3
SKIP_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT"}
TOP_K = 1000
# Было 20 (здесь 50). Измерение потолка (reports/ceiling_scan.txt) показало, что
# слой способен достать правильный стих у 41% мест эталона, но при выдаче в 20
# кандидатов достаёт 9%: полнота слоя определялась этой константой, а не
# качеством векторов. Тысяча кандидатов на абзац -- это 3% корпуса, файл
# кандидатов растёт линейно и остаётся в сотнях МБ.
TOP_R = 10


def load_units(path: str):
    ids, lemmas, poss = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["id"])
            lemmas.append(r["lemmas"])
            poss.append(r.get("pos", ["X"] * len(r["lemmas"])))
    return ids, lemmas, poss


def build(nlp, ids, lemmas, poss, freq: Counter, total: int) -> np.ndarray:
    dim = nlp.vocab.vectors.shape[1]
    out = np.zeros((len(ids), dim), dtype=np.float32)
    for i, (lem, pos) in enumerate(zip(lemmas, poss)):
        acc = np.zeros(dim, dtype=np.float32)
        wsum = 0.0
        for l, p in zip(lem, pos):
            if p in SKIP_POS:
                continue
            lx = nlp.vocab[l]
            if not lx.has_vector:
                continue
            w = SIF_A / (SIF_A + freq[l] / total)
            acc += w * lx.vector
            wsum += w
        if wsum > 0:
            out[i] = acc / wsum
    return out


def drop_first_pc(mats: list[np.ndarray]) -> None:
    stacked = np.vstack(mats)
    stacked = stacked - stacked.mean(axis=0, keepdims=True)
    # первая главная компонента через степенной метод -- SVD на 45 тыс. x 300
    # считается и так, но степенной метод не держит вторую копию матрицы
    v = np.random.default_rng(0).normal(size=stacked.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(30):
        v = stacked.T @ (stacked @ v)
        v /= np.linalg.norm(v) + 1e-12
    for m in mats:
        m -= np.outer(m @ v, v)


def normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def main(source_jsonl: str, target_jsonl: str, out_tsv: str,
         csls: str = "0") -> None:
    nlp = spacy.load("ru_core_news_lg", disable=["ner", "parser", "tagger"])
    s_ids, s_lem, s_pos = load_units(source_jsonl)
    t_ids, t_lem, t_pos = load_units(target_jsonl)
    print(f"источник {len(s_ids)}, цель {len(t_ids)}", file=sys.stderr)

    freq: Counter = Counter()
    for lst in s_lem:
        freq.update(lst)
    for lst in t_lem:
        freq.update(lst)
    total = sum(freq.values())

    S = build(nlp, s_ids, s_lem, s_pos, freq, total)
    T = build(nlp, t_ids, t_lem, t_pos, freq, total)
    drop_first_pc([S, T])
    S, T = normalize(S), normalize(T)

    use_csls = csls != "0"
    chunk = 512
    # средний косинус стиха с его лучшими абзацами -- та же поправка на
    # хабность, что в rerank_attribution.py, но встроенная в слой
    src_top = np.zeros(len(s_ids), dtype=np.float32)
    if use_csls:
        best = np.full((len(s_ids), TOP_R), -1.0, dtype=np.float32)
        for a in range(0, len(t_ids), chunk):
            sims = (T[a:a + chunk] @ S.T).T  # (n_src, chunk)
            merged = np.concatenate([best, sims], axis=1)
            part = np.partition(merged, -TOP_R, axis=1)[:, -TOP_R:]
            best = part
        src_top = best.mean(axis=1)

    n_rows = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("score\tn\tngram\ttarget_id\tsource_id\n")
        for a in range(0, len(t_ids), chunk):
            sims = T[a:a + chunk] @ S.T
            if use_csls:
                k = min(TOP_R, sims.shape[1])
                tgt_top = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
                sims = 2 * sims - src_top[None, :] - tgt_top[:, None]
            for r in range(sims.shape[0]):
                row = sims[r]
                top = np.argpartition(row, -TOP_K)[-TOP_K:]
                top = top[np.argsort(-row[top])]
                for j in top:
                    f.write(f"{row[j]:.5f}\t\t\t{t_ids[a+r]}\t{s_ids[j]}\n")
                    n_rows += 1
            if (a // chunk) % 4 == 0:
                print(f"{a}/{len(t_ids)}", file=sys.stderr, flush=True)
    print(f"строк: {n_rows} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
