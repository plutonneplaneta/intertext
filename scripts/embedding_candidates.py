"""Кандидаты по семантической близости: топ-K библейских стихов на каждый
абзац романа, по косинусу нормализованных векторов E5 (просто скалярное
произведение). Формат вывода совпадает с find_candidates.py (score, n,
ngram, target_id, source_id, target_text, source_text), чтобы eval_recall.py
работал без изменений — n/ngram здесь не значат ничего, оставлены пустыми.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

TOP_K = 1000
# Было 20 (здесь 50). Измерение потолка (reports/ceiling_scan.txt) показало, что
# слой способен достать правильный стих у 41% мест эталона, но при выдаче в 20
# кандидатов достаёт 9%: полнота слоя определялась этой константой, а не
# качеством векторов. Тысяча кандидатов на абзац -- это 3% корпуса, файл
# кандидатов растёт линейно и остаётся в сотнях МБ.


def load_raw_text(tsv_path: str, id_col: str = "verse_id") -> dict[str, str]:
    out = {}
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            out[row[id_col]] = row["text"]
    return out


def main(source_npz: str, target_npz: str, source_tsv: str, target_tsv: str, out_tsv: str) -> None:
    src = np.load(source_npz, allow_pickle=True)
    tgt = np.load(target_npz, allow_pickle=True)
    src_ids, src_emb = src["ids"], src["embeddings"]
    tgt_ids, tgt_emb = tgt["ids"], tgt["embeddings"]
    print(f"источник: {len(src_ids)}, цель: {len(tgt_ids)}", file=sys.stderr)

    src_text = load_raw_text(source_tsv)
    tgt_text = load_raw_text(target_tsv)

    # нормализованы при кодировании -> косинус = скалярное произведение
    sims = tgt_emb @ src_emb.T  # (n_target, n_source)
    print(f"матрица сходства: {sims.shape}", file=sys.stderr)

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("score\tn\tngram\ttarget_id\tsource_id\ttarget_text\tsource_text\n")
        for i, tid in enumerate(tgt_ids):
            row = sims[i]
            top_idx = np.argpartition(row, -TOP_K)[-TOP_K:]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            for j in top_idx:
                sid = src_ids[j]
                score = float(row[j])
                tt = tgt_text.get(tid, "").replace("\t", " ").replace("\n", " ")
                st = src_text.get(sid, "").replace("\t", " ").replace("\n", " ")
                f.write(f"{score:.4f}\t\t\t{tid}\t{sid}\t{tt}\t{st}\n")

    print(f"готово -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
