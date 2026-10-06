"""Нормировка балла на длину абзаца: не деление (короткий абзац дал бы
абсурдный всплеск на шуме), а Z-оценка относительно фона внутри своего
диапазона длины -- насколько балл абзаца необычен для абзацев ЕГО объёма,
а не для романа в целом.

Фон (среднее и стандартное отклонение по корзине длины) считается ТОЛЬКО
по не-эталонным абзацам -- иначе настоящие цитаты подмешались бы в
оценку своего же фона и искусственно сблизили бы себя с ним.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

N_BINS = 10


def load_lengths(novel_tsv: str) -> dict[str, int]:
    out = {}
    for row in csv.DictReader(open(novel_tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        out[row["verse_id"]] = len(row["text"].split())
    return out


def load_max_scores(path: str, all_ids: set[str]) -> dict[str, float]:
    """max score по каждому target_id; 0.0, если ни разу не встретился как
    target вовсе (метод не нашёл для него ни одного кандидата)."""
    out = {tid: 0.0 for tid in all_ids}
    for row in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        tid, s = row["target_id"], float(row["score"])
        if tid in out and s > out[tid]:
            out[tid] = s
    return out


def make_bins(lengths: dict[str, int], n_bins: int) -> dict[str, int]:
    """id -> номер корзины по квантилям длины."""
    ids_sorted = sorted(lengths, key=lambda i: lengths[i])
    n = len(ids_sorted)
    bin_of = {}
    for rank, tid in enumerate(ids_sorted):
        bin_of[tid] = min(n_bins - 1, rank * n_bins // n)
    return bin_of


def zscore_by_bin(scores: dict[str, float], bin_of: dict[str, int], gold_ids: set[str]) -> dict[str, float]:
    by_bin_bg: dict[int, list[float]] = defaultdict(list)
    for tid, b in bin_of.items():
        if tid not in gold_ids:  # фон -- только не-эталон
            by_bin_bg[b].append(scores[tid])

    stats = {}
    for b, vals in by_bin_bg.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var ** 0.5 or 1e-6
        stats[b] = (mean, std)

    return {tid: (scores[tid] - stats[bin_of[tid]][0]) / stats[bin_of[tid]][1] for tid in scores}


def main(novel_tsv: str, gold_tsv: str, method_name: str, candidates_tsv: str) -> None:
    lengths = load_lengths(novel_tsv)
    all_ids = set(lengths.keys())

    gold_ids = set()
    for row in csv.DictReader(open(gold_tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["novel_verse_id"] and float(row["match_score"]) >= 0.6:
            gold_ids.add(row["novel_verse_id"])

    raw = load_max_scores(candidates_tsv, all_ids)
    bin_of = make_bins(lengths, N_BINS)
    z = zscore_by_bin(raw, bin_of, gold_ids)

    # корреляция длины с сырым баллом и с Z-оценкой -- проверка, что
    # нормировка действительно убрала эффект длины
    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        return cov / (sx * sy) if sx and sy else 0.0

    ids = list(all_ids)
    len_list = [lengths[i] for i in ids]
    raw_list = [raw[i] for i in ids]
    z_list = [z[i] for i in ids]
    print(f"[{method_name}] корреляция длина~сырой балл: {corr(len_list, raw_list):.3f}, "
          f"длина~Z-оценка: {corr(len_list, z_list):.3f}")

    gold_z = sorted((z[i] for i in gold_ids if i in z), reverse=True)
    non_gold_z = sorted((z[i] for i in all_ids - gold_ids), reverse=True)
    print(f"[{method_name}] Z-оценка медиана: эталон={gold_z[len(gold_z)//2]:.2f} "
          f"не-эталон={non_gold_z[len(non_gold_z)//2]:.2f}")
    print(f"[{method_name}] Z-оценка топ-3: эталон={[round(x,2) for x in gold_z[:3]]} "
          f"не-эталон={[round(x,2) for x in non_gold_z[:3]]}")

    out_path = candidates_tsv.replace(".tsv", "_zscore.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("target_id\tlength\traw_max\tzscore\tis_gold\n")
        for tid in all_ids:
            f.write(f"{tid}\t{lengths[tid]}\t{raw[tid]:.3f}\t{z[tid]:.3f}\t{int(tid in gold_ids)}\n")
    print(f"-> {out_path}\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
