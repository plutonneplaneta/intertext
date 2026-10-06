"""Синтаксический слой, вариант 2: тот же признак (общее ребро дерева разбора),
но несколько способов превратить совпадения в балл -- чтобы отделить вклад
самого признака от вклада нормировки на длину.

Базовый балл первой версии -- сумма редкости общих рёбер. Он растёт с длиной
абзаца механически: больше рёбер -> больше слагаемых (корреляция с длиной
0,44). Пост-обработка (Z-оценка по корзине длины) лечит это снаружи; здесь
проверяется, не лучше ли встроить нормировку в сам балл:

  sum_idf     сумма idf общих рёбер (как в syntax_candidates.py)
  cos_idf     косинус между idf-векторами рёбер абзаца и стиха -- деление на
              обе нормы убирает зависимость от длины по построению
  cover_src   доля веса стиха, воспроизведённая в абзаце: sum_idf / ||стих||_1.
              От длины абзаца не зависит вовсе, а содержательно это ровно то,
              что значит «цитата»: стих воспроизведён целиком, а не задет одним
              словом
  n_shared    сколько общих редких рёбер (без веса)
  max_edge    редкость самого редкого общего ребра -- одно очень редкое
              совпадение подозрительнее десяти средних

Кандидаты -- объединение топ-K по каждому варианту, а не топ-K по базовому:
иначе выборка была бы отобрана базовым баллом и остальные варианты
оценивались бы на его условиях.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict

TOP_K = 200
RARITY_DF_RATIO = 0.02


def load_units(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append((rec["id"], [tuple(e) for e in rec["edges"]]))
    return out


def main(source_jsonl: str, target_jsonl: str, out_tsv: str) -> None:
    source = load_units(source_jsonl)
    target = load_units(target_jsonl)
    print(f"источник: {len(source)}, цель: {len(target)}", file=sys.stderr)

    edge_sets_src = [set(e) for _, e in source]
    edge_sets_tgt = [set(e) for _, e in target]
    n_docs = len(edge_sets_src) + len(edge_sets_tgt)
    df: Counter = Counter()
    for s in edge_sets_src:
        df.update(s)
    for s in edge_sets_tgt:
        df.update(s)
    idf = {e: math.log(n_docs / c) for e, c in df.items()}
    rare = {e for e, c in df.items() if c / n_docs <= RARITY_DF_RATIO}
    print(f"рёбер {len(df)}, редких {len(rare)}", file=sys.stderr)

    # нормы по всем рёбрам единицы (не только редким): знаменатель косинуса
    # должен отражать весь объём единицы, иначе длинный абзац опять в выигрыше
    src_l1 = [sum(idf[e] for e in s) or 1e-9 for s in edge_sets_src]
    src_l2 = [math.sqrt(sum(idf[e] ** 2 for e in s)) or 1e-9 for s in edge_sets_src]
    tgt_l2 = [math.sqrt(sum(idf[e] ** 2 for e in s)) or 1e-9 for s in edge_sets_tgt]

    index: dict[tuple, list[int]] = defaultdict(list)
    for j, s in enumerate(edge_sets_src):
        for e in s:
            if e in rare:
                index[e].append(j)

    n_rows = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tsum_idf\tcos_idf\tcover_src\tn_shared\tmax_edge\n")
        for i, (tid, _) in enumerate(target):
            acc_sum: dict[int, float] = defaultdict(float)
            acc_sq: dict[int, float] = defaultdict(float)
            acc_n: dict[int, int] = defaultdict(int)
            acc_max: dict[int, float] = defaultdict(float)
            for e in edge_sets_tgt[i]:
                hits = index.get(e)
                if not hits:
                    continue
                w = idf[e]
                for j in hits:
                    acc_sum[j] += w
                    acc_sq[j] += w * w
                    acc_n[j] += 1
                    if w > acc_max[j]:
                        acc_max[j] = w
            if not acc_sum:
                continue

            variants = {}
            for j, s in acc_sum.items():
                variants[j] = (
                    s,
                    acc_sq[j] / (tgt_l2[i] * src_l2[j]),
                    s / src_l1[j],
                    float(acc_n[j]),
                    acc_max[j],
                )
            keep: set[int] = set()
            for col in range(5):
                top = sorted(variants, key=lambda j: -variants[j][col])[:TOP_K]
                keep.update(top)
            for j in keep:
                v = variants[j]
                f.write(f"{tid}\t{source[j][0]}\t{v[0]:.4f}\t{v[1]:.6f}\t"
                        f"{v[2]:.6f}\t{int(v[3])}\t{v[4]:.4f}\n")
                n_rows += 1
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(target)}, строк {n_rows}", file=sys.stderr, flush=True)

    print(f"строк: {n_rows} -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
