"""Лексический слой, вариант 2: к точному совпадению леммной n-граммы
добавлены (а) n-граммы с одним пропуском и (б) признаки помимо максимума.

Зачем пропуск. Первая версия ловит только дословную формулу. Достоевский
чаще меняет одно слово внутри устойчивого оборота -- и вся n-грамма
рассыпается. Шаблон с одной «дыркой» внутри (не по краям: край и так даёт
более короткую точную n-грамму) ловит такую подстановку, не открывая шлюз
для случайных совпадений: остальные позиции по-прежнему должны сойтись
буквально, и среди них должна быть редкая лемма.

Зачем признаки помимо максимума. max по абзацу -- максимум из N попыток,
и он растёт с N (то есть с длиной) даже на чистом шуме. Количество
совпадений и покрытие стиха от длины абзаца зависят иначе, так что вместе
они позволяют отличить «длинный абзац случайно задел стих» от «стих
воспроизведён».

  best_idf       максимальная суммарная редкость совпавшей n-граммы (база)
  best_n         длина этой n-граммы
  best_gap_idf   то же среди шаблонов с одним пропуском (0, если таких нет)
  n_matches      сколько различных n-грамм совпало с этим стихом
  sum_top3       сумма трёх лучших различных совпадений
  cover_src      best_idf / суммарная редкость лемм стиха -- какую долю стиха
                 объясняет совпадение; от длины абзаца не зависит
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict

MIN_N, MAX_N = 3, 6
GAP_N = (5, 6)          # шаблоны с пропуском только для длинных окон
RARITY_DF_RATIO = 0.02
MIN_SCORE = 6.0
TOP_K = 200


def h(parts) -> int:
    return int.from_bytes(
        hashlib.blake2b("\x00".join(parts).encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def load_units(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append((rec["id"], rec["lemmas"]))
    return out


def ngrams(lemmas, n):
    return [tuple(lemmas[i:i + n]) for i in range(len(lemmas) - n + 1)]


def gap_patterns(ng: tuple[str, ...]):
    """Шаблоны с одной внутренней «дыркой»: (a,*,c,d) и т.п."""
    for k in range(1, len(ng) - 1):
        yield ng[:k] + ("\x01",) + ng[k + 1:]


def main(source_jsonl: str, target_jsonl: str, out_tsv: str) -> None:
    source = load_units(source_jsonl)
    target = load_units(target_jsonl)
    print(f"источник: {len(source)}, цель: {len(target)}", file=sys.stderr)

    n_docs = len(source) + len(target)
    df: Counter[str] = Counter()
    for _, lem in source:
        df.update(set(lem))
    for _, lem in target:
        df.update(set(lem))
    idf = {l: math.log(n_docs / c) for l, c in df.items()}
    rare = {l for l, c in df.items() if c / n_docs <= RARITY_DF_RATIO}
    print(f"лемм {len(idf)}, редких {len(rare)}", file=sys.stderr)

    src_l1 = [sum(idf.get(l, 0.0) for l in set(lem)) or 1e-9 for _, lem in source]

    # индексы по хешу шаблона -- тюплы строк в словаре на миллионы записей
    # съедают гигабайты, хеш занимает 8 байт
    exact: dict[int, list[int]] = defaultdict(list)
    gapped: dict[int, list[int]] = defaultdict(list)
    for j, (_, lem) in enumerate(source):
        for n in range(MIN_N, MAX_N + 1):
            for ng in ngrams(lem, n):
                if not any(l in rare for l in ng):
                    continue
                exact[h(ng)].append(j)
                if n in GAP_N:
                    for pat in gap_patterns(ng):
                        if any(l in rare for l in pat if l != "\x01"):
                            gapped[h(pat)].append(j)
    print(f"точных шаблонов {len(exact)}, с пропуском {len(gapped)}", file=sys.stderr)

    n_rows = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tbest_idf\tbest_n\tbest_gap_idf\t"
                "n_matches\tsum_top3\tcover_src\n")
        for i, (tid, lem) in enumerate(target):
            # по каждому стиху -- список баллов различных совпавших n-грамм
            hits_exact: dict[int, list[float]] = defaultdict(list)
            best_n: dict[int, int] = defaultdict(int)
            hits_gap: dict[int, float] = defaultdict(float)

            for n in range(MIN_N, MAX_N + 1):
                for ng in ngrams(lem, n):
                    score = sum(idf.get(l, 0.0) for l in ng)
                    if score < MIN_SCORE:
                        continue
                    for j in exact.get(h(ng), ()):
                        hits_exact[j].append(score)
                        if n > best_n[j]:
                            best_n[j] = n
                    if n in GAP_N:
                        for pat in gap_patterns(ng):
                            sc = score - idf.get(ng[pat.index("\x01")], 0.0)
                            if sc < MIN_SCORE:
                                continue
                            for j in gapped.get(h(pat), ()):
                                if sc > hits_gap[j]:
                                    hits_gap[j] = sc

            cands = set(hits_exact) | set(hits_gap)
            if not cands:
                continue
            feats = {}
            for j in cands:
                lst = sorted(hits_exact.get(j, []), reverse=True)
                best = lst[0] if lst else 0.0
                feats[j] = (
                    best, best_n.get(j, 0), hits_gap.get(j, 0.0),
                    len(lst), sum(lst[:3]), best / src_l1[j],
                )
            keep: set[int] = set()
            for col in (0, 2, 3, 4, 5):
                keep.update(sorted(feats, key=lambda j: -feats[j][col])[:TOP_K])
            for j in keep:
                v = feats[j]
                f.write(f"{tid}\t{source[j][0]}\t{v[0]:.4f}\t{v[1]}\t{v[2]:.4f}\t"
                        f"{v[3]}\t{v[4]:.4f}\t{v[5]:.6f}\n")
                n_rows += 1
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(target)}, строк {n_rows}", file=sys.stderr, flush=True)

    print(f"строк: {n_rows} -> {out_tsv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
