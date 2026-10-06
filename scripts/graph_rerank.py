"""Переранжирование по графу кандидатов, а не по каждой паре в отдельности.

До сих пор балл пары (абзац, стих) считался независимо от остальных пар. Но
кандидаты образуют граф, и у заимствований есть структура, которую этот граф
удерживает, а независимое решение теряает:

  1. Цитата почти никогда не приходится на один стих. Тихомиров ссылается на
     диапазоны и на параллельные места (MAR.4.22;LUK.8.16-17;MAT.10.16-17).
     Значит если абзац тянется к стиху, соседние стихи той же главы и
     параллельные места в других Евангелиях -- тоже кандидаты, и согласие
     между ними само по себе признак.
  2. Абзацы романа идут подряд. Исповедь Мармеладова разлита по десяткам
     абзацев и черпает из одного пласта; если предыдущий абзац тянется к
     Евангелию от Луки, у следующего это повышает правдоподобие Луки.

Отсюда распространение по графу (одна итерация, вида
  новый_балл = балл + alpha * (среднее по соседям-стихам)
             + beta  * (среднее по соседям-абзацам)),
где граф стихов -- соседство в главе плюс рёбра «параллельное место», а граф
абзацев -- соседство в романе.

Параллельные места НЕ берутся из внешнего справочника: они строятся из самого
корпуса как пары стихов с высоким взаимным лексическим сходством внутри
Евангелий. Справочник перекрёстных ссылок был бы лучше, но его нет, а
синоптические параллели по лексике видны и так.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import sparse

from oracle_diagnostics import load_gold, verse_matches
from retrieval_scorers import _idf_from, _l2_rows, _token_matrix, load_ann

GOSPELS = {"MAT", "MAR", "LUK", "JOH"}
PARALLEL_MIN_SIM = 0.55
PARALLEL_TOP = 5


def verse_neighbour_graph(s_ids: list[str], s_lem: list[list[str]],
                          radius: int = 2) -> sparse.csr_matrix:
    """Граф стихов: соседство в главе (вес убывает с расстоянием) плюс
    параллельные места внутри Евангелий по лексическому сходству."""
    n = len(s_ids)
    chap = [".".join(i.split(".")[:3]) for i in s_ids]
    rows, cols, data = [], [], []
    for j in range(n):
        for d in range(1, radius + 1):
            for k in (j - d, j + d):
                if 0 <= k < n and chap[k] == chap[j]:
                    rows.append(j)
                    cols.append(k)
                    data.append(1.0 / d)
    n_adj = len(rows)

    # параллельные места: только внутри Евангелий, только сильное сходство
    gosp = [i for i, sid in enumerate(s_ids) if sid.split(".")[1] in GOSPELS]
    if gosp:
        vocab = {l: i for i, l in enumerate({l for j in gosp for l in s_lem[j]})}
        M = _token_matrix([s_lem[j] for j in gosp], vocab)
        idf = _idf_from(M)
        Mn = _l2_rows(M @ sparse.diags(idf))
        book = [s_ids[j].split(".")[1] for j in gosp]
        CH = 512
        for a in range(0, len(gosp), CH):
            block = np.asarray((Mn[a:a + CH] @ Mn.T).todense(), dtype=np.float32)
            for r in range(block.shape[0]):
                jg = a + r
                row = block[r]
                row[jg] = 0.0
                cand = np.argpartition(row, -PARALLEL_TOP)[-PARALLEL_TOP:]
                for kg in cand:
                    if row[kg] < PARALLEL_MIN_SIM or book[kg] == book[jg]:
                        continue
                    rows.append(gosp[jg])
                    cols.append(gosp[kg])
                    data.append(float(row[kg]))
    print(f"граф стихов: {n_adj} рёбер соседства, "
          f"{len(rows)-n_adj} рёбер параллельных мест", file=sys.stderr)
    G = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    deg = np.asarray(G.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sparse.diags((1.0 / deg).astype(np.float32)) @ G


def xref_graph(path: str, s_ids: list[str], min_votes: int = 0) -> sparse.csr_matrix:
    """Граф из внешнего справочника перекрёстных ссылок (build_xref_graph.py).
    В отличие от рёбер «параллельное место», выведенных из лексики, он содержит
    связи Ветхого и Нового Заветов, у которых общих слов почти нет."""
    idx = {s: i for i, s in enumerate(s_ids)}
    rows, cols, data = [], [], []
    n_kept = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            v = int(row["votes"])
            if v < min_votes:
                continue
            a_, b_ = idx.get(row["verse_a"]), idx.get(row["verse_b"])
            if a_ is None or b_ is None:
                continue
            w = 1.0 + np.log1p(max(v, 0))
            rows += [a_, b_]
            cols += [b_, a_]
            data += [w, w]
            n_kept += 1
    print(f"граф справочника: {n_kept} рёбер (голосов >= {min_votes})", file=sys.stderr)
    G = sparse.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)),
                          shape=(len(s_ids), len(s_ids)))
    deg = np.asarray(G.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sparse.diags((1.0 / deg).astype(np.float32)) @ G


def paragraph_neighbour_pairs(pars: list[str]) -> dict[str, list[str]]:
    """Соседи абзаца -- предыдущий и следующий по нумерации внутри главы."""
    def key(p):
        parts = p.split(".")
        return (parts[1], parts[2], int(parts[3]))
    order = sorted(pars, key=key)
    pos = {p: i for i, p in enumerate(order)}
    out = {}
    for p in pars:
        i = pos[p]
        nb = []
        for k in (i - 1, i + 1):
            if 0 <= k < len(order):
                q = order[k]
                if key(q)[:2] == key(p)[:2]:
                    nb.append(q)
        out[p] = nb
    return out


def main(table_tsv: str, bible_ann: str, gold_tsv: str,
         score_col: str = "tfidf_rare_pct",
         xref_path: str | None = None) -> None:
    s_ids, s_lem, _, _, _ = load_ann(bible_ann)
    sidx = {s: i for i, s in enumerate(s_ids)}
    G = verse_neighbour_graph(s_ids, s_lem)
    GX = xref_graph(xref_path, s_ids) if xref_path else None

    # баллы из таблицы объединения: (абзац, стих) -> балл выбранной колонки
    by_par: dict[str, dict[int, float]] = defaultdict(dict)
    recs: dict[int, str] = {}
    with open(table_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            j = sidx.get(row["source_id"])
            if j is None:
                continue
            by_par[row["target_id"]][j] = float(row[score_col])
            recs[int(row["rec_id"])] = row["target_id"]

    pars = sorted(by_par)
    par_nb = paragraph_neighbour_pairs(pars)
    n = len(s_ids)

    dense = {}
    for p in pars:
        v = np.zeros(n, dtype=np.float32)
        for j, sc in by_par[p].items():
            v[j] = sc
        dense[p] = v

    gold = load_gold(gold_tsv)
    correct = [{j for j, sid in enumerate(s_ids) if verse_matches(sid, g[1])}
               for g in gold]
    par_of_rec = [g[0] for g in gold]

    # Эталон Тихомирова часто задаёт ДИАПАЗОН стихов (28 мест из 64), и метод,
    # размазывающий балл по соседям, мог бы выглядеть хорошо просто потому, что
    # любой стих диапазона считается правильным. Поэтому всё считается ещё и
    # только по местам с одиночными ссылками -- там этой поддавки нет.
    single_only = {k for k, g in enumerate(gold)
                   if all(vf == vt for _, _, vf, vt in g[1])}

    def ranks_for(scores: dict[str, np.ndarray],
                  subset: set[int] | None = None) -> np.ndarray:
        out = []
        for k, g in enumerate(gold):
            if subset is not None and k not in subset:
                continue
            p = par_of_rec[k]
            if p not in scores or not correct[k]:
                continue
            sc = scores[p]
            pool = np.array(sorted(by_par[p]))
            best = sc[list(correct[k] & set(pool.tolist()))] if (correct[k] & set(pool.tolist())) else None
            if best is None or len(best) == 0:
                continue
            b = best.max()
            vals = sc[pool]
            out.append(int((vals > b).sum()) + (int((vals == b).sum()) + 1) / 2.0)
        return np.array(out, dtype=float)

    print(f"\nграфовое распространение по колонке {score_col}, "
          f"{len(pars)} абзацев\n")
    print(f"{'вариант':38s} {'мед.ранг':>9s} {'топ-10':>7s} {'топ-50':>7s} "
          f"{'топ-200':>8s} {'одиночн.мед':>11s} {'одн.топ200':>10s}")
    print(f"{'':38s} {'--- все места ---':>36s} {'-- только одиночные ссылки --':>22s}")

    def report(label, scores):
        r = ranks_for(scores)
        rs = ranks_for(scores, single_only)
        print(f"{label:38s} {np.median(r):9.0f} {np.mean(r<=10):7.2f} "
              f"{np.mean(r<=50):7.2f} {np.mean(r<=200):8.2f} "
              f"{np.median(rs):11.0f} {np.mean(rs<=200):9.2f}")
        return r

    base = report("без распространения", dense)
    base_single = ranks_for(dense, single_only)

    results = {"без распространения": base}
    singles = {"без распространения": base_single}
    for alpha in (0.3, 0.6, 1.0):
        sc = {p: dense[p] + alpha * (G @ dense[p]) for p in pars}
        results[f"+ соседи-стихи alpha={alpha}"] = report(
            f"+ соседи-стихи alpha={alpha}", sc)
        singles[f"+ соседи-стихи alpha={alpha}"] = ranks_for(sc, single_only)

    for beta in (0.3, 0.6):
        sc = {}
        for p in pars:
            nb = par_nb.get(p, [])
            add = np.zeros(n, dtype=np.float32)
            for q in nb:
                if q in dense:
                    add += dense[q]
            if nb:
                add /= len(nb)
            sc[p] = dense[p] + beta * add
        results[f"+ соседи-абзацы beta={beta}"] = report(
            f"+ соседи-абзацы beta={beta}", sc)
        singles[f"+ соседи-абзацы beta={beta}"] = ranks_for(sc, single_only)

    if GX is not None:
        for alpha in (0.3, 0.6, 1.0):
            sc = {p: dense[p] + alpha * (GX @ dense[p]) for p in pars}
            results[f"+ справочник ссылок alpha={alpha}"] = report(
                f"+ справочник ссылок alpha={alpha}", sc)
            singles[f"+ справочник ссылок alpha={alpha}"] = ranks_for(sc, single_only)
        for alpha in (0.6, 1.0):
            sc = {p: dense[p] + alpha * (G @ dense[p]) + alpha * (GX @ dense[p])
                  for p in pars}
            results[f"+ соседство И справочник alpha={alpha}"] = report(
                f"+ соседство И справочник alpha={alpha}", sc)
            singles[f"+ соседство И справочник alpha={alpha}"] = ranks_for(sc, single_only)

    best_alpha = 0.6
    sc = {}
    for p in pars:
        nb = [q for q in par_nb.get(p, []) if q in dense]
        add = np.zeros(n, dtype=np.float32)
        for q in nb:
            add += dense[q]
        if nb:
            add /= len(nb)
        v = dense[p] + 0.3 * add
        sc[p] = v + best_alpha * (G @ v)
    results["оба соседства"] = report("оба соседства (0.6 / 0.3)", sc)
    singles["оба соседства"] = ranks_for(sc, single_only)

    print()
    rng = np.random.default_rng(0)
    for label, r in results.items():
        if label == "без распространения":
            continue
        m = min(len(r), len(base))
        d = [np.median(r[:m][i]) - np.median(base[:m][i])
             for i in (rng.integers(0, m, m) for _ in range(5000))]
        print(f"{label} против базы: {np.median(r)-np.median(base):+.0f} "
              f"(бутстрап 95%: {np.percentile(d,2.5):+.0f}..{np.percentile(d,97.5):+.0f})")

    print("\nтолько места с одиночными ссылками (нет поддавки от диапазонов),\n"
          f"{len(base_single)} мест; здесь медиана ранга почти не двигается, поэтому\n"
          "смотрим на долю мест в первых 200 стихах:")
    for label, r in singles.items():
        if label == "без распространения":
            continue
        m = min(len(r), len(base_single))
        a_, b_ = r[:m] <= 200, base_single[:m] <= 200
        d = [float(a_[i].mean() - b_[i].mean())
             for i in (rng.integers(0, m, m) for _ in range(5000))]
        print(f"  {label}: топ-200 {b_.mean():.2f} -> {a_.mean():.2f}, "
              f"разница {a_.mean()-b_.mean():+.2f} "
              f"(бутстрап 95%: {np.percentile(d,2.5):+.2f}..{np.percentile(d,97.5):+.2f})")


if __name__ == "__main__":
    main(*sys.argv[1:6])
