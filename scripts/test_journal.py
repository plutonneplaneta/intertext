"""Журнал всех проверок: одна таблица на все сравнения, с единой статистикой и поправкой.

Каждая строка спецификации -- одно парное сравнение двух наборов признаков на одних и тех же
местах эталона: файл рангов (RANKS_OUT из fusion_retrieval.py / eval_heldout.py) и метки
нового и базового набора. Для всех считается одно и то же: средний Δ log-ранга (новый минус
базовый), перестановочный p по знакам разностей (двусторонний, 20 000), число мест.

Поправка: Холм внутри семейства (колонка family) и Бонферрони на ВСЕ проверки журнала.
Семейства заданы заранее в reports/prereg_independent_test.md (H1, H2); остальные --
разведочные, их p описательные.

Спецификация -- TSV с колонками:
  id, family, status, prespecified, note, new_file, new_label, base_file, base_label
status: актуально | отозвано (с причиной в note).
Пути -- относительно каталога со спецификацией.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np


def load(path: str) -> dict[str, dict[int, int]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["label"]] = {int(k): v for k, v in d["ranks"].items() if v}
    return out


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    running = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def main(spec_tsv: str, out_tsv: str, manual_tsv: str = "") -> None:
    base_dir = os.path.dirname(os.path.abspath(spec_tsv))
    rng = np.random.default_rng(0)
    cache: dict[str, dict] = {}
    rows = []
    with open(spec_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            for k in ("new_file", "base_file"):
                p = os.path.join(base_dir, r[k])
                if p not in cache:
                    cache[p] = load(p)
            new = cache[os.path.join(base_dir, r["new_file"])][r["new_label"]]
            base = cache[os.path.join(base_dir, r["base_file"])][r["base_label"]]
            keys = sorted(set(new) & set(base))
            d = np.log([new[k] for k in keys]) - np.log([base[k] for k in keys])
            obs = d.mean()
            perm = (rng.choice([-1, 1], size=(20000, len(d))) * d).mean(1)
            p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 20001
            rows.append({**r, "n": len(keys), "delta": obs, "ratio": float(np.exp(obs)), "p": p})

    if manual_tsv:
        # строки без парных рангов (сравнение со случайным порядком и т.п.): значения посчитаны
        # отдельно и записаны как есть: id, family, status, prespecified, n, delta, ratio, p, note
        with open(manual_tsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                rows.append({**r, "n": int(r["n"]), "delta": float(r["delta"]),
                             "ratio": float(r["ratio"]), "p": float(r["p"])})

    live = [i for i, r in enumerate(rows) if r["status"] == "актуально"]
    fam_adj = {}
    for fam in sorted({rows[i]["family"] for i in live}):
        idx = [i for i in live if rows[i]["family"] == fam]
        for i, a in zip(idx, holm([rows[i]["p"] for i in idx])):
            fam_adj[i] = a
    n_all = len(live)
    with open(out_tsv, "w", encoding="utf-8") as g:
        g.write("id\tfamily\tstatus\tprespecified\tn\tdelta_logrank\trank_ratio\tp\tp_holm_family\tp_bonferroni_all\tnote\n")
        for i, r in enumerate(rows):
            holm_p = f"{fam_adj[i]:.4f}" if i in fam_adj else ""
            bon = f"{min(1.0, r['p'] * n_all):.4f}" if i in fam_adj else ""
            g.write(f"{r['id']}\t{r['family']}\t{r['status']}\t{r['prespecified']}\t{r['n']}\t{r['delta']:+.3f}\t"
                    f"{r['ratio']:.2f}\t{r['p']:.4f}\t{holm_p}\t{bon}\t{r['note']}\n")
    print(f"проверок {len(rows)}, актуальных {n_all} -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
