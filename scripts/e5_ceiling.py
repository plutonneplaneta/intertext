"""Пункт 3: обученный кодировщик предложений (multilingual-e5-base) вместо
статических векторов -- потолок полноты и ранги, в тех же столбцах, что
ceiling_scan.py, чтобы числа были сравнимы напрямую.

Пилот считал этот слой по абзацу целиком и выдавал TOP_K=20 кандидатов, отчего
его полнота была 5%. Здесь меряется потолок: исчерпывающий счёт всех 31 102
стихов, единица -- и абзац, и предложение с максимумом на абзац.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from oracle_diagnostics import load_gold, verse_matches


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return list(d["ids"]), d["embeddings"].astype(np.float32)


def main(bible_npz: str, novel_npz: str, sent_npz: str, gold_tsv: str) -> None:
    s_ids, S = load_npz(bible_npz)
    t_ids, T = load_npz(novel_npz)
    q_ids, Q = load_npz(sent_npz)
    print(f"стихов {len(s_ids)}, абзацев {len(t_ids)}, предложений {len(q_ids)}",
          file=sys.stderr)

    gold = load_gold(gold_tsv)
    correct = [{j for j, sid in enumerate(s_ids) if verse_matches(sid, g[1])}
               for g in gold]
    t_pos = {t: i for i, t in enumerate(t_ids)}
    sent_of = defaultdict(list)
    for i, q in enumerate(q_ids):
        sent_of[q.split("#", 1)[0]].append(i)

    rows = {"emb_e5/абзац": [], "emb_e5/предл": []}
    for k, g in enumerate(gold):
        if not correct[k]:
            continue
        par = g[0]
        ti = t_pos.get(par)
        variants = {}
        if ti is not None:
            variants["emb_e5/абзац"] = S @ T[ti]
        idx = sent_of.get(par, [])
        if idx:
            variants["emb_e5/предл"] = (Q[idx] @ S.T).max(axis=0)
        for name, sc in variants.items():
            best = sc[list(correct[k])].max()
            greater = int((sc > best).sum())
            equal = int((sc == best).sum())
            rows[name].append((greater + (equal + 1) / 2.0, int(best > -1e9)))

    print(f"\n{'вариант':14s} {'виден':>6s} {'медиана':>8s} {'кв.25':>7s} "
          f"{'кв.75':>7s} {'топ-50':>7s} {'топ-200':>8s} {'топ-1000':>9s}")
    for name, vals in rows.items():
        if not vals:
            continue
        r = np.array([v[0] for v in vals], dtype=float)
        h = float(np.mean([v[1] for v in vals]))
        print(f"{name:14s} {h:6.2f} {np.median(r):8.0f} {np.percentile(r,25):7.0f} "
              f"{np.percentile(r,75):7.0f} {np.mean(r<=50)*h:7.2f} "
              f"{np.mean(r<=200)*h:8.2f} {np.mean(r<=1000)*h:9.2f}")
    print("\n(«виден» у плотного скорера всегда 1,00: косинус ненулевой у всех\n"
          " стихов. Смысл имеет ранг.)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
