"""Графовые признаки пары (абзац, стих), считаемые по рёбрам разбора.

Два признака из программы отчёта (WL-ядро с весами по редкости и MCS):

  wl_h1w   Weisfeiler-Lehman, один шаг. Метка узла -- лемма вместе с
           отсортированным набором (тип связи, направление, лемма соседа) среди
           СОДЕРЖАТЕЛЬНЫХ соседей. Вес метки -- сумма idf леммы и её соседей:
           чем реже слова, согласованные вокруг узла, тем весомее совпадение.
           Признак -- косинус взвешенных множеств меток предложения и стиха.
           Изолированные узлы метки не дают: это уже мешок слов (tfidf_rare).
  mcs_*    Максимальный общий связный подграф предложения и стиха. Рёбра в
           аннотации лемматические (узел = лемма, не токен), поэтому метки узлов
           уникальны, и MCS считается ТОЧНО: узлы -- общие леммы, рёбра -- общие
           рёбра, ответ -- наибольшая компонента связности пересечения.
           mcs_soft_idf  рёбра без типа связи, вес компоненты -- сумма idf узлов;
           mcs_hard_n    рёбра с совпавшим типом связи, число узлов компоненты.

Единица -- предложение, на абзац берётся максимум, как и в остальном конвейере.
Служебные части речи отброшены, как в gold_findability.py.

Считается только для пар из таблицы пула (fusion_retrieval.py), у остальных
стихов признака нет, поэтому процентиль -- внутри пула абзаца, а не среди всех
31 102 стихов, как у прежних скореров.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict

FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT",
                "NUM", "ADV"}


def load_units(path: str):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            content = {l for l, p in zip(r["lemmas"], r.get("pos", [])) if p not in FUNCTION_POS}
            edges = [(h, d, c) for h, d, c in r["edges"] if h in content and c in content and h != c]
            out[r["id"]] = (content, edges)
    return out


class Unit:
    __slots__ = ("nodes", "soft", "hard", "wl", "wl_total")

    def __init__(self, content, edges, idf):
        nb = defaultdict(set)
        self.soft = set()
        self.hard = set()
        for h, d, c in edges:
            nb[h].add((d, ">", c))
            nb[c].add((d, "<", h))
            self.soft.add(frozenset((h, c)))
            self.hard.add((frozenset((h, c)), d))
        self.nodes = set(nb)
        self.wl = {}
        for n, ctx in nb.items():
            key = (n, tuple(sorted(ctx)))
            self.wl[key] = idf.get(n, 0.0) + sum(idf.get(m, 0.0) for m in {x[2] for x in ctx})
        self.wl_total = sum(self.wl.values())


def largest_component(nodes: set[str], edges: set, weight) -> tuple[float, int]:
    """Наибольшая по весу компонента связности графа (nodes, edges)."""
    adj = defaultdict(set)
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    seen, best_w, best_n = set(), 0.0, 0
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        w = sum(weight(x) for x in comp)
        if w > best_w:
            best_w, best_n = w, len(comp)
    return best_w, best_n


def pair_features(a: Unit, b: Unit, idf) -> tuple[float, float, float]:
    shared = a.nodes & b.nodes
    if len(shared) < 2:
        return 0.0, 0.0, 0.0
    wl = 0.0
    if a.wl_total and b.wl_total:
        common = set(a.wl) & set(b.wl)
        wl = sum(a.wl[k] for k in common) / math.sqrt(a.wl_total * b.wl_total)
    w = lambda x: idf.get(x, 0.0)
    soft_edges = {e for e in a.soft & b.soft if e <= shared}
    mcs_soft, _ = largest_component(shared, soft_edges, w)
    hard_edges = {e for e in a.hard & b.hard}
    hard_pairs = {e for e, _ in hard_edges}
    _, mcs_hard_n = largest_component(shared, hard_pairs, w)
    return wl, mcs_soft, float(mcs_hard_n)


def main(bible_ann: str, sent_ann: str, table_tsv: str, out_tsv: str) -> None:
    bible_raw = load_units(bible_ann)
    n_verses = len(bible_raw)
    df = Counter()
    for content, _ in bible_raw.values():
        df.update(content)
    idf = {l: math.log(n_verses / c) for l, c in df.items()}

    bible = {k: Unit(c, e, idf) for k, (c, e) in bible_raw.items()}
    sents = defaultdict(list)
    for k, (c, e) in load_units(sent_ann).items():
        sents[k.split("#", 1)[0]].append(Unit(c, e, idf))

    pairs = []
    with open(table_tsv, encoding="utf-8", newline="") as f:
        seen = set()
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            key = (r["target_id"], r["source_id"])
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    print(f"пар в пуле: {len(pairs)}", file=sys.stderr)

    feats = {}
    for i, (par, sid) in enumerate(pairs):
        v = bible.get(sid)
        best = (0.0, 0.0, 0.0)
        if v is not None:
            for s in sents.get(par, []):
                f3 = pair_features(s, v, idf)
                best = tuple(max(x, y) for x, y in zip(best, f3))
        feats[(par, sid)] = best
        if (i + 1) % 20000 == 0:
            print(f"{i+1}/{len(pairs)}", file=sys.stderr, flush=True)

    by_par = defaultdict(list)
    for (par, sid), v in feats.items():
        by_par[par].append(sid)
    names = ("wl_h1w", "mcs_soft_idf", "mcs_hard_n")
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\t" + "\t".join(f"{n}_raw\t{n}_ppct" for n in names) + "\n")
        for par, sids in by_par.items():
            ppct = []
            for j in range(3):
                vals = sorted(feats[(par, s)][j] for s in sids)
                m: dict[float, float] = {}
                for i, x in enumerate(vals):
                    m.setdefault(x, i / len(vals))
                ppct.append(m)
            for s in sids:
                row = feats[(par, s)]
                cells = []
                for j in range(3):
                    cells += [f"{row[j]:.6f}", f"{ppct[j][row[j]]:.6f}"]
                f.write(f"{par}\t{s}\t" + "\t".join(cells) + "\n")
    nz = [sum(1 for v in feats.values() if v[j] > 0) for j in range(3)]
    print(f"ненулевых: " + ", ".join(f"{n}={c}" for n, c in zip(names, nz)) + f" -> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
