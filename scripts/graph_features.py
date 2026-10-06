"""Графовые признаки поверх дерева разбора и графа кандидатов.

Синтаксический слой пилота -- это граф, урезанный до одного ребра: признак
есть, если у абзаца и стиха совпала тройка (лемма головы, тип связи, лемма
зависимого). Потолок такого признака 0,34, и измерение показало, что само
требование совпадения ТИПА связи стоит семи процентных пунктов даром
(soft_edges: 0,41). Здесь проверяются признаки, которые используют топологию
дерева, а не отдельное ребро:

  dep_pair2/3     пары содержательных лемм, соединённых путём в дереве длины
                  не больше 2 или 3, без порядка и без типов связей. Это
                  графовый аналог skipbigram (лучшего по точности варианта:
                  медиана ранга 54), но соседство считается по дереву, а не по
                  окну в тексте: «сказал» и «дщерь» могут стоять через десять
                  слов и быть связаны напрямую, а могут стоять рядом и не быть
                  связаны вовсе
  dep_path2       путь длины 2 со стянутой серединой: (лемма, лемма) через
                  одно слово. Ловит подстановку среднего слова -- то, что
                  n-грамма и одиночное ребро теряют целиком
  collapsed       ребра со стянутым предлогом: вместо (идти, obl, дом) плюс
                  (дом, case, в) -- одно (идти, obl:в, дом). Предлог у
                  Достоевского и в Синодальном переводе часто разный при том
                  же отношении
  cooc_expand     расширение запроса по графу совместной встречаемости лемм:
                  лемма тянет за собой соседей по графу «встречались в одном
                  стихе». Попытка поднять потолок там, где общих слов нет
                  вовсе

Всё -- мешки признаков с весом по редкости и косинусом, чтобы числа были
сравнимы с остальными вариантами в ceiling_scan.py.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque

import numpy as np
from scipy import sparse

FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT"}


def load_graph_ann(path: str):
    ids, lemmas, poss, heads, rels, adps = [], [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["id"])
            lemmas.append(r["lemmas"])
            poss.append(r.get("pos", ["X"] * len(r["lemmas"])))
            heads.append(r.get("heads", [-1] * len(r["lemmas"])))
            rels.append(r.get("rels", ["dep"] * len(r["lemmas"])))
            adps.append(r.get("adp_of", [""] * len(r["lemmas"])))
    return ids, lemmas, poss, heads, rels, adps


def _adjacency(heads: list[int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in heads]
    for child, h in enumerate(heads):
        if h is not None and 0 <= h < len(heads) and h != child:
            adj[child].append(h)
            adj[h].append(child)
    return adj


def dep_pairs(lemmas, poss, heads, max_dist: int, with_dist: bool = False):
    """Неупорядоченные пары содержательных лемм на расстоянии <= max_dist
    по дереву. Обход в ширину от каждой вершины -- деревья короткие
    (предложение), поэтому это дешевле, чем кажется."""
    adj = _adjacency(heads)
    content = [i for i, p in enumerate(poss) if p not in FUNCTION_POS]
    out = []
    content_set = set(content)
    for u in content:
        seen = {u: 0}
        q = deque([u])
        while q:
            x = q.popleft()
            d = seen[x]
            if d == max_dist:
                continue
            for y in adj[x]:
                if y in seen:
                    continue
                seen[y] = d + 1
                q.append(y)
                if y in content_set and y > u:
                    a, b = lemmas[u], lemmas[y]
                    key = (a, b) if a <= b else (b, a)
                    out.append(key + ((d + 1,) if with_dist else ()))
    return out


def dep_path2(lemmas, poss, heads):
    """Пары, соединённые ровно через одно слово: середина выброшена."""
    adj = _adjacency(heads)
    out = []
    for mid in range(len(lemmas)):
        nb = adj[mid]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                u, v = nb[i], nb[j]
                if poss[u] in FUNCTION_POS or poss[v] in FUNCTION_POS:
                    continue
                a, b = lemmas[u], lemmas[v]
                out.append((a, b) if a <= b else (b, a))
    return out


def collapsed_edges(lemmas, poss, heads, rels, adps):
    """Ребро со стянутым предлогом зависимого: (голова, связь:предлог, зависимое)."""
    out = []
    for child, h in enumerate(heads):
        if not (0 <= h < len(lemmas)) or h == child:
            continue
        if poss[child] in FUNCTION_POS or poss[h] in FUNCTION_POS:
            continue
        rel = rels[child]
        adp = adps[child] if child < len(adps) else ""
        out.append((lemmas[h], f"{rel}:{adp}" if adp else rel, lemmas[child]))
    return out


def cooc_expand(lemmas, poss, neighbours: dict[str, list[str]], top: int = 3):
    """Запрос плюс соседи каждой леммы по графу совместной встречаемости."""
    out = []
    for l, p in zip(lemmas, poss):
        if p in FUNCTION_POS:
            continue
        out.append(l)
        out.extend(neighbours.get(l, ())[:top])
    return out


def build_cooc_neighbours(docs: list[list[str]], poss: list[list[str]],
                          min_df: int = 3, top: int = 3) -> dict[str, list[str]]:
    """Граф «леммы встречались в одном стихе», вес по PMI; для каждой леммы --
    top самых сильных соседей. Частые леммы отбрасываются: их соседство ничего
    не различает, а граф от них разрастается."""
    df: dict[str, int] = defaultdict(int)
    for d, p in zip(docs, poss):
        for l in {x for x, q in zip(d, p) if q not in FUNCTION_POS}:
            df[l] += 1
    vocab = {l: i for i, l in enumerate(l for l, c in df.items() if c >= min_df)}
    if not vocab:
        return {}
    rows, cols = [], []
    for d, p in zip(docs, poss):
        idx = sorted({vocab[x] for x, q in zip(d, p)
                      if q not in FUNCTION_POS and x in vocab})
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                rows.append(idx[a])
                cols.append(idx[b])
    n = len(vocab)
    C = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n, n)).tocsr()
    C = C + C.T
    cnt = np.array([df[l] for l in vocab], dtype=np.float32)
    total = float(len(docs))
    inv = {i: l for l, i in vocab.items()}
    neigh: dict[str, list[str]] = {}
    C = C.tolil()
    for i in range(n):
        row = C.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        pmi = np.log((row.data * total) / np.maximum(cnt[i] * cnt[row.col], 1e-9) + 1e-9)
        order = np.argsort(-pmi)[:top]
        neigh[inv[i]] = [inv[row.col[k]] for k in order]
    return neigh
