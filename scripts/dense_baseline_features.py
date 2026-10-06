"""Сильные стандартные базы как признаки пар пула (для сравнения с surprisal).

Что просят рецензенты: более сильные плотные модели и реранкер, а не только E5-base.
Для каждой пары (абзац, стих) пула:

  dense   косинус вектора стиха и каждого предложения абзаца, максимум по предложениям --
          так же, как у `e5` в основной таблице. Модели: multilingual-e5-large, bge-m3,
          ai-forever/sbert_large_nlu_ru.
  ce      кросс-энкодер (bge-reranker-v2-m3): оценка пары (ближайшее по E5-base
          предложение, стих) -- единственный способ, которым пара «читается» целиком, как в
          surprisal, но обученной на релевантность моделью.

Процентиль -- внутри пула абзаца (колонка <name>_ppct), как у surprisal и графовых признаков:
у прежних скореров процентиль среди всех 31 102 стихов, здесь считать по всем стихам
дорого для кросс-энкодера и не нужно для остальных.

Использование:
  dense_baseline_features.py dense <модель> <префикс_запроса> <префикс_стиха> <имя> <bible.tsv> <sents.tsv> <table.tsv> <out.tsv>
  dense_baseline_features.py ce <модель> <имя> <bible.tsv> <sents.tsv> <table.tsv> <bible_e5.npz> <sent_e5.npz> <out.tsv>
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np


def read_tsv(path):
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(r["verse_id"])
            texts.append(r["text"])
    return ids, texts


def read_pairs(table):
    pairs, seen = [], set()
    with open(table, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            k = (r["target_id"], r["source_id"])
            if k not in seen:
                seen.add(k)
                pairs.append(k)
    return pairs


def write(out, name, pairs, vals):
    by_par = defaultdict(list)
    for (p, s), v in zip(pairs, vals):
        by_par[p].append((s, v))
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"target_id\tsource_id\t{name}_raw\t{name}_ppct\n")
        for p, lst in by_par.items():
            v = [x[1] for x in lst]
            m: dict[float, float] = {}
            for i, x in enumerate(sorted(v)):
                m.setdefault(x, i / len(v))
            for s, x in lst:
                f.write(f"{p}\t{s}\t{x:.6f}\t{m[x]:.6f}\n")
    print(f"{name}: {len(pairs)} пар -> {out}")


def dense(model_name, qpre, ppre, name, bible_tsv, sents_tsv, table, out):
    import torch
    from sentence_transformers import SentenceTransformer

    dev = next((f"cuda:{i}" for i in range(torch.cuda.device_count())
                if "3060" in torch.cuda.get_device_name(i)), "cpu")
    model = SentenceTransformer(model_name, device=dev)
    model.half()
    pairs = read_pairs(table)
    pars = {p for p, _ in pairs}
    vids, vtext = read_tsv(bible_tsv)
    sids, stext = read_tsv(sents_tsv)
    keep = [i for i, s in enumerate(sids) if s.split("#", 1)[0] in pars]
    V = model.encode([f"{ppre}{t}" for t in vtext], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
    Q = model.encode([f"{qpre}{stext[i]}" for i in keep], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
    vpos = {v: i for i, v in enumerate(vids)}
    sents_of = defaultdict(list)
    for j, i in enumerate(keep):
        sents_of[sids[i].split("#", 1)[0]].append(j)
    vals = []
    for p, s in pairs:
        idx = sents_of.get(p)
        vals.append(float((Q[idx] @ V[vpos[s]]).max()) if idx and s in vpos else 0.0)
    write(out, name, pairs, vals)


def ce(model_name, name, bible_tsv, sents_tsv, table, bible_npz, sent_npz, out):
    import torch
    from sentence_transformers import CrossEncoder

    dev = next((f"cuda:{i}" for i in range(torch.cuda.device_count())
                if "3060" in torch.cuda.get_device_name(i)), "cpu")
    model = CrossEncoder(model_name, device=dev, max_length=192)
    model.model.half()
    pairs = read_pairs(table)
    vids, vtext = read_tsv(bible_tsv)
    vt = dict(zip(vids, vtext))
    sids, stext = read_tsv(sents_tsv)
    st = dict(zip(sids, stext))
    b = np.load(bible_npz, allow_pickle=True)
    q = np.load(sent_npz, allow_pickle=True)
    # каждое обращение к NpzFile["..."] заново читает и распаковывает массив с диска:
    # внутри цикла по 90 тысячам пар это были часы, поэтому массивы берутся один раз
    b_ids, b_emb, q_ids, q_emb = list(b["ids"]), b["embeddings"], list(q["ids"]), q["embeddings"]
    bpos = {s: i for i, s in enumerate(b_ids)}
    sents_of = defaultdict(list)
    for i, s in enumerate(q_ids):
        sents_of[s.split("#", 1)[0]].append(i)
    chosen = []
    for p, s in pairs:
        idx = sents_of.get(p)
        if not idx or s not in bpos:
            chosen.append(None)
            continue
        sims = q_emb[idx] @ b_emb[bpos[s]]
        chosen.append(q_ids[idx[int(np.argmax(sims))]])
    todo = [i for i, c in enumerate(chosen) if c]
    scores = model.predict([(st[chosen[i]], vt[pairs[i][1]]) for i in todo], batch_size=64,
                           show_progress_bar=False)
    vals = [0.0] * len(pairs)
    for i, sc in zip(todo, scores):
        vals[i] = float(sc)
    write(out, name, pairs, vals)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dense":
        dense(*sys.argv[2:10])
    else:
        ce(*sys.argv[2:10])
