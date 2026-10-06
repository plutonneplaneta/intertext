"""Перенос замороженной GNN основного проекта как признак (Phase 6).

Берётся обученная в другом проекте автора сеть (граф разбора: токены -- узлы, зависимости и
порядок -- рёбра, виртуальный узел, локальная передача сообщений плюс полное
внимание), которая предсказывает вовлечённость. Здесь она только инференс:
вектор графа до головы регрессии (forward(..., return_pooled=True)) для каждого
стиха и для предложений эталонных абзацев. Признак пары -- косинус центрированных
векторов, на абзац максимум по предложениям, как у остальных скореров.

Ожидание, оговорённое заранее: перенос маловероятно полезен (другая целевая
функция, другая единица, другой домен). Поэтому рядом обязателен контроль --
та же архитектура со СЛУЧАЙНЫМИ весами и теми же замороженными векторами лемм:
если обученная не лучше случайной, «перенос» ничего не передаёт, и всё, что
видно, дают векторы лемм.

Вектор центрируется по среднему по стихам: у сырых векторов графа общая
составляющая велика, и косинус без центрирования почти не различает пары.

Код сети -- копия app/analysis/{graphs,graph_model}.py основного проекта
(~/intertext/pl_gnn), боевое дерево не трогается.

Вход: чекпойнт, стихи, предложения романа (TSV id/text), таблица пула, эталон.
Выход: TSV (target_id, source_id, <name>_raw, <name>_pct) по парам пула и
печать медианы ранга правильного стиха ТОЛЬКО по этому признаку.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "intertext" / "pl_gnn"))
sys.path.insert(0, str(Path(__file__).parent))


def read_tsv(path: str) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(r["verse_id"])
            texts.append(r["text"])
    return ids, texts


def embed(texts, nlp, network, vocabulary, device, batch_size=64):
    import torch
    from torch_geometric.loader import DataLoader

    from app.analysis import graphs
    from app.analysis.graph_model import to_data

    items, keep = [], []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        graph = graphs.build_graph(doc, vocabulary)
        if len(graph.lemmas) <= 1:
            continue
        items.append(to_data(graph))
        keep.append(i)
    out = []
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=batch_size):
            _, pooled = network(batch.to(device), return_pooled=True)
            out.append(pooled.cpu().numpy())
    return np.array(keep), np.concatenate(out).astype(np.float32)


def main(model_path: str, bible_tsv: str, sents_tsv: str, table_tsv: str, gold_tsv: str,
         out_tsv: str, name: str, mode: str = "trained", seed: int = 0, device: str = "cpu") -> None:
    import spacy
    import torch

    from app.analysis.graph_model import EngagementGraphNet, ModelSettings, load_model
    from oracle_diagnostics import load_gold, verse_matches

    if device == "3060":
        device = next((f"cuda:{i}" for i in range(torch.cuda.device_count())
                       if "3060" in torch.cuda.get_device_name(i)), "cpu")
    torch.manual_seed(seed)
    network, vocabulary, _ = load_model(Path(model_path))
    if mode == "random":
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        settings = ModelSettings(**saved["settings"])
        settings.lemma_vectors = saved.get("lemma_vectors")
        network = EngagementGraphNet(settings)
    network = network.to(device).eval()
    print(f"режим {mode}, seed {seed}, устройство {device}", file=sys.stderr)

    pairs = []
    seen = set()
    with open(table_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            k = (r["target_id"], r["source_id"])
            if k not in seen:
                seen.add(k)
                pairs.append(k)
    pars = {p for p, _ in pairs}

    nlp = spacy.load("ru_core_news_lg")
    s_ids, s_text = read_tsv(bible_tsv)
    keep_s, S = embed(s_text, nlp, network, vocabulary, device)
    s_ids = [s_ids[i] for i in keep_s]
    print(f"стихов с графом: {len(s_ids)}", file=sys.stderr, flush=True)

    q_ids_all, q_text_all = read_tsv(sents_tsv)
    sel = [i for i, q in enumerate(q_ids_all) if q.split("#", 1)[0] in pars]
    keep_q, Q = embed([q_text_all[i] for i in sel], nlp, network, vocabulary, device)
    q_ids = [q_ids_all[sel[i]] for i in keep_q]
    print(f"предложений с графом: {len(q_ids)}", file=sys.stderr, flush=True)

    mu = S.mean(0, keepdims=True)
    S = S - mu
    Q = Q - mu
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9

    sid_pos = {s: i for i, s in enumerate(s_ids)}
    sents_of = defaultdict(list)
    for i, q in enumerate(q_ids):
        sents_of[q.split("#", 1)[0]].append(i)

    n = len(s_ids)
    agg, pct = {}, {}
    for par in pars:
        idx = sents_of.get(par)
        if not idx:
            continue
        v = (Q[idx] @ S.T).max(0)
        order = np.argsort(v, kind="stable")
        r = np.empty(n, dtype=np.float32)
        r[order] = np.arange(n, dtype=np.float32) / n
        agg[par], pct[par] = v, r

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write(f"target_id\tsource_id\t{name}_raw\t{name}_pct\n")
        for par, sid in pairs:
            j = sid_pos.get(sid)
            if par in agg and j is not None:
                f.write(f"{par}\t{sid}\t{agg[par][j]:.6f}\t{pct[par][j]:.6f}\n")
            else:
                f.write(f"{par}\t{sid}\t0\t0\n")

    ranks = []
    for par, group, _ in load_gold(gold_tsv):
        if par not in agg:
            continue
        correct = [i for i, s in enumerate(s_ids) if verse_matches(s, group)]
        if not correct:
            continue
        order = np.argsort(-agg[par], kind="stable")
        pos = np.empty(n, dtype=int)
        pos[order] = np.arange(1, n + 1)
        ranks.append(min(pos[c] for c in correct))
    r = np.array(ranks, float)
    print(f"{name}: один признак, мест {len(r)}, медиана ранга {np.median(r):.0f} из {n}, "
          f"топ-200 {np.mean(r <= 200):.2f}, топ-1000 {np.mean(r <= 1000):.2f} -> {out_tsv}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a[:7], mode=a[7] if len(a) > 7 else "trained",
         seed=int(a[8]) if len(a) > 8 else 0, device=a[9] if len(a) > 9 else "cpu")
