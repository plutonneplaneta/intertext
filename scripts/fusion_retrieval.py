"""Обучаемое объединение шести скореров на уровне предложения.

Диагностика (oracle_diagnostics.py) показала, где потолок. Точные слои дают
правильному стиху ненулевой балл лишь в 11% (n-граммы) и 34% (рёбра) случаев
из 64 -- это их предел, и объединение пилота застряло на 37,5% именно поэтому.
Плотные скореры (bm25, tfidf_cos, char4, emb) видят правильный стих почти
всегда, но ставят его в среднем на тысячный ранг из 31 102. Значит расти можно
только за счёт плотных, и только улучшая ранжирование.

Здесь это и проверяется: логрегрессия по шести скорерам сразу, каждый -- в
двух видах, сырой балл и его процентиль среди всех 31 102 стихов. Процентиль
нужен потому, что шкалы у скореров несопоставимы (BM25 -- десятки, косинус --
доли единицы, n-граммы -- ноль у 99% стихов), а логрегрессия линейна по
признакам: без приведения к общей шкале один скорер с большим разбросом
задавил бы остальные.

Единица сопоставления -- предложение: балл пары (абзац, стих) равен максимуму
по предложениям абзаца. Это единственное изменение, которое в прошлой итерации
дало эффект за пределами шума.

Оценка -- leave-one-target-out по абзацам эталона. Пул кандидатов -- объединение
топ-K по каждому скореру, то есть ни один скорер не отбирает выборку под себя.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

from oracle_diagnostics import load_gold, verse_matches
from retrieval_scorers import SCORERS, ScorerBank, load_ann

POOL_K = int(os.environ.get("POOL_K", 300))


def load_e5(bible_npz: str, sent_npz: str):
    """Готовые векторы multilingual-e5-base (scripts/embed.py). Отдельный
    скорер, а не замена статическим: измерение показало, что E5 лучше в
    верхушке списка (топ-200 0,25 против 0,22), а статические -- в хвосте
    (топ-1000 0,44 против 0,34), то есть они дополняют друг друга."""
    b = np.load(bible_npz, allow_pickle=True)
    q = np.load(sent_npz, allow_pickle=True)
    return (list(b["ids"]), b["embeddings"].astype(np.float32),
            list(q["ids"]), q["embeddings"].astype(np.float32))


def build_features(bible_ann: str, novel_ann: str, novel_sent_ann: str,
                   gold_tsv: str, out_tsv: str,
                   bible_e5_npz: str | None = None,
                   sent_e5_npz: str | None = None,
                   bible_graph_ann: str | None = None,
                   sent_graph_ann: str | None = None) -> None:
    if not (bible_graph_ann and sent_graph_ann) and not os.environ.get("ALLOW_NO_GRAPH"):
        # без аннотации с топологией скорер dep_pair2 молча равен нулю на всех парах, и
        # «двенадцать скореров» оказываются одиннадцатью (так вышла медиана 411 вместо 224)
        raise SystemExit("нужны аннотации с топологией (bible_graph_ann, sent_graph_ann): "
                         "передайте *_ann.jsonl (в них есть поле heads) или задайте ALLOW_NO_GRAPH=1")
    t_ids, t_lem, t_pos, t_edges, t_raw = load_ann(novel_ann)
    q_ids, q_lem, q_pos, q_edges, q_raw = load_ann(novel_sent_ann)

    bank = ScorerBank(bible_ann, [t_lem, q_lem], bible_graph_ann=bible_graph_ann)

    dep_docs = None
    if bible_graph_ann and sent_graph_ann:
        from graph_features import dep_pairs, load_graph_ann
        from retrieval_scorers import DEP_PAIR_DIST
        gq_ids, gq_lem, gq_pos, gq_heads, _, _ = load_graph_ann(sent_graph_ann)
        pos_of = {q: i for i, q in enumerate(gq_ids)}
        dep_docs = {}
        for i, q in enumerate(q_ids):
            k = pos_of.get(q)
            if k is not None:
                dep_docs[i] = dep_pairs(gq_lem[k], gq_pos[k], gq_heads[k],
                                        DEP_PAIR_DIST)
        print(f"пары по дереву подключены для {len(dep_docs)} предложений",
              file=sys.stderr)
    bank.prepare_vectors([(t_ids, t_lem, t_pos), (q_ids, q_lem, q_pos)])
    print(f"стихов {len(bank.s_ids)}, абзацев {len(t_ids)}, предложений {len(q_ids)}",
          file=sys.stderr)

    gold = load_gold(gold_tsv)
    gold_pars = sorted({g[0] for g in gold})
    sent_of_par = defaultdict(list)
    for i, q in enumerate(q_ids):
        sent_of_par[q.split("#", 1)[0]].append(i)

    # какие стихи правильны для какой записи эталона
    correct_by_rec = []
    for par, group, frag in gold:
        correct_by_rec.append(
            {j for j, sid in enumerate(bank.s_ids) if verse_matches(sid, group)})
    recs_of_par = defaultdict(list)
    for k, (par, _, _) in enumerate(gold):
        recs_of_par[par].append(k)

    Q_vec = bank.T_vecs[1]
    n_src = len(bank.s_ids)

    e5 = None
    if bible_e5_npz and sent_e5_npz:
        e5_sids, e5_S, e5_qids, e5_Q = load_e5(bible_e5_npz, sent_e5_npz)
        assert e5_sids == bank.s_ids, "порядок стихов в npz не совпал с аннотацией"
        e5_qpos = {q: i for i, q in enumerate(e5_qids)}
        e5 = (e5_S, e5_Q, e5_qpos)
        print(f"E5 подключён: {e5_S.shape}", file=sys.stderr)

    scorer_names = list(SCORERS) + (["e5"] if e5 else [])
    rows = []
    for pi, par in enumerate(gold_pars):
        agg = {name: np.zeros(n_src, dtype=np.float32) for name in scorer_names}
        for qi in sent_of_par.get(par, []):
            sc = bank.score(q_lem[qi], q_raw[qi], q_edges[qi], Q_vec[qi],
                            dep_doc=(dep_docs or {}).get(qi))
            for name in SCORERS:
                np.maximum(agg[name], sc[name], out=agg[name])
            if e5 is not None:
                k = e5[2].get(q_ids[qi])
                if k is not None:
                    np.maximum(agg["e5"], e5[0] @ e5[1][k], out=agg["e5"])

        # процентиль балла среди всех стихов -- общая шкала для всех скореров
        pct = {}
        for name in scorer_names:
            order = np.argsort(agg[name], kind="stable")
            r = np.empty(n_src, dtype=np.float32)
            r[order] = np.arange(n_src, dtype=np.float32) / n_src
            pct[name] = r

        natural: set[int] = set()
        for name in scorer_names:
            k = min(POOL_K, n_src)
            natural.update(np.argpartition(agg[name], -k)[-k:].tolist())

        # ВАЖНО. Правильные стихи подмешиваются в пул намеренно -- иначе
        # переранжировщику нечего ранжировать у тех мест, где извлечение стих
        # не достало, и наборы признаков сравнивались бы на разных выборках.
        # Но из-за этого «достижимость 100%» в такой таблице гарантирована
        # построением, а не измерена. Настоящая достижимость пула -- колонка
        # in_natural_pool, и только она сравнима с полнотой пилота.
        pool = set(natural)
        for k in recs_of_par[par]:
            pool |= correct_by_rec[k]

        for j in sorted(pool):
            feats = [agg[n][j] for n in scorer_names] + [pct[n][j] for n in scorer_names]
            labels = [1 if j in correct_by_rec[k] else 0 for k in recs_of_par[par]]
            in_nat = [int(bool(natural & correct_by_rec[k])) for k in recs_of_par[par]]
            rows.append((par, bank.s_ids[j], feats, labels, recs_of_par[par], in_nat))
        print(f"{pi+1}/{len(gold_pars)} {par}: пул {len(pool)}", file=sys.stderr, flush=True)

    names = [f"{n}_raw" for n in scorer_names] + [f"{n}_pct" for n in scorer_names]
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\trec_id\tin_natural_pool\t"
                + "\t".join(names) + "\tlabel\n")
        for par, sid, feats, labels, recs, in_nat in rows:
            for rec_id, lab, nat in zip(recs, labels, in_nat):
                f.write(f"{par}\t{sid}\t{rec_id}\t{nat}\t"
                        + "\t".join(f"{v:.6f}" for v in feats) + f"\t{lab}\n")
    n_pos = sum(sum(labels) for *_, labels, _, _ in rows)
    nat_ok = {rec for *_, recs, in_nat in rows
              for rec, nat in zip(recs, in_nat) if nat}
    n_recs = len({rec for *_, recs, _ in rows for rec in recs})
    print(f"строк пула {len(rows)}, положительных пар {n_pos} -> {out_tsv}")
    print(f"ДОСТИЖИМОСТЬ БЕЗ ПОДМЕШИВАНИЯ: {len(nat_ok)}/{n_recs} мест эталона "
          f"(топ-{POOL_K} по каждому из {len(scorer_names)} скореров)")


# ------------------------------------------------------------------ оценка


def fit_logreg(X, y, l2=2.0, lr=0.5, iters=2000):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        w -= lr * g
    return w


def evaluate(table_tsv: str, feature_sets: list[tuple[str, list[str]]]) -> None:
    with open(table_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        names = [c for c in reader.fieldnames
                 if c not in ("target_id", "source_id", "rec_id", "label")]
        names = [c for c in names if c != "in_natural_pool"]
        raw = [(r["target_id"], int(r["rec_id"]),
                [float(r[c]) for c in names], int(r["label"]),
                int(r.get("in_natural_pool", 1))) for r in reader]

    by_rec = defaultdict(list)
    nat_of_rec = {}
    for par, rec, feats, lab, nat in raw:
        by_rec[rec].append((par, feats, lab))
        nat_of_rec[rec] = nat
    par_of_rec = {rec: v[0][0] for rec, v in by_rec.items()}
    recs = [r for r, v in by_rec.items() if any(x[2] == 1 for x in v)]
    n_nat = sum(1 for r in recs if nat_of_rec.get(r))
    print(f"\nмест эталона в таблице: {len(recs)} из {len(by_rec)}")
    print(f"из них извлечение достало правильный стих САМО (без подмешивания): "
          f"{n_nat} -- это и есть полнота извлечения; остальные попали в пул "
          f"принудительно, чтобы наборы признаков сравнивались на одной выборке")
    print(f"{'набор признаков':30s} {'мед.ранг':>9s} {'кв.25':>7s} {'кв.75':>7s} "
          f"{'топ-1':>6s} {'топ-10':>7s} {'топ-50':>7s} {'топ-200':>8s}")

    results = {}
    for label, cols in feature_sets:
        idx = [names.index(c) for c in cols]
        ranks = {}
        for held in recs:
            train = [(f, l) for r, v in by_rec.items()
                     if par_of_rec[r] != par_of_rec[held] for _, f, l in v]
            Xtr = np.array([[f[i] for i in idx] for f, _ in train])
            ytr = np.array([l for _, l in train], dtype=float)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            w = fit_logreg((Xtr - mu) / sd, ytr)
            te = by_rec[held]
            Xte = (np.array([[f[i] for i in idx] for _, f, _ in te]) - mu) / sd
            s = np.hstack([Xte, np.ones((len(te), 1))]) @ w
            yte = np.array([l for _, _, l in te])
            order = np.argsort(-s, kind="stable")
            hit = np.where(yte[order] == 1)[0]
            ranks[held] = int(hit[0]) + 1 if len(hit) else None
        v = np.array([r for r in ranks.values() if r is not None], dtype=float)
        results[label] = ranks
        if os.environ.get("RANKS_OUT"):
            import json
            with open(os.environ["RANKS_OUT"], "a", encoding="utf-8") as fo:
                fo.write(json.dumps({"label": label,
                                     "ranks": {str(k): v for k, v in ranks.items()}},
                                    ensure_ascii=False) + "\n")
        print(f"{label:30s} {np.median(v):9.0f} {np.percentile(v,25):7.0f} "
              f"{np.percentile(v,75):7.0f} {np.mean(v<=1):6.2f} {np.mean(v<=10):7.2f} "
              f"{np.mean(v<=50):7.2f} {np.mean(v<=200):8.2f}")

    base = feature_sets[0][0]
    print()
    for label in list(results)[1:]:
        keys = [k for k in results[label]
                if results[label][k] is not None and results[base][k] is not None]
        a = np.array([results[label][k] for k in keys], float)
        b = np.array([results[base][k] for k in keys], float)
        rng = np.random.default_rng(0)
        d = [np.median(a[i]) - np.median(b[i])
             for i in (rng.integers(0, len(keys), len(keys)) for _ in range(5000))]
        print(f"{label} против «{base}»: медиана ранга "
              f"{np.median(a)-np.median(b):+.0f} "
              f"(бутстрап 95%: {np.percentile(d,2.5):+.0f}..{np.percentile(d,97.5):+.0f})")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build_features(*sys.argv[2:11])
    else:
        # ПЕРВЫЙ переданный набор -- база для всех сравнений. Раньше наборы
        # вставлялись в начало (insert(0)), из-за чего базой становился
        # ПОСЛЕДНИЙ переданный: в абляции «выбросить один скорер из двенадцати»
        # все интервалы считались против «без e5» вместо полного набора. Теперь
        # порядок сохраняется, и база предсказуема.
        table = sys.argv[2]
        specs = sys.argv[3:]
        sets = []
        for spec in specs:
            lab, cols = spec.split("=", 1)
            sets.append((lab, cols.split(",")))
        if not sets:
            # набор по умолчанию -- только когда ничего не передали; раньше он
            # добавлялся всегда с устаревшей подписью «все шесть», хотя строился
            # из текущего SCORERS (их уже двенадцать), и оказывался дубликатом
            sets = [(f"все скореры ({len(SCORERS)}), сырые + процентили",
                     [f"{n}_raw" for n in SCORERS] + [f"{n}_pct" for n in SCORERS])]
        evaluate(table, sets)
