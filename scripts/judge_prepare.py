"""Подготовка слепой ручной проверки непомеченных находок.

Вопрос: насколько ТОЧНЫ находки систем на абзацах, которые комментарий не разбирает? Эталон
неполон (комментатор отмечает не всё), поэтому «ложное срабатывание» по эталону может быть
настоящей неотмеченной аллюзией. Её можно установить только чтением.

Системы (логрегрессия обучена на «Преступлении и наказании», на пулах той же конструкции, что
и тестовые: объединение топ-K по скореру, K = POOL_K; см. ниже), по одной модели на систему:
  S1  база из 12 скореров
  S2  база + surprisal 1,5B
  S3  один surprisal 1,5B

Для каждого романа (C&P и «Карамазовы») берутся пары (абзац, стих) из пулов НЕЭТАЛОННЫХ абзацев
выборки обнаружения (600 случайных, seed 0), и каждая система выдаёт N лучших пар по своему
баллу (по всем абзацам вместе). Объединение пар систем -- кандидаты на чтение.

Контроль самого читающего: в тот же список вслепую подмешаны (а) золотые пары (абзац и один из
стихов, документированных комментарием) и (б) случайные пары (случайный стих к случайному
неэталонному абзацу). Они нужны, чтобы оценить, находит ли читающий известные аллюзии и не
видит ли аллюзий там, где их заведомо нет.

Выход: слепой список (item_id в случайном порядке, роман, текст абзаца, ссылка и текст стиха --
БЕЗ системы, ранга, балла и типа) и ключ (item_id -> система, ранги, тип). Ключ читающему не
показывается до фиксации ответов; его SHA-256 записывается в предварительную фиксацию.

Использование (все пути -- на сервере):
  judge_prepare.py <cp_table.tsv> <out_blind.tsv> <out_key.tsv>
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from collections import defaultdict

import numpy as np

from fusion_retrieval import fit_logreg
from retrieval_scorers import SCORERS

W = "/tmp/intertext-work"
N_TOP = 40          # находок на систему на роман
N_GOLD = 20         # золотых контрольных пар на роман
N_RANDOM = 12       # случайных контрольных пар на роман
TOP20 = 1 - 20 / 31102 - 1e-9
SEED = 7

NAMES = list(SCORERS) + ["e5"]
BASE = [f"{n}_raw" for n in NAMES] + [f"{n}_pct" for n in NAMES]
SUR = ["sur_delta_raw", "sur_delta_ppct", "sur_gain_raw", "sur_gain_ppct"]
SYSTEMS = {"S1": BASE, "S2": BASE + SUR, "S3": SUR}


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def ppct_inplace(rows, key_col, out_col, group_col="target_id"):
    by = defaultdict(dict)
    for r in rows:
        by[r[group_col]][r["source_id"]] = float(r[key_col])
    for par, d in by.items():
        vals = sorted(d.values())
        m = {}
        for i, x in enumerate(vals):
            m.setdefault(x, i / len(vals))
        for r in rows:
            if r[group_col] == par:
                r[out_col] = str(m[float(r[key_col])])


def train_models(cp_table):
    rows = read(cp_table)
    keep = [r for r in rows if r["label"] == "1" or any(float(r[f"{n}_pct"]) >= TOP20 for n in NAMES)]
    ppct_inplace(keep, "sur_delta_raw", "sur_delta_ppct")
    ppct_inplace(keep, "sur_gain_raw", "sur_gain_ppct")
    print(f"обучающая таблица: {len(rows)} строк -> {len(keep)} в пуле топ-20", file=sys.stderr)
    y = np.array([int(r["label"]) for r in keep], float)
    models = {}
    for name, cols in SYSTEMS.items():
        X = np.array([[float(r[c]) for c in cols] for r in keep])
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        models[name] = (cols, mu, sd, fit_logreg((X - mu) / sd, y))
    return models


def score_pool(models, rows):
    out = {}
    for name, (cols, mu, sd, w) in models.items():
        X = np.array([[float(r[c]) for c in cols] for r in rows])
        out[name] = np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w
    return out


def window(par_text: str, verse_text: str, limit: int = 1100) -> str:
    if len(par_text) <= limit:
        return par_text
    stems = {w[:5] for w in re.findall(r"[а-яё]+", verse_text.lower()) if len(w) >= 4}
    sents = re.split(r"(?<=[.!?…])\s+", par_text)
    best = max(range(len(sents)), key=lambda i: len({w[:5] for w in re.findall(r"[а-яё]+", sents[i].lower())} & stems))
    lo = hi = best
    size = len(sents[best])
    while True:
        grew = False
        if lo > 0 and size + len(sents[lo - 1]) <= limit:
            lo -= 1; size += len(sents[lo]); grew = True
        if hi < len(sents) - 1 and size + len(sents[hi + 1]) <= limit:
            hi += 1; size += len(sents[hi]); grew = True
        if not grew:
            break
    return ("… " if lo > 0 else "") + " ".join(sents[lo:hi + 1]) + (" …" if hi < len(sents) - 1 else "")


def main(cp_table: str, out_blind: str, out_key: str) -> None:
    rng = random.Random(SEED)
    models = train_models(cp_table)
    bible = {r["verse_id"]: r["text"] for r in read("../data/source/bible_verses.tsv")}
    novels = {
        "C&P": dict(text="../data/target/cp_fb2.tsv", pool=f"{W}/det_pool_cp.tsv", sur=f"{W}/det_sur_cp.tsv",
                    meta=f"{W}/det_meta_cp.tsv", gold="../data/gold/gold_standard_v7.tsv"),
        "BK": dict(text="../data/target/bk_rvb.tsv", pool=f"{W}/det_pool_bk.tsv", sur=f"{W}/det_sur_bk.tsv",
                   meta=f"{W}/det_meta_bk.tsv", gold="../data/gold/gold_academic_bk_v1.tsv"),
    }
    items = {}   # (novel, par, verse) -> dict(systems={S:rank})
    for novel, cfg in novels.items():
        text = {r["verse_id"]: r["text"] for r in read(cfg["text"])}
        meta = {r["target_id"]: r["is_gold"] for r in read(cfg["meta"])}
        sur = {(r["target_id"], r["source_id"]): r for r in read(cfg["sur"])}
        rows, seen = [], set()
        for r in read(cfg["pool"]):
            k = (r["target_id"], r["source_id"])
            if k in seen or meta.get(r["target_id"]) != "0" or k not in sur:
                continue
            seen.add(k)
            r.update({c: sur[k][c] for c in SUR})
            rows.append(r)
        scores = score_pool(models, rows)
        print(f"{novel}: неэталонных пар в пулах {len(rows)} (абзацев {len({r['target_id'] for r in rows})})", file=sys.stderr)
        for name in SYSTEMS:
            order = np.argsort(-scores[name], kind="stable")[:N_TOP]
            for rank, i in enumerate(order, 1):
                k = (novel, rows[i]["target_id"], rows[i]["source_id"])
                items.setdefault(k, {"systems": {}, "type": "find"})["systems"][name] = rank
        # контроль: золотые пары
        from oracle_diagnostics import load_gold, verse_matches
        gold = load_gold(cfg["gold"])
        for par, group, _frag in rng.sample(gold, min(N_GOLD, len(gold))):
            vs = [v for v in bible if verse_matches(v, group)]
            if vs and par in text:
                items.setdefault((novel, par, rng.choice(vs)), {"systems": {}, "type": "gold"})
        # контроль: случайные пары
        pars = sorted({r["target_id"] for r in rows})
        for _ in range(N_RANDOM):
            items.setdefault((novel, rng.choice(pars), rng.choice(list(bible))), {"systems": {}, "type": "random"})
        novels[novel]["text_map"] = text

    keys = sorted(items)
    rng.shuffle(keys)
    ids = {}
    for k in keys:
        ids[k] = hashlib.sha1(f"{SEED}|{k}".encode()).hexdigest()[:8]
    with open(out_blind, "w", encoding="utf-8") as f:
        f.write("item_id\tnovel\tparagraph\tverse_ref\tverse_text\n")
        for k in keys:
            novel, par, verse = k
            ptxt = novels[novel]["text_map"][par]
            f.write(f"{ids[k]}\t{novel}\t{window(ptxt, bible[verse]).replace(chr(9), ' ')}\t{verse}\t{bible[verse]}\n")
    with open(out_key, "w", encoding="utf-8") as f:
        f.write("item_id\tnovel\tparagraph_id\tverse_id\ttype\tsystems\n")
        for k in keys:
            it = items[k]
            f.write(f"{ids[k]}\t{k[0]}\t{k[1]}\t{k[2]}\t{it['type']}\t"
                    f"{','.join(f'{s}:{r}' for s, r in sorted(it['systems'].items()))}\n")
    cnt = defaultdict(int)
    for k in keys:
        cnt[items[k]["type"]] += 1
    print(f"пар для чтения {len(keys)}: " + ", ".join(f"{t}={n}" for t, n in sorted(cnt.items())))
    print("sha256 ключа:", hashlib.sha256(open(out_key, "rb").read()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
