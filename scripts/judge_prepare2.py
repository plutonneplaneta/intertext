"""Слепая ручная проверка, раунд 2: исправленный протокол отбора пар.

Что пошло не так в раунде 1 (reports/manual_check.txt): списки S1 и S2 совпали целиком и состояли
из пар «однословная реплика + Быт 1:1» («Он задыхался.», «Знаю.», «Что?»). Модель обучена
ранжировать стихи ВНУТРИ абзаца эталона (длинного), а протокол ранжировал пары по сырому баллу
сразу по всем абзацам; на коротких репликах баллы выходят за пределы обучающих. Раунд оказался
неинформативным для главного сравнения. Исправления, заданные ДО чтения раунда 2:

  1. Допускаются только абзацы длиной не менее MIN_WORDS = 15 слов: на однословных репликах
     аллюзии не бывает и оценка по ним ничего не говорит о системах.
  2. Не более CAP_PER_PAR = 3 пар на абзац от одной системы (в раунде 1 у S3 21 пара приходилась на
     один абзац).
  3. Золотые контроли -- только записи с лексическим мостиком (>= 2 общих основ слов длиной от 5
     знаков между абзацем и стихом) и из диапазона берётся стих с наибольшим пересечением: рубрика
     требует текстового свидетельства, и проверять чувствительность на записях, где его заведомо
     нет (биографические, богословские отсылки), бессмысленно.
  4. Случайные контроли -- «трудные»: стих из пула того же абзаца на местах 50-100 по баллу S1
     (лексически похож, но не лучший), чтобы проверить специфичность читающего на правдоподобных
     неверных парах, а не на заведомо нелепых.

Остальное как в judge_prepare.py: те же системы S1-S3, N_TOP = 40 на систему на роман, ключ
запечатан.
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
from collections import defaultdict

import numpy as np

import judge_prepare as jp

MIN_WORDS = 15
CAP_PER_PAR = 3
N_TOP = 40
N_GOLD = 20
N_HARD = 12
SEED = 11


def stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]+", text.lower()) if len(w) >= 5}


def main(cp_table: str, out_blind: str, out_key: str) -> None:
    rng = random.Random(SEED)
    models = jp.train_models(cp_table)
    bible = {r["verse_id"]: r["text"] for r in jp.read("../data/source/bible_verses.tsv")}
    from oracle_diagnostics import load_gold, verse_matches
    W = jp.W
    novels = {
        "C&P": dict(text="../data/target/cp_fb2.tsv", pool=f"{W}/det_pool_cp.tsv", sur=f"{W}/det_sur_cp.tsv",
                    meta=f"{W}/det_meta_cp.tsv", gold="../data/gold/gold_standard_v7.tsv"),
        "BK": dict(text="../data/target/bk_rvb.tsv", pool=f"{W}/det_pool_bk.tsv", sur=f"{W}/det_sur_bk.tsv",
                   meta=f"{W}/det_meta_bk.tsv", gold="../data/gold/gold_academic_bk_v1.tsv"),
    }
    items, texts = {}, {}
    for novel, cfg in novels.items():
        text = {r["verse_id"]: r["text"] for r in jp.read(cfg["text"])}
        texts[novel] = text
        meta = {r["target_id"]: (r["is_gold"], int(r["length"])) for r in jp.read(cfg["meta"])}
        sur = {(r["target_id"], r["source_id"]): r for r in jp.read(cfg["sur"])}
        rows, seen = [], set()
        for r in jp.read(cfg["pool"]):
            k = (r["target_id"], r["source_id"])
            m = meta.get(r["target_id"])
            if k in seen or not m or m[0] != "0" or m[1] < MIN_WORDS or k not in sur:
                continue
            seen.add(k)
            r.update({c: sur[k][c] for c in jp.SUR})
            rows.append(r)
        scores = jp.score_pool(models, rows)
        pars = sorted({r["target_id"] for r in rows})
        print(f"{novel}: допустимых абзацев {len(pars)}, пар {len(rows)}", file=sys.stderr)
        for name in jp.SYSTEMS:
            taken, per_par = 0, defaultdict(int)
            for i in np.argsort(-scores[name], kind="stable"):
                par = rows[i]["target_id"]
                if per_par[par] >= CAP_PER_PAR:
                    continue
                per_par[par] += 1
                taken += 1
                it = items.setdefault((novel, par, rows[i]["source_id"]), {"systems": {}, "type": "find"})
                it["systems"][name] = taken
                if taken == N_TOP:
                    break
        # золотые контроли с мостиком, лучший по пересечению стих
        gold = load_gold(cfg["gold"])
        cand = []
        for par, group, _ in gold:
            if par not in text:
                continue
            ps = stems(text[par])
            best = max((v for v in bible if verse_matches(v, group)),
                       key=lambda v: len(ps & stems(bible[v])), default=None)
            if best and len(ps & stems(bible[best])) >= 2:
                cand.append((par, best))
        for par, v in rng.sample(cand, min(N_GOLD, len(cand))):
            items.setdefault((novel, par, v), {"systems": {}, "type": "gold"})
        print(f"  золотых с мостиком: {len(cand)}", file=sys.stderr)
        # трудные случайные: стих из пула абзаца на местах 50-100 по S1
        s1 = scores["S1"]
        by_par = defaultdict(list)
        for i, r in enumerate(rows):
            by_par[r["target_id"]].append((s1[i], r["source_id"]))
        for par in rng.sample(pars, N_HARD):
            ranked = [v for _, v in sorted(by_par[par], reverse=True)]
            if len(ranked) > 100:
                items.setdefault((novel, par, rng.choice(ranked[50:100])), {"systems": {}, "type": "hard"})

    keys = sorted(items)
    rng.shuffle(keys)
    ids = {k: hashlib.sha1(f"{SEED}|{k}".encode()).hexdigest()[:8] for k in keys}
    with open(out_blind, "w", encoding="utf-8") as f:
        f.write("item_id\tnovel\tparagraph\tverse_ref\tverse_text\n")
        for k in keys:
            novel, par, verse = k
            f.write(f"{ids[k]}\t{novel}\t{jp.window(texts[novel][par], bible[verse]).replace(chr(9), ' ')}\t"
                    f"{verse}\t{bible[verse]}\n")
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
    for s in ("S1", "S2", "S3"):
        print(f"  {s}: {sum(1 for k in keys if s in items[k]['systems'])} пар")
    print("пересечение S1 и S2:", sum(1 for k in keys if "S1" in items[k]["systems"] and "S2" in items[k]["systems"]))
    print("sha256 ключа:", hashlib.sha256(open(out_key, "rb").read()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
