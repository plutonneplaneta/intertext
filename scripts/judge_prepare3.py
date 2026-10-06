"""Слепая ручная проверка, раунд 3: отбор находок ВНУТРИ пула абзаца.

Почему: в раундах 1-2 пары ранжировались по сырому баллу между абзацами, и в топ попадали
«хаб»-стихи (Откр 22:11/22:21, 1 Пар 2:52), которые модель любит независимо от абзаца. Модель же
обучена ранжировать стихи внутри пула абзаца. Раунд 3 следует этому:

  1. для каждого допустимого абзаца (>= MIN_WORDS слов, неэталонный) система выбирает один
     лучший стих из пула этого абзаца (top-1 по её баллу);
  2. абзацы упорядочиваются по заметности лучшего стиха в собственном пуле:
     z = (макс. балл - средний балл пула) / стд. балла пула;
  3. берутся N_TOP абзацев с наибольшим z на систему на роман; один и тот же стих у одной системы
     не более CAP_PER_VERSE раз.

Пары, уже прочитанные в раундах 1-2, исключаются заранее (читающий их помнит); система берёт тогда
лучший из оставшихся стихов абзаца. Золотые контроли: с мостиком в >= GOLD_STEMS = 2 общих основ, ранее не читавшиеся
(порог 3 дал бы всего 7 пар, поэтому оставлен порог раунда 2; чувствительность дополнительно
печатается по силе мостика); трудные случайные -- как в раунде 2.
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from collections import defaultdict

import numpy as np

import judge_prepare as jp

MIN_WORDS = 15
N_TOP = 40
CAP_PER_VERSE = 3
N_GOLD = 20
GOLD_STEMS = 2
N_HARD = 12
SEED = 13
W = jp.W
PREV_KEYS = [f"{W}/judge_key.tsv", f"{W}/judge2_key.tsv"]


def stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]+", text.lower()) if len(w) >= 5}


def prev_pairs() -> set:
    out = set()
    for p in PREV_KEYS:
        for r in jp.read(p):
            out.add((r["novel"], r["paragraph_id"], r["verse_id"]))
    return out


def main(cp_table: str, out_blind: str, out_key: str) -> None:
    rng = random.Random(SEED)
    seen_before = prev_pairs()
    print(f"ранее прочитано пар: {len(seen_before)}", file=sys.stderr)
    models = jp.train_models(cp_table)
    bible = {r["verse_id"]: r["text"] for r in jp.read("../data/source/bible_verses.tsv")}
    from oracle_diagnostics import load_gold, verse_matches
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
        by_par = defaultdict(list)
        for i, r in enumerate(rows):
            by_par[r["target_id"]].append(i)
        pars = sorted(by_par)
        print(f"{novel}: допустимых абзацев {len(pars)}, пар {len(rows)}", file=sys.stderr)
        for name in jp.SYSTEMS:
            sc = scores[name]
            cand = []
            for par, idx in by_par.items():
                allv = sc[idx]
                if len(allv) < 10 or allv.std() == 0:
                    continue
                fresh = [i for i in idx if (novel, par, rows[i]["source_id"]) not in seen_before]
                if not fresh:
                    continue
                best = max(fresh, key=lambda i: sc[i])
                cand.append(((sc[best] - allv.mean()) / allv.std(), par, rows[best]["source_id"]))
            cand.sort(reverse=True)
            taken, per_verse = 0, defaultdict(int)
            for z, par, verse in cand:
                if per_verse[verse] >= CAP_PER_VERSE:
                    continue
                per_verse[verse] += 1
                taken += 1
                it = items.setdefault((novel, par, verse), {"systems": {}, "type": "find"})
                it["systems"][name] = taken
                if taken == N_TOP:
                    break
        # золотые контроли с сильным мостиком, ранее не читавшиеся
        gold = load_gold(cfg["gold"])
        cands = []
        for par, group, _ in gold:
            if par not in text:
                continue
            ps = stems(text[par])
            best = max((v for v in bible if verse_matches(v, group)),
                       key=lambda v: len(ps & stems(bible[v])), default=None)
            if best and len(ps & stems(bible[best])) >= GOLD_STEMS and (novel, par, best) not in seen_before:
                cands.append((par, best))
        for par, v in rng.sample(cands, min(N_GOLD, len(cands))):
            items.setdefault((novel, par, v), {"systems": {}, "type": "gold"})
        print(f"  золотых с мостиком >= {GOLD_STEMS}: {len(cands)}", file=sys.stderr)
        # трудные случайные: места 50-100 в пуле абзаца по S1
        s1 = scores["S1"]
        ranked_by_par = {}
        for par, idx in by_par.items():
            ranked = [rows[i]["source_id"] for i in sorted(idx, key=lambda i: -s1[i])]
            if len(ranked) > 100:
                ranked_by_par[par] = ranked
        hard_pars = rng.sample(sorted(ranked_by_par), N_HARD)
        for par in hard_pars:
            v = rng.choice(ranked_by_par[par][50:100])
            if (novel, par, v) not in seen_before:
                items.setdefault((novel, par, v), {"systems": {}, "type": "hard"})

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
    for a, b in (("S1", "S2"), ("S1", "S3"), ("S2", "S3")):
        print(f"пересечение {a} и {b}:", sum(1 for k in keys if a in items[k]["systems"] and b in items[k]["systems"]))
    vc = defaultdict(int)
    for k in keys:
        if "S1" in items[k]["systems"]:
            vc[k[2]] += 1
    print("S1: разных стихов", len(vc), "самые частые", sorted(vc.items(), key=lambda x: -x[1])[:4])
    print("sha256 ключа:", hashlib.sha256(open(out_key, "rb").read()).hexdigest())
    print("sha256 слепого списка:", hashlib.sha256(open(out_blind, "rb").read()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
