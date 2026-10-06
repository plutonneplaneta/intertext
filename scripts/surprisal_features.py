"""Surprisal / условная перплексия как признак заимствования (Phase 10).

Идея из отчёта (по мотивам ReCaLL): если предложение романа восходит к стиху,
то стих в контексте должен делать его для языковой модели менее неожиданным,
чем то же предложение без контекста:

    delta = log P(предложение | стих) - log P(предложение)

В отчёте это названо экстраполяцией, а не подтверждённым фактом; здесь проверяется
дёшево и один раз. Один проход вперёд, без генерации.

Оговорки, принятые заранее:
  * Считается только по парам пула (не по всем 31 102 стихам): полное пространство
    нереалистично, как и предупреждает отчёт. Процентиль -- внутри пула абзаца.
  * Для каждой пары берётся предложение абзаца, ближайшее к стиху по E5 (как в
    sentence_pairs.py), чтобы не считать все предложения абзаца.
  * Базовая (не инструктивная) модель: измеряется предсказание текста, а не
    следование промпту. Модель могла видеть и Библию, и роман -- это общая
    оговорка к любому такому сигналу.
  * Есть общий эффект «любой стих помогает любому предложению» (общие слова);
    логрегрессия и процентиль внутри абзаца его частично снимают, но не полностью.

Признаки: sur_delta (разность в нат, суммарно) и sur_gain (то же на токен
предложения).
"""
from __future__ import annotations

import csv
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np

MAX_VERSE, MAX_SENT = 64, 64

MODE = os.environ.get("SUR_MODE", "orig")


def variant(verse: str, sentence: str, vid: str) -> str:
    """Вариант контекста для анализа механизма.

    shuffle -- слова стиха перемешаны (лексика та же, порядок и синтаксис разрушены);
    mask    -- слова стиха, общие с предложением (по первым пяти буквам), заменены на «…»:
               остаётся только связь, не опирающаяся на общие слова.
    """
    if MODE == "orig":
        return verse
    words = verse.split()
    if MODE == "shuffle":
        rnd = random.Random(vid)
        rnd.shuffle(words)
        return " ".join(words)
    if MODE == "mask":
        stems = {w[:5] for w in re.findall(r"[а-яё]+", sentence.lower()) if len(w) >= 4}
        out = []
        for w in words:
            core = re.sub(r"[^а-яё]", "", w.lower())
            out.append("…" if len(core) >= 4 and core[:5] in stems else w)
        return " ".join(out)
    raise SystemExit(f"SUR_MODE? {MODE}")


def read_tsv(path):
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            ids.append(r["verse_id"])
            texts.append(r["text"])
    return ids, texts


def main(model_name: str, bible_tsv: str, sents_tsv: str, bible_npz: str, sent_npz: str,
         table_tsv: str, out_tsv: str, limit: int = 0, batch: int = 32, reuse: str = "") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = next((f"cuda:{i}" for i in range(torch.cuda.device_count())
                if "3060" in torch.cuda.get_device_name(i)), "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(dev).eval()
    print(f"модель {model_name}, устройство {dev}", file=sys.stderr, flush=True)

    b = np.load(bible_npz, allow_pickle=True)
    q = np.load(sent_npz, allow_pickle=True)
    b_pos = {s: i for i, s in enumerate(b["ids"])}
    S, Q = b["embeddings"], q["embeddings"]
    sents_of = defaultdict(list)
    q_ids = list(q["ids"])
    for i, sid in enumerate(q_ids):
        sents_of[sid.split("#", 1)[0]].append(i)

    bible_ids, bible_text = read_tsv(bible_tsv)
    verse_text = dict(zip(bible_ids, bible_text))
    sid_all, sent_text_all = read_tsv(sents_tsv)
    sent_text = dict(zip(sid_all, sent_text_all))

    pairs, seen = [], set()
    with open(table_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            k = (r["target_id"], r["source_id"])
            if k not in seen:
                seen.add(k)
                pairs.append(k)
    if limit:
        pairs = pairs[:limit]

    chosen = []
    for par, sid in pairs:
        idx = sents_of.get(par)
        if not idx or sid not in b_pos:
            chosen.append(None)
            continue
        sims = Q[idx] @ S[b_pos[sid]]
        chosen.append(q_ids[idx[int(np.argmax(sims))]])
    cache: dict[tuple[str, str], tuple[float, float]] = {}
    if reuse:
        with open(reuse, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                cache[(r["target_id"], r["source_id"])] = (float(r["sur_delta_raw"]), float(r["sur_gain_raw"]))
        # кэш годится, только если предложение выбрано тем же способом (E5 по тем же векторам)
        chosen = [None if (pairs[i] in cache) else c for i, c in enumerate(chosen)]
        print(f"из кэша взято {sum(1 for p in pairs if p in cache)} пар", file=sys.stderr)
    print(f"пар {len(pairs)}, считается заново {sum(1 for c in chosen if c)}, уникальных предложений {len({c for c in chosen if c})}",
          file=sys.stderr, flush=True)

    nl = tok("\n", add_special_tokens=False)["input_ids"]
    enc_cache: dict[str, list[int]] = {}

    def enc(text: str, cap: int) -> list[int]:
        key = f"{cap}|{text}"
        if key not in enc_cache:
            enc_cache[key] = tok(text, add_special_tokens=False)["input_ids"][:cap]
        return enc_cache[key]

    @torch.no_grad()
    def score(items: list[tuple[list[int], list[int]]]) -> list[float]:
        """Сумма логвероятностей токенов второй части при первой как контексте."""
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            seqs = [c + s for c, s in chunk]
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for j, x in enumerate(seqs):
                ids[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            ids_d = ids.to(dev)
            hidden = lm.model(input_ids=ids_d, attention_mask=mask.to(dev)).last_hidden_state
            for j, (c, s_) in enumerate(chunk):
                # голова словаря только на позициях предложения: полные логиты
                # по словарю в 152 тыс. токенов на батч не помещаются в 12 ГБ
                h = hidden[j, len(c) - 1:len(c) - 1 + len(s_)]
                lp = torch.log_softmax(lm.lm_head(h).float(), dim=-1)
                tgt = ids_d[j, len(c):len(c) + len(s_)]
                out.append(float(lp.gather(-1, tgt.unsqueeze(-1)).sum()))
        return out

    uniq = sorted({c for c in chosen if c})
    t0 = time.time()
    base_lp = dict(zip(uniq, score([(nl, enc(sent_text[u], MAX_SENT)) for u in uniq])))
    print(f"без контекста: {len(uniq)} предложений за {time.time()-t0:.0f}с", file=sys.stderr, flush=True)

    # пары сортируем по длине, чтобы выравнивание давало меньше пустых мест
    order = sorted((i for i, c in enumerate(chosen) if c),
                   key=lambda i: len(enc(verse_text[pairs[i][1]], MAX_VERSE)) +
                   len(enc(sent_text[chosen[i]], MAX_SENT)))
    items = [(enc(variant(verse_text[pairs[i][1]], sent_text[chosen[i]], pairs[i][1]), MAX_VERSE) + nl,
              enc(sent_text[chosen[i]], MAX_SENT))
             for i in order]
    cond = []
    for k in range(0, len(items), batch * 40):
        cond += score(items[k:k + batch * 40])
        print(f"{min(k + batch * 40, len(items))}/{len(items)} за {time.time()-t0:.0f}с",
              file=sys.stderr, flush=True)

    delta = {p: v for p, v in cache.items()}
    for i, c in zip(order, cond):
        sent = chosen[i]
        n = max(1, len(enc(sent_text[sent], MAX_SENT)))
        d = c - base_lp[sent]
        delta[pairs[i]] = (d, d / n)

    by_par = defaultdict(list)
    for p in pairs:
        by_par[p[0]].append(p[1])
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tsur_delta_raw\tsur_delta_ppct\tsur_gain_raw\tsur_gain_ppct\n")
        for par, sids in by_par.items():
            cols = [[delta.get((par, s), (0.0, 0.0))[j] for s in sids] for j in range(2)]
            pct = []
            for vals in cols:
                m: dict[float, float] = {}
                for i, x in enumerate(sorted(vals)):
                    m.setdefault(x, i / len(vals))
                pct.append(m)
            for k, s in enumerate(sids):
                f.write(f"{par}\t{s}\t{cols[0][k]:.4f}\t{pct[0][cols[0][k]]:.6f}\t"
                        f"{cols[1][k]:.4f}\t{pct[1][cols[1][k]]:.6f}\n")
    d = np.array([v[0] for v in delta.values()])
    print(f"delta: среднее {d.mean():.2f}, sd {d.std():.2f}, доля >0 {np.mean(d > 0):.2f} -> {out_tsv}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a[:7], limit=int(a[7]) if len(a) > 7 else 0, reuse=a[8] if len(a) > 8 else "")
