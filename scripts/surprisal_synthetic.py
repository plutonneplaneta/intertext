"""Проверка контаминации surprisal: находит ли он стих по предложению, которого нет в обучающих данных.

Три условия на каждое место эталона, один и тот же набор кандидатов (стих места и
299 стихов-отвлекающих из пула абзаца, не входящих в правильные):

  real  предложение романа, содержащее фрагмент из эталона (иначе -- ближайшее по E5);
  syn   синтетическая аллюзия (gen_synthetic_allusions.py): смысл стиха без его слов;
  neu   нейтральное предложение без аллюзий: контроль -- ранг стиха должен быть шансовым.

Для каждого условия -- ранг стиха среди 300 кандидатов по delta = log P(предл | стих) −
log P(предл) и, для сравнения, по косинусу E5. Шанс: медиана 150, топ-10 0,033, топ-30 0,10.

Как читать. Если запоминание связей из литературы -- источник сигнала, то на syn он
должен пропасть, а на real остаться. Если syn находится сопоставимо с real (а neu -- на
уровне шанса), сигнал опирается на текст, а не на выученные пары. Оговорка: у real есть
лексический мостик, у syn он убран по построению, поэтому syn может быть труднее и по
причинам, не связанным с памятью -- это сравнивается с E5, у которого запоминания нет.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

import numpy as np

N_DISTRACTORS = 299


def main(model_name: str, gold_tsv: str, bible_tsv: str, sents_tsv: str, syn_tsv: str,
         table_tsv: str, bible_npz: str, sent_npz: str, out_prefix: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_contamination import load_text_map
    from oracle_diagnostics import load_gold, verse_matches

    dev = next((f"cuda:{i}" for i in range(torch.cuda.device_count())
                if "3060" in torch.cuda.get_device_name(i)), "cpu")
    bible = load_text_map(bible_tsv)
    sents = load_text_map(sents_tsv)
    gold = load_gold(gold_tsv)
    b = np.load(bible_npz, allow_pickle=True)
    b_pos = {s: i for i, s in enumerate(b["ids"])}
    S = b["embeddings"]
    q = np.load(sent_npz, allow_pickle=True)
    q_pos = {s: i for i, s in enumerate(q["ids"])}

    pool = defaultdict(list)
    with open(table_tsv, encoding="utf-8", newline="") as f:
        seen = set()
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            k = (r["target_id"], r["source_id"])
            if k not in seen:
                seen.add(k)
                pool[r["target_id"]].append(r["source_id"])

    syn = {}
    with open(syn_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            syn[(int(r["rec_id"]), r["kind"])] = r["text"]

    e5 = SentenceTransformer("intfloat/multilingual-e5-base", device=dev)
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(dev).eval()
    nl = tok("\n", add_special_tokens=False)["input_ids"]

    def enc(t, cap):
        return tok(t, add_special_tokens=False)["input_ids"][:cap]

    @torch.no_grad()
    def lp(items, batch=32):
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            width = max(len(c) + len(s) for c, s in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (c, s) in enumerate(chunk):
                x = c + s
                ids[j, :len(x)] = torch.tensor(x)
                mask[j, :len(x)] = 1
            ids_d = ids.to(dev)
            hidden = lm.model(input_ids=ids_d, attention_mask=mask.to(dev)).last_hidden_state
            for j, (c, s) in enumerate(chunk):
                h = hidden[j, len(c) - 1:len(c) - 1 + len(s)]
                lps = torch.log_softmax(lm.lm_head(h).float(), dim=-1)
                out.append(float(lps.gather(-1, ids_d[j, len(c):len(c) + len(s)].unsqueeze(-1)).sum()))
        return out

    rng = random.Random(3)
    rows = []
    for k, (par, group, frag) in enumerate(gold):
        verse = next((s for s in bible if verse_matches(s, group)), None)
        if verse is None or par not in pool or (k, "syn") not in syn:
            continue
        correct = {s for s in bible if verse_matches(s, group)}
        cands = [s for s in pool[par] if s not in correct]
        rng.shuffle(cands)
        cand = [verse] + cands[:N_DISTRACTORS]
        key = frag.strip("… ").lower()[:25]
        in_par = [sid for sid in sents if sid.split("#", 1)[0] == par]
        real = next((sid for sid in in_par if key and key in sents[sid].lower()), None)
        if real is None and in_par and all(s in q_pos for s in in_par):
            sims = [float(q["embeddings"][q_pos[s]] @ S[b_pos[verse]]) for s in in_par]
            real = in_par[int(np.argmax(sims))]
        if real is None:
            continue
        texts = {"real": sents[real], "syn": syn[(k, "syn")], "neu": syn[(k, "neu")]}
        embs = e5.encode([f"query: {t}" for t in texts.values()], normalize_embeddings=True)
        for (cond, text), emb in zip(texts.items(), embs):
            base = lp([(nl, enc(text, 64))])[0]
            conds = lp([(enc(bible[v], 64) + nl, enc(text, 64)) for v in cand])
            delta = np.array(conds) - base
            cos = np.array([float(emb @ S[b_pos[v]]) for v in cand])
            r_sur = int((delta > delta[0]).sum()) + 1
            r_e5 = int((cos > cos[0]).sum()) + 1
            rows.append((k, par, cond, r_sur, r_e5))
        print(f"{k}/{len(gold)}", file=sys.stderr, flush=True)

    with open(f"{out_prefix}.tsv", "w", encoding="utf-8") as f:
        f.write("rec_id\tpar\tcond\trank_surprisal\trank_e5\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"модель {model_name}; кандидатов {N_DISTRACTORS + 1} на место; шанс: медиана 150, топ-10 0,033, топ-30 0,10")
    for cond in ("real", "syn", "neu"):
        for name, j in (("surprisal", 3), ("e5", 4)):
            v = np.array([r[j] for r in rows if r[2] == cond], float)
            print(f"  {cond:4s} {name:9s} n={len(v):2d} медиана {np.median(v):5.0f} "
                  f"топ-10 {np.mean(v <= 10):.2f} топ-30 {np.mean(v <= 30):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:10])
