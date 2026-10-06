"""Проверки LLM-верификатора на запоминание вместо чтения текста.

Четыре режима (все на локальной модели, OpenAI-совместимый эндпоинт):

  pair      пара (абзац, стих) с рубрикой 0-3 И типом заимствования -- как в
            llm_verify.py, плюс поле «тип». Вход -- hard_decoys_v7.tsv, результат
            разбивается по типу обманки (G золотые, A/B/C/R обманки).
  repeat    те же пары типов G и A, REPEATS прогонов при температуре 0,7: доля
            перекидываний вердикта между прогонами и согласие с детерминированным.
  blind     «холодный» режим против якорения: модель получает фрагмент и пять
            пронумерованных стихов (один золотой, три обманки A/B/C, один
            случайный), не зная, какой предложил конвейер, и выбирает лучший или
            «нет». Доля верного выбора против шанса 1/5.
  complete  без стиха вообще: фрагмент романа и вопрос «из какого места Библии».
            Совпадение книги/главы с эталоном -- мера того, что связь просто
            выучена. Сравнивать с тем, что показывает pair на тех же золотых.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
from collections import defaultdict

from llm_verify import load_text_map
from oracle_diagnostics import load_gold, verse_matches

REPEATS = 3

TYPED_PROMPT = """Ты помогаешь филологу проверять гипотезы о библейских цитатах и аллюзиях у Достоевского. Тебе дан фрагмент романа и стих из Библии (Синодальный перевод). Оцени связь между ними СТРОГО по шкале:

0 — связи нет: общих слов нет или их совпадение случайно.
1 — общая лексика или тема, но это не заимствование.
2 — вероятный перефраз: смысл близок, но слова в основном разные.
3 — явная цитата или близкий парафраз: есть общие характерные слова или обороты, которые вряд ли совпали случайно.

И укажи тип связи: «дословное», «близкий перефраз», «тематическая перекличка», «структурное эхо» или «нет».

Фрагмент романа:
«{target}»

Стих ({source_id}):
«{source}»

Ответь СТРОГО в формате JSON без пояснений вокруг:
{{"score": <0-3>, "тип": "<тип связи>", "цитата_из_романа": "<общие слова, если есть>", "цитата_из_стиха": "<общие слова, если есть>"}}"""

BLIND_PROMPT = """Ты помогаешь филологу. Дан фрагмент романа Достоевского и пять стихов Библии (Синодальный перевод). Возможно, фрагмент восходит к одному из них, а возможно, ни к одному. Выбери стих, к которому фрагмент восходит с наибольшей вероятностью, и оцени связь по шкале: 0 — связи нет, 1 — общая тема или лексика, 2 — вероятный перефраз, 3 — явная цитата. Если связи нет ни с одним, выбери 0.

Фрагмент романа:
«{target}»

Стихи:
{verses}

Ответь СТРОГО в формате JSON без пояснений вокруг:
{{"выбор": <номер 1-5 или 0>, "score": <0-3>}}"""

COMPLETE_PROMPT = """Фрагмент из романа Достоевского «Преступление и наказание» содержит цитату или отсылку к Библии. Назови место в Библии (Синодальный перевод), к которому он восходит, не пересказывая текст.

Фрагмент: «{fragment}»

Ответь СТРОГО в формате JSON без пояснений вокруг:
{{"книга": "<русское название книги>", "глава": <номер главы>}}"""

BOOK_CODES = [("матфе", "MAT"), ("марк", "MAR"), ("лук", "LUK"), ("иоанн", "JOH"),
              ("псалм", "PSA"), ("пс", "PSA"), ("бытие", "GEN"), ("исаи", "ISA"),
              ("исход", "EXO"), ("иезекииль", "EZE"), ("михе", "MIC"), ("деяни", "ACT"),
              ("иаков", "JAM"), ("захари", "ZEC"), ("иов", "JOB"), ("притч", "PRO"),
              ("иеремии", "JER"), ("ефес", "EPH"), ("рим", "ROM")]


def book_code(name: str) -> str:
    n = name.lower().strip()
    for key, code in BOOK_CODES:
        if key in n:
            return code
    return ""


def jparse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def ask(base_url: str, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
    import urllib.request
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as e:
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return "ОШИБКА: " + err


def read_pairs(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["target_id"], r["source_id"], r["decoy_type"])
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def run_pair(pairs, novel, bible, out, url, model, temperature=0.0, runs=1, types=None):
    with open(out, "w", encoding="utf-8") as f:
        f.write("target_id\tsource_id\tdecoy_type\trun\tscore\ttype\n")
        for i, (t, s, ty) in enumerate(pairs):
            if types and ty not in types:
                continue
            for r in range(runs):
                raw = ask(url, model, TYPED_PROMPT.format(target=novel[t], source_id=s, source=bible[s]),
                          temperature)
                v = jparse(raw) or {}
                score = v.get("score", "")
                f.write(f"{t}\t{s}\t{ty}\t{r}\t{score}\t{str(v.get('тип', '')).replace(chr(9), ' ')}\n")
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(pairs)}", file=sys.stderr, flush=True)


def run_blind(pairs, novel, bible, out, url, model):
    by_t = defaultdict(lambda: defaultdict(list))
    for t, s, ty in pairs:
        by_t[t][ty].append(s)
    rng = random.Random(5)
    with open(out, "w", encoding="utf-8") as f:
        f.write("target_id\tgold_pos\tchosen\tscore\tchosen_type\n")
        for t, d in sorted(by_t.items()):
            if not d["G"] or not d["A"]:
                continue
            cands = [(d["G"][0], "G")] + [(d[k][0], k) for k in ("A", "B", "C", "R") if d[k]][:4]
            rng.shuffle(cands)
            listing = "\n".join(f"{i+1}. ({s}) «{bible[s]}»" for i, (s, _) in enumerate(cands))
            raw = ask(url, model, BLIND_PROMPT.format(target=novel[t], verses=listing))
            v = jparse(raw) or {}
            ch = v.get("выбор", 0)
            try:
                ch = int(ch)
            except (TypeError, ValueError):
                ch = 0
            gold_pos = [i for i, (_, ty) in enumerate(cands) if ty == "G"][0] + 1
            ctype = cands[ch - 1][1] if 1 <= ch <= len(cands) else "none"
            f.write(f"{t}\t{gold_pos}\t{ch}\t{v.get('score', '')}\t{ctype}\n")


def run_complete(gold_tsv, out, url, model):
    gold = load_gold(gold_tsv)
    with open(out, "w", encoding="utf-8") as f:
        f.write("target_id\tbook_ok\tchapter_ok\tanswer\n")
        for par, group, frag in gold:
            raw = ask(url, model, COMPLETE_PROMPT.format(fragment=frag.strip("… ")))
            v = jparse(raw) or {}
            code = book_code(str(v.get("книга", "")))
            try:
                ch = int(v.get("глава"))
            except (TypeError, ValueError):
                ch = -1
            book_ok = any(code == b for b, c, vf, vt in group)
            chap_ok = any(code == b and ch == c for b, c, vf, vt in group)
            f.write(f"{par}\t{int(book_ok)}\t{int(chap_ok)}\t{str(v.get('книга', '')).replace(chr(9), ' ')} {ch}\n")


if __name__ == "__main__":
    mode, pairs_path, novel_tsv, bible_tsv, gold_tsv, out, url, model = sys.argv[1:9]
    novel, bible = load_text_map(novel_tsv), load_text_map(bible_tsv)
    pairs = read_pairs(pairs_path)
    if mode == "pair":
        run_pair(pairs, novel, bible, out, url, model)
    elif mode == "repeat":
        run_pair(pairs, novel, bible, out, url, model, temperature=0.7, runs=REPEATS, types={"G", "A"})
    elif mode == "blind":
        run_blind(pairs, novel, bible, out, url, model)
    elif mode == "complete":
        run_complete(gold_tsv, out, url, model)
    else:
        raise SystemExit(f"режим? {mode}")
    print(f"готово: {mode} -> {out}")
