"""LLM-верификатор кандидатных пар (методика из отчёта LLM/графы): не
судья "аллюзия/не аллюзия" в открытой формулировке (задокументированное
падение согласия с людьми на таких формулировках), а рубрицированная
порядковая шкала с явно расписанными уровнями -- и обоснование цитатой,
а не голая метка, чтобы вердикт можно было проверить, а не поверить на
слово.

Шкала:
  0 -- связи нет (общие слова случайны или их нет);
  1 -- общая лексика/тема, не заимствование;
  2 -- вероятный перефраз (смысл близок, слова разные);
  3 -- явная цитата или близкий парафраз (общие характерные слова/обороты).

Модель -- локальный тестовый контейнер (llama.cpp, тот же образ и веса,
что и боевая служба основного проекта, отдельный порт 8001, не трогает
боевой контур). OpenAI-совместимый /v1/chat/completions.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request

RUBRIC_PROMPT = """Ты помогаешь филологу проверять гипотезы о библейских цитатах и аллюзиях у Достоевского. Тебе дан фрагмент романа и стих из Библии (Синодальный перевод). Оцени связь между ними СТРОГО по шкале:

0 — связи нет: общих слов нет или их совпадение случайно.
1 — общая лексика или тема, но это не заимствование (общеупотребительные слова, бытовая ситуация).
2 — вероятный перефраз: смысл близок, но слова в основном разные — Достоевский мог иметь в виду этот стих, не пересказывая его дословно.
3 — явная цитата или близкий парафраз: есть общие характерные слова или обороты, которые вряд ли совпали случайно.

Фрагмент романа:
«{target}»

Стих ({source_id}):
«{source}»

Ответь СТРОГО в формате JSON без пояснений вокруг:
{{"score": <0-3>, "цитата_из_романа": "<общие слова/оборот из фрагмента, если есть>", "цитата_из_стиха": "<общие слова/оборот из стиха, если есть>"}}"""


def call_llm(base_url: str, model: str, prompt: str, temperature: float = 0.0, timeout: int = 60) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_verdict(text: str) -> dict | None:
    # модель иногда оборачивает JSON в ```json ... ``` или добавляет текст вокруг
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        obj["score"] = int(obj.get("score", -1))
        if obj["score"] not in (0, 1, 2, 3):
            return None
        return obj
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def load_text_map(tsv_path: str, id_col: str = "verse_id") -> dict[str, str]:
    out = {}
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            out[row[id_col]] = row["text"]
    return out


def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            pairs.append((row["target_id"], row["source_id"]))
    return pairs


def main(
    pairs_path: str, novel_tsv: str, bible_tsv: str, out_path: str,
    base_url: str = "http://localhost:8001/v1", model: str = "qwen-test",
) -> None:
    pairs = load_pairs(pairs_path)
    novel_text = load_text_map(novel_tsv)
    bible_text = load_text_map(bible_tsv)
    print(f"пар для проверки: {len(pairs)}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("target_id\tsource_id\tllm_score\tцитата_романа\tцитата_стиха\traw\n")
        t0 = time.time()
        n_ok, n_fail = 0, 0
        for i, (tid, sid) in enumerate(pairs):
            target = novel_text.get(tid, "")
            source = bible_text.get(sid, "")
            if not target or not source:
                continue
            prompt = RUBRIC_PROMPT.format(target=target, source_id=sid, source=source)
            try:
                raw = call_llm(base_url, model, prompt)
                verdict = parse_verdict(raw)
            except Exception as e:  # сеть/таймаут -- не срывать весь прогон
                print(f"ошибка на {tid}/{sid}: {e}", file=sys.stderr)
                verdict, raw = None, str(e)
            if verdict:
                n_ok += 1
                cit_r = verdict.get("цитата_из_романа", "").replace("\t", " ").replace("\n", " ")
                cit_s = verdict.get("цитата_из_стиха", "").replace("\t", " ").replace("\n", " ")
                out.write(f"{tid}\t{sid}\t{verdict['score']}\t{cit_r}\t{cit_s}\t{raw[:200].__repr__()}\n")
            else:
                n_fail += 1
                out.write(f"{tid}\t{sid}\t\t\t\t{str(raw)[:200].__repr__()}\n")
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"{i+1}/{len(pairs)} за {elapsed:.0f}с (успешно разобрано: {n_ok}, не разобрано: {n_fail})",
                      file=sys.stderr, flush=True)

    print(f"готово: {n_ok} разобрано, {n_fail} не разобрано -> {out_path}")


if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) > 5:
        kwargs["base_url"] = sys.argv[5]
    if len(sys.argv) > 6:
        kwargs["model"] = sys.argv[6]
    main(*sys.argv[1:5], **kwargs)
