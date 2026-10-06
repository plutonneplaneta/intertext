"""Стенд для проверки, помогает ли LLM-обогащение, -- с контролями от
контаминации.

ЗАЧЕМ КОНТРОЛИ. «Преступление и наказание» -- один из самых
прокомментированных романов на свете, и опознания Тихомирова в основном
совпадают с общей традицией комментирования, которая есть в сети целиком.
Языковая модель, которую спросить «какой стих стоит за „Се человек!“ в
исповеди Мармеладова», ответит «Ин 19:5» -- но не потому, что сопоставила
текст, а потому, что читала примечание. Поэтому измерение «LLM подняла
полноту с 73% до 95%» само по себе ничего не говорит о МЕТОДЕ: оно измеряет,
насколько модель помнит именно этот эталон.

Отсюда три режима, и осмысленно только их сравнение:

  rerank      модель переранжирует шортлист из K стихов по эталонному абзацу.
              Требуется не ответ, а МЕХАНИЗМ: выписать общие слова из обоих
              текстов. Цитаты проверяются программно -- если слов нет в
              текстах, ответ был припоминанием, а не сопоставлением.
  obfuscated  то же, но в абзаце заменены имена персонажей. Узнать «какое это
              место романа» становится труднее, а содержание остаётся. Если
              качество падает -- исходное было узнаванием, а не выводом.
  negative    те же запросы по абзацам, которые Тихомиров НЕ комментировал.
              Доля абзацев, где модель уверенно называет источник, -- это
              ложная тревога. Базовая доля цитатных абзацев в романе 1,1%, так
              что модель, видящая заимствование в каждом пятом абзаце,
              бесполезна независимо от полноты на эталоне.

Без режимов obfuscated и negative числа режима rerank интерпретировать нельзя.

ЗАПУСК. Нужен ключ (ANTHROPIC_API_KEY или профиль `ant auth login`).
Сначала --dry-run: он печатает оценку стоимости и пишет запросы на диск, не
обращаясь к API.

    python3 scripts/llm_enrich.py rerank     --dry-run
    python3 scripts/llm_enrich.py negative   --dry-run
    python3 scripts/llm_enrich.py rerank                # реальный прогон

Счёт идёт через Batch API (вдвое дешевле, задержка до часа не важна).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

MODEL = "claude-opus-5"
TOP_K = 50
MAX_TOKENS = 2000
N_NEGATIVE = 400
SEED = 0

# Имена персонажей -- сильнейшая подсказка «какое это место романа».
# В режиме obfuscated они заменяются, остальное остаётся как есть.
CHARACTER_NAMES = [
    "Раскольников", "Родион Романович", "Родион", "Родя",
    "Мармеладов", "Семён Захарыч", "Катерина Ивановна",
    "Соня", "Сонечка", "Софья Семёновна", "Софья Семеновна",
    "Разумихин", "Дмитрий Прокофьич", "Порфирий Петрович", "Порфирий",
    "Свидригайлов", "Аркадий Иванович", "Лужин", "Пётр Петрович",
    "Петр Петрович", "Дуня", "Дунечка", "Авдотья Романовна",
    "Пульхерия Александровна", "Лизавета", "Алёна Ивановна",
    "Алена Ивановна", "Настасья", "Заметов", "Лебезятников", "Амалия",
]
PLACEHOLDERS = ["Н.", "М.", "К.", "С.", "Р.", "П.", "Д.", "А.", "Л.", "Б."]

SCHEMA = {
    "type": "object",
    "properties": {
        "verse_id": {
            "type": "string",
            "description": "id стиха из списка кандидатов, например b.MAT.27.39; "
                           "либо пустая строка, если заимствования нет",
        },
        "kind": {
            "type": "string",
            "enum": ["дословная цитата", "парафраз",
                     "мотив без общих слов", "нет заимствования"],
        },
        "words_in_paragraph": {
            "type": "array", "items": {"type": "string"},
            "description": "слова или оборот, ВЫПИСАННЫЕ ИЗ АБЗАЦА дословно; "
                           "пустой список, если общих слов нет",
        },
        "words_in_verse": {
            "type": "array", "items": {"type": "string"},
            "description": "слова или оборот, ВЫПИСАННЫЕ ИЗ СТИХА дословно",
        },
        "confidence": {"type": "number",
                       "description": "от 0 до 1, насколько уверены"},
    },
    "required": ["verse_id", "kind", "words_in_paragraph",
                 "words_in_verse", "confidence"],
    "additionalProperties": False,
}

PROMPT = """Перед тобой абзац из русского романа XIX века и список стихов
Синодального перевода Библии -- кандидатов, отобранных автоматическим поиском.

Задача: определить, воспроизводит ли абзац один из этих стихов, и если да --
какой именно.

Требования, от которых зависит смысл ответа:

1. Опирайся ТОЛЬКО на два текста, которые видишь. Не опирайся на то, что тебе
   может быть известно об этом романе, о его комментариях или о том, какие
   места в нём принято считать библейскими. Если узнаёшь отрывок -- это не
   основание для ответа.
2. Выпиши общие слова: отдельно те, что стоят в абзаце, и те, что стоят в
   стихе. Выписывай дословно, как в тексте. Их сверят с источниками
   программно.
3. Если общих слов нет, а сходство только смысловое -- так и скажи
   («мотив без общих слов»), не подбирая слова, которых в текстах нет.
4. Если ни один стих не подходит -- «нет заимствования» с пустым verse_id.
   Это нормальный и ожидаемый ответ: большинство абзацев романа ничего не
   цитируют.

АБЗАЦ:
{paragraph}

КАНДИДАТЫ:
{candidates}
"""


def load_tsv(path: str, key: str = "verse_id", val: str = "text") -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r[key]: r[val] for r in
                csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def obfuscate(text: str) -> str:
    """Заменяет имена персонажей на нейтральные инициалы. Порядок -- от длинных
    к коротким, иначе «Родион» съест «Родион Романович»."""
    mapping = {}
    for i, name in enumerate(sorted(CHARACTER_NAMES, key=len, reverse=True)):
        mapping[name] = PLACEHOLDERS[i % len(PLACEHOLDERS)]
    out = text
    for name in sorted(mapping, key=len, reverse=True):
        out = re.sub(re.escape(name), mapping[name], out)
    return out


def shortlists(fusion_tsv: str, top_k: int) -> dict[str, list[str]]:
    """Шортлист = топ-K по среднему процентилю всех скореров. Среднее, а не
    обученные веса: шортлист не должен зависеть от того, чему модель училась на
    этом же эталоне."""
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    with open(fusion_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        pct_cols = [c for c in reader.fieldnames if c.endswith("_pct")]
        for r in reader:
            s = sum(float(r[c]) for c in pct_cols) / len(pct_cols)
            rows[r["target_id"]][r["source_id"]] = s
    return {t: [s for s, _ in sorted(v.items(), key=lambda kv: -kv[1])[:top_k]]
            for t, v in rows.items()}


def build_requests(mode: str, novel: dict, bible: dict,
                   lists: dict[str, list[str]], gold_pars: list[str]):
    rng = random.Random(SEED)
    if mode == "negative":
        pool = [p for p in lists if p not in set(gold_pars)]
        # Отрицательный контроль на десятке абзацев бессмыслен: доля ложных
        # тревог оценивается с точностью +-30%. Поэтому здесь отказ, а не
        # молчаливый прогон на том, что нашлось: таблица признаков по эталону
        # содержит нецитатных абзацев ровно ноль или один.
        if len(pool) < 100:
            print(f"В таблице признаков только {len(pool)} нецитатных абзацев -- "
                  f"для отрицательного контроля нужно хотя бы 100.\n"
                  f"Соберите шортлисты по выборке нецитатных абзацев:\n"
                  f"    python3 scripts/build_negative_shortlists.py ...\n"
                  f"и передайте полученную таблицу через --fusion.", file=sys.stderr)
            return []
        targets = rng.sample(pool, min(N_NEGATIVE, len(pool)))
    else:
        targets = gold_pars

    out = []
    for p in targets:
        text = novel.get(p, "")
        if mode == "obfuscated":
            text = obfuscate(text)
        cands = lists.get(p, [])[:TOP_K]
        cand_text = "\n".join(f"{sid}: {bible.get(sid,'')}" for sid in cands)
        out.append({
            "custom_id": f"{mode}--{p}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": PROMPT.format(
                    paragraph=text, candidates=cand_text)}],
                "output_config": {"format": {"type": "json_schema",
                                             "schema": SCHEMA}},
            },
        })
    return out


def estimate(requests: list[dict]) -> None:
    chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests)
    tok_in = chars / 2.6          # оценка по символам: count_tokens требует ключа
    tok_out = len(requests) * 300
    price = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
             "claude-haiku-4-5": (1.0, 5.0)}
    pin, pout = price.get(MODEL, (5.0, 25.0))
    full = tok_in / 1e6 * pin + tok_out / 1e6 * pout
    print(f"запросов: {len(requests)}; входных токенов ~{tok_in/1000:.0f} тыс. "
          f"(оценка по символам, не count_tokens)")
    print(f"модель {MODEL}: ~${full:.2f}, через Batch API ~${full/2:.2f}")


def verify(results_path: str, novel: dict, bible: dict, gold_tsv: str) -> None:
    """Проверка ответов: выписанные слова должны реально стоять в текстах."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from oracle_diagnostics import load_gold, verse_matches
    gold = load_gold(gold_tsv)
    refs_by_par = defaultdict(list)
    for par, group, _ in gold:
        refs_by_par[par].append(group)

    n = ok_verse = quoted_ok = asserted = 0
    for line in open(results_path, encoding="utf-8"):
        rec = json.loads(line)
        mode, par = rec["custom_id"].split("--", 1)
        ans = rec["answer"]
        n += 1
        if ans.get("kind") != "нет заимствования" and ans.get("verse_id"):
            asserted += 1
        vid = ans.get("verse_id", "")
        if vid and any(verse_matches(vid, g) for g in refs_by_par.get(par, [])):
            ok_verse += 1
        pt = novel.get(par, "").lower()
        vt = bible.get(vid, "").lower()
        wp = [w for w in ans.get("words_in_paragraph", []) if w]
        wv = [w for w in ans.get("words_in_verse", []) if w]
        if wp and wv and all(w.lower() in pt for w in wp) and all(w.lower() in vt for w in wv):
            quoted_ok += 1
    print(f"ответов: {n}")
    print(f"  назвала источник (не «нет заимствования»): {asserted} "
          f"({asserted/max(n,1):.0%})")
    print(f"  стих совпал с эталоном: {ok_verse} ({ok_verse/max(n,1):.0%})")
    print(f"  выписанные слова действительно есть в обоих текстах: {quoted_ok} "
          f"({quoted_ok/max(n,1):.0%})")
    print("\nЕсли «назвала источник» высоко на отрицательном контроле -- находки\n"
          "дешёвые. Если «стих совпал» высоко, а «слова есть в текстах» низко --\n"
          "это припоминание комментария, а не сопоставление текста.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rerank", "obfuscated", "negative", "verify"])
    ap.add_argument("--novel", default="data/target/cp_fb2.tsv")
    ap.add_argument("--bible", default="data/source/bible_verses.tsv")
    ap.add_argument("--gold", default="data/gold/gold_standard_v7.tsv")
    ap.add_argument("--fusion", required=False,
                    help="таблица признаков из fusion_retrieval.py build")
    ap.add_argument("--results", help="для режима verify: jsonl с ответами")
    ap.add_argument("--out", default="llm_requests.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    novel = load_tsv(a.novel)
    bible = load_tsv(a.bible)

    if a.mode == "verify":
        verify(a.results, novel, bible, a.gold)
        return

    if not a.fusion:
        ap.error("нужен --fusion: шортлисты берутся из таблицы признаков")
    gold_pars = sorted({r["novel_verse_id"] for r in csv.DictReader(
        open(a.gold, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        if r["novel_verse_id"] and float(r["match_score"]) >= 0.6})
    lists = shortlists(a.fusion, TOP_K)
    requests = build_requests(a.mode, novel, bible, lists, gold_pars)
    if not requests:
        return
    estimate(requests)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {a.out}")

    if a.dry_run:
        print("--dry-run: к API не обращались")
        return

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[
        Request(custom_id=r["custom_id"],
                params=MessageCreateParamsNonStreaming(**r["params"]))
        for r in requests])
    print(f"batch {batch.id}, статус {batch.processing_status}")
    print(f"результаты: python3 scripts/llm_enrich.py verify "
          f"--results <файл> ; выгрузка батча -- client.messages.batches.results('{batch.id}')")


if __name__ == "__main__":
    main()
