"""Синтетические аллюзии для проверки контаминации сигнала surprisal.

Если surprisal находит стих потому, что языковая модель запомнила связи «место
романа ↔ стих» из научной литературы, то на тексте, которого в обучающих данных
быть не могло, сигнал пропадёт. Поэтому для каждого места эталона локальная модель
пишет НОВОЕ предложение в духе Достоевского, отсылающее к тому же стиху смыслом,
но без его слов и оборотов («syn»), и нейтральное предложение без религиозных мотивов
(«neu», контроль на то, что стиль генератора сам не даёт ложных находок).

Стих -- первый из процитированных для записи (фиксированный выбор, без подбора под
какой-либо скорер). Генерация идёт на тестовом контейнере T-lite (порт 8002),
боевой контур не затрагивается.

Выход TSV: rec_id, par, verse_id, kind, text, shared (сколько слов длиннее 3 букв
общих у предложения и стиха -- проверка, что лексического мостика нет).
"""
from __future__ import annotations

import re
import sys

from llm_contamination import ask, load_text_map
from oracle_diagnostics import load_gold, verse_matches

SYN_PROMPT = """Ты пишешь прозу в духе Достоевского («Преступление и наказание»).

Библейский стих (Синодальный перевод): «{verse}»

Напиши ОДНО предложение (до 35 слов), которое отсылает к смыслу этого стиха, как делал бы Достоевский: через образ или ситуацию, без цитирования. Не используй слов и оборотов из стиха. Верни только предложение, без кавычек и пояснений."""

NEU_PROMPT = """Ты пишешь прозу в духе Достоевского («Преступление и наказание»).

Напиши ОДНО предложение (до 35 слов) о бытовой сцене в Петербурге: хозяйка, лавка, дорога, погода, долги. Без религиозных мотивов и библейских отсылок. Верни только предложение, без кавычек и пояснений."""


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[а-яёa-z]+", text.lower()) if len(w) > 3}


def main(gold_tsv: str, bible_tsv: str, out_tsv: str, url: str, model: str) -> None:
    bible = load_text_map(bible_tsv)
    ids = list(bible)
    gold = load_gold(gold_tsv)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("rec_id\tpar\tverse_id\tkind\ttext\tshared\n")
        for k, (par, group, _) in enumerate(gold):
            verse = next((s for s in ids if verse_matches(s, group)), None)
            if verse is None:
                continue
            for kind, prompt in (("syn", SYN_PROMPT.format(verse=bible[verse])), ("neu", NEU_PROMPT)):
                text = ask(url, model, prompt, temperature=0.8, max_tokens=120)
                text = text.strip().strip("«»\"").replace("\t", " ").replace("\n", " ")
                if text.startswith("ОШИБКА"):
                    continue
                shared = len(words(text) & words(bible[verse]))
                f.write(f"{k}\t{par}\t{verse}\t{kind}\t{text}\t{shared}\n")
    print(f"-> {out_tsv}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
