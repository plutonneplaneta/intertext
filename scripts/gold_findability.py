"""Оценка реального потолка: сколько мест эталона в принципе можно найти
сопоставлением текста.

Все цифры полноты в этом проекте делились на 64 -- как если бы каждое
документированное место было текстовым заимствованием. Чтение недостижимых
мест показало, что это не так: Тихомиров комментирует и то, что текстом не
выражено. Примеры из числа недостижимых:

  «Содом-с, безобразнейший...» -> Быт 19:12-16. «Содом» употреблён как имя
      нарицательное («шум, беспорядок»); с текстом главы о Лоте не совпадает
      ни одно слово. Комментарий объясняет этимологию, а не цитату.
  «За одну жизнь» -> Ин 11:47. Указан богословский архетип «арифметической
      теории» (Каиафа: лучше одному умереть за народ). Общих слов нет.
  «Не там смотрите... в четвёртом Евангелии» -> Ин 6. Это УКАЗАНИЕ на
      Евангелие, а не цитата из него.
  «Это был Новый Завет в русском переводе» -> Ин 11:25. Комментарий -- о
      личном экземпляре Достоевского, сведение биографическое.

Такие места не найдёт никакой метод, сопоставляющий текст, и держать их в
знаменателе -- значит занижать любую оценку и гнаться за недостижимым.

Объективная мера здесь -- сколько СОДЕРЖАТЕЛЬНЫХ ЛЕММ общих у абзаца романа и
у стихов, на которые указывает эталон, и каков их суммарный вес по редкости.
Ноль общих редких лемм означает, что лексического мостика нет вовсе; тогда
найти место можно только по смыслу, а на практике -- никак.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict

FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "AUX", "PRON", "PUNCT",
                "NUM", "ADV"}
RARE_DF_RATIO = 0.02


def load_seq(path: str):
    """Леммы в порядке текста (со служебными): нужны для поиска совпадений
    подряд -- дословная цитата обычно включает предлоги и союзы."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = r["lemmas"]
    return out


def load_ann(path: str):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = [l for l, p in zip(r["lemmas"], r.get("pos", []))
                            if p not in FUNCTION_POS]
    return out


def main(bible_ann: str, novel_ann: str, gold_tsv: str, fusion_tsv: str) -> None:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from oracle_diagnostics import load_gold, verse_matches

    s_lem = load_ann(bible_ann)
    t_lem = load_ann(novel_ann)
    n_src = len(s_lem)
    df = Counter()
    for lem in s_lem.values():
        df.update(set(lem))
    idf = {l: math.log(n_src / c) for l, c in df.items()}
    rare = {l for l, c in df.items() if c / n_src <= RARE_DF_RATIO}

    gold = load_gold(gold_tsv)
    nat = {}
    with open(fusion_tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            nat[int(r["rec_id"])] = int(r["in_natural_pool"])

    # Мера «общих редких лемм» сама по себе слишком узка: дословная цитата из
    # ЧАСТЫХ слов («Се человек», «Ищите и обрящете», «Она Бога узрит») даёт по
    # ней ноль, хотя мостик есть и притом самый прочный -- совпадение подряд.
    # Поэтому считаются три типа мостика.
    t_seq = load_seq(novel_ann)
    s_seq = load_seq(bible_ann)

    def longest_run(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        best = 0
        for x in a:
            cur = [0] * (len(b) + 1)
            for j, y in enumerate(b, 1):
                if x == y:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
            prev = cur
        return best

    rows = []
    for k, (par, group, frag) in enumerate(gold):
        pl = set(t_lem.get(par, []))
        pseq = t_seq.get(par, [])
        best = {"shared_rare": 0, "idf": 0.0, "shared_any": 0, "run": 0, "verse": ""}
        for vid, vl in s_lem.items():
            if not verse_matches(vid, group):
                continue
            shared = pl & set(vl)
            sr = shared & rare
            w = sum(idf.get(l, 0.0) for l in sr)
            run = longest_run(pseq, s_seq.get(vid, []))
            key = (len(sr), run, len(shared))
            cur = (best["shared_rare"], best["run"], best["shared_any"])
            if key > cur:
                best = {"shared_rare": len(sr), "idf": w, "shared_any": len(shared),
                        "run": run, "verse": vid}
        rows.append({"rec": k, "par": par, "shared_rare": best["shared_rare"],
                     "shared_any": best["shared_any"], "run": best["run"],
                     "idf_weight": round(best["idf"], 1), "verse": best["verse"],
                     "reachable": nat.get(k, 0), "fragment": frag[:45]})

    n = len(rows)
    print(f"мест эталона: {n}\n")
    print(f"{'общих редких лемм':>18s} {'мест':>6s} {'из них достижимо':>17s}")
    buckets = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["shared_rare"] <= hi]
        if not sel:
            continue
        lbl = f"{lo}" if lo == hi else (f"{lo}+" if hi == 99 else f"{lo}-{hi}")
        print(f"{lbl:>18s} {len(sel):6d} {sum(r['reachable'] for r in sel):17d}")

    print(f"\n{'тип мостика':44s} {'мест':>6s} {'достижимо':>10s}")
    def show(label, pred):
        sel = [r for r in rows if pred(r)]
        print(f"{label:44s} {len(sel):6d} {sum(r['reachable'] for r in sel):10d}")
        return sel
    show("есть общая редкая лемма", lambda r: r["shared_rare"] >= 1)
    show("совпадение >= 2 лемм подряд", lambda r: r["run"] >= 2)
    show("совпадение >= 3 лемм подряд", lambda r: r["run"] >= 3)
    bridged = show("хоть что-то: редкая лемма ИЛИ >= 2 подряд",
                   lambda r: r["shared_rare"] >= 1 or r["run"] >= 2)
    only_run = show("  только подряд, редких лемм нет",
                    lambda r: r["shared_rare"] == 0 and r["run"] >= 2)
    none = show("НИ РЕДКОЙ ЛЕММЫ, НИ ДВУХ ПОДРЯД",
                lambda r: r["shared_rare"] == 0 and r["run"] < 2)

    print("\nМест, где дословная цитата есть, но составлена из ЧАСТЫХ слов")
    print("(их теряет требование редкой леммы в find_candidates.py):\n")
    for r in sorted(only_run, key=lambda x: -x["run"]):
        print(f"  [{r['rec']:2d}] {r['par']:16s} подряд={r['run']} "
              f"дост={r['reachable']}  «{r['fragment']}»")

    print("\nМест без всякого лексического мостика -- сопоставлением текста")
    print("не находятся ни при каком пороге и ни при каком скорере:\n")
    for r in sorted(none, key=lambda x: x["par"]):
        print(f"  [{r['rec']:2d}] {r['par']:16s} общих лемм={r['shared_any']} "
              f"дост={r['reachable']}  «{r['fragment']}»")

    unreached = [r for r in bridged if not r["reachable"]]
    print("\nМеста С мостиком, но НЕ достигнутые пулом (цель дальнейшей работы):\n")
    for r in sorted(unreached, key=lambda x: x["par"]):
        print(f"  [{r['rec']:2d}] {r['par']:16s} редких={r['shared_rare']} "
              f"подряд={r['run']} стих={r['verse']}  «{r['fragment']}»")
    if len(sys.argv) > 5:
        with open(sys.argv[5], "w", encoding="utf-8") as f:
            f.write("rec_id\tpar\tverse\tshared_rare\trun\tfragment\n")
            for r in sorted(unreached, key=lambda x: x["rec"]):
                f.write(f"{r['rec']}\t{r['par']}\t{r['verse']}\t{r['shared_rare']}\t"
                        f"{r['run']}\t{r['fragment'].replace(chr(9), ' ')}\n")

    reachable = sum(r["reachable"] for r in rows)
    nb, nn = len(bridged), len(none)
    rb = sum(r["reachable"] for r in bridged)
    rn = sum(r["reachable"] for r in none)
    print(f"\nИТОГ. Достижимо сейчас: {reachable}/{n} ({reachable/n:.0%}).")
    print(f"  из {nb} мест с лексическим мостиком достаётся {rb} ({rb/nb:.0%})")
    print(f"  из {nn} мест без мостика достаётся {rn} "
          f"({rn/nn:.0%} -- плотные скореры кое-что берут по смыслу)")
    print(f"Осталось взять: {nb - rb} мест с мостиком -- это настоящая цель;")
    print(f"остальные {nn - rn} без мостика для текстового метода недостижимы.")


if __name__ == "__main__":
    main(*sys.argv[1:5])
