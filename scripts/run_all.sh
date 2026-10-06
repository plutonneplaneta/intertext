#!/usr/bin/env bash
# Полный прогон второй итерации с нуля: от текстов до таблиц в reports/.
# На четырёх ядрах CPU занимает около 10 минут, из них разбор spaCy -- 1,5.
#
# Промежуточные файлы (аннотации, списки кандидатов) -- сотни МБ, в git не
# хранятся, пишутся в $WORK. Задайте WORK, если не хотите /tmp.
set -euo pipefail

cd "$(dirname "$0")/.."
WORK="${WORK:-/tmp/intertext-work}"
mkdir -p "$WORK" reports

CSL=data/csl_normalize.tsv
BIBLE=data/source/bible_verses.tsv
NOVEL=data/target/cp_fb2.tsv
GOLD=data/gold/gold_standard_v7.tsv   # v4 + Псалтирь + абзацы + точечные поправки
GOLD_V5=data/gold/gold_standard_v5.tsv
GOLD_V6=data/gold/gold_standard_v6.tsv
NP="${NP:-4}"

echo "== эталон: перевод ссылок из синодальной нумерации в нумерацию корпуса =="
# Корпус в масоретском счёте (как KJV), Тихомиров -- в синодальном; без этого
# шесть мест эталона из 64 указывают на другие псалмы. См. reports/psalm_numbering_bug.txt
python3 scripts/renumber_gold.py data/gold/gold_standard_v4.tsv "$BIBLE" "$GOLD_V5"

echo "== эталон: перепривязка к абзацам того текста, на котором работает конвейер =="
# Эталон строился по тексту с ilibrary.ru, где разбиение на абзацы крупнее;
# 14 записей из 65 указывали не на тот абзац. См. reports/gold_alignment_bug.txt
python3 scripts/realign_gold.py "$GOLD_V5" "$NOVEL" "$GOLD_V6"
python3 scripts/audit_gold_alignment.py "$GOLD_V6" "$NOVEL" "$WORK/gold_audit.tsv" \
        > reports/gold_alignment_bug.txt 2>&1

echo "== эталон: точечные поправки с обоснованием =="
python3 scripts/apply_gold_corrections.py "$GOLD_V6" \
        data/gold/manual_corrections.tsv "$GOLD"
python3 scripts/check_gold_refs.py "$GOLD" > reports/gold_refs_crosscheck.txt 2>&1

echo "== разбор spaCy (леммы, рёбра, границы предложений) =="
python3 scripts/nlp_annotate.py "$BIBLE" "$WORK/bible_ann.jsonl" verse_id "$CSL" "$NP"
python3 scripts/nlp_annotate.py "$NOVEL" "$WORK/novel_ann.jsonl" verse_id "$CSL" "$NP"
python3 scripts/split_sentences.py "$WORK/novel_ann.jsonl" "$WORK/novel_sents.tsv"
python3 scripts/nlp_annotate.py "$WORK/novel_sents.tsv" "$WORK/novel_sent_ann.jsonl" \
        verse_id "$CSL" "$NP"

echo "== кандидаты: базовые три слоя по абзацам =="
python3 scripts/find_candidates.py     "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" "$WORK/cand_lex.tsv"
python3 scripts/syntax_candidates.py   "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" "$WORK/cand_syn.tsv"
python3 scripts/semantic_static.py     "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" "$WORK/cand_sem.tsv" 0

echo "== кандидаты: те же слои по предложениям, свод к абзацу =="
python3 scripts/find_candidates.py     "$WORK/bible_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "$WORK/cand_lex_sent.tsv"
python3 scripts/syntax_candidates.py   "$WORK/bible_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "$WORK/cand_syn_sent.tsv"
python3 scripts/semantic_static.py     "$WORK/bible_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "$WORK/cand_sem_sent.tsv" 0
python3 scripts/aggregate_sentences.py "$WORK/cand_lex_sent.tsv" "$WORK/cand_lex_sentmax.tsv"
python3 scripts/aggregate_sentences.py "$WORK/cand_syn_sent.tsv" "$WORK/cand_syn_sentmax.tsv"

echo "== кандидаты: расширенные наборы признаков =="
python3 scripts/lexical_candidates_v2.py "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" "$WORK/cand_lex_v2.tsv"
python3 scripts/syntax_candidates_v2.py  "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" "$WORK/cand_syn_v2.tsv"

echo "== reports/length_normalization: сетка нормировок =="
python3 scripts/normalize_experiments.py "$NOVEL" "$GOLD" reports/length_normalization.tsv \
        "лексика=$WORK/cand_lex.tsv" "синтаксис=$WORK/cand_syn.tsv" "семантика=$WORK/cand_sem.tsv" \
        > reports/length_normalization.txt

echo "== reports/length_matched_auc: AUC при сопоставимой длине =="
python3 scripts/length_matched_auc.py "$NOVEL" "$GOLD" \
        "лексика (абзац)=$WORK/cand_lex.tsv" "лексика (предложения)=$WORK/cand_lex_sentmax.tsv" \
        "синтаксис (абзац)=$WORK/cand_syn.tsv" "синтаксис (предложения)=$WORK/cand_syn_sentmax.tsv" \
        "семантика (абзац)=$WORK/cand_sem.tsv" > reports/length_matched_auc.txt

echo "== reports/detection_vs_length: прирост слоя над «только длина» =="
(cd scripts && python3 evaluate_layers.py "../$NOVEL" "../$GOLD" \
        "син:sum_idf,cos_idf,cover_src=$WORK/cand_syn_v2.tsv" \
        "лекс:best_idf,best_gap_idf,cover_src=$WORK/cand_lex_v2.tsv" \
        "сем:score=$WORK/cand_sem.tsv") > reports/detection_vs_length.txt 2>&1

echo "== reports/attribution_reranking: поправка на хабность =="
{
  for pair in "синтаксис (абзац):cand_syn" "синтаксис (предложения):cand_syn_sentmax" \
              "лексика (абзац):cand_lex" "лексика (предложения):cand_lex_sentmax" \
              "семантика (абзац):cand_sem"; do
    echo "########## ${pair%%:*} ##########"
    (cd scripts && python3 rerank_attribution.py "$WORK/${pair##*:}.tsv" "../$GOLD") 2>/dev/null
    echo
  done
} > reports/attribution_reranking.txt

echo "== reports/combined_attribution: наборы признаков, LOO =="
(cd scripts && python3 build_combined_table_v2.py "../$GOLD" "$WORK/table_v2.tsv" \
  "лекс:best_idf,best_gap_idf,n_matches,sum_top3,cover_src:хаб=$WORK/cand_lex_v2.tsv" \
  "син:sum_idf,cos_idf,cover_src,n_shared,max_edge:хаб=$WORK/cand_syn_v2.tsv" \
  "синс:score:хаб=$WORK/cand_syn_sent.tsv" \
  "сем:score:хаб=$WORK/cand_sem.tsv" \
  "семс:score:хаб=$WORK/cand_sem_sent.tsv")
{
  echo "ЧАСТЬ 1. Таблица пилота из репозитория (эмбеддинги E5), три сырых признака."
  echo "Медиана -- настоящая; combine_and_evaluate.py печатал sorted[n//2] (158 вместо 90,5)."
  (cd scripts && python3 combine_and_evaluate_v2.py ../data/gold/combined_table.tsv \
     "только лексика=lex" "только эмбеддинги=emb" "только синтаксис=syn" \
     "комбинация трёх=lex,emb,syn")
  echo
  echo "ЧАСТЬ 2. Своя таблица: семантика на статических векторах, пул кандидатов шире."
  echo "Наборы признаков сравнимы между собой; с частью 1 эти числа НЕ сравнимы."
} > reports/combined_attribution.txt 2>&1
(cd scripts && python3 combine_and_evaluate_v2.py "$WORK/table_v2.tsv" \
  "база: три слоя, сырые баллы=лекс_best_idf,сем_score,син_sum_idf" \
  "+ синтаксис по предложениям=лекс_best_idf,сем_score,син_sum_idf,синс_score" \
  "+ семантика по предложениям=лекс_best_idf,сем_score,син_sum_idf,семс_score" \
  "+ оба предложенческих=лекс_best_idf,сем_score,син_sum_idf,синс_score,семс_score" \
  "+ поправка на хабность=лекс_best_idf,сем_score,син_sum_idf,син_minus_src,сем_minus_src,лекс_minus_src" \
  ) >> reports/combined_attribution.txt 2>&1

echo "== reports/oracle_ceiling: потолок полноты каждого скорера =="
{
  echo "Диагностика потолка: где стоит правильный стих, если снять все пороги и индексы."
  echo "Исчерпывающий счёт 31 102 стихов против каждого из 64 мест эталона."
  echo
  (cd scripts && python3 oracle_diagnostics.py "$WORK/bible_ann.jsonl" \
     "$WORK/novel_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "../$GOLD" \
     "$WORK/oracle.tsv") 2>/dev/null
} > reports/oracle_ceiling.txt

echo "== reports/fusion_attribution: объединение всех скореров, атрибуция =="
(cd scripts && python3 fusion_retrieval.py build "$WORK/bible_ann.jsonl" \
   "$WORK/novel_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "../$GOLD" \
   "$WORK/fusion_table.tsv" "" "" "$WORK/bible_ann.jsonl" "$WORK/novel_sent_ann.jsonl")
# Последние два аргумента -- аннотации с топологией (поле heads); без них скорер dep_pair2
# молча даёт нуль на всех парах, и «одиннадцать скореров» оказываются десятью живыми
# (ошибка нашлась при воспроизведении медианы 220: без них выходило 411, с ними 224).
# Файлы *_ann.jsonl уже содержат heads, отдельная аннотация не нужна.
python3 - "$WORK/fusion_attribution_args.txt" <<'PY'
import sys
sys.path.insert(0, "scripts")
from retrieval_scorers import SCORERS
cols = lambda sub: ",".join([f"{n}_raw" for n in sub] + [f"{n}_pct" for n in sub])
dense = ["bm25", "tfidf_cos", "char4", "emb", "ngram", "edges"]
args = [f"все скореры ({len(SCORERS)})={cols(SCORERS)}"]   # ПЕРВЫЙ набор -- база сравнений
args += [f"шесть плотных+точных, процентили={','.join(f'{n}_pct' for n in dense)}"]
args += [f"четыре плотных, процентили={','.join(f'{n}_pct' for n in ('bm25', 'tfidf_cos', 'char4', 'emb'))}"]
args += ["только tfidf_cos=tfidf_cos_pct"]
args += ["пилот: n-граммы + рёбра + векторы=ngram_raw,edges_raw,emb_raw"]
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(args) + "\n")
PY
{
  echo "Все наборы признаков -- на ОДНОМ пуле кандидатов (объединение топ-300 по"
  echo "каждому скореру), leave-one-target-out."
  echo
  mapfile -t FA < "$WORK/fusion_attribution_args.txt"
  (cd scripts && python3 fusion_retrieval.py eval "$WORK/fusion_table.tsv" "${FA[@]}")
} > reports/fusion_attribution.txt 2>&1

echo "== reports/fusion_detection: объединение скореров против длины =="
(cd scripts && python3 fusion_detection.py "$WORK/bible_ann.jsonl" \
   "$WORK/novel_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "../$GOLD" \
   "$WORK/detect_table.tsv")
(cd scripts && python3 eval_fusion_detection.py "$WORK/detect_table.tsv") \
   > reports/fusion_detection.txt 2>&1

echo "== reports/ceiling_scan: потолок полноты шестнадцати вариантов признака =="
{
  echo "Сканирование потолка полноты по шестнадцати вариантам признака."
  echo "Единица -- предложение, максимум на абзац; 31 102 стиха на каждое из 64 мест."
  echo
  (cd scripts && python3 ceiling_scan.py "$WORK/bible_ann.jsonl" \
     "$WORK/novel_sent_ann.jsonl" "../$GOLD" "$WORK/ceiling.tsv") 2>/dev/null
} > reports/ceiling_scan.txt

echo "== reports/topk_fix: что даёт поднятие TOP_K слоя семантики =="
python3 scripts/semantic_static.py "$WORK/bible_ann.jsonl" "$WORK/novel_ann.jsonl" \
        "$WORK/cand_sem_k1000.tsv" 0
{
  echo "Полнота против эталона, тот же eval_recall.py, что у пилота."
  echo
  for f in cand_lex cand_syn cand_sem cand_sem_k1000; do
    echo "--- $f ---"
    python3 scripts/eval_recall.py "$GOLD" "$WORK/$f.tsv" 2>/dev/null | head -3
  done
} > reports/topk_fix.txt

echo "== графовые признаки: нужна аннотация с топологией разбора =="
python3 scripts/nlp_annotate.py "$BIBLE" "$WORK/bible_g.jsonl" verse_id "$CSL" "$NP"
python3 scripts/nlp_annotate.py "$WORK/novel_sents.tsv" "$WORK/novel_sent_g.jsonl" \
        verse_id "$CSL" "$NP"
{
  echo "Графовые признаки: потолок полноты, в одной таблице с неграфовыми."
  echo
  (cd scripts && python3 ceiling_scan.py "$WORK/bible_ann.jsonl" \
     "$WORK/novel_sent_ann.jsonl" "../$GOLD" "$WORK/ceiling_graph.tsv" \
     "$WORK/bible_g.jsonl" "$WORK/novel_sent_g.jsonl") 2>/dev/null
} > reports/graph_features.txt

echo "== reports/graph_rerank: распространение балла по графу стихов =="
{
  echo "Граф стихов: соседство в главе + рёбра параллельных мест из корпуса."
  echo
  (cd scripts && python3 graph_rerank.py "$WORK/fusion_table.tsv" \
     "$WORK/bible_ann.jsonl" "../$GOLD" tfidf_rare_pct)
} > reports/graph_rerank.txt 2>&1

echo "== reports/passage_retrieval: пассаж против стиха как единица =="
{
  echo "Пассаж как единица извлечения; цена -- число прочитанных стихов."
  echo
  (cd scripts && python3 passage_retrieval.py "$WORK/bible_ann.jsonl" \
     "$WORK/novel_sent_ann.jsonl" "../$GOLD") 2>/dev/null
} > reports/passage_retrieval.txt

echo "== reports/collapsed_edges: чистый тест стягивания предлогов =="
{
  echo "Три варианта ключа ребра на одном и том же наборе рёбер."
  echo
  (cd scripts && python3 collapsed_edges_test.py "$WORK/bible_g.jsonl" \
     "$WORK/novel_sent_g.jsonl" "../$GOLD") 2>/dev/null
} > reports/collapsed_edges.txt

echo "== reports/xref_graph: внешний граф перекрёстных ссылок =="
# 340 тыс. связей, CC-BY; в репозиторий не кладём -- регенерируется одной командой
if [ ! -f "$WORK/xrefs.zip" ]; then
  curl -sSL -m 300 -o "$WORK/xrefs.zip" \
       https://a.openbible.info/data/cross-references.zip || true
fi
if [ -f "$WORK/xrefs.zip" ]; then
  python3 scripts/build_xref_graph.py "$WORK/xrefs.zip" "$BIBLE" "$WORK/xref_edges.tsv" 0
  {
    echo "Граф из openbible.info cross-references (CC-BY) против соседства стихов."
    echo
    (cd scripts && python3 graph_rerank.py "$WORK/fusion_table.tsv" \
       "$WORK/bible_ann.jsonl" "../$GOLD" tfidf_rare_pct "$WORK/xref_edges.tsv")
  } > reports/xref_graph.txt 2>&1
else
  echo "справочник ссылок не скачался (сетевая политика?) -- пункт пропущен"
fi

echo "== reports/e5_encoder: обученный кодировщик против статических векторов =="
python3 scripts/embed.py "$BIBLE" 'passage:' "$WORK/bible_e5.npz"
python3 scripts/embed.py "$WORK/novel_sents.tsv" 'query:' "$WORK/novel_sent_e5.npz"
python3 scripts/embed.py "$NOVEL" 'query:' "$WORK/novel_e5.npz"
{
  echo "multilingual-e5-base по предложениям: потолок и вклад в объединение."
  echo
  (cd scripts && python3 e5_ceiling.py "$WORK/bible_e5.npz" "$WORK/novel_e5.npz" \
     "$WORK/novel_sent_e5.npz" "../$GOLD") 2>/dev/null
} > reports/e5_encoder.txt

echo "== таблица признаков + e5: то же объединение, плюс обученный кодировщик =="
# Отдельная таблица от fusion_table.tsv (строка 125) -- та строится ДО того, как
# появляются e5-векторы (см. выше), эта -- после. Используется gold_findability,
# rank_models и абляцией ниже: везде, где ожидается «двенадцать» (одиннадцать
# SCORERS + e5), а не «одиннадцать» из ранней таблицы.
(cd scripts && python3 fusion_retrieval.py build "$WORK/bible_ann.jsonl" \
   "$WORK/novel_ann.jsonl" "$WORK/novel_sent_ann.jsonl" "../$GOLD" \
   "$WORK/fusion_table_e5.tsv" "$WORK/bible_e5.npz" "$WORK/novel_sent_e5.npz" \
   "$WORK/bible_ann.jsonl" "$WORK/novel_sent_ann.jsonl")

echo "== reports/gold_findability: реальный потолок эталона =="
{
  echo "Что из эталона в принципе можно найти сопоставлением текста."
  echo
  (cd scripts && python3 gold_findability.py "$WORK/bible_ann.jsonl" \
     "$WORK/novel_ann.jsonl" "../$GOLD" "$WORK/fusion_table_e5.tsv") 2>/dev/null
} > reports/gold_findability.txt

echo "== reports/rank_models: попарная регрессия и каскад против поточечной =="
(cd scripts && python3 rank_models.py "$WORK/fusion_table_e5.tsv") \
   > reports/rank_models.txt 2>&1

# Абляция «выбросить по одному скореру» считается 1 ч 20 мин, поэтому по
# умолчанию пропущена. ABLATION=1 ./scripts/run_all.sh -- включить.
if [ -n "${ABLATION:-}" ]; then
  echo "== reports/ablation: выброс по одному скореру из двенадцати =="
  python3 - "$WORK/ablation_args.txt" <<'PY'
import sys
sys.path.insert(0, "scripts")
from retrieval_scorers import SCORERS
names = list(SCORERS) + ["e5"]
cols = lambda sub: ",".join([f"{n}_raw" for n in sub] + [f"{n}_pct" for n in sub])
args = [f"все скореры={cols(names)}"]          # ПЕРВЫЙ набор -- база сравнений
args += [f"без {d}={cols([n for n in names if n != d])}" for d in names]
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(args) + "\n")
PY
  {
    echo "Выброс по одному скореру, одна таблица признаков, LOO по абзацам."
    echo
    mapfile -t A < "$WORK/ablation_args.txt"
    (cd scripts && python3 fusion_retrieval.py eval "$WORK/fusion_table_e5.tsv" "${A[@]}")
  } > reports/ablation.txt 2>&1
fi

echo "готово. Результаты -- в reports/, промежуточные файлы -- в $WORK"
