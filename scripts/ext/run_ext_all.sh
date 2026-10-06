#!/usr/bin/env bash
# Шаги 1-7 протокола на материалах A (Мильтон — KJV) и B (Мильтон — классики) в заявленном порядке.
# Вход: после fetch_ext.sh. Все отчёты -- reports/ext/. Python с numpy и scipy (venv: uv venv; uv pip install numpy scipy).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
for m in A B B2 C; do $PY scripts/ext/run_scorers.py $m; done
for m in A B C; do $PY scripts/ext/ext_step1_tasks.py $m > /dev/null; done
for m in A B C; do
  $PY scripts/ext/ext_step2_audit.py $m > /dev/null
done
python3 scripts/ext/ext_apply_corrections.py data/ext/gold/gold_milton_bible_v0.tsv data/ext/gold/corrections_A.tsv data/ext/gold/gold_milton_bible_v1.tsv
python3 scripts/ext/ext_apply_corrections.py data/ext/gold/gold_milton_classics_v0.tsv data/ext/gold/corrections_B.tsv data/ext/gold/gold_milton_classics_v1.tsv
python3 scripts/ext/ext_apply_corrections.py data/ext/gold/gold_bunyan_v0.tsv data/ext/gold/corrections_C.tsv data/ext/gold/gold_bunyan_v1.tsv data/ext/gold/additions_C.tsv
for m in A B C; do $PY scripts/ext/ext_step2_report.py $m > /dev/null; done
for m in A B B2 C; do $PY scripts/ext/ext_step3_ceiling.py $m v1 > /dev/null; $PY scripts/ext/ext_step3_ceiling.py $m v0 > /dev/null; $PY scripts/ext/ext_step3_v03.py $m v1 > /dev/null; $PY scripts/ext/ext_step3_v03.py $m v0 > /dev/null; done
for m in A B C; do $PY scripts/ext/ext_step4_gold_design.py $m v1 > /dev/null; done
for m in A B C; do $PY scripts/ext/ext_step5_stats.py $m > /dev/null; done
for m in A B B2 C; do $PY scripts/ext/ext_step6_controls.py $m > /dev/null; done
$PY scripts/ext/synth_defects.py > /dev/null
$PY scripts/ext/ext_step7_repro.py > /dev/null
$PY scripts/ext/ext_order.py A B C
$PY scripts/ext/ext_assemble.py

# Материал D (русский): все шаги с EXT_LANG=ru; LLM-судья запускается отдельно на сервере (llm_judge_build.py, llm_judge_run.py, llm_judge_analyze.py)
export EXT_LANG=ru
$PY scripts/ext/run_scorers.py D
$PY scripts/ext/ext_step1_tasks.py D > /dev/null
$PY scripts/ext/ext_step2_audit.py D > /dev/null
python3 scripts/ext/ext_apply_corrections.py data/ext/gold/gold_onegin_v0.tsv data/ext/gold/corrections_D.tsv data/ext/gold/gold_onegin_v1.tsv
$PY scripts/ext/ext_step2_report.py D > /dev/null
$PY scripts/ext/ext_step3_v03.py D v1 > /dev/null; $PY scripts/ext/ext_step3_v03.py D v0 > /dev/null
$PY scripts/ext/ext_step4_gold_design.py D > /dev/null; $PY scripts/ext/ext_step5_stats.py D > /dev/null; $PY scripts/ext/ext_step6_controls.py D > /dev/null
$PY scripts/ext/ext_order.py D > /dev/null; $PY scripts/ext/ext_step7_repro_d.py > /dev/null
unset EXT_LANG
$PY scripts/ext/ext_assemble.py

# Часть 4 (закрытие ограничений): плотные скореры считаются на сервере (dense_export.py -> dense_scores.py), оценка локально; судья на C: llm_judge_build_c.py
# unset EXT_LANG; $PY scripts/ext/ext_dense_eval.py A <dir>; ... ; EXT_LANG=ru $PY scripts/ext/ext_dense_eval.py D <dir>; $PY scripts/ext/synth_defects2.py
