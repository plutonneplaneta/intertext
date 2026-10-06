#!/usr/bin/env bash
# Скачивание сырых материалов для проверки протокола на других корпусах (Часть II PROTOCOL.md)
# и сборка всех входных таблиц. Сырые HTML/тексты -- в $WORK (в git не кладутся).
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK="${WORK:-/tmp/intertext-ext}"
mkdir -p "$WORK/milton" "$WORK/cl" "$WORK/out" data/ext/source data/ext/gold

# Milton Reading Room (Dartmouth): текст и примечания, 12 книг. Страницы публичные, robots.txt не запрещает.
for b in $(seq 1 12); do
  curl -sfL -m 60 "https://milton.host.dartmouth.edu/reading_room/pl/book_$b/text.shtml"      -o "$WORK/milton/b$b.html"
  curl -sfL -m 60 "https://milton.host.dartmouth.edu/reading_room/pl/book_$b/annotation.js"   -o "$WORK/milton/a$b.js"
done
# KJV (bible-corpus, bibledatabase.com) и классики (Gutenberg)
curl -sfL -m 120 https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/English.xml -o "$WORK/English.xml"
for p in aen:228 ili:6130 ody:3160 met:21765 met2:26073; do
  curl -sfL -m 120 "https://www.gutenberg.org/cache/epub/${p##*:}/pg${p##*:}.txt" -o "$WORK/cl/${p%%:*}.txt"
done

python3 scripts/extract_bible.py "$WORK/English.xml" data/ext/source/kjv_verses.tsv
python3 scripts/ext/extract_milton.py "$WORK/milton" "$WORK/out"
python3 scripts/ext/extract_classics.py "$WORK/cl" "$WORK/out/classics_books.tsv"
python3 scripts/ext/build_gold_milton.py "$WORK/out/pl_lines.tsv" "$WORK/out/pl_notes.tsv" \
        data/ext/source/kjv_verses.tsv data/ext/gold/gold_milton_bible_v0.tsv data/ext/gold/gold_milton_classics_v0.tsv

# Материал C (Баньян — KJV): Gutenberg #131, разбор по prereg_material_C.md, наивный эталон v0, поправки шага 2
mkdir -p "$WORK/pp"
curl -sfL -m 120 https://www.gutenberg.org/cache/epub/131/pg131.txt -o "$WORK/pp/131.txt"
python3 scripts/ext/extract_bunyan.py "$WORK/pp/131.txt" data/ext/source/kjv_verses.tsv data/ext/target/pp_lines.tsv data/ext/gold/gold_bunyan_v0.tsv
python3 scripts/ext/ext_bunyan_fix.py "$WORK/pp/131.txt" data/ext/source/kjv_verses.tsv data/ext/gold/gold_bunyan_v0.tsv data/ext/gold/corrections_C.tsv data/ext/gold/additions_C.tsv
# источник B2 (очищенный, prereg_followup.md П1а)
python3 scripts/ext/extract_classics.py "$WORK/cl" "$WORK/out/classics_books_clean.tsv" clean

# Материал D (Онегин — поэты по Лотману): Лотман (pushkin-lit.ru, нужен браузерный User-Agent), «Онегин» (Викитека ПСС 1977), PoetryCorpus (GitHub)
mkdir -p "$WORK/lot" "$WORK/ru/on"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
for c in 1 2 3 4 5 6 7 8; do for p in 1 2 3 4; do
  curl -sfL -A "$UA" -m 60 "https://pushkin-lit.ru/pushkin/articles/lotman/onegin-kommentarij/onegin-comments-$c-$p.htm" -o "$WORK/lot/c$c-$p.html"; sleep 1
  [ "$(wc -c < "$WORK/lot/c$c-$p.html")" -gt 10000 ] || { echo "c$c-$p.html короткий -- повторить" >&2; exit 1; }
done; done
for c in 1 2 3 4 5 6 7 8; do
  python3 - "$c" "$WORK/ru/on" <<'PYEOF'
import sys, json, urllib.parse, subprocess
c, d = sys.argv[1], sys.argv[2]
t = 'Евгений Онегин (Пушкин)/ПСС 1977 (СО)/Глава ' + c
u = 'https://ru.wikisource.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main&format=json&titles=' + urllib.parse.quote(t)
b = subprocess.run(['curl', '-sfL', '-m', '60', '-A', 'Mozilla/5.0 research-bot', u], capture_output=True).stdout
open(f'{d}/ch{c}.wiki', 'w', encoding='utf8').write(list(json.loads(b)['query']['pages'].values())[0]['revisions'][0]['slots']['main']['*'])
PYEOF
done
curl -sfL -m 120 -o "$WORK/ru/pc.zip" https://github.com/IlyaGusev/PoetryCorpus/archive/refs/heads/master.zip && unzip -q -o "$WORK/ru/pc.zip" -d "$WORK/ru"
EXT_LANG=ru python3 scripts/ext/extract_onegin_d.py "$WORK/lot" "$WORK/ru/on" "$WORK/ru/PoetryCorpus-master/datasets/corpus/all.xml" data/ext/target/on_lines.tsv data/ext/source/ru_units.tsv data/ext/gold/gold_onegin_v0.tsv
