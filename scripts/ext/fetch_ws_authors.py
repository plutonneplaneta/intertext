# -*- coding: utf-8 -*-
"""Тексты авторов (общественное достояние) с ru.wikisource через API: обход категории автора (глубина <=3), страницы (ns 0) с «(Фамилия)» в названии,
вики-текст пакетами по 12 страниц, очистка шаблонов. Возобновляемо: <каталог>/<автор>.jsonl (title, text).
Использование: fetch_ws_authors.py <каталог> "Автор=Категория:Название;Фамилия" ..."""
import sys, os, re, json, time, subprocess, urllib.parse
D = sys.argv[1]; os.makedirs(D, exist_ok=True)
SKIP = re.compile(r'Литература о|Критика|Статьи об|Письма|Импорт|Вики|Персоналии|Изображен|Аудио|Переводы|Пародии|Стихи для детей|Биограф|Воспоминан|Рецензии|Страницы|Файл')
def api(**kw):
    """Викимедиа ограничивает частоту (429): не чаще одного запроса в 1,2 с, при 429 -- пауза и повтор (требования к ботам)."""
    u = 'https://ru.wikisource.org/w/api.php?format=json&' + urllib.parse.urlencode(kw)
    for k in range(8):
        time.sleep(1.2)
        r = subprocess.run(['curl', '-sL', '-m', '90', '-A', 'intertext-study/1.0 (research; contact via repo owner)', '-w', '\n%{http_code}', u], capture_output=True)
        body, _, code = r.stdout.rpartition(b'\n')
        if code == b'200':
            try: return json.loads(body)
            except Exception: pass
        time.sleep(10 * (k + 1) if code == b'429' else 5)
    return {}
def members(cat, kind):
    out = []; cont = {}
    while True:
        d = api(action='query', list='categorymembers', cmtitle=cat, cmtype=kind, cmlimit=500, **cont)
        out += [x['title'] for x in d.get('query', {}).get('categorymembers', [])]
        if 'continue' in d: cont = d['continue']
        else: return out
def clean(w):
    w = re.sub(r'<ref[^>]*>.*?</ref>', '', w, flags=re.S); w = re.sub(r'<!--.*?-->', '', w, flags=re.S)
    for _ in range(4): w = re.sub(r'\{\{[^{}]*\}\}', '', w)
    w = re.sub(r'\[\[(?:Категория|Category|Файл|File)[^\]]*\]\]', '', w); w = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', w)
    w = re.sub(r"<[^>]+>|'''?|__[A-Z]+__|^=+.*?=+\s*$", '', w, flags=re.M)
    return re.sub(r'\n{3,}', '\n\n', w).strip()
for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1); cat, sur = rest.split(';')
    out = f'{D}/{name}.jsonl'
    if os.path.exists(out): print(name, 'уже есть'); continue
    seen = set(); queue = [(cat, 0)]; pages = []
    while queue:
        c, dep = queue.pop(0)
        if c in seen: continue
        seen.add(c)
        for t in members(c, 'page'):
            if f'({sur})' in t or f'({sur}' in t: pages.append(t)
        if dep < 3:
            for sc in members(c, 'subcat'):
                if not SKIP.search(sc): queue.append((sc, dep + 1))
    pages = sorted(set(pages)); print(name, 'категорий', len(seen), 'страниц', len(pages), flush=True)
    with open(out, 'w', encoding='utf8') as f:
        for i in range(0, len(pages), 12):
            batch = pages[i:i + 12]
            d = api(action='query', prop='revisions', rvprop='content', rvslots='main', titles='|'.join(batch), redirects=1)
            for p in d.get('query', {}).get('pages', {}).values():
                if p.get('revisions'):
                    w = clean(p['revisions'][0]['slots']['main']['*'])
                    if len(w) > 400: f.write(json.dumps({'title': p['title'], 'text': w}, ensure_ascii=False) + '\n')
print('DONE', flush=True)
