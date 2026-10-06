# -*- coding: utf-8 -*-
"""«Двенадцать стульев» (Викитека, 41 глава) -> novel_ds.jsonl (chapter, text). Вежливо: пауза 3 с, при 429 -- ожидание."""
import sys, json, time, subprocess, urllib.parse, re
sys.argv += [] ; OUT = sys.argv[1]
def roman(n):
    r = ''
    for v, s in ((40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')):
        while n >= v: r += s; n -= v
    return r
def api(**kw):
    u = 'https://ru.wikisource.org/w/api.php?format=json&' + urllib.parse.urlencode(kw)
    for k in range(8):
        time.sleep(3)
        r = subprocess.run(['curl', '-sL', '-m', '90', '-A', 'intertext-study/1.0 (research)', '-w', '\n%{http_code}', u], capture_output=True)
        body, _, code = r.stdout.rpartition(b'\n')
        if code == b'200':
            try: return json.loads(body)
            except Exception: pass
        time.sleep(10 * (k + 1))
    return {}
def clean(w):
    w = re.sub(r'<ref[^>]*>.*?</ref>', '', w, flags=re.S); w = re.sub(r'<!--.*?-->', '', w, flags=re.S)
    for _ in range(4): w = re.sub(r'\{\{[^{}]*\}\}', '', w)
    w = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', w); w = re.sub(r"<[^>]+>|'''?", '', w)
    return re.sub(r'\n{3,}', '\n\n', w).strip()
with open(OUT, 'w', encoding='utf8') as f:
    for n in range(1, 42):
        t = f'Двенадцать стульев (Ильф и Петров)/Глава {roman(n)}'
        d = api(action='query', prop='revisions', rvprop='content', rvslots='main', titles=t, redirects=1)
        p = list(d.get('query', {}).get('pages', {}).values())
        if not p or 'revisions' not in p[0]: print('нет главы', n, flush=True); continue
        f.write(json.dumps({'chapter': n, 'text': clean(p[0]['revisions'][0]['slots']['main']['*'])}, ensure_ascii=False) + '\n')
print('готово')
