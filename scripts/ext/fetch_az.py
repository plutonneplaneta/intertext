# -*- coding: utf-8 -*-
"""Скачивание собраний сочинений авторов (общественное достояние) с az.lib.ru (robots.txt допускает), возобновляемо.
Использование: fetch_az.py <каталог> автор1=путь1 автор2=путь2 ...  Например: Chekhov=c/chehow_a_p
Каждый файл собрания (text_NNNN.shtml) -> <каталог>/<автор>/<файл>.txt (UTF-8, теги убраны). Пауза 1 с между запросами."""
import sys, os, re, time, subprocess, html, http.client, threading, queue
D = sys.argv[1]
def get(url):
    """az.lib.ru (Apache 1.3) не закрывает соединение и не присылает Content-Length: читаем, пока идут данные, и останавливаемся
    после 4 с тишины; ответ считается полным, если получено >500 байт и код 200."""
    host, path = re.match(r'http://([^/]+)(/.*)', url).groups()
    for _ in range(3):
        try:
            c = http.client.HTTPConnection(host, timeout=20); c.request('GET', path, headers={'User-Agent': 'Mozilla/5.0 research-bot (intertext study)', 'Connection': 'close'})
            r = c.getresponse(); ok = r.status == 200; c.sock.settimeout(4); buf = b''
            while True:
                try: ch = r.read(65536)
                except Exception: break
                if not ch: break
                buf += ch
            c.close()
            if ok and len(buf) > 500: return buf
        except Exception: pass
        time.sleep(3)
    return b''
def dec(b):
    for enc in ('utf-8', 'cp1251', 'koi8-r'):
        try:
            t = b.decode(enc)
            if re.search('[А-Яа-я]{4}', t): return t
        except Exception: pass
    return b.decode('cp1251', 'replace')
for spec in sys.argv[2:]:
    name, path = spec.split('=')
    os.makedirs(f'{D}/{name}', exist_ok=True)
    idx = dec(get(f'http://az.lib.ru/{path}/'))
    files = []
    for m in re.finditer(r'(?i)href="?(text_\d+[^"> ]*\.shtml)"?[^>]*>(.*?)</a>', idx, re.S):
        if m.group(1) not in [f for f, _ in files]: files.append((m.group(1), re.sub(r'<[^>]+>', '', m.group(2)).strip()))
    print(name, len(files), 'файлов', flush=True)
    with open(f'{D}/{name}/_index.tsv', 'w', encoding='utf8') as f:
        for fn, ti in files: f.write(f'{fn}\t{ti}\n')
    todo = [(fn, ti) for fn, ti in files if not (os.path.exists(f'{D}/{name}/{fn}.txt') and os.path.getsize(f'{D}/{name}/{fn}.txt') > 200)]
    q = queue.Queue()
    for x in todo: q.put(x)
    def work():
        while True:
            try: fn, ti = q.get_nowait()
            except queue.Empty: return
            raw = get(f'http://az.lib.ru/{path}/{fn}')
            if not raw: print('не получен', name, fn, flush=True); continue
            t = dec(raw); t = re.sub(r'(?is)<script.*?</script>|<style.*?</style>', ' ', t)
            t = re.sub(r'(?i)<br\s*/?>|</p>|</dd>|</h\d>|</li>', '\n', t); t = html.unescape(re.sub(r'<[^>]+>', ' ', t))
            open(f'{D}/{name}/{fn}.txt', 'w', encoding='utf8').write(t); time.sleep(0.7)
    ths = [threading.Thread(target=work) for _ in range(3)]
    [t.start() for t in ths]; [t.join() for t in ths]
print('DONE', flush=True)
