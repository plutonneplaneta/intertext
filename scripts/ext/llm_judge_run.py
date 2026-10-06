# -*- coding: utf-8 -*-
"""Запуск LLM-судьи на сервере (только stdlib). Читает items.jsonl, free_recall.jsonl; пишет results.jsonl, recall_results.jsonl (возобновляемо).
Использование: python3 llm_judge_run.py <каталог> [url]. Модель: T-lite в исследовательском контейнере (порт 8002)."""
import sys, os, json, time, urllib.request
D = sys.argv[1]; URL = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:8002/v1/chat/completions'
SYS = ('Ты — филолог-пушкинист. Оцени, перекликается ли текст А (строфа «Евгения Онегина») с текстом Б (отрывок стихотворения или другая строфа). '
       'Шкала: 0 — связи нет (общие слова и общая тема не считаются); 1 — слабое тематическое сходство; 2 — заметная перекличка: общий образ, оборот или мотив, '
       'который можно показать цитатами; 3 — прямая цитата, парафраз или явное заимствование. Если оценка 1 и выше, приведи ТОЧНЫЕ цитаты из обоих текстов, подтверждающие оценку. '
       'Ответ — строго JSON: {"score": 0-3, "quote_a": "...", "quote_b": "..."}')
def call(messages, temp, max_tokens=300):
    body = json.dumps({'model': 'x', 'messages': messages, 'temperature': temp, 'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r: return json.loads(r.read())['choices'][0]['message']['content']
        except Exception as e:
            err = str(e); time.sleep(3)
    return 'ERROR: ' + err
def run(infile, outfile, mk):
    done = set()
    if os.path.exists(outfile):
        for l in open(outfile, encoding='utf8'): done.add(json.loads(l)['id'])
    n = 0
    with open(outfile, 'a', encoding='utf8') as f:
        for l in open(infile, encoding='utf8'):
            it = json.loads(l)
            if it['id'] in done: continue
            msgs, temp = mk(it)
            raw = call(msgs, temp); n += 1
            f.write(json.dumps({'id': it['id'], 'raw': raw}, ensure_ascii=False) + '\n'); f.flush()
            if n % 25 == 0: print(outfile, n, time.strftime('%H:%M:%S'), flush=True)
def judge(it): return ([{'role': 'system', 'content': it.get('system', SYS)}, {'role': 'user', 'content': f'Текст А:\n{it["stanza"]}\n\nТекст Б:\n{it["unit_window"]}'}], it['temp'])
def recall(it): return ([{'role': 'user', 'content': it['prompt'] + it['stanza']}], 0.0) if 'prompt' in it else ([{'role': 'user', 'content': 'Ниже строфа из «Евгения Онегина» Пушкина. Назови поэта и произведение (или другую строфу «Онегина»), к которым, по-твоему, отсылает эта строфа. Ответ — строго JSON: {"author": "...", "work": "..."}\n\n' + it['stanza']}], 0.0)
if os.path.exists(f'{D}/free_recall.jsonl'): run(f'{D}/free_recall.jsonl', f'{D}/recall_results.jsonl', recall)
run(f'{D}/items.jsonl', f'{D}/results.jsonl', judge)
print('DONE', flush=True)
