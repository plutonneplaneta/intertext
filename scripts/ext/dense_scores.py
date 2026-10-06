# -*- coding: utf-8 -*-
"""Плотные скореры на сервере (prereg_closure.md, З1). Окна по 128 слов (шаг 128), оценка единицы = максимум косинуса по парам окон.
Использование: dense_scores.py <каталог> <mat> <модель: e5l|bgem3|sbertru>"""
import sys, os, time, numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from sentence_transformers import SentenceTransformer
D, mat, key = sys.argv[1:4]
NAMES = {'e5l': 'intfloat/multilingual-e5-large', 'bgem3': 'BAAI/bge-m3', 'sbertru': 'ai-forever/sbert_large_nlu_ru'}
dev = None
for i in range(torch.cuda.device_count()):
    if '3060' in torch.cuda.get_device_name(i): dev = f'cuda:{i}'
if dev is None: raise SystemExit('RTX 3060 не найдена по имени (индексы torch и nvidia-smi различаются)')
model = SentenceTransformer(NAMES[key], device=dev); model.max_seq_length = 384
if dev.startswith('cuda'): model.half()
W = 128
def chunks(items):
    out, own = [], []
    for k, t in enumerate(items):
        w = t.split()
        for s in range(0, max(len(w), 1), W): out.append(' '.join(w[s:s + W])); own.append(k)
    return out, np.array(own)
def rd(p): return [l.rstrip('\n').split('\t', 1) for l in open(p, encoding='utf8') if l.strip()]
T = rd(f'{D}/targets_{mat}.tsv'); U = rd(f'{D}/units_{mat}.tsv')
tc, town = chunks([t[1] for t in T]); uc, uown = chunks([u[1] for u in U])
pq, pp = ('query: ', 'passage: ') if key == 'e5l' else ('', '')
t0 = time.time()
te = model.encode([pq + c for c in tc], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
ue = model.encode([pp + c for c in uc], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
tt = torch.from_numpy(te).to(dev); ut_ = torch.from_numpy(ue).to(dev)
ustart = np.r_[0, np.flatnonzero(np.diff(uown)) + 1]; tstart = np.r_[0, np.flatnonzero(np.diff(town)) + 1]
S = np.zeros((len(T), len(U)), dtype=np.float32)
for ti in range(len(T)):
    rows = np.flatnonzero(town == ti)
    sim = (tt[rows] @ ut_.T).float().cpu().numpy()          # окна цели x окна источника
    S[ti] = np.maximum.reduceat(sim.max(axis=0), ustart)
uid = [u[0] for u in U]; upos = {u: i for i, u in enumerate(uid)}
for ti, t in enumerate(T):
    if t[0] in upos: S[ti, upos[t[0]]] = -1.0           # самосовпадение (D)
np.save(f'{D}/dense_{mat}_{key}.npy', S)
print(mat, key, S.shape, f'{time.time() - t0:.0f}с', flush=True)
