import sys,math,collections,json
sys.path.insert(0,'/path/to/workdir/indep/work')
from common import *
import numpy as np
lines,gold,notes,src,samp=load()
RARE_SRC=int(sys.argv[1]) if len(sys.argv)>1 else 5
RARE_PL=int(sys.argv[2]) if len(sys.argv)>2 else 15
MINLEN=4
cl={u:clean_src(u,t) for u,t in src.items()}
units=sorted(cl)
ustem={u:set(stems(cl[u])) for u in units}
def seq(text):  # token sequence with stems, stopwords kept as None
    out=[]
    for w in toks(text):
        out.append(None if (w in STOP or len(w)<3) else stem(w))
    return out
def bigrams(text):
    s=seq(text); return {(a,b) for a,b in zip(s,s[1:]) if a and b}
ubig={u:bigrams(cl[u]) for u in units}
df_src=collections.Counter(w for u in units for w in ustem[u])
# PL paragraph df
paras={}
for b,ls in lines.items():
    for l,p,t in ls: paras.setdefault((b,p),[]).append(t)
pst={k:set(stems(' '.join(v))) for k,v in paras.items()}
df_pl=collections.Counter(w for s in pst.values() for w in s)
def rare(w): return len(w)>=MINLEN and 1<=df_src[w]<=RARE_SRC and df_pl[w]<=RARE_PL
# idf-weighted for tfidf cosine
N=len(units)
idf={w:math.log(N/df_src[w]) for w in df_src}
utf={}
for u in units:
    c=collections.Counter(stems(cl[u])); utf[u]=c
unorm={u:math.sqrt(sum((c[w]*idf[w])**2 for w in c)) for u,c in utf.items()}
def tfidf_scores(ptext):
    c=collections.Counter(stems(ptext)); pn=math.sqrt(sum((c[w]*idf.get(w,math.log(N)))**2 for w in c))
    sc={}
    for u in units:
        d=sum(c[w]*idf[w]*utf[u][w]*idf[w] for w in c if w in utf[u]); sc[u]=d/(pn*unorm[u]) if pn else 0
    return sc
res=[]; ranks=collections.defaultdict(list); nonzero=collections.defaultdict(int)
null_rates=[]
for g in gold:
    key=(g['book'],para_text(lines,g['book'],g['line'])[0])
    ptext=' '.join(paras[key]); ps=pst[key]; pb=bigrams(ptext)
    u=g['unit']
    feats={}
    feats['shared_content_stems']={x:len(ps&ustem[x]) for x in units}
    feats['shared_rare_stems']={x:len([w for w in ps&ustem[x] if rare(w)]) for x in units}
    feats['shared_bigrams']={x:len(pb&ubig[x]) for x in units}
    feats['tfidf_cosine']=tfidf_scores(ptext)
    for f,sc in feats.items():
        if sc[u]>0: nonzero[f]+=1
        r=1+sum(1 for x in units if sc[x]>sc[u]); ranks[f].append(r)
    rl=sorted(w for w in ps&ustem[u] if rare(w)); bg=sorted(pb&ubig[u])
    null_rates.append(np.mean([feats['shared_rare_stems'][x]>0 for x in units if x!=u]))
    # anchored
    nt=notes[g['note']]; lt=' '.join(x[2] for x in lines[g['book']] if x[0]==g['line'])
    anch=set(stems(nt+' '+lt))
    rl_anch=[w for w in rl if w in anch]
    bg_anch=[b for b in bg if b[0] in anch and b[1] in anch]
    res.append(dict(rid=g['rid'],rl=rl,bg=[' '.join(b) for b in bg],rl_a=rl_anch,bg_a=[' '.join(b) for b in bg_anch],ref=g['ref'],note=g['note'],unit=u))
tot=len(gold)
print('RARE_SRC',RARE_SRC,'RARE_PL',RARE_PL)
for f in nonzero: print(f,'nonzero',nonzero[f],'/',tot,'| top1',sum(r==1 for r in ranks[f]),'top5',sum(r<=5 for r in ranks[f]),'top15',sum(r<=15 for r in ranks[f]),'median rank',np.median(ranks[f]))
hb=[r for r in res if r['rl'] or r['bg']]
print('bridge literal',len(hb),'rare only',sum(1 for r in res if r['rl'] and not r['bg']),'big only',sum(1 for r in res if r['bg'] and not r['rl']),'both',sum(1 for r in res if r['rl'] and r['bg']))
print('rare-lemma bridge',sum(1 for r in res if r['rl']),'bigram bridge',sum(1 for r in res if r['bg']))
print('anchored bridge',sum(1 for r in res if r['rl_a'] or r['bg_a']))
print('null rate for rare-lemma bridge over wrong units: mean',round(float(np.mean(null_rates)),3))
json.dump(res,open(BASE+'work/step3_res.json','w'),indent=1)
json.dump(dict(nonzero=nonzero,ranks=ranks),open(BASE+'work/step3_ceil.json','w'))
# extra: null for bigram and either, plus tsv
nb=[];ne=[]
for g in gold:
    key=(g['book'],para_text(lines,g['book'],g['line'])[0]); ps=pst[key]; pb=bigrams(' '.join(paras[key]))
    w=[x for x in units if x!=g['unit']]
    nb.append(np.mean([len(pb&ubig[x])>0 for x in w]))
    ne.append(np.mean([(len(pb&ubig[x])>0) or any(rare(q) for q in ps&ustem[x]) for x in w]))
print('null bigram',round(float(np.mean(nb)),3),'null either',round(float(np.mean(ne)),3))
import csv
TH=f'rare_lemma:len>={MINLEN},df_src<={RARE_SRC}/75 units,df_PL<={RARE_PL}/368 paras; bigram:adjacent non-stop stems'
with open(BASE+'results_step3.tsv','w') as f:
    f.write('record_id\thas_bridge\tbridge_type\tthresholds_used\thas_bridge_anchored\tshared_rare_lemmas\tshared_bigrams\tanchored_evidence\n')
    for r in res:
        t='both' if r['rl'] and r['bg'] else 'rare_lemma' if r['rl'] else 'bigram' if r['bg'] else 'none'
        a=r['rl_a']+r['bg_a']
        f.write('\t'.join([r['rid'],str(int(t!='none')),t,TH,str(int(bool(a))),','.join(r['rl'][:8]),','.join(r['bg'][:8]),','.join(a)])+'\n')
