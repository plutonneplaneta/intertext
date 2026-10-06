import re,csv,time,collections
t0=time.time()
SUF=['ing','edly','ed','es','s','ly','ment','ness','eth','est','er','e']
def stem(w):
    for s in ['ingly','ing','edly','ed','eth','est','ies','es','s','ly','e']:
        if w.endswith(s) and len(w)-len(s)>=3:
            return w[:-len(s)]
    return w
def toks(t,skip=False):
    ws=re.findall(r"[a-z]+",t.lower().replace("’","'").replace("'"," "))
    return [stem(w) for w in ws]
src={}
for r in csv.DictReader(open('data/source_classics_books.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE):
    src[r['unit_id']]=toks(r['text'])
N=len(src);df=collections.Counter()
for u,t in src.items():
    for s in set(t): df[s]+=1
RARE=max(3,int(N*0.01)); BI=int(N*0.10)  # 3, 7
paras=collections.defaultdict(list); lineof={}
for l in open('data/target_paradise_lost_lines.tsv',encoding='utf-8'):
    p=l.rstrip('\n').split('\t')
    if len(p)<4 or not p[0].isdigit(): continue
    paras[(p[0],p[2])].append(p[3]); lineof[(p[0],int(p[1]))]=(p[0],p[2])
out=[];cnt=collections.Counter()
for r in csv.DictReader(open('data/gold_raw.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE):
    key=lineof[(r['poem_book'],int(r['poem_line']))]
    pt=toks(' '.join(paras[key])); ut=src[r['source_unit']]
    rare={s for s in set(pt)&set(ut) if len(s)>=4 and df[s]<=RARE}
    ok=lambda s:len(s)>=4 and df[s]<=BI
    pb={(a,b) for a,b in zip(pt,pt[1:]) if ok(a) and ok(b)}
    ub={(a,b) for a,b in zip(ut,ut[1:]) if ok(a) and ok(b)}
    bi=pb&ub
    ty='both' if rare and bi else 'rare' if rare else 'bigram' if bi else 'none'
    cnt[ty]+=1
    out.append((r['record_id'],int(ty!='none'),ty,sorted(rare),sorted(bi)))
w=open('results_step3.tsv','w');w.write('record_id\thas_bridge\tbridge_type\n')
for o in out:w.write(f'{o[0]}\t{o[1]}\t{o[2]}\n')
print(N,RARE,BI,cnt,sum(c for k,c in cnt.items() if k!='none'))
for o in out:print(o[0],o[2],o[3][:4],o[4][:2])
print(time.time()-t0)
