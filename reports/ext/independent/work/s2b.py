import collections,re
rows=[l.rstrip('\n').split('\t') for l in open('data/target_paradise_lost_lines.tsv')]
paras=collections.defaultdict(list)
line2para={}
for b,l,p,t in rows:
    b,l,p=int(b),int(l),int(p)
    if p not in paras[b]: paras[b].append(p)
    line2para[(b,l)]=p
print({b:(v==list(range(1,2*len(v),2))) for b,v in paras.items()})
# duplicates of line numbers?
c=collections.Counter((r[0],r[1]) for r in rows); print([k for k,v in c.items() if v>1][:5])
g=[l.rstrip('\n').split('\t') for l in open('data/gold_raw.tsv')][1:]
bad=0
for r in g:
    b,ln=int(r[3]),int(r[4]); p=line2para.get((b,ln))
    if p is None: print('noline',r[0],b,ln); continue
    rank=paras[b].index(p)+1
    pl=int(r[2].split('.p')[1]); 
    if rank!=pl: print('mismatch',r[0],r[2],rank,p); bad+=1
print(len(g),bad)
