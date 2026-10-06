import csv,collections
rows=[l.rstrip('\n').split('\t') for l in open('data/target_paradise_lost_lines.tsv')]
print(rows[0], len(rows))
bad=[r for r in rows if len(r)!=4]; print(len(bad))
paras=collections.defaultdict(list)
for b,l,p,t in rows:
    if int(p) not in paras[int(b)]: paras[int(b)].append(int(p))
for b in sorted(paras): print(b,len(paras[b]),paras[b][:6],max(int(r[1]) for r in rows if int(r[0])==b))
