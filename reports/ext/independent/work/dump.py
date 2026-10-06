import collections
rows=[l.rstrip('\n').split('\t') for l in open('data/target_paradise_lost_lines.tsv')]
tl={(int(r[0]),int(r[1])):r[3] for r in rows}
notes={}
for l in list(open('data/notes_full_text.tsv'))[1:]:
    k,v=l.rstrip('\n').split('\t',1); notes[k]=v
samp=set(open('data/manual_sample_ids.txt').read().split())
for l in list(open('data/gold_raw.tsv'))[1:]:
    r=l.rstrip('\n').split('\t')
    b,ln=int(r[3]),int(r[4])
    print(f"## {r[0]} {'[S]' if r[0] in samp else ''} note={r[1]} PL {b}.{ln} ref='{r[5]}' unit={r[6]}")
    print("LINE:",tl.get((b,ln)))
    print("NOTE:",notes.get(r[1],'MISSING')[:1500]); print()
