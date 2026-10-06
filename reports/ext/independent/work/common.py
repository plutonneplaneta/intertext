import re,collections
BASE='/path/to/workdir/indep/'
def load():
    rows=[l.rstrip('\n').split('\t') for l in open(BASE+'data/target_paradise_lost_lines.tsv')]
    lines=collections.defaultdict(list)   # book -> [(lineno, paraid, text)]
    for b,l,p,t in rows: lines[int(b)].append((int(l),int(p),t))
    gold=[]
    for l in list(open(BASE+'data/gold_raw.tsv'))[1:]:
        r=l.rstrip('\n').split('\t')
        gold.append(dict(rid=r[0],note=r[1],place=r[2],book=int(r[3]),line=int(r[4]),ref=r[5],unit=r[6],head=r[7]))
    notes={}
    for l in list(open(BASE+'data/notes_full_text.tsv'))[1:]:
        k,v=l.rstrip('\n').split('\t',1); notes[k]=v
    src={}
    for l in list(open(BASE+'data/source_classics_books.tsv'))[1:]:
        k,v=l.rstrip('\n').split('\t',1); src[k]=v
    samp=open(BASE+'data/manual_sample_ids.txt').read().split()
    return lines,gold,notes,src,samp
def para_text(lines,book,line):
    p=[x[1] for x in lines[book] if x[0]==line][0]
    return p,[x for x in lines[book] if x[1]==p]
def clean_src(uid,t):
    if uid=='Iliad.24':
        i=t.find('END OF THE ILIAD'); t=t[:i] if i>0 else t
    t=re.sub(r'\[\d+\]|\[[IVX]+\.\d+(-\d+)?\]','',t)
    return t
STOP=set("""a about above after again against all also am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself thee thou thy thine ye hath doth art shall may might must upon unto nor yet though thus whose whence hence thence one let say said still much many every ever never ere oft e'er ne'er whom whilst amid amidst through thro' tho' till like""".split())
def norm_tok(w):
    w=w.lower().replace('’',"'")
    w=re.sub(r"'s$","",w)
    w=w.replace("'","")
    return w
def stem(w):
    w=w.replace('v','u').replace('j','i')
    for suf in ('ings','ing','edst','eth','est','ed','es','ies','s','ly','d','th'):
        if w.endswith(suf) and len(w)-len(suf)>=3:
            w=w[:-len(suf)]; break
    w=re.sub(r'(ie|y|e)$','',w) if len(w)>3 else w
    return w
def toks(text):
    return [norm_tok(w) for w in re.findall(r"[A-Za-z][A-Za-z'’]*",text)]
def stems(text,drop_stop=True):
    out=[]
    for w in toks(text):
        if drop_stop and (w in STOP or len(w)<3): continue
        out.append(stem(w))
    return out
