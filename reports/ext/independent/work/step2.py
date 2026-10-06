import sys,re,collections
sys.path.insert(0,'/path/to/workdir/indep/work')
from common import *
lines,gold,notes,src,samp=load()
CANON={'Aeneid':[756,804,718,705,871,901,817,731,818,908,915,952],
 'Iliad':[611,877,461,544,909,529,482,565,713,579,848,471,837,522,746,867,761,617,424,503,611,515,897,804],
 'Odyssey':[444,434,497,847,493,331,347,586,566,574,640,453,402,525,557,481,508,405,604,394,423,501,372,548],
 'Metamorphoses':[779,875,733,803,678,721,865,884,797,739,795,628,968,851,879]}
flags=collections.defaultdict(list); info={}
# 1 alignment
for g in gold:
    p,pl=para_text(lines,g['book'],g['line'])
    rank=sorted({x[1] for x in lines[g['book']]}).index(p)+1
    if f"PL.{g['book']}.p{rank:03d}"!=g['place']: flags[g['rid']].append('place_mismatch')
    lemma=re.split(r'\.(?:\s|$)',notes[g['note']].strip(),1)[0]
    gl=g['head']
    if lemma.lower().startswith('lines'): info[g['rid']]=('lemma_generic',); continue
    ls=set(stems(lemma,False)); 
    ls={s for s in ls if len(s)>=3}
    inline=set(stems(' '.join(x[2] for x in pl if x[0]==g['line']),False))
    inpara=set(stems(' '.join(x[2] for x in pl),False))
    nearby=set(stems(' '.join(x[2] for x in lines[g['book']] if abs(x[0]-g['line'])<=3),False))
    if not ls: continue
    f_line=len(ls&inline)/len(ls); f_par=len(ls&inpara)/len(ls); f_near=len(ls&nearby)/len(ls)
    if f_par<0.5: flags[g['rid']].append('lemma_not_in_paragraph')
    elif f_line<0.5: flags[g['rid']].append('lemma_not_in_poem_line')
    info[g['rid']]=(lemma,round(f_line,2),round(f_par,2))
# 2 ref validity
def parse_ref(ref):
    m=re.match(r'(Aeneid|Iliad|Odyssey|Metamorphoses)\s*(?:book\s*)?(\d+)?\s*[.:]?\s*(.*)$',ref)
    return m.groups()
cites=collections.defaultdict(set)
for g in gold:
    auth,bk,rest=parse_ref(g['ref'])
    if g['unit'] not in src: flags[g['rid']].append('unit_missing_in_source')
    if auth!=g['unit'].split('.')[0] or (bk and bk!=g['unit'].split('.')[1]): flags[g['rid']].append('ref_unit_mismatch')
    if not bk: flags[g['rid']].append('ref_no_book')
    if not rest.strip(): flags[g['rid']].append('book_level_ref')
    nums=[int(x) for x in re.findall(r'\d+',rest)]
    if len(nums)>=2 and nums[0]>nums[1] and not re.search(r'\d+-\d+',rest) is None and nums[0]>nums[1]:
        # allow abbreviated ranges like 484-85
        if not (nums[1]<nums[0] and len(str(nums[1]))<len(str(nums[0]))): flags[g['rid']].append('ref_range_reversed')
    if rest.strip() and bk and nums:
        n=CANON[auth][int(bk)-1]
        if max(nums)>n: flags[g['rid']].append('ref_line_beyond_book')
    # ref present in note text (independent route)
    nt=re.sub(r'\s+','',notes[g['note']]).lower()
    rt=re.sub(r'\s+','',g['ref']).lower()
    ok=rt in nt or rt.replace(':','.') in nt
    if not ok:
        ok2 = re.sub(r'\s+','',g['ref'].replace('book ','')).lower() in nt
        if not ok2: flags[g['rid']].append('ref_not_in_note')
# 3 duplicates & note-id
seen=collections.Counter((g['place'],g['unit']) for g in gold)
cnt=collections.Counter(g['note'] for g in gold)
for g in gold:
    if seen[(g['place'],g['unit'])]>1: flags[g['rid']].append('dup_place_unit')
    if cnt[g['note']]>1: flags[g['rid']].append('shared_note_multi_ref')
    nums=[int(x) for x in re.findall(r'\d+',g['note'].split(':',1)[1])]
    if nums:
        lo,hi=min(nums),max(nums)
        if not(lo-0<=g['line']<=hi) : flags[g['rid']].append('note_id_line_mismatch')
# 4 note cites not in gold
for nid in set(g['note'] for g in gold):
    gu={g['unit'] for g in gold if g['note']==nid}
    cs=set()
    for m in re.finditer(r'(Aeneid|Iliad|Odyssey|Metamorphoses)\s*(?:book\s*)?(\d+)',notes[nid]): cs.add(f'{m.group(1)}.{m.group(2)}')
    miss=cs-gu
    if miss:
        for g in gold:
            if g['note']==nid: flags[g['rid']].append('note_cites_unit_not_in_gold:'+'|'.join(sorted(miss)))
# 5 name check
for g in gold:
    lemma=re.split(r'\.(?:\s|$)',notes[g['note']].strip(),1)[0]
    PN={'Orion','Oechalia','Oeta','Medusa','Raphael','Maia','Laertes','Chimera','Pythian','Olympian','Hesperian','Cerberian','Hyacinthin'}
    cap=[w for w in re.findall(r"\b[A-Z][a-z]{3,}",lemma) if w in PN]
    t=clean_src(g['unit'],src[g['unit']]).lower()
    author=g['unit'].split('.')[0]
    for w in cap:
        wl=w.lower()
        if wl[:5] in t: continue
        other=[u for u in src if u.startswith(author) and wl[:5] in src[u].lower()]
        if other: flags[g['rid']].append(f'weak_name_{w}_absent_in_unit_present_in:{",".join(o.split(".")[1] for o in other[:4])}')
        else: flags[g['rid']].append(f'weak_name_{w}_absent_in_author')
import json

# range spans crossing paragraphs
for g in gold:
    nums=[int(x) for x in re.findall(r'\d+',g['note'].split(':',1)[1])]
    if len(nums)>=2:
        ps={x[1] for x in lines[g['book']] if min(nums)<=x[0]<=max(nums)}
        if len(ps)>1: flags[g['rid']].append('note_range_spans_paragraphs')
for rid in ('R011','R066','R096'): flags[rid].append('ref_line_doubtful_external')
json.dump({k:v for k,v in flags.items()},open(BASE+'work/auto_flags.json','w'),indent=1)
json.dump({k:list(v) for k,v in info.items()},open(BASE+'work/info.json','w'))
for g in gold:
    print(g['rid'],g['ref'],'|',';'.join(flags.get(g['rid'],[])) ,info.get(g['rid']))
#f.split(':')[0].split('_')[0]+'_'+f.split(':')[0].split('_')[1] if f.startswith('name_') else f.split(':')[0] for v in flags.values() for f in set(v))
print(tot)
