"""Frozen S4.5 inputs: rosters, four-cell construction, background states and schedules.

CPU only; no model loading. Implements "RI participation in measured structures:
implementation specification" (26 September 2026). Every decision object written by
this module is content-addressed; measurement code refuses inputs whose hashes drift.
"""
import collections
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,write_lines,span_tokens
from stage4_plan import hid,masks as base_masks
from s42_plan import mean_roles,key_levels

POLICY=dict(version=1,protocol='ri_structure_final_protocol_20260926',seed=20260926,
    metric='logit(a)-logit(b) at first divergent answer token; shared prefix teacher forced; a,b fixed by x00',
    cells=dict(x00='original facts, original query (gold a)',x10='swapped mothers, original query (gold b)',
               x01='original facts, alternative query (gold b)',x11='swapped mothers, alternative query (gold a)'),
    tau=.1,coherent_fraction=.7,heterogeneous_fraction=.2,bootstrap_draws=20000,
    fidelity_F=[.8,1.2],fidelity_L=.2,accuracy_drop=.05,mean_min_families=10,
    max_selected=3,max_pairs_per_candidate=2,
    background='retained heads live at every original test-block position; other head-output slices replaced by role means; MLPs, embeddings, demonstrations, shared normalization and answer prefix live',
    mean_bank='all 1024 heads; discovery x00/x10 cells, both orders; leave-one-family-out in discovery, fixed full bank on held-out',
    routes='S4.1/S4.3 executors inside the declared background: source replaced at site, post-normalization branches frozen, selective live intermediates, receiver channel recaptured and injected into a fresh run',
    gamma='I_t(B_h_plus)-I_t(B_h_minus); difference of matched intervention effects',
    scope='single-hop contextual task; head-subset mechanisms conditional on the live background')

def hname(h):return f'L{h//32}H{h%32}'
def ids(names):return sorted(hid(n) for n in names)

STRUCTURES={
 'T1':dict(retained=['L17H1','L18H18','L18H19','L27H6'],anchor=dict(source='L17H1',site='query_writer_union',live=['L18H18','L18H19'],receiver='L27H6',channel='Q'),historical=dict(I=1.635,J=1.511)),
 'T2':dict(retained=['L15H25','L16H1','L16H21','L18H18','L18H19'],anchor=dict(source='L15H25',site='query_child_last',live=['L16H1','L16H21'],receiver='L18H18',channel='Q'),historical=dict(I=.172,J=.179)),
 'T3':dict(retained=['L17H1','L18H18','L27H6'],anchor=dict(source='L17H1',site='query_writer_union',live=['L18H18'],receiver='L27H6',channel='Q'),historical=dict(I=.876,J=.809)),
 'T4':dict(retained=['L8H15','L18H18'],anchor=dict(source='L8H15',site='query_sentence',live=[],receiver='L18H18',channel='V'),historical=dict(I=.465,J=.442)),
 'T5':dict(retained=['L20H1','L27H6'],anchor=dict(source='L20H1',site='colon',live=[],receiver='L27H6',channel='Q'),historical=dict(I=-.236,J=-.223))}
C33=['L6H24','L8H15','L13H10','L13H18','L14H23','L14H26','L15H3','L15H25','L16H1','L16H21','L16H31','L17H1','L17H3','L17H17','L17H24',
     'L18H18','L18H19','L18H24','L19H16','L19H22','L20H1','L20H7','L21H6','L21H18','L21H23','L22H5','L23H15','L24H19','L25H17','L26H23','L27H6','L30H13','L30H18']
R17=['L1H27','L8H8','L9H6','L9H17','L11H4','L11H7','L12H2','L16H4','L16H16','L16H24','L17H5','L21H25','L23H10','L25H18','L26H31','L27H23','L30H3']
C50=sorted(C33+R17,key=hid)
CANDIDATES=['L1H27','L11H4','L17H5','L23H10','L25H18','L9H16']
REFERENCES=['L18H19','L8H15']
# Named attachments (specification section 9). Sources precede receivers in layer order.
ATTACHMENTS=[
 (['T1','T3','T4'],['L1H27','L9H16','L11H4','L17H5'],dict(source='h',site='all_sentence',receiver='L18H18',channel='V')),
 (['T2'],['L1H27','L9H16','L11H4'],dict(source='h',site='all_sentence',receiver='L16H1',channel='V')),
 (['T2'],['L17H5'],dict(source='h',site='colon',receiver='L18H18',channel='Q')),
 (['T1','T2','T3','T4'],['L23H10','L25H18'],dict(source='L18H18',site='colon',receiver='h',channel='Q')),
 (['T5'],['L1H27','L9H16','L11H4','L17H5'],dict(source='h',site='all_sentence',receiver='L20H1',channel='V')),
 (['T5'],['L23H10','L25H18'],dict(source='h',site='colon',receiver='L27H6',channel='Q'))]
CELLS=['x00','x10','x01','x11']
GOLD_SIDE=dict(x00='a',x10='b',x01='b',x11='a')

def check_rosters():
    for t,s in STRUCTURES.items():
        if not set(s['retained'])<=set(C33):raise ValueError('T set outside C33: '+t)
        a=s['anchor']
        if not {a['source'],a['receiver']}|set(a['live'])<=set(s['retained']):raise ValueError('Anchor outside retained set: '+t)
        if hid(a['source'])//32>=hid(a['receiver'])//32:raise ValueError('Noncausal anchor: '+t)
        if any(not hid(a['source'])//32<hid(h)//32<hid(a['receiver'])//32 for h in a['live']):raise ValueError('Live head outside source-receiver band: '+t)
        if a['channel']=='Q' and a['site']!='colon' and not a['live']:raise ValueError('Direct cross-position Q route: '+t)
    if len(C50)!=50 or len(set(C33)&set(R17)):raise ValueError('C50 roster')
    if not set(STRUCTURES['T3']['retained'])<set(STRUCTURES['T1']['retained']):raise ValueError('T3 must nest in T1')
    if set(CANDIDATES)&set(C33):raise ValueError('Candidate inside C33')
    for ts,hs,att in ATTACHMENTS:
        for h in hs:
            a=attachment(ts[0],h)
            if hid(a['source'])//32>=hid(a['receiver'])//32:raise ValueError(f'Noncausal attachment {h}')
            if a['channel']=='Q' and a['site']!='colon':raise ValueError('Q attachment must originate at the colon')
def attachment(t,h):
    hits=[att for ts,hs,att in ATTACHMENTS if t in ts and h in hs]
    if len(hits)!=1:raise ValueError(f'No unique attachment for {t},{h}')
    a=dict(hits[0]);a['source']=h if a['source']=='h' else a['source'];a['receiver']=h if a['receiver']=='h' else a['receiver']
    a['key']=f"{a['source']}|{a['site']}|{a['receiver']}|{a['channel']}";return a
check_rosters()

def control_rosters(original59):
    """One frozen same-layer alternative receiver per possible attachment; Random(seed), sorted eligible IDs."""
    rng=random.Random(POLICY['seed']);excluded=set(ids(C50))|{hid(h) for s in STRUCTURES.values() for h in s['retained']}|set(original59)
    keys=sorted({attachment(t,h)['key'] for ts,hs,_ in ATTACHMENTS for t in ts for h in hs})
    rosters={}
    for key in keys:
        src,site,recv,ch=key.split('|');li=hid(recv)//32;h=hid(src) if src in CANDIDATES else hid(recv)
        eligible=sorted(x for x in range(li*32,li*32+32) if x not in excluded and x!=h and x!=hid(recv))
        if not eligible:raise ValueError('No eligible control receiver: '+key)
        rosters[key]=dict(control=hname(rng.choice(eligible)),eligible=len(eligible))
    return rosters

# ---------------------------------------------------------------- cells --------
def divergence(tok,prompt,a,b):
    base=tok.encode(prompt,add_special_tokens=False);tails=[]
    for name in (a,b):
        full=tok.encode(prompt+' '+name+'.',add_special_tokens=False)
        if full[:len(base)]!=base:raise ValueError('Tokenizer changed prompt boundary')
        tails.append(full[len(base):])
    k=next((i for i,(x,y) in enumerate(zip(*tails)) if x!=y),None)
    if k is None:raise ValueError('Candidates have no divergent token')
    return dict(input_ids=base+tails[0][:k],n=len(base),prefix=tails[0][:k],tok_a=tails[0][k],tok_b=tails[1][k])

def annotate_row(tok,row):
    enc=tok(row['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    row=dict(row);row['token_ids']=list(enc['input_ids']);row['offsets']=[list(x) for x in enc['offset_mapping']]
    row['tokens']=tok.convert_ids_to_tokens(row['token_ids']);return row

def cell_masks(row):
    m=base_masks(row);text=row['prompt'];facts=[f for f in row['facts'] if f['block']=='test'];union=set()
    for f in facts:
        a,b=f['tail_span'];end=text.find('.',f['head_span'][1])
        union.update(span_tokens(row['offsets'],(a,end+1)))
    m['all_sentence']=sorted(union)
    if set(m['all_sentence'])&set(m['question']) or m['colon'][0] in m['all_sentence']:raise ValueError('All-fact mask touches question/colon')
    return m

def alternative_fact(row):
    query=row['query_source'];alt=[f for f in row['facts'] if f['block']=='test' and f['tail'] in row['candidates'] and f['head']!=query]
    if len(alt)!=1:raise ValueError('Ambiguous alternative query: '+row['id'])
    return alt[0]

def changed_query_row(tok,row):
    """Alternative queried child. Every derived field is rebuilt; nothing is inherited stale."""
    fact=alternative_fact(row);old='Question: Who is the mother of '+row['query_source']+'?';i=row['prompt'].rfind(old)
    if i<0 or i<row['test_start']:raise ValueError('Final question not found in test block: '+row['id'])
    new={k:row[k] for k in ['family','facts','candidates','demo_ids','test_start','order'] if k in row}
    new.update(id=row['id']+'/query_change',variant=row['variant']+'_query_change',
               prompt=row['prompt'][:i]+row['prompt'][i:].replace(old,'Question: Who is the mother of '+fact['head']+'?',1),
               query_source=fact['head'],question_entity=fact['head'],gold=fact['tail'])
    if new['prompt'][:row['test_start']]!=row['prompt'][:row['test_start']]:raise ValueError('Demonstrations changed')
    for f in new['facts']:
        if f['block']=='test' and (new['prompt'][slice(*f['head_span'])]!=f['head'] or new['prompt'][slice(*f['tail_span'])]!=f['tail']):raise ValueError('Fact spans moved')
    if new['prompt'].count('Question:')!=row['prompt'].count('Question:'):raise ValueError('Question count changed')
    return annotate_row(tok,new)

def build_cell(tok,row,cell,names):
    a,b=names;row=dict(row,query_source=row.get('question_entity',row.get('query_source')))
    if row['gold']!=(a if GOLD_SIDE[cell]=='a' else b) or set(row['candidates'])!={a,b}:raise ValueError('Cell gold/candidate identity: '+row['id']+' '+cell)
    dv=divergence(tok,row['prompt'],a,b)
    if dv['n']!=len(row['token_ids']) or dv['input_ids'][:dv['n']]!=row['token_ids']:raise ValueError('Annotation/tokenizer mismatch: '+row['id'])
    m=cell_masks(row);keys=mean_roles(row,m)
    if m['colon'][0]!=dv['n']-1:raise ValueError('Colon is not the final prompt token')
    return dict(cell=cell,id=row['id'],variant=row['variant'],prompt=row['prompt'],gold=row['gold'],gold_side=GOLD_SIDE[cell],query_source=row['query_source'],
        token_ids=row['token_ids'],offsets=row['offsets'],input_ids=dv['input_ids'],n=dv['n'],shared_prefix=dv['prefix'],tok_a=dv['tok_a'],tok_b=dv['tok_b'],
        masks=m,mean_keys=keys,test_start=row['test_start'])

def aligned(x,y):
    if len(x['token_ids'])!=len(y['token_ids']):raise ValueError('Unaligned twin tokens: '+x['id'])
    if sum(p!=q for p,q in zip(x['token_ids'],y['token_ids']))>6:raise ValueError('More than six differing tokens: '+x['id'])
    if x['masks']!=y['masks'] or x['mean_keys']!=y['mean_keys']:raise ValueError('Role alignment differs across twins: '+x['id'])
    if x['shared_prefix']!=y['shared_prefix'] or (x['tok_a'],x['tok_b'])!=(y['tok_a'],y['tok_b']):raise ValueError('Prefix differs across twins: '+x['id'])

def build_pair(tok,base,corr,split,saved=None,exact_all=False):
    if base['family']!=corr['family'] or base['order']!=corr['order'] or base['demo_ids']!=corr['demo_ids']:raise ValueError('Twin mismatch: '+base['id'])
    a=base['gold'];b=next(c for c in base['candidates'] if c!=a)
    if corr['gold']!=b or set(corr['candidates'])!={a,b}:raise ValueError('Twins must exchange the same two answers')
    cells=dict(x00=build_cell(tok,base,'x00',(a,b)),x10=build_cell(tok,corr,'x10',(a,b)),
               x01=build_cell(tok,changed_query_row(tok,base),'x01',(a,b)),x11=build_cell(tok,changed_query_row(tok,corr),'x11',(a,b)))
    aligned(cells['x00'],cells['x10']);aligned(cells['x01'],cells['x11'])
    if cells['x01']['query_source']!=alternative_fact(base)['head'] or cells['x01']['masks']['_query_first']==cells['x00']['masks']['_query_first']:raise ValueError('Query change did not move the queried fact')
    return dict(id=base['id'],family=base['family'],order=base['order'],split=split,names=[a,b],exact_all=exact_all,
                shared_prefix=cells['x00']['shared_prefix'],saved_baselines=saved,cells=cells)

def dataset_rows(path,split):
    rows=[r for r in read_lines(path) if r['split']==split and r['variant'] in ('base','corrupted') and r['world']=='kinship']
    out=[]
    for r in rows:
        p=r['prompt'];start=p.rfind('\n\n')+2
        out.append(dict(id=r['id'],family=r['family'],variant=r['variant'],order=r['option_order'],prompt=p,facts=r['facts'],candidates=r['candidates'],
                        gold=r['gold'],query_source=r['question_entity'],question_entity=r['question_entity'],demo_ids=r['demo_ids'],test_start=start,split=r['split']))
    return out

def twins(rows):
    by=collections.defaultdict(dict)
    for r in rows:by[(r['family'],r['order'])][r['variant']]=r
    for k,v in sorted(by.items()):
        if set(v)!={'base','corrupted'}:raise ValueError('Incomplete twin: '+str(k))
        yield v['base'],v['corrupted']

def working_families(baseline_rows):
    ok=collections.defaultdict(list)
    for r in baseline_rows:
        if r['variant'] in ('base','corrupted'):ok[r['family']].append(bool(r['correct']))
    return sorted(f for f,v in ok.items() if len(v)==4 and all(v))

# ---------------------------------------------------------------- states -------
def state(live,label):
    if live is None:return dict(id='full',label=label,live=None)
    live=sorted(set(int(h) for h in live))
    if any(not 0<=h<1024 for h in live):raise ValueError('Head outside model')
    return dict(id=hashlib.sha256(json.dumps(live).encode()).hexdigest()[:16],label=label,live=live)

def states_for(B,extra=()):
    """The at most seven B states plus references and collective removal."""
    B=set(ids(B));out={'B':state(B,'B')}
    for h in CANDIDATES:
        x=hid(h);out[f'B-{h}']=state(B-{x},f'B-{h}');out[f'B+{h}']=state(B|{x},f'B+{h}')
    for h in REFERENCES:out[f'B-{h}']=state(B-{hid(h)},f'B-{h}')
    return out

def rb(B,original59):return sorted(set(ids(B))&set(original59))

class Schedule:
    def __init__(self,inputs,phase,population,cells=CELLS):
        self.plan_hash=digest(inputs/'plan.json');self.phase=phase;self.population=population;self.cells=list(cells)
        self.states={};self.jobs=[];self.meta={}
    def add_state(self,s):
        old=self.states.setdefault(s['id'],s)
        if old['live']!=s['live']:raise ValueError('State hash collision')
        return s['id']
    def behavior(self,s,baseline='mean',cells=None):
        sid=self.add_state(s);cells=list(cells or self.cells);jid=f'behavior|{sid}|{baseline}|{"".join(cells)}'
        self._add(dict(id=jid,kind='behavior',state=sid,baseline=baseline,cells=cells));return jid
    def route(self,s,anchor,direction):
        sid=self.add_state(s);jid=f'route|{sid}|{anchor}|{direction}'
        a=STRUCTURES[anchor]['anchor']
        self._add(dict(id=jid,kind='route',state=sid,anchor=anchor,direction=direction,source=hid(a['source']),site=a['site'],
                       live=ids(a['live']),receiver=hid(a['receiver']),channel=a['channel']));return jid
    def attachment(self,s,att,role,direction,candidate,structure,control_of=None):
        sid=self.add_state(s);jid=f'attachment|{sid}|{att["key"]}|{role}|{direction}'
        self._add(dict(id=jid,kind='attachment',state=sid,role=role,direction=direction,candidate=candidate,structure=structure,key=att['key'],
                       source=hid(att['source']),site=att['site'],receiver=hid(att['receiver']),channel=att['channel'],control_of=control_of));return jid
    def _add(self,job):
        if any(j['id']==job['id'] for j in self.jobs):return
        self.jobs.append(job)
    def write(self,path,**meta):
        json_write(path,dict(policy=POLICY,plan_hash=self.plan_hash,phase=self.phase,population=self.population,
                             states=self.states,jobs=self.jobs,**meta))
        return path

def stage_a(inputs,plan,with_c50=False):
    s=Schedule(inputs,'stage_a_c50' if with_c50 else 'stage_a','core')
    if with_c50:s.behavior(state(ids(C50),'C50'));return s
    s.behavior(state(None,'full'));s.behavior(state([],'empty'))
    for t,st in STRUCTURES.items():s.behavior(state(ids(st['retained']),t))
    s.behavior(state(ids(C33),'C33'));return s

def stage_b(inputs,plan,B):
    s=Schedule(inputs,'stage_b','core');st=states_for(B)
    for k,v in st.items():s.behavior(v)
    s.behavior(state(set(ids(B))-set(rb(B,plan['original59'])),'B-R'));return s

def stage_c(inputs,plan,B):
    s=Schedule(inputs,'stage_c','core',cells=['x00','x10']);st=states_for(B)
    for k in ['B']+[f'B{sign}{h}' for h in CANDIDATES for sign in '-+']:
        for t in STRUCTURES:s.route(st[k],t,'noise')
    for t in STRUCTURES:s.route(state(None,'full'),t,'noise')  # full-background regression against historical anchors (gate work)
    return s

def stage_d(inputs,plan,B,selection):
    s=Schedule(inputs,'stage_d','core',cells=['x00','x10']);st=states_for(B);B_ids=set(ids(B))
    for sel in selection['selected']:
        h=sel['candidate']
        if sel['structure']=='B':continue
        t=sel['structure']
        for k in [f'B-{h}',f'B+{h}']:
            for d in ['noise','restore']:s.route(st[k],t,d)
        att=attachment(t,h);ctrl=plan['control_rosters'][att['key']]['control']
        diag=state(B_ids|{hid(h),hid(ctrl)},f'B+{h}+{ctrl}');catt=dict(att,receiver=ctrl,key=att['key']+'|control='+ctrl)
        for d in ['noise','restore']:
            s.attachment(diag,att,'target',d,h,t);s.attachment(diag,catt,'control',d,h,t,control_of=att['key'])
    return s

def behavior_states(plan,B,selection):
    st=states_for(B);B_ids=set(ids(B))
    mean=[state(None,'full'),state([],'empty'),st['B'],state(B_ids-set(rb(B,plan['original59'])),'B-R')]
    donor=[st['B'],state(B_ids-set(rb(B,plan['original59'])),'B-R')]
    for sel in selection['selected']:
        h=sel['candidate'];other=st[f'B-{h}'] if hid(h) in B_ids else st[f'B+{h}']
        if other['id'] not in {x['id'] for x in mean}:mean.append(other);donor.append(other)
    return mean,donor

def full_discovery(inputs,plan,B,selection):
    s=Schedule(inputs,'full_discovery','discovery',cells=['x00','x10']);mean,donor=behavior_states(plan,B,selection)
    for x in mean:s.behavior(x,'mean')
    for x in donor:s.behavior(x,'donor')
    return s

def validation(inputs,plan,freeze):
    B=freeze['B']['members'];sel=freeze['selection'];s=Schedule(inputs,'validation','heldout');st=states_for(B);mean,donor=behavior_states(plan,B,sel)
    for x in mean:s.behavior(x,'mean',CELLS)
    for x in donor:s.behavior(x,'donor',['x00','x10'])
    d=stage_d(inputs,plan,B,sel)
    for k,v in d.states.items():s.states.setdefault(k,v)
    for j in d.jobs:s._add(j)
    return s

# ---------------------------------------------------------------- prepare ------
def prepare(stage3_inputs,s42_inputs,candidates_json,out,tok,dataset=None,baseline=None,compatibility=None):
    """Discovery inputs: 178 pairs x four cells, frozen rosters, control rosters, plan hashes."""
    if out.exists():raise FileExistsError('Frozen inputs already exist; choose a new version')
    prompts={r['id']:r for r in read_lines(stage3_inputs/'prompts.jsonl.gz')}
    old=list(read_lines(s42_inputs/'pairs.jsonl.gz'));oplan=json_read(s42_inputs/'plan.json')
    if len(old)!=178 or digest(s42_inputs/'pairs.jsonl.gz')!=oplan['pair_hash']:raise ValueError('S4.2 population/hash')
    cand=json_read(candidates_json);original59=sorted(x['layer']*32+x['head'] for x in cand['candidates'])
    if len(original59)!=59:raise ValueError('Original RI candidate JSON must hold 59 heads')
    pairs=[]
    for p in old:
        if any(p[k]['split']!='discovery' for k in ['clean','corr']):raise ValueError('Held-out leakage')
        base,corr=prompts[p['clean']['id']],prompts[p['corr']['id']]
        for r in (base,corr):
            if r.get('split','discovery')!='discovery':raise ValueError('Held-out leakage')
        base=dict(base,demo_ids=p['clean']['demo_ids'],question_entity=p['clean']['question_entity'],split='discovery')
        corr=dict(corr,demo_ids=p['corr']['demo_ids'],question_entity=p['corr']['question_entity'],split='discovery')
        for r in (base,corr):
            fresh=annotate_row(tok,r)
            if fresh['token_ids']!=r['token_ids'] or [list(x) for x in r['offsets']]!=fresh['offsets']:raise ValueError('Historical annotation not reproduced: '+r['id'])
        q=build_pair(tok,base,corr,'discovery',saved=p['saved_baselines'],exact_all=p['exact_all'])
        if {k:v for k,v in q['cells']['x00']['masks'].items() if k!='all_sentence'}!=p['masks']:raise ValueError('Historical masks not reproduced: '+p['id'])
        if q['cells']['x00']['mean_keys']!=p['mean_keys'] or q['cells']['x00']['token_ids']!=p['original_ids']['clean'] or q['shared_prefix']!=p['shared_prefix']:raise ValueError('Historical roles/ids not reproduced')
        pairs.append(q)
    families=sorted({p['family'] for p in pairs});core=sorted({p['family'] for p in pairs if p['exact_all']})
    if len(families)!=89 or len(core)!=20 or core!=oplan['core_families']:raise ValueError('Population mismatch')
    heldout=dict(sealed=True,dataset=str(dataset) if dataset else None,baseline=str(baseline) if baseline else None,
                 dataset_hash=digest(dataset) if dataset else None,baseline_hash=digest(baseline) if baseline else None)
    hashes={str(stage3_inputs/'prompts.jsonl.gz'):digest(stage3_inputs/'prompts.jsonl.gz'),str(s42_inputs/'pairs.jsonl.gz'):digest(s42_inputs/'pairs.jsonl.gz'),str(candidates_json):digest(candidates_json)}
    exceptions={}
    if compatibility:
        # S4.3 family-034 accommodation, kept explicit: pairs whose historical baseline drifted >0.05 on the
        # documented GPU keep the diagnostic's deterministic value as an alternative reference.
        diag=json_read(compatibility)
        for r in diag['rows']:
            if r['historical_margin_error']>.05:
                exceptions[r['pair_id']]=dict(margin=r['current_margin'],clean_logit=r['current_clean_logit'],corr_logit=r['current_corr_logit'],
                    historical_margin_error=r['historical_margin_error'],gpu=diag['runtime']['gpu'],source_sha256=digest(compatibility),reference_tolerance=1e-3)
        hashes[str(compatibility)]=digest(compatibility)
    return freeze_inputs(out,pairs,oplan['model'],core,oplan['gate_families'],original59,heldout,hashes,exceptions)

def freeze_inputs(out,pairs,model,core,gate,original59,heldout,source_hashes,exceptions=None):
    families=sorted({p['family'] for p in pairs})
    keys=sorted({k for p in pairs for c in p['cells'].values() for key in c['mean_keys'] if key for k in key_levels(key)})
    out.mkdir(parents=True);write_lines(out/'pairs.jsonl.gz',pairs)
    plan=dict(policy=POLICY,model=model,families=families,core_families=core,gate_families=gate,
        mean_keys=keys,original59=original59,structures=STRUCTURES,C33=C33,R17=R17,C50=C50,candidates=CANDIDATES,references=REFERENCES,
        attachments=[dict(structures=ts,candidates=hs,attachment=att) for ts,hs,att in ATTACHMENTS],control_rosters=control_rosters(original59),
        heldout=heldout,pair_hash=digest(out/'pairs.jsonl.gz'),source_hashes=source_hashes,compatibility_exceptions=exceptions or {})
    json_write(out/'plan.json',plan)
    json_write(out/'structure_registry.json',dict(structures={t:dict(s,retained_ids=ids(s['retained'])) for t,s in STRUCTURES.items()},C33=C33,C50=C50,R17=R17,
        nesting='T3 is nested in T1 (comparison, not independent replication)',structural_zero_comparators=['T1','T2','T3']))
    json_write(out/'candidate_registry.json',dict(candidates=CANDIDATES,references=REFERENCES,original59=[hname(h) for h in original59],
        RI31_note='RI membership always comes from the original 59-head candidate JSON; R(B)=B intersect original59',
        in_original59={h:hid(h) in set(original59) for h in CANDIDATES},control_rosters=plan['control_rosters']))
    stage_a(out,plan).write(out/'stage_a.json');stage_a(out,plan,True).write(out/'stage_a_c50.json')
    return plan

def prepare_heldout(inputs,freeze_path,dataset,baseline_jsonl,out,tok):
    """Seal the 87 held-out families only after the freeze manifest exists and validates."""
    plan=json_read(inputs/'plan.json');freeze=json_read(freeze_path)
    from s45_freeze import validate_freeze
    validate_freeze(inputs,freeze)
    if out.exists():raise FileExistsError('Held-out inputs already exist')
    if plan['heldout']['dataset_hash'] and digest(dataset)!=plan['heldout']['dataset_hash']:raise ValueError('Dataset changed since discovery freeze')
    rows=dataset_rows(dataset,'validation');fams=working_families(list(read_lines(baseline_jsonl)))
    fams=[f for f in fams if any(r['family']==f for r in rows)]
    if len(fams)!=87 or set(fams)&set(plan['families']):raise ValueError(f'Expected 87 sealed held-out families, got {len(fams)}')
    pairs=[build_pair(tok,annotate_row(tok,b),annotate_row(tok,c),'heldout') for b,c in twins([r for r in rows if r['family'] in fams])]
    if len(pairs)!=174:raise ValueError('Expected 174 held-out pairs')
    out.mkdir(parents=True);write_lines(out/'pairs.jsonl.gz',pairs)
    json_write(out/'plan.json',dict(plan,heldout=dict(plan['heldout'],sealed=False,opened_with_freeze=digest(freeze_path)),
        families=fams,core_families=[],gate_families=fams[:3],pair_hash=digest(out/'pairs.jsonl.gz'),discovery_plan_hash=digest(inputs/'plan.json')))
    validation(out,json_read(out/'plan.json'),freeze).write(out/'validation.json',freeze_hash=digest(freeze_path))
    return json_read(out/'plan.json')

# ---------------------------------------------------------------- loading ------
def load(inputs,schedule=None):
    plan=json_read(inputs/'plan.json')
    if plan['policy']!=POLICY or digest(inputs/'pairs.jsonl.gz')!=plan['pair_hash']:raise ValueError('Changed inputs/policy')
    if plan['structures']!=STRUCTURES or plan['C33']!=C33 or plan['C50']!=C50 or plan['candidates']!=CANDIDATES:raise ValueError('Roster drift between code and frozen plan')
    pairs=list(read_lines(inputs/'pairs.jsonl.gz'))
    if {p['family'] for p in pairs}!=set(plan['families']) or len(pairs)!=2*len(plan['families']):raise ValueError('Input population')
    if schedule is None:return plan,pairs
    s=json_read(schedule)
    if s['plan_hash']!=digest(inputs/'plan.json') or s['policy']!=POLICY:raise ValueError('Schedule identity')
    if s['population']=='core':pairs=[p for p in pairs if p['family'] in plan['core_families']]
    elif s['population'] not in ['discovery','heldout']:raise ValueError('Population')
    elif s['population']=='heldout' and any(p['split']!='heldout' for p in pairs):raise ValueError('Held-out schedule on discovery data')
    for sid,st in s['states'].items():
        if state(st['live'],st['label'])['id']!=sid:raise ValueError('State identity')
    seen=set()
    for j in s['jobs']:
        if j['id'] in seen or j['state'] not in s['states']:raise ValueError('Job identity')
        seen.add(j['id'])
        if j['kind']=='behavior' and (j['baseline'] not in ['mean','donor'] or not set(j['cells'])<=set(CELLS)):raise ValueError('Behavior job')
        if j['kind'] in ['route','attachment']:
            if j['direction'] not in ['noise','restore'] or j['source']//32>=j['receiver']//32:raise ValueError('Route job')
            live=s['states'][j['state']]['live']
            if live is not None and (j['source'] not in live or j['receiver'] not in live or any(h not in live for h in j.get('live',[]))):raise ValueError('Route component not live in its background: '+j['id'])
            if 'Q' in j['channel'] and j['site']!='colon' and not j.get('live'):raise ValueError('Direct cross-position Q route')
            for p in pairs:
                if any(j['site'] not in p['cells'][c]['masks'] for c in ['x00','x10']):raise ValueError('Missing site mask')
    return plan,s,pairs

def assigned(pairs,shard,shards):
    if not 0<=shard<shards:raise ValueError('Invalid shard')
    fams=sorted({p['family'] for p in pairs});chosen=set(fams[shard::shards])
    return [p for p in pairs if p['family'] in chosen]

def record_keys(s,pairs,shard,shards):
    keys=set()
    for p in assigned(pairs,shard,shards):
        for j in s['jobs']:
            if j['kind']=='behavior':keys.update((p['id'],j['id'],c) for c in j['cells'])
            else:keys.add((p['id'],j['id'],'x00' if j['direction']=='noise' else 'x10'))
    return keys
