"""Frozen S4.2 inputs and declarative interventions; CPU only, no model loading."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,write_lines,span_tokens
from stage4_plan import hid

POLICY=dict(version=1,seed=20260923,tau=.1,coherent_fraction=.7,heterogeneous_fraction=.2,
    bootstrap_draws=20000,g1_common_cap=128,g1_extension_contrasts=16,
    mean_min_families=10,mean_scope='test facts/question/colon; demonstrations and answer prefix live',
    mean_weighting='events within family, then equal families; both orders and variants',
    mean_evaluation='leave-one-family-out',
    K_rule='smallest prefix with symmetric donor clean-minus-corrupt gap fraction in [0.3,0.8]; otherwise closest to 0.55',
    mean_reverse='corrupted-recipient mean replacement; positive restore sign is not literal clean restoration',
    split_rule='retained group interaction/conditional effect: prior-sign halves and pair of largest absolute conditional effects; within original cap',
    sensitivity_rule='repeat selected extended contrasts in both directions, including those not retained after extension; other core-only G1 claims remain provisional',
    scope='S4.2 G1/blocking/G2/G3 only; no mediated search or circuit/heldout execution')

def atom(h,site='colon',donor='other'):
    return dict(head=hid(h) if isinstance(h,str) else int(h),site=site,donor=donor)

def config(atoms=()):
    aa=sorted({(a['head'],a['site'],a['donor']) for a in atoms})
    if any(d not in ['other','self'] for h,s,d in aa):raise ValueError('Bad donor policy')
    if any(not 0<=h<1024 for h,s,d in aa):raise ValueError('Head outside model')
    # Different roles of the same head are allowed only with the same donor;
    # actual overlapping token positions are checked again by the engine.
    if any(len({d for hh,s,d in aa if hh==h})>1 for h,s,d in aa):raise ValueError('Conflicting donors for a head')
    identity=json.dumps(aa,separators=(',',':'))
    return dict(id='intact' if not aa else hashlib.sha256(identity.encode()).hexdigest()[:20],
                atoms=[dict(head=h,site=s,donor=d) for h,s,d in aa],observe=False,directions=[])

class Registry:
    def __init__(self):self.configs={};self.contrasts=[]
    def add(self,atoms=(),directions=('noise',),observe=False):
        c=config(atoms);old=self.configs.setdefault(c['id'],c)
        if old['atoms']!=c['atoms']:raise ValueError('Configuration hash collision')
        old['directions']=sorted(set(old['directions'])|set(directions));old['observe']|=observe
        return c['id']
    def contrast(self,name,terms,panel,**kw):
        cc=collections.defaultdict(float)
        for cid,weight in terms:cc[cid]+=weight
        self.contrasts.append(dict(id=name,panel=panel,terms=[[c,w] for c,w in sorted(cc.items()) if w],**kw))

def group_atoms(heads,site='colon',donor='other'):return [atom(h,site,donor) for h in heads]

def mean_roles(row,masks):
    """Keys contain syntax/role/slot only, never query relevance or name identity."""
    offsets=row['offsets'];text=row['prompt'];n=len(offsets);keys=[None]*n
    def buckets(pos):return ['single'] if len(pos)==1 else ['first']+['interior']*(len(pos)-2)+['last']
    def assign(pos,slot,role,span=False):
        bs=buckets(pos) if span else ['fixed']*len(pos)
        for j,b in zip(pos,bs):
            if keys[j] is not None:raise ValueError('Overlapping mean roles')
            keys[j]=[slot,role,b]
    for fi in range(4):
        for role in ['mother','child']:assign(masks[f'fact{fi}_{role}'],str(fi),role,True)
        for k,j in enumerate(masks[f'fact{fi}_template']):assign([j],str(fi),f'fact_template_{k}')
        assign(masks[f'fact{fi}_punctuation'],str(fi),'fact_period')
    qstart=text.rfind('Question:');a=text.index(row['query_source'],qstart)
    qname=span_tokens(offsets,(a,a+len(row['query_source'])))
    assign(qname,'question','question_name',True)
    qrest=[j for j in masks['question'] if keys[j] is None]
    for k,j in enumerate(qrest):assign([j],'question',f'question_template_{k}')
    assign(masks['colon'],'answer','colon')
    # Only fixed syntax remains: separators and Answer label. All name spans were assigned.
    for j,(a,b) in enumerate(offsets):
        if b>row['test_start'] and keys[j] is None:
            assign([j],'structure','fixed_token_'+str(row['token_ids'][j]))
    if any(keys[j] is None for j,(a,b) in enumerate(offsets) if b>row['test_start']):raise ValueError('Missing test role')
    return keys

def key_levels(key):
    slot,role,bucket=key
    return list(dict.fromkeys(['|'.join([slot,role,bucket]),'|'.join(['*',role,bucket]),'|'.join(['*',role,'*']),'*|*|*']))

def prepare(stage4,stage3_inputs,stage3_runs,proposal,out):
    if out.exists():raise FileExistsError('Frozen inputs already exist; choose a new version')
    old=json_read(stage4/'plan.json');pp=json_read(proposal);pairs=list(read_lines(stage4/'pairs.jsonl.gz'))
    if digest(stage4/'pairs.jsonl.gz')!=old['pair_hash']:raise ValueError('S4.1 pair hash')
    if len(pairs)!=178 or {p['family'] for p in pairs}!=set(old['design']['discovery_families']):raise ValueError('Population mismatch')
    prompts={x['id']:x for x in read_lines(stage3_inputs/'prompts.jsonl.gz')}
    prior=json_read(stage3_runs/'manifest.json')
    if prior['input_hash']!=digest(stage3_inputs/'plan.json') or prior['model']!=old['model']:raise ValueError('Stage-3 provenance mismatch')
    if not json_read(stage3_runs/'gate_passed.json')['passed']:raise ValueError('Prior global gate failed')
    singles=sorted({hid(x[0]) for x in pp['group_measurements'] if len(x)==1}|{hid('L26H31')})
    provenance={str(stage4/'plan.json'):digest(stage4/'plan.json'),str(proposal):digest(proposal),
                str(stage3_runs/'manifest.json'):digest(stage3_runs/'manifest.json')}
    for p in pairs:
        if any(p[k]['split']!='discovery' for k in ['clean','corr']):raise ValueError('Heldout leakage')
        p['mean_keys']=mean_roles(prompts[p['clean']['id']],p['masks'])
        if p['mean_keys']!=mean_roles(prompts[p['corr']['id']],p['masks']):raise ValueError('Mean role mismatch')
        p['saved_single']={}
        for h,v in p['saved_P'].items():p['saved_single'][f'{h}:all']=dict(effect=-v,result=None,baseline_margin=p['saved_baselines'][0],origin='saved_stage2_P',
            limitation='Historical Scope-P reference stores margin effect, not individual patched logits')
        for rank,heads in enumerate(prior['shards']):
            need=set(singles)&set(heads)
            if not need:continue
            if not json_read(stage3_runs/f'done_{rank}.json')['complete'] or not json_read(stage3_runs/f'gate_{rank}.json')['passed']:raise ValueError('Prior shard incomplete')
            for ci in range((len(heads)+prior['chunk_size']-1)//prior['chunk_size']):
                hs=heads[ci*prior['chunk_size']:(ci+1)*prior['chunk_size']]
                if not need.intersection(hs):continue
                path=stage3_runs/f'replica_{rank}'/'pairs'/f"family_{p['family']:03d}_order{p['order']}_chunk{ci:02d}.npz"
                mark=json_read(path.with_name(path.name+'.ok.json'))
                if mark['sha256']!=digest(path):raise ValueError('Prior chunk hash mismatch')
                if mark['identity']!=prior['identity']+f':{rank}:'+p['id']+':'+str(ci):raise ValueError('Prior chunk identity')
                with np.load(path,allow_pickle=False) as z:
                    if str(z['pair_id'])!=p['id'] or z['heads'].tolist()!=hs:raise ValueError('Prior pair/head mismatch')
                    if int(z['n'])!=len(p['original_ids']['clean']):raise ValueError('Prior mask mismatch')
                    for j,h in enumerate(hs):
                        if h not in need:continue
                        f=z['scopeF_readout'][j];b=p['saved_baselines'][0] if z['scopeF_reused'][j] else float(z['baseline'][0])
                        p['saved_single'][f'{h}:colon']=dict(effect=float(b-f[0]),result=dict(margin=float(f[0]),clean_logit=float(f[1]),corr_logit=float(f[2])),
                            baseline_margin=b,origin='saved_stage3_F',source_sha256=mark['sha256'])
        if any(f'{h}:colon' not in p['saved_single'] for h in singles):raise ValueError('Missing reusable F measurement')
    out.mkdir(parents=True);write_lines(out/'pairs.jsonl.gz',pairs)
    keys=sorted({k for p in pairs for key in p['mean_keys'] if key for k in key_levels(key)})
    plan=dict(policy=POLICY,model=old['model'],proposal=pp,core_families=old['design']['core_families'],
        gate_families=old['gate_families'],families=sorted({p['family'] for p in pairs}),mean_keys=keys,
        pair_hash=digest(out/'pairs.jsonl.gz'),source_hashes=provenance)
    json_write(out/'plan.json',plan)
    reg=initial(plan);write_schedule(out/'initial.json',out,reg,'core','donor','initial')
    return plan

def add_g1(reg,plan):
    for ms in plan['proposal']['group_measurements']:reg.add(group_atoms(ms))
    for g in plan['proposal']['groups']:
        ms=g['members'];whole=reg.add(group_atoms(ms));singles=[reg.add(group_atoms([h])) for h in ms]
        reg.contrast('G1:'+g['id']+':whole',[(whole,1)],'G1',group=g['id'],kind='whole')
        reg.contrast('G1:'+g['id']+':interaction',[(whole,1)]+[(x,-1) for x in singles],'G1',group=g['id'],kind='interaction')
        for h in ms:
            minus=reg.add(group_atoms([x for x in ms if x!=h]))
            reg.contrast('G1:'+g['id']+':conditional:'+h,[(whole,1),(minus,-1)],'G1',group=g['id'],kind='conditional',head=hid(h))

def initial(plan):
    pp=plan['proposal'];r=Registry();r.add((),('noise','restore'),True);add_g1(r,plan)
    for w in pp['blocking']:
        for site in w['source_masks']:
            source=[atom(w['source'],site)];clamp=group_atoms(w['receivers'],donor='self')
            src=r.add(source,('noise','restore'));blocked=r.add(source+clamp,('noise','restore'));selfc=r.add(clamp,('noise','restore'))
            tag='block:'+w['source']+':'+site
            r.contrast(tag,[(src,1),(blocked,-1)],'blocking',kind='blocking',source=hid(w['source']),site=site)
            r.contrast(tag+':source',[(src,1)],'blocking',kind='source',source=hid(w['source']),site=site)
            r.contrast(tag+':self',[(selfc,1)],'control',kind='self_clamp')
    for i in range(1,5):r.add(group_atoms(pp['g2']['K_prefix_order'][:i]),('noise','restore'))
    g3=pp['g3'];backgrounds=g3['backgrounds']+[g3['downstream_attention_null']]
    intact_h=r.add([atom(g3['head'])],observe=True)
    for i,hs in enumerate(backgrounds):
        bg=r.add(group_atoms(hs),observe=True);ab=r.add(group_atoms(hs)+[atom(g3['head'])],observe=True)
        r.contrast(f'G3:{i}:conditional',[(ab,1),(bg,-1)],'G3',kind='conditional',background=hs)
        r.contrast(f'G3:{i}:change',[(ab,1),(bg,-1),(intact_h,-1)],'G3',kind='change',background=hs)
    return r

def conditional(plan,K):
    r=Registry();pp=plan['proposal'];r.add();ki=r.add(group_atoms(K))
    for name in pp['g2']['ri31']:
        bg=[h for h in K if h!=name];b=r.add(group_atoms(bg));a=r.add(group_atoms(bg)+[atom(name,'all')]);single=r.add([atom(name,'all')])
        r.contrast('G2:'+name+':conditional',[(a,1),(b,-1)],'G2',kind='conditional',head=hid(name))
        r.contrast('G2:'+name+':change',[(a,1),(b,-1),(single,-1)],'G2',kind='change',head=hid(name))
    cohorts=[('positive',pp['g2']['positive']),('negative',pp['g2']['negative']),('residual23',pp['g2']['residual23'])]
    cohorts +=[(f'reference{i+1}',h) for i,h in enumerate(pp['g2']['reference_cohorts'])]
    for name,hs in cohorts:
        intact=r.add(group_atoms(hs,'all'));joint=r.add(group_atoms(K)+group_atoms(hs,'all'))
        r.contrast('G2group:'+name+':intact',[(intact,1)],'G2group',kind='intact',cohort=name)
        r.contrast('G2group:'+name+':weakened',[(joint,1),(ki,-1)],'G2group',kind='weakened',cohort=name)
        r.contrast('G2group:'+name+':change',[(joint,1),(ki,-1),(intact,-1)],'G2group',kind='change',cohort=name)
    return r

def write_schedule(path,inputs,reg,population,mode,phase,**kw):
    json_write(path,dict(plan_hash=digest(inputs/'plan.json'),policy=POLICY,population=population,mode=mode,phase=phase,
        configs=sorted(reg.configs.values(),key=lambda c:c['id']),contrasts=reg.contrasts,**kw))

def load(inputs,schedule=None):
    plan=json_read(inputs/'plan.json')
    if plan['policy']!=POLICY or digest(inputs/'pairs.jsonl.gz')!=plan['pair_hash']:raise ValueError('Changed inputs/policy')
    pairs=list(read_lines(inputs/'pairs.jsonl.gz'))
    if len(pairs)!=178 or {p['family'] for p in pairs}!=set(plan['families']):raise ValueError('Input population')
    if schedule is None:return plan,pairs
    s=json_read(schedule)
    if s['plan_hash']!=digest(inputs/'plan.json') or s['policy']!=POLICY:raise ValueError('Schedule identity')
    if s['population'] not in ['core','all'] or s['mode'] not in ['donor','mean']:raise ValueError('Population/baseline')
    if s['population']=='core':pairs=[p for p in pairs if p['family'] in plan['core_families']]
    ids=set()
    for c in s['configs']:
        if config(c['atoms'])['id']!=c['id'] or c['id'] in ids:raise ValueError('Invalid/duplicate config')
        ids.add(c['id'])
        if not set(c['directions'])<=set(['noise','restore']) or not c['directions']:raise ValueError('Direction')
        for a in c['atoms']:
            if any(a['site'] not in p['masks'] for p in pairs):raise ValueError('Unknown site')
    if len({c['id'] for c in s['contrasts']})!=len(s['contrasts']):raise ValueError('Duplicate contrast')
    for c in s['contrasts']:
        if any(cid not in ids for cid,w in c['terms']):raise ValueError('Missing contrast component')
        if any(not np.isfinite(w) for cid,w in c['terms']):raise ValueError('Invalid contrast coefficient')
    return plan,s,pairs

def assigned(pairs,shard,shards):
    if not 0<=shard<shards:raise ValueError('Invalid shard')
    families=sorted({p['family'] for p in pairs});selected=set(families[shard::shards])
    return [p for p in pairs if p['family'] in selected]

def reusable(p,c,direction,mode):
    if mode!='donor' or direction!='noise' or c['observe'] or len(c['atoms'])!=1:return None
    a=c['atoms'][0]
    return p['saved_single'].get(f"{a['head']}:{a['site']}") if a['donor']=='other' else None
