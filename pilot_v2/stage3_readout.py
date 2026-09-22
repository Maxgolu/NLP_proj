"""Stage 3, optional Section 1.5: controlled representation readout (Patchscopes-style).

Question: what do the causally localized early sites of the two writer heads contain?
  L17H1  -> the query-fact ``is'' token (main site) and the query-fact period (second site)
  L15H25 -> the last token of the queried child in the query fact (main site) and ``is'' (second)
For each core prompt (20 common families x 2 orders) the residual stream at the site is
captured after the writer's own block, injected into a fixed readout prompt at the same
layer, and the next-token log-probabilities of the candidate names are recorded.

Source conditions (what is injected):
  intact_clean       residual of the clean prompt
  head_site          clean prompt, writer head replaced by the corrupted donor at the site only
  head_span          clean prompt, writer head replaced from the first differing token to the site
  intact_corrupted   residual of the corrupted twin (entity swap: mothers exchanged)
  irrelevant         clean prompt, same token role in the OTHER answer-side fact (distractor)
  none               readout prompt alone (constant per readout prompt)

The readout decodes a multi-component residual; it cannot be attributed to one head except
through the intact-vs-head-replaced contrast. Decoded text is exploratory; probabilities of
the fixed candidate set are the measurement. Nothing here is a causal claim about a channel.

Subcommands (CPU unless stated):
  prepare  build the frozen plan from saved Stage-3 tables (no model, no tokenizer)
  run      GPU: gate + measurements (2 GPUs = one replica), restartable
  analyze  family-level summaries from the saved records
"""
import argparse
import collections
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
from stage3_common import *

ROOT=Path(__file__).resolve().parent
CODE_FILES=['stage3_readout.py','stage3_common.py','stage3_engine.py','stage3_measure.py','stage2_engine.py','stage1_scan.py','stage1_audit.py','model_lock_olmo2.json']

READOUT_POLICY=dict(
    version=1,seed=20260922,
    writers={'L17H1':dict(head=17*32+1,layer=17,sites=['query_is','query_period']),
             'L15H25':dict(head=15*32+25,layer=15,sites=['query_child_last','query_is'])},
    capture='output of decoder block <layer> (residual stream after the writer\'s own layer) at the site position',
    injection='replace the residual at the placeholder token of the readout prompt at the same block output',
    placeholder=' x',
    readouts={'identity':dict(prefix='cat -> cat ; 1135 -> 1135 ; hello -> hello ;',suffix=' ->'),
              'mother_of':dict(prefix='Fact about a person:',suffix='\nQuestion: Who is the mother of this person?\nAnswer:')},
    conditions=['intact_clean','head_site','head_span','intact_corrupted','irrelevant'],
    candidates='first token of " <name>" for the query mother, the other candidate mother, the remaining distinct mothers and the queried child (last)',
    gate=dict(identity_injection_tolerance=0.02,self_patch_tolerance=0.001,saved_importance_tolerance=0.05),
    population='20 common discovery families x 2 orders (the 40 common pairs); the 87 validation families are not used')


# ----------------------------------------------------------------------------- prepare (CPU)
def site_positions(rows,pair):
    """Derive site indices from the saved single-position profile rows of one pair (any head).

    rows: position_profiles.csv rows for this family/order (one head is enough; positions are
    head-independent). Returns dict site -> j, plus the first differing token and the fact
    index of the other answer-side candidate.
    """
    clean=pair['clean'];byj={int(r['j']):r for r in rows}
    js=sorted(byj)
    query_fact=None;cand_fact=None
    facts=[f for f in clean['facts'] if f['block']=='test']
    other=next(c for c in clean['candidates'] if c!=clean['gold'])
    for i,f in enumerate(facts):
        if f['head']==clean['question_entity']:query_fact=i
        elif f['tail']==other:cand_fact=i
    if query_fact is None or cand_fact is None:raise ValueError('Cannot identify query/candidate facts: '+pair['id'])
    def ents(r):return json.loads(r['entities'])
    def last_of(fact,role):
        c=[j for j in js if any(e['fact']==fact and e['role']==role and e['last'] for e in ents(byj[j]))]
        if len(c)!=1:raise ValueError(f'Ambiguous {role} span for fact {fact}: {pair["id"]}')
        return c[0]
    # The saved 'fact' index in the profile is the index within the prompt's full fact list.
    prompt_facts=clean['facts'];offset=[k for k,f in enumerate(prompt_facts) if f['block']=='test'][0]
    q=query_fact+offset;c=cand_fact+offset
    def after(j,text):
        k=j+1
        if k not in byj or str(byj[k]['text']).strip()!=text:raise ValueError(f'Expected {text!r} after position {j}: {pair["id"]}')
        return k
    def period_after(j):
        for k in js:
            if k>j and byj[k]['punctuation'] in (True,'True') and not ents(byj[k]):return k
        raise ValueError('No period after child: '+pair['id'])
    q_mother_last=last_of(q,'target');q_child_last=last_of(q,'source')
    c_mother_last=last_of(c,'target');c_child_last=last_of(c,'source')
    sites=dict(query_is=after(q_mother_last,'is'),query_period=period_after(q_child_last),query_child_last=q_child_last)
    controls=dict(query_is=after(c_mother_last,'is'),query_period=period_after(c_child_last),query_child_last=c_child_last)
    return dict(sites=sites,controls=controls,first_diff=js[0],query_fact=q,candidate_fact=c)


def prepare(inputs,run_dir,out):
    pairs=[p for p in read_lines(inputs/'pairs.jsonl.gz') if p['exact_all']]
    if len(pairs)!=40:raise ValueError('Expected the 40 common pairs')
    profile=collections.defaultdict(list);saved={}
    with (run_dir/'analysis/position_profiles.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key=(int(r['family']),int(r['order']))
            if r['head']=='L17H1':profile[key].append(r)
            if r['head'] in READOUT_POLICY['writers']:saved[(r['head'],int(r['family']),int(r['order']),int(r['j']))]=float(r['importance'])
    items=[]
    for p in pairs:
        key=(p['family'],p['order']);pos=site_positions(profile[key],p)
        clean=p['clean'];facts=[f for f in clean['facts'] if f['block']=='test']
        mothers=[f['tail'] for f in facts];other=next(c for c in clean['candidates'] if c!=clean['gold'])
        # Distinct names only: in chain families the queried child is itself a mother elsewhere.
        names=dict(query_mother=clean['gold'],other_candidate=other,
                   other_mothers=[m for m in mothers if m not in (clean['gold'],other,clean['question_entity'])],
                   query_child=clean['question_entity'])
        for w,spec in READOUT_POLICY['writers'].items():
            for site in spec['sites']:
                j=pos['sites'][site];cj=pos['controls'][site]
                items.append(dict(pair_id=p['id'],family=p['family'],order=p['order'],writer=w,head=spec['head'],layer=spec['layer'],
                                  site=site,j=j,control_j=cj,first_diff=pos['first_diff'],names=names,
                                  saved_site_importance=saved.get((w,p['family'],p['order'],j))))
    for it in items:
        if it['saved_site_importance'] is None:raise ValueError('Saved single-position importance missing for '+it['pair_id'])
    plan=dict(policy=READOUT_POLICY,items=items,pairs=[p['id'] for p in pairs],
              source_run=run_dir.name,source_hashes={'position_profiles.csv':digest(run_dir/'analysis/position_profiles.csv'),
              'pairs.jsonl.gz':digest(inputs/'pairs.jsonl.gz')},model=json_read(inputs/'plan.json')['model'],
              forwards=dict(source_captures=len(pairs)*(2+2*len(READOUT_POLICY['writers'])),
                            readouts=len(items)*len(READOUT_POLICY['conditions'])*len(READOUT_POLICY['readouts'])+len(READOUT_POLICY['readouts'])))
    out.mkdir(parents=True,exist_ok=False)
    json_write(out/'readout_plan.json',plan)
    import shutil;shutil.copy(inputs/'pairs.jsonl.gz',out/'pairs.jsonl.gz')
    print(json.dumps(dict(items=len(items),**plan['forwards']),indent=1))


# ----------------------------------------------------------------------------- GPU helpers
class Readout:
    def __init__(self,e):
        self.e=e;t=e.t;tok=e.tok
        ph=tok.encode(READOUT_POLICY['placeholder'],add_special_tokens=False)
        if len(ph)!=1:raise ValueError('Placeholder must be one token')
        self.placeholder=ph[0];self.prompts={}
        for name,r in READOUT_POLICY['readouts'].items():
            pre=tok.encode(r['prefix'],add_special_tokens=False);suf=tok.encode(r['suffix'],add_special_tokens=False)
            full=tok.encode(r['prefix']+READOUT_POLICY['placeholder']+r['suffix'],add_special_tokens=False)
            if full!=pre+ph+suf:raise ValueError('Readout prompt tokenization is not compositional: '+name)
            self.prompts[name]=dict(ids=t.tensor([full],device=e.device),pos=len(pre))

    def hidden(self,ids,layer,positions,patch=None):
        """Residual stream at the output of block `layer` at `positions`; optional head patch context."""
        e=self.e;t=e.t;store={}
        def hook(mod,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            store['h']=h[0,positions].detach().float().cpu()
        handle=e.layers[layer].register_forward_hook(hook)
        try:
            ctx=patch if patch is not None else _null()
            with ctx,t.no_grad():e.model(ids,use_cache=False)
        finally:handle.remove()
        return store['h']

    def inject(self,name,layer,vector,candidate_ids,top=5):
        e=self.e;t=e.t;p=self.prompts[name]
        def hook(mod,inp,out):
            h=out[0] if isinstance(out,tuple) else out
            h=h.clone();h[0,p['pos']]=vector.to(h.device,h.dtype)
            return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
        handle=e.layers[layer].register_forward_hook(hook) if vector is not None else None
        try:
            with t.no_grad():logits=e.model(p['ids'],use_cache=False).logits[0,-1].float()
        finally:
            if handle:handle.remove()
        lp=t.log_softmax(logits,-1).cpu();tops=t.topk(lp,top)
        return dict(candidates=[float(lp[c]) for c in candidate_ids],top_tokens=[e.tok.decode([int(i)]) for i in tops.indices],
                    top_logprobs=[float(v) for v in tops.values])


import contextlib
@contextlib.contextmanager
def _null():yield


def first_token(tok,name):
    ids=tok.encode(' '+name,add_special_tokens=False)
    return ids[0]


def run_worker(out):
    from stage1_scan import load_model
    from stage3_engine import Stage3Engine
    from stage3_measure import checked_capture
    import transformers
    m=json_read(out/'manifest.json');plan=json_read(Path(m['plan']))
    if plan['policy']!=READOUT_POLICY:raise ValueError('Plan policy and code differ')
    pairs={p['id']:p for p in read_lines(Path(m['plan']).parent/'pairs.jsonl.gz')}
    def state(status,**kw):json_write(out/'state.json',dict(status=status,**kw))
    state('loading model');start=time.monotonic();t,tok,model,lock=load_model()
    if lock!=plan['model']:raise ValueError('Loaded model lock differs')
    e=Stage3Engine(t,tok,model);ro=Readout(e)
    json_write(out/'runtime.json',dict(torch=t.__version__,transformers=transformers.__version__,load_seconds=time.monotonic()-start,
                                       attention=model.config._attn_implementation,device_map={k:str(v) for k,v in getattr(model,'hf_device_map',{}).items()}))
    # ---------------- gate ----------------
    state('gating');g=READOUT_POLICY['gate'];report=dict(passed=True,checks=[])
    for name,p in ro.prompts.items():
        for layer in sorted({it['layer'] for it in plan['items']}):
            own=ro.hidden(p['ids'],layer,[p['pos']])[0]
            base=ro.inject(name,layer,None,[0]);back=ro.inject(name,layer,own,[0])
            drift=max(abs(a-b) for a,b in zip(base['top_logprobs'],back['top_logprobs']))
            if base['top_tokens']!=back['top_tokens'] or drift>g['identity_injection_tolerance']:
                raise ValueError(f'Identity injection changed the readout ({name}, layer {layer}): {drift}')
            report['checks'].append(dict(check='identity_injection',readout=name,layer=layer,drift=drift))
    probe=[it for it in plan['items'] if it['site']==READOUT_POLICY['writers'][it['writer']]['sites'][0]]
    probe=[probe[0],probe[len(probe)//2],probe[-1]]
    for it in probe:
        pair=pairs[it['pair_id']];spec=e.encode(pair);n=spec['n'];j=it['j'];h=it['head'];layer=it['layer']
        clean=checked_capture(e,spec['clean'][:,:n],[h],[j]);corr=checked_capture(e,spec['corr'][:,:n],[h],[j])
        z=clean['heads'][h]['z'];z1=corr['heads'][h]['z']
        intact=ro.hidden(spec['clean'][:,:n],layer,[j])[0]
        selfp=ro.hidden(spec['clean'][:,:n],layer,[j],e.vector_patch(h,[j],z))[0]
        d=float((intact-selfp).abs().max())
        if d>g['self_patch_tolerance']:raise ValueError(f'Self patch changed the residual: {d}')
        base=e.readout(spec['clean'],spec['g'],spec['d']);patched=e.patched(spec,h,[j],z1)
        imp=float(base[0]-patched[0]);saved=it['saved_site_importance']
        if abs(imp-saved)>g['saved_importance_tolerance']:raise ValueError(f'Saved single-position importance not replicated for {it["pair_id"]} {it["writer"]}: {imp} vs {saved}')
        report['checks'].append(dict(check='self_patch_and_saved_importance',pair=it['pair_id'],writer=it['writer'],self_patch_drift=d,importance=imp,saved=saved))
    json_write(out/'gate.json',report)
    if m['gate_only']:state('gate complete (gate-only run)');return
    # ---------------- measurements ----------------
    path=out/'readout_records.jsonl.gz';done=set()
    if path.exists():
        for r in read_lines(path):done.add((r['pair_id'],r['writer'],r['site']))
    records=[]
    def flush():
        nonlocal records
        if not records:return
        existing=list(read_lines(path)) if path.exists() else []
        write_lines(path,existing+records);records=[]
    by_pair=collections.defaultdict(list)
    for it in plan['items']:by_pair[it['pair_id']].append(it)
    for pi,(pid,items) in enumerate(by_pair.items()):
        if all((it['pair_id'],it['writer'],it['site']) in done for it in items):continue
        state('measuring',pair=pi+1,total=len(by_pair));pair=pairs[pid];spec=e.encode(pair);n=spec['n']
        names=items[0]['names'];cand_names=[names['query_mother'],names['other_candidate'],*names['other_mothers'],names['query_child']]
        cand_ids=[first_token(tok,x) for x in cand_names]
        positions=sorted({it['j'] for it in items}|{it['control_j'] for it in items});layers=sorted({it['layer'] for it in items})
        vec={}
        for layer in layers:
            hc=ro.hidden(spec['clean'][:,:n],layer,positions);hr=ro.hidden(spec['corr'][:,:n],layer,positions)
            for k,j in enumerate(positions):vec[('intact_clean',layer,j)]=hc[k];vec[('intact_corrupted',layer,j)]=hr[k]
        for it in items:
            key=(it['pair_id'],it['writer'],it['site'])
            if key in done:continue
            h=it['head'];layer=it['layer'];j=it['j']
            corr=checked_capture(e,spec['corr'][:,:n],[h],list(range(it['first_diff'],j+1)));z1=corr['heads'][h]['z']
            site_vec=ro.hidden(spec['clean'][:,:n],layer,[j],e.vector_patch(h,[j],z1[-1:]))[0]
            span_vec=ro.hidden(spec['clean'][:,:n],layer,[j],e.vector_patch(h,list(range(it['first_diff'],j+1)),z1))[0]
            sources=dict(intact_clean=vec[('intact_clean',layer,j)],head_site=site_vec,head_span=span_vec,
                         intact_corrupted=vec[('intact_corrupted',layer,j)],irrelevant=vec[('intact_clean',layer,it['control_j'])])
            for cond,v in sources.items():
                for name in ro.prompts:
                    r=ro.inject(name,layer,v,cand_ids)
                    records.append(dict(pair_id=pid,family=it['family'],order=it['order'],writer=it['writer'],site=it['site'],j=j,
                                        condition=cond,readout=name,candidate_names=cand_names,candidate_logprobs=r['candidates'],
                                        top_tokens=r['top_tokens'],top_logprobs=r['top_logprobs'],
                                        head_site_vs_intact_norm=float((site_vec-vec[('intact_clean',layer,j)]).norm()) if cond=='head_site' else None))
            for name in ro.prompts:
                none=ro.inject(name,layer,None,cand_ids)  # readout prompt alone; constant logits, pair-specific candidates
                records.append(dict(pair_id=pid,family=it['family'],order=it['order'],writer=it['writer'],site=it['site'],j=j,condition='none',
                                    readout=name,candidate_names=cand_names,candidate_logprobs=none['candidates'],
                                    top_tokens=none['top_tokens'],top_logprobs=none['top_logprobs']))
        flush()
    flush()
    json_write(out/'done.json',dict(complete=True,records=sum(1 for _ in read_lines(path))));state('complete')


def controller(a):
    plan_path=a.plan.resolve();plan=json_read(plan_path)
    if plan['policy']!=READOUT_POLICY:raise ValueError('Plan policy and code differ')
    import torch
    from stage1_scan import RUNS
    if torch.cuda.device_count()!=2:raise ValueError('Allocate exactly 2 GPUs (one model replica)')
    stable=dict(plan_hash=digest(plan_path),code_hashes={n:digest(ROOT/n) for n in CODE_FILES},gate_only=a.gate_only)
    identity=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    m=dict(**stable,identity=identity,plan=str(plan_path),model=plan['model'])
    if not re.fullmatch('[A-Za-z0-9_-]+',a.name):raise ValueError('Invalid run name')
    out=RUNS/a.name
    if out.exists():
        if not a.resume or json_read(out/'manifest.json')!=m:raise ValueError('Resume requires identical plan, code, mode and name')
    else:out.mkdir(parents=True);json_write(out/'manifest.json',m)
    with run_lock(out):
        (out/'done.json').unlink(missing_ok=True)
        run_worker(out)
    print('READOUT RUN FINISHED:',out,flush=True)


# ----------------------------------------------------------------------------- analyze (CPU)
def analyze(out):
    plan=json_read(Path(json_read(out/'manifest.json')['plan']))
    if not json_read(out/'done.json').get('complete'):raise ValueError('Run incomplete')
    if not json_read(out/'gate.json').get('passed'):raise ValueError('Gate did not pass')
    recs=list(read_lines(out/'readout_records.jsonl.gz'))
    expected=len(plan['items'])*(len(READOUT_POLICY['conditions'])+1)*len(READOUT_POLICY['readouts'])
    if len(recs)!=expected:raise ValueError(f'Expected {expected} records, found {len(recs)}')
    res=out/'analysis';res.mkdir(exist_ok=True)
    rows=[]
    for r in recs:
        lp=r['candidate_logprobs'];names=r['candidate_names']
        qm,oc=lp[0],lp[1];others=lp[2:-1];child=lp[-1]
        top=int(np.argmax(lp));child_index=len(lp)-1
        rows.append(dict(pair_id=r['pair_id'],family=r['family'],order=r['order'],writer=r['writer'],site=r['site'],condition=r['condition'],
                         readout=r['readout'],lp_query_mother=qm,lp_other_candidate=oc,lp_other_mothers_mean=float(np.mean(others)),lp_query_child=child,
                         contrast_candidates=qm-oc,contrast_all_mothers=qm-float(np.mean([oc,*others])),query_mother_is_top_candidate=top==0,
                         other_candidate_is_top_candidate=top==1,query_child_is_top_candidate=top==child_index,
                         mother_in_top5=any(len(t.strip())>=2 and names[0].startswith(t.strip()) for t in r['top_tokens']),
                         top_tokens='|'.join(x.replace('\n','\\n') for x in r['top_tokens'])))
    with (res/'readout_events.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    keys=lambda r:(r['writer'],r['site'],r['readout'],r['condition'])
    groups=collections.defaultdict(list)
    for r in rows:groups[keys(r)].append(r)
    for k,g in sorted(groups.items()):
        fam=collections.defaultdict(list)
        for r in g:fam[r['family']].append(r)
        def fmean(field):return float(np.mean([np.mean([x[field] for x in v]) for v in fam.values()]))
        summary.append(dict(writer=k[0],site=k[1],readout=k[2],condition=k[3],families=len(fam),prompts=len(g),
                            contrast_candidates=fmean('contrast_candidates'),contrast_all_mothers=fmean('contrast_all_mothers'),
                            lp_query_mother=fmean('lp_query_mother'),lp_other_candidate=fmean('lp_other_candidate'),lp_query_child=fmean('lp_query_child'),
                            frac_query_mother_top=fmean('query_mother_is_top_candidate'),frac_other_candidate_top=fmean('other_candidate_is_top_candidate'),
                            frac_query_child_top=fmean('query_child_is_top_candidate')))
    with (res/'readout_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    # Paired contrasts per family: intact minus head-replaced, and entity swap as the reference shift.
    paired=[]
    look={(r['pair_id'],r['writer'],r['site'],r['readout'],r['condition']):r for r in rows}
    for (pid,w,s,ro_) in sorted({(r['pair_id'],r['writer'],r['site'],r['readout']) for r in rows}):
        get=lambda c:look[(pid,w,s,ro_,c)]
        ic=get('intact_clean');paired.append(dict(pair_id=pid,family=ic['family'],order=ic['order'],writer=w,site=s,readout=ro_,
            intact_contrast=ic['contrast_candidates'],
            head_site_shift=get('head_site')['contrast_candidates']-ic['contrast_candidates'],
            head_span_shift=get('head_span')['contrast_candidates']-ic['contrast_candidates'],
            entity_swap_shift=get('intact_corrupted')['contrast_candidates']-ic['contrast_candidates'],
            irrelevant_shift=get('irrelevant')['contrast_candidates']-ic['contrast_candidates'],
            none_contrast=get('none')['contrast_candidates']))
    with (res/'readout_paired.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
    agg=[]
    pg=collections.defaultdict(list)
    for r in paired:pg[(r['writer'],r['site'],r['readout'])].append(r)
    for k,g in sorted(pg.items()):
        fam=collections.defaultdict(list)
        for r in g:fam[r['family']].append(r)
        fm=lambda field:np.array([np.mean([x[field] for x in v]) for v in fam.values()])
        swap=fm('entity_swap_shift');site=fm('head_site_shift');span=fm('head_span_shift');irr=fm('irrelevant_shift')
        agg.append(dict(writer=k[0],site=k[1],readout=k[2],families=len(fam),
                        intact_contrast=float(fm('intact_contrast').mean()),none_contrast=float(fm('none_contrast').mean()),
                        entity_swap_shift=float(swap.mean()),entity_swap_negative_fraction=float((swap<0).mean()),
                        head_site_shift=float(site.mean()),head_site_negative_fraction=float((site<0).mean()),
                        head_span_shift=float(span.mean()),head_span_negative_fraction=float((span<0).mean()),
                        irrelevant_shift=float(irr.mean()),
                        head_span_over_swap=float(span.mean()/swap.mean()) if abs(swap.mean())>1e-9 else None,
                        note='shift = contrast(condition) - contrast(intact clean); contrast = logprob(query mother) - logprob(other candidate); negative shift = toward the swapped mother'))
    with (res/'readout_family_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(agg[0]));w.writeheader();w.writerows(agg)
    json_write(out/'summary.json',dict(complete=True,records=len(recs),items=len(plan['items']),
               interpretation='Descriptive readout of a multi-component residual; head attribution only via intact vs head-replaced contrasts; not a channel or circuit claim.'))
    print('READOUT ANALYSIS WRITTEN:',res,flush=True)
    for a in agg:print({k:(round(v,3) if isinstance(v,float) else v) for k,v in a.items() if k!='note'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--inputs',type=Path,default=ROOT.parent/'results/stage3_inputs_v1')
    p.add_argument('--run',type=Path,default=ROOT.parent/'results/stage3_v1');p.add_argument('--out',type=Path,default=ROOT.parent/'results/stage3_readout_inputs_v1')
    r=sub.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--name',required=True)
    r.add_argument('--gate-only',action='store_true');r.add_argument('--resume',action='store_true')
    z=sub.add_parser('analyze');z.add_argument('run_dir',type=Path)
    a=ap.parse_args()
    if a.cmd=='prepare':prepare(a.inputs,a.run,a.out)
    elif a.cmd=='run':controller(a)
    else:analyze(a.run_dir)
