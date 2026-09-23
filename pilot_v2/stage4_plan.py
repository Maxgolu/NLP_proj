"""CPU-only immutable Stage-4.1 input/configuration compiler."""
import csv
import json
import re
import shutil
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,write_lines,span_tokens

POLICY=dict(version=1, protocol='stage4_design_v2', seed=20260923,
            metric='first divergent clean-minus-corrupt logits; prefix teacher forced',
            endpoint='one receiver head at original final colon; K/V at source positions',
            freeze='post-normalization residual branches; source attention live',
            tau=.1, coherent_fraction=.7, heterogeneous_fraction=.2,
            gate_inert=.001, gate_replication=.05, gate_reconstruction=.02,
            full_route_cap=96, controls_top=24)

def hid(name):
    m=re.fullmatch(r'L(\d+)H(\d+)',name)
    if not m: raise ValueError(name)
    return int(m[1])*32+int(m[2])

def route(source,site,kind,receiver,channel='',panel='',**extra):
    d=dict(source=source,site=site,kind=kind,receiver=receiver,channel=channel,panel=panel,**extra)
    d['id']=f'{source}:{site}:{kind}:{receiver}:{channel}'
    return d

def masks(row):
    text=row['prompt']; offsets=row['offsets']; n=len(offsets)
    facts=[f for f in row['facts'] if f['block']=='test']
    q=next(i for i,f in enumerate(facts) if f['head']==row['query_source'])
    other=next(i for i,f in enumerate(facts) if i!=q and f['tail'] in row['candidates'])
    result={'colon':[n-1], 'all':list(range(n))}; mothers=[];children=[];templates=[];punct=[]
    def ix(a,b): return span_tokens(offsets,(a,b))
    for i,f in enumerate(facts):
        a,b=f['tail_span']; c,d=f['head_span']
        if text[a:b]!=f['tail'] or text[c:d]!=f['head']: raise ValueError('Fact span mismatch')
        end=text.find('.',d)
        if end<0: raise ValueError('Fact punctuation missing')
        im=ix(a,b);ic=ix(c,d);ip=ix(end,end+1)
        matches=list(re.finditer(r'\bis\b',text[b:c]))
        if len(matches)!=1: raise ValueError('Ambiguous is site')
        match=matches[0];ii=ix(b+match.start(),b+match.end())
        whole=ix(a,end+1);tmp=sorted(set(whole)-set(im+ic+ip))
        for k,v in [('mother',im),('child',ic),('template',tmp),('punctuation',ip)]:
            result[f'fact{i}_{k}']=v
        mothers+=im;children+=ic;templates+=tmp;punct+=ip
        role=dict(mother=im,child_last=[ic[-1]],is_token=ii,period=ip,sentence=whole,
                  writer_union=sorted(set(im+ii+ip)))
        if i in [q,other]:
            prefix='query' if i==q else 'other'
            result.update({prefix+'_'+k:v for k,v in role.items()})
    result.update(all_mother=sorted(set(mothers)),all_child=sorted(set(children)),
                  all_template=sorted(set(templates)),all_punctuation=sorted(set(punct)))
    qs=text.rfind('Question:');ans=text.rfind('Answer:')
    result['question']=ix(qs,ans)
    result['nonquery_mother']=sorted(set(mothers)-set(result['query_mother']))
    test=ix(row['test_start'],len(text))
    result['earlier_remaining']=sorted(set(test)-set(result['nonquery_mother'])-{n-1})
    result['_query_first']=q<other
    return result

def prepare(stage3,design,out):
    if out.exists(): raise FileExistsError('Use a new input directory: '+str(out))
    pairs=list(read_lines(stage3/'pairs.jsonl.gz')); prompts={p['id']:p for p in read_lines(stage3/'prompts.jsonl.gz')}
    d=json_read(design/'manifest_proposal.json'); refs=np.load(stage3/'references.npz')
    sourceplan=json_read(stage3/'plan.json')
    if len(pairs)!=178 or sorted({p['family'] for p in pairs})!=d['discovery_families']: raise ValueError('Discovery population mismatch')
    for i,p in enumerate(pairs):
        if any(p[x]['split']!='discovery' for x in ['clean','corr']): raise ValueError('Held-out leakage')
        clean,corr=[prompts[p[k]['id']] for k in ['clean','corr']]
        if len(clean['token_ids'])!=len(corr['token_ids']):raise ValueError('Unaligned original tokens')
        p['masks']=masks(clean);cm=masks(corr)
        if p['masks']!=cm:raise ValueError('Role alignment differs across donor and recipient')
        p['original_ids']={k:prompts[p[k]['id']]['token_ids'] for k in ['clean','corr']}
        p['saved_baselines']=refs['baselines'][i].tolist()
        p['saved_P']={str(int(h)):float(v) for h,v in zip(refs['heads'],refs['scopeP_delta'][i])}
        p['saved_F']={str(int(h)):float(v) for h,v in zip(refs['heads'],refs['scopeF_delta'][i]) if np.isfinite(v)}
        if p['exact_all']:
            for h in map(hid,d['coverage_candidates']): p['saved_P'][str(h)]=float(refs['exact'][i,h])
        # Saved logits are an independent endpoint gate, in fixed clean/corrupt token order.
        if len(p['clean_logits'])!=2:raise ValueError('Missing individual-logit reference')
    out.mkdir(parents=True)
    write_lines(out/'pairs.jsonl.gz',pairs)
    plan=dict(policy=POLICY,model=sourceplan['model'],design=d,
              pair_hash=digest(out/'pairs.jsonl.gz'),gate_families=d['core_families'][:2]+[168],
              source_hashes={str(stage3/'pairs.jsonl.gz'):digest(stage3/'pairs.jsonl.gz'),
                             str(stage3/'references.npz'):digest(stage3/'references.npz'),
                             str(design/'manifest_proposal.json'):digest(design/'manifest_proposal.json')})
    json_write(out/'plan.json',plan)
    configs=[]
    for h in map(hid,d['coverage_candidates']):
        for site in ['all','colon']: configs.append(route(h,site,'single',h,panel='coverage'))
    write_schedule(out/'coverage.json',out,configs,'all',['noise'],phase='coverage')
    write_schedule(out/'seed.json',out,seed_configs(d),'core',['noise'],phase='seed',requires_coverage=True)
    return plan

def seed_configs(design):
    pool=list(map(hid,design['initial_route_pool']))
    # The four other strong-P heads are sources; the remaining 21 are answer-position candidates.
    readers=[h for h in pool if h not in [hid(s) for s in ['L13H10','L14H26','L15H25','L17H1']]]
    configs=[]
    sources=[('L17H1',['query_mother','query_is_token','query_period'],'P1'),
             ('L15H25',['query_child_last'],'P2'),
             ('L13H10',['query_sentence'],'P4'),('L14H26',['query_sentence'],'P4'),
             ('L17H3',['nonquery_mother','earlier_remaining'],'P4')]
    for name,sites,panel in sources:
        h=hid(name);li=h//32
        for site in sites:
            for v in readers:
                if v//32>li:
                    for ch in ['K','V']: configs.append(route(h,site,'head',v,ch,panel))
            for mlp in [li,li+1]:configs.append(route(h,site,'mlp',mlp,panel=panel))
    colon=['L16H1','L16H21','L17H24','L18H18','L18H19','L19H16','L20H1','L23H15','L25H17','L17H3']
    for name in colon:
        h=hid(name);li=h//32
        for v in readers:
            if v//32>li:
                for ch in ['Q','KV']:configs.append(route(h,'colon','head',v,ch,'P3'))
        for mlp in [li,li+1]:configs.append(route(h,'colon','mlp',mlp,panel='P3'))
        configs.append(route(h,'colon','bypass',31,panel='P3'))
    return sorted({c['id']:c for c in configs}.values(),key=lambda c:(c['source'],c['site'],c['kind'],c['receiver'],c['channel']))

def write_schedule(path,inputs,configs,population,directions,**meta):
    ids=[c['id'] for c in configs]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate configuration IDs')
    json_write(path,dict(policy=POLICY,plan_hash=digest(inputs/'plan.json'),configs=configs,
                         population=population,directions=directions,**meta))

def load(inputs,schedule):
    plan=json_read(inputs/'plan.json'); sc=json_read(schedule)
    if plan['policy']!=POLICY or sc['policy']!=POLICY:raise ValueError('Policy mismatch')
    if sc['plan_hash']!=digest(inputs/'plan.json') or plan['pair_hash']!=digest(inputs/'pairs.jsonl.gz'):raise ValueError('Changed inputs')
    pairs=list(read_lines(inputs/'pairs.jsonl.gz'))
    if sc['population']=='core':pairs=[p for p in pairs if p['exact_all']]
    elif sc['population']!='all':raise ValueError('Unknown population')
    for c in sc['configs']:
        h=c['source'];li=h//32
        if not 0<=h<1024:raise ValueError('Source outside model')
        if c['kind']=='head' and c['receiver']//32<=li:raise ValueError('Noncausal head route')
        if c['kind']=='head' and (c['channel'] not in ['Q','K','V','KV','QKV'] or not 0<=c['receiver']<1024):raise ValueError('Bad channel/receiver')
        if c['kind']=='head' and 'Q' in c['channel'] and c['site']!='colon':raise ValueError('Direct cross-position Q route is invalid')
        if c['kind']=='mlp' and not li<=c['receiver']<32:raise ValueError('Noncausal MLP route')
        if c['kind'] not in ['single','head','mlp','bypass']:raise ValueError('Unknown kind')
        if c['kind']=='bypass' and c['site']!='colon':raise ValueError('Only colon bypass planned')
        for p in pairs:
            if c['site'] not in p['masks'] or not p['masks'][c['site']]:raise ValueError('Missing source mask')
    return plan,sc,pairs

def retained(values):
    a=np.asarray(values,dtype=float);mean=float(a.mean());ma=float(np.abs(a).mean())
    sign=float((np.sign(a)==np.sign(mean)).mean());fraction=float((np.abs(a)>=POLICY['tau']).mean())
    coherent=abs(mean)>=POLICY['tau'] and sign>=POLICY['coherent_fraction']
    hetero=ma>=POLICY['tau'] and fraction>=POLICY['heterogeneous_fraction']
    return dict(mean=mean,mean_absolute=ma,median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()),
                sign_fraction=sign,above_tau_fraction=fraction,retained=coherent or hetero,
                classification='coherent' if coherent else 'heterogeneous' if hetero else 'below_rule')
