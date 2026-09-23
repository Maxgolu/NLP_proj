"""Exact coverage verification, family-first summaries, and bounded follow-up schedules."""
import collections
import csv
from pathlib import Path
import numpy as np
from stage3_common import json_write,json_read,digest
from stage4_plan import load,retained,route,seed_configs,write_schedule,hid,POLICY

def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',encoding='utf-8',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def gather(inputs,schedule,runs):
    from stage4_run import verify_run,expected_keys
    plan,sc,pairs=load(inputs,schedule);rows=[];identities=[]
    for run in runs:
        rows+=verify_run(run,inputs,schedule);identities.append(json_read(run/'manifest.json')['identity'])
    if len({str(x['code']) for x in identities})!=1:raise ValueError('Mixed source implementations')
    keys=[(r['pair_id'],r['config_id'],r['direction']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_keys(sc,pairs,0,1):raise ValueError('Missing/overlapping shards')
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['config_id'],r['direction'],r['family'])].append(r)
    fam=[]
    for (cid,di,f),rr in sorted(grouped.items()):
        if len(rr)!=2 or len({r['order'] for r in rr})!=2:raise ValueError('Incomplete family orders')
        fam.append(dict(config_id=cid,direction=di,family=f,effect=float(np.mean([r['effect'] for r in rr])),
                        shared_prefix=any(bool(r['prefix']) for r in rr)))
    grouped=collections.defaultdict(list)
    for r in fam:grouped[(r['config_id'],r['direction'])].append(r)
    summaries=[]; bootstrap={}; configurations={c['id']:c for c in sc['configs']}
    for (cid,di),rr in sorted(grouped.items()):
        stats=retained([r['effect'] for r in rr]);stats.update(config_id=cid,direction=di,families=len(rr))
        no_prefix=[r['effect'] for r in rr if not r['shared_prefix']]
        stats['no_prefix_mean']=float(np.mean(no_prefix)) if no_prefix else None
        count=len(rr)
        if count not in bootstrap:
            bootstrap[count]=np.random.default_rng(POLICY['seed']).integers(0,count,size=(20000,count))
        values=np.array([r['effect'] for r in sorted(rr,key=lambda x:x['family'])])
        interval=np.quantile(values[bootstrap[count]].mean(axis=1),[.025,.975])
        stats['descriptive_ci_low'],stats['descriptive_ci_high']=map(float,interval)
        cfg=configurations[cid];h=cfg['source'];receiver=cfg['receiver']
        stats.update(source_head=f'L{h//32}H{h%32}',source_site=cfg['site'],receiver_kind=cfg['kind'],
                     receiver=f'L{receiver//32}H{receiver%32}' if cfg['kind']=='head' else str(receiver),
                     channel=cfg['channel'],panel=cfg['panel'])
        summaries.append(stats)
    return plan,sc,rows,fam,summaries

def analyze(inputs,schedule,runs,out):
    if out.exists():raise FileExistsError('Use a new analysis directory')
    plan,sc,rows,fam,summaries=gather(inputs,schedule,runs);out.mkdir(parents=True)
    write_csv(out/'family_effects.csv',fam);write_csv(out/'route_summary.csv',summaries)
    write_csv(out/'events.csv',[dict(pair_id=r['pair_id'],config_id=r['config_id'],direction=r['direction'],family=r['family'],order=r['order'],
                  effect=r['effect'],origin=r['origin'],channel_norm=r['channel_norm'],prefix=str(r['prefix']),
                  baseline_clean_logit=r['baseline']['clean_logit'],baseline_corr_logit=r['baseline']['corr_logit'],
                  endpoint_clean_logit=r['result']['clean_logit'] if r['result'] else None,
                  endpoint_corr_logit=r['result']['corr_logit'] if r['result'] else None) for r in rows])
    # Predeclared context diagnostic: report paired-order/query-fact strata, no post-hoc name fishing.
    strata=[];gg=collections.defaultdict(list)
    for r in rows:gg[(r['config_id'],r['direction'],r['query_first'],r['family'])].append(r['effect'])
    sf=collections.defaultdict(list)
    for (cid,di,q,f),v in gg.items():sf[(cid,di,q)].append(float(np.mean(v)))
    for (cid,di,q),v in sorted(sf.items()):
        if len(v)>=10:strata.append(dict(config_id=cid,direction=di,query_first=q,families=len(v),**retained(v)))
    write_csv(out/'context_summary.csv',strata)
    json_write(out/'verification.json',dict(complete=True,records=len(rows),configuration_count=len(sc['configs']),plan_hash=digest(inputs/'plan.json'),schedule_hash=digest(schedule),
                  configs=sc['configs'],population=sc['population'],family_effects_hash=digest(out/'family_effects.csv'),
                  run_manifests={str(r):digest(r/'manifest.json') for r in runs},note='Discovery summaries; retention is prioritization, not confirmation'))
    print(f'Analysis: {len(rows)} records, {len(summaries)} contrasts; {out}')

def compare(analyses,out):
    """Joint KV and source-union interactions, paired by family on identical populations."""
    if out.exists():raise FileExistsError(out)
    values={};configs={};identities=set();populations=set()
    for folder in analyses:
        meta=json_read(folder/'verification.json')
        if digest(folder/'family_effects.csv')!=meta['family_effects_hash']:raise ValueError('Changed family effects')
        identities.add(meta['plan_hash']);populations.add(meta['population'])
        configs.update({c['id']:c for c in meta['configs']})
        with (folder/'family_effects.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=(row['config_id'],row['direction'],int(row['family']));v=float(row['effect'])
                if key in values and abs(values[key]-v)>1e-8:raise ValueError('Conflicting duplicate family measurement')
                values[key]=v
    if len(identities)!=1 or len(populations)!=1:raise ValueError('Compare the same input plan and population; do not mix core/all')
    outputs=[];skipped=[]
    for cid,c in sorted(configs.items()):
        terms=None;label=None
        if c['kind']=='head' and c['channel']=='KV':
            terms=[route(c['source'],c['site'],'head',c['receiver'],ch)['id'] for ch in ['K','V']];label='KV_minus_K_minus_V'
        if c['site']=='query_writer_union':
            terms=[route(c['source'],site,c['kind'],c['receiver'],c['channel'])['id'] for site in ['query_mother','query_is_token','query_period']];label='union_minus_three_sites'
        if terms is None:continue
        for direction in ['noise','restore']:
            fams={f for cc,di,f in values if cc==cid and di==direction}
            if not fams:continue
            if any({f for cc,di,f in values if cc==term and di==direction}!=fams for term in terms):
                skipped.append(dict(config_id=cid,direction=direction,reason='Missing component on identical family population'));continue
            for f in sorted(fams):
                outputs.append(dict(config_id=cid,direction=direction,family=f,contrast=label,
                                    effect=values[(cid,direction,f)]-sum(values[(term,direction,f)] for term in terms)))
    groups=collections.defaultdict(list)
    for r in outputs:groups[(r['config_id'],r['direction'],r['contrast'])].append(r['effect'])
    summary=[dict(config_id=cid,direction=di,contrast=kind,families=len(v),**retained(v)) for (cid,di,kind),v in sorted(groups.items())]
    out.mkdir(parents=True);write_csv(out/'interaction_family.csv',outputs);write_csv(out/'interaction_summary.csv',summary)
    json_write(out/'unavailable.json',skipped)
    print('Compared paired interactions:',len(summary),'unavailable:',len(skipped))

class NoFollowup(ValueError):
    """Valid scientific boundary: no eligible work, not a computational failure."""


def next_schedule(inputs,schedule,runs,out,mode,prior_extension=None):
    if out.exists():raise FileExistsError(out)
    plan,sc,rows,fam,summaries=gather(inputs,schedule,runs)
    byid={c['id']:c for c in sc['configs']};sm={x['config_id']:x for x in summaries if x['direction']=='noise'}
    d=plan['design'];configs=[];meta=dict(parent_schedule_hash=digest(schedule),parent_runs={str(p):digest(p/'done.json') for p in runs})
    population='core';directions=['noise']
    if mode=='after-coverage':
        if sc.get('phase')!='coverage':raise ValueError('Expected coverage phase')
        qualifies=[];localize=[];direct=[]
        for name in d['coverage_candidates']:
            h=hid(name);p=sm[route(h,'all','single',h)['id']];f=sm[route(h,'colon','single',h)['id']]
            if p['retained'] or f['retained']:
                qualifies.append(h)
                if abs(p['mean'])>=.1 and abs(f['mean']/p['mean'])<.8:localize.append(h)
                elif abs(p['mean'])<.1 and p['retained']:localize.append(h)  # cancelling total: unresolved location
                if f['retained']:direct.append(h)
        meta.update(coverage_retained=qualifies,coverage_colon=direct)
        if localize:
            for h in localize:
                for site in ['all_mother','all_child','all_template','all_punctuation','question','colon']:
                    configs.append(route(h,site,'single',h,panel='coverage_localize'))
            phase='localize_roles'
        else:
            configs=seed_with_coverage(d,direct,[]);phase='seed'
        meta['phase']=phase
    elif mode=='after-localize':
        if sc.get('phase') not in ['localize_roles','localize_slots']:raise ValueError('Expected localization phase')
        direct=sc.get('coverage_colon',[]);sites=list(sc.get('coverage_source_sites',[]))
        meta.update(coverage_retained=sc.get('coverage_retained',[]),coverage_colon=direct)
        for cid,st in sm.items():
            if not st['retained']:continue
            cfg=byid[cid];h=cfg['source'];site=cfg['site']
            if sc['phase']=='localize_roles' and site.startswith('all_'):
                role=site[4:]
                for slot in range(4):configs.append(route(h,f'fact{slot}_{role}','single',h,panel='coverage_localize'))
            else:sites.append([h,site])
        meta['coverage_source_sites']=sites
        if configs:meta['phase']='localize_slots'
        else:configs=seed_with_coverage(d,direct,sites);meta['phase']='seed'
    elif mode=='refine':
        if sc.get('phase')!='seed':raise ValueError('Refine follows a completed seed run')
        selected=sorted((x for x in summaries if x['direction']=='noise' and x['retained']),key=lambda x:(-x['mean_absolute'],x['config_id']))
        for st in selected:
            c=byid[st['config_id']]
            if c['kind']=='head' and c['channel'] in ['K','V','KV']:
                for ch in (['KV'] if c['channel'] in ['K','V'] else ['K','V']):configs.append(route(c['source'],c['site'],'head',c['receiver'],ch,c['panel']))
            # Joint primary-writer mask; a union must be measured, never summed.
            if c['source']==hid('L17H1') and c['kind']!='bypass':
                configs.append(route(c['source'],'query_writer_union',c['kind'],c['receiver'],c['channel'],c['panel']))
            if c['source']==hid('L15H25') and c['kind']!='bypass':
                configs.append(route(c['source'],'query_is_token',c['kind'],c['receiver'],c['channel'],'control',control_of=c['id']))
        rng=np.random.default_rng(POLICY['seed']);forbidden=set(map(hid,d['initial_route_pool']+d['coverage_candidates']+d['ri31']+d['ri_dropped']+['L3H11','L9H22']))
        for st in selected[:POLICY['controls_top']]:
            c=byid[st['config_id']]
            # Colon source: another fact position is an impossible direct-Q spatial control, not a matched route.
            wrong=c['site'].replace('query_','other_') if c['site'].startswith('query_') else 'query_sentence'
            if c['kind']!='bypass' and 'Q' not in c['channel']:
                configs.append(route(c['source'],wrong,c['kind'],c['receiver'],c['channel'],'control',control_of=c['id']))
            if c['kind']=='head':
                li=c['receiver']//32;pool=[li*32+i for i in range(32) if li*32+i not in forbidden]
                for h in rng.choice(pool,2,replace=False):configs.append(route(c['source'],c['site'],'head',int(h),c['channel'],'control',control_of=c['id']))
        meta['phase']='refine'
    elif mode=='extend':
        if sc.get('phase') not in ['seed','refine']:raise ValueError('Extend seed or refinement, separately')
        selected=sorted((x for x in summaries if x['direction']=='noise' and x['retained'] and byid[x['config_id']].get('panel')!='control'),key=lambda x:(-x['mean_absolute'],x['config_id']))
        # Prioritize at least one route per source and RI contributor, before global effect ranking.
        picked=[];covered=set();ri=set(map(hid,d['ri31']))
        for st in selected:
            c=byid[st['config_id']];tags={('source',c['source'])}
            if c['source'] in ri:tags.add(('ri',c['source']))
            if c['kind']=='head' and c['receiver'] in ri:tags.add(('ri',c['receiver']))
            if tags-covered:picked.append(c);covered|=tags
        picked+=[byid[x['config_id']] for x in selected if byid[x['config_id']] not in picked]
        # Reserve a small explicit part of the shared budget for joint-channel/union follow-ups.
        # 72 seed + up to 24 (or unused remainder) refinement contrasts, <=96 in total.
        previous=[]
        if sc['phase']=='refine':
            if prior_extension is None:raise ValueError('Refinement extension requires --prior-extension')
            _,prior,_=load(inputs,prior_extension)
            if prior.get('phase')!='extension' or prior.get('parent_schedule_hash')!=sc.get('parent_schedule_hash'):
                raise ValueError('Prior extension must originate from this refinement\'s seed schedule')
            previous=prior['configs'];meta['prior_extension_hash']=digest(prior_extension)
            budget=POLICY['full_route_cap']-len(previous)
        else:budget=72
        prior_ids={c['id'] for c in previous};picked=[c for c in picked if c['id'] not in prior_ids]
        configs=picked[:budget];meta.update(phase='extension',overflow_ids=[x['id'] for x in picked[budget:]],remaining_full_route_budget=budget-len(configs))
        population='all';directions=['noise','restore']
    else:raise ValueError(mode)
    configs=sorted({c['id']:c for c in configs}.values(),key=lambda c:c['id'])
    if not configs:raise NoFollowup('No follow-up configurations meet this rule; report boundary, do not submit an empty job')
    write_schedule(out,inputs,configs,population,directions,**meta)
    load(inputs,out)
    print(f'Wrote {out}: {len(configs)} configurations, phase {meta["phase"]}')

def seed_with_coverage(design,colon_heads,source_sites):
    configs=seed_configs(design);readers=[hid(h) for h in design['initial_route_pool'] if h not in ['L13H10','L14H26','L15H25','L17H1']]+colon_heads
    # Add newly supported recipients to every compatible source mask; preserve temporal/channel semantics.
    templates={(c['source'],c['site'],c['channel'],c['panel']) for c in configs if c['kind']=='head'}
    for h in colon_heads:
        for src,site,ch,panel in sorted(templates):
            if h//32>src//32:configs.append(route(src,site,'head',h,ch,panel))
        source_sites=source_sites+[[h,'colon']]
    for src,site in source_sites:
        for target in readers:
            if target//32>src//32:
                for ch in (['Q','KV'] if site=='colon' else ['K','V']):configs.append(route(src,site,'head',target,ch,'coverage_route'))
        for mlp in [src//32,min(31,src//32+1)]:configs.append(route(src,site,'mlp',mlp,panel='coverage_route'))
        if site=='colon':configs.append(route(src,site,'bypass',31,panel='coverage_route'))
    return list({c['id']:c for c in configs}.values())
