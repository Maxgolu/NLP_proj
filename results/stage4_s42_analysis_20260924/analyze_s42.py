"""Paired, family-weighted S4.2 research analysis; saved data only."""
from pathlib import Path
import sys,json,csv,collections,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PKG=ROOT/'results/stage4_s42_v1';RUN=ROOT/'results/stage4_s42_all_v1'
sys.path.insert(0,str(PKG))
from stage3_common import json_read,json_write,read_lines,digest,head_name
from s42_plan import load,config,group_atoms,atom,hid
from stage4_plan import retained
plan,pairs=load(PKG/'inputs');pp=plan['proposal'];core=set(plan['core_families'])
family_order=sorted({p['family'] for p in pairs});pmap={p['id']:p for p in pairs}
summaries={ph:json_read(RUN/'analysis'/ph/'summary.json') for ph in ['initial','conditional','refinement','extension','sensitivity']}
index={};configs={};phase_rows={}
for ph in summaries:
    sc=json_read(PKG/'inputs/initial.json' if ph=='initial' else RUN/'schedules'/f'{ph}.json')
    configs.update({c['id']:c for c in sc['configs']});rows=[]
    for rank in range(3):
        for path in sorted((RUN/'runs'/f'{ph}_r{rank}'/'chunks').glob('*.json')):
            if path.with_suffix('.ok').exists():rows+=json_read(path)
    phase_rows[ph]=rows
    for r in rows:
        key=(r['mode'],r['pair_id'],r['config_id'],r['direction'])
        if key in index:assert abs(index[key]['effect']-r['effect'])<1e-10
        index[key]=r

boot={}
def stats(values):
    x=np.array(values,float);n=len(x)
    if n not in boot:boot[n]=np.random.default_rng(20260923).integers(0,n,size=(20000,n))
    q=np.quantile(x[boot[n]].mean(1),[.025,.975]);qa=np.quantile(np.abs(x)[boot[n]].mean(1),[.025,.975])
    return dict(families=n,**retained(x),ci_low=float(q[0]),ci_high=float(q[1]),ma_ci_low=float(qa[0]),ma_ci_high=float(qa[1]))
def csvwrite(name,rows):
    if not rows:return
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def pop(name):return [p for p in pairs if name=='all89' or (p['family'] in core)==(name=='core20')]
def get(p,cid,mode='donor',di='noise'):
    r=index.get((mode,p['id'],cid,di))
    if r:return r['effect']
    c=configs.get(cid)
    if mode=='donor' and di=='noise' and c and len(c['atoms'])==1:
        a=c['atoms'][0]
        if a['donor']=='other':
            s=p['saved_single'].get(f"{a['head']}:{a['site']}")
            if s:return s['effect']
    raise KeyError((p['id'],cid,mode,di))
def fam(vals,ps):
    d=collections.defaultdict(list)
    for p,v in zip(ps,vals):d[p['family']].append(v)
    assert all(len(v)==2 for v in d.values())
    return np.array([np.mean(d[f]) for f in sorted(d)])
def contrast(terms,ps,mode='donor',di='noise'):
    return fam([sum(w*get(p,c,mode,di) for c,w in terms) for p in ps],ps)
def cid(hs,site='colon'):
    c=config(group_atoms(hs,site));configs.setdefault(c['id'],c);return c['id']

# Full ledger with explicit population, direction and baseline. No promotion of unmeasured cells.
ledger=[]
for ph,ss in summaries.items():
    for r in ss:ledger.append(dict(phase=ph,mode='mean' if ph=='sensitivity' else 'donor',**r))
csvwrite('contrast_ledger.csv',ledger)

groups=[];nfull=[];cond=[]
for g in pp['groups']:
    name=g['id'];ms=g['members'];whole=cid(ms);singles=[cid([h]) for h in ms]
    for population in ['core20','additional69','all89']:
        ps=pop(population)
        try:
            effect=contrast([(whole,1)],ps);N=contrast([(whole,1)]+[(c,-1) for c in singles],ps)
        except KeyError:continue
        nfull.append(dict(group=name,population=population,kind='interaction',mode='donor',direction='noise',**stats(N)))
        groups.append(dict(group=name,population=population,kind='whole',mode='donor',direction='noise',**stats(effect)))
    for h in ms:
        c=next(r for r in summaries['initial'] if r['id']==f'G1:{name}:conditional:{h}' and r['direction']=='noise')
        cond.append(dict(group=name,head=h,**{k:v for k,v in c.items() if k not in ['group','head']}))
csvwrite('g1_interactions_cpu_extended.csv',nfull);csvwrite('g1_whole_populations.csv',groups);csvwrite('g1_conditional_core.csv',cond)

# Empirical endpoint accuracy and order strata for the proposed groups, both baselines/directions.
task=[];order=[]
for g in pp['groups']:
    c=cid(g['members'])
    for mode in ['donor','mean']:
        for di in ['noise','restore']:
            for population in ['core20','additional69','all89']:
                ps=pop(population)
                try:rr=[index[(mode,p['id'],c,di)] for p in ps]
                except KeyError:continue
                margins=np.array([r['result']['margin'] for r in rr]);eff=fam([r['effect'] for r in rr],ps)
                task.append(dict(group=g['id'],mode=mode,direction=di,population=population,pairs=len(ps),mean_margin=float(margins.mean()),
                    clean_candidate_preferred=float((margins>0).mean()),recipient_accuracy=float((margins*(1 if di=='noise' else -1)>0).mean()),**stats(eff)))
                for before in [True,False]:
                    tmp=[(p,r) for p,r in zip(ps,rr) if bool(p['masks']['_query_first'])==before]
                    dd=collections.defaultdict(list)
                    for p,r in tmp:dd[p['family']].append(r['effect'])
                    gap=float(np.mean([p['saved_baselines'][0]-p['saved_baselines'][1] for p,r in tmp]))
                    st=stats([np.mean(v) for v in dd.values()])
                    order.append(dict(group=g['id'],mode=mode,direction=di,population=population,query_first=before,full_gap=gap,normalized_mean=st['mean']/gap,**st))
csvwrite('g1_task_performance.csv',task);csvwrite('g1_order_effects.csv',order)

# Matched receiver blocking, residual effect and ratio bootstrapped jointly by family.
blocking=[]
for w in pp['blocking']:
    for site in w['source_masks']:
        src=config([atom(w['source'],site)])['id'];blocked=config([atom(w['source'],site)]+group_atoms(w['receivers'],donor='self'))['id']
        for mode in ['donor','mean']:
            for di in ['noise','restore']:
                for population in ['core20','additional69','all89']:
                    ps=pop(population)
                    try:s=contrast([(src,1)],ps,mode,di);b=contrast([(src,1),(blocked,-1)],ps,mode,di)
                    except KeyError:continue
                    st=stats(b);ix=boot[len(b)];den=s[ix].mean(1);rat=b[ix].mean(1)/den
                    blocking.append(dict(source=w['source'],site=site,population=population,mode=mode,direction=di,source_mean=float(s.mean()),
                        blocked_residual_mean=float((s-b).mean()),ratio=float(b.mean()/s.mean()),ratio_ci_low=float(np.quantile(rat,.025)),
                        ratio_ci_high=float(np.quantile(rat,.975)),bootstrap_denominator_min_abs=float(np.abs(den).min()),**st))
csvwrite('blocking_paired.csv',blocking)

# RI audit: all 31 core results, selected extensions, historical scope comparison.
with (ROOT/'results/stage3_v1/analysis/head_profiles.csv').open() as f:old={r['head']:r for r in csv.DictReader(f)}
ri=[]
for name in pp['g2']['ri31']:
    for phase in ['conditional','extension','sensitivity']:
        for di in ['noise','restore']:
            found=[r for r in summaries[phase] if r['id']==f'G2:{name}:conditional' and r['direction']==di]
            if not found:continue
            r=found[0];change=next(x for x in summaries[phase] if x['id']==f'G2:{name}:change' and x['direction']==di)
            ri.append(dict(head=name,phase=phase,direction=di,mode='mean' if phase=='sensitivity' else 'donor',prior_P=float(old[name]['stage2_P']),prior_F=float(old[name]['stage3_F']),
                conditional_mean=r['mean'],conditional_ma=r['mean_absolute'],conditional_class=r['classification'],conditional_retained=r['retained'],
                ci_low=r['descriptive_ci_low'],ci_high=r['descriptive_ci_high'],change_mean=change['mean'],change_ma=change['mean_absolute'],change_retained=change['retained']))
csvwrite('ri31_audit.csv',ri)

# Residual group vs each fixed reference: paired magnitude contrasts, not a randomization null.
references=[];residual_families=[];residual_nonadd=[]
K=json_read(RUN/'K_selection.json')['K'];kc=cid(K)
for mode in ['donor','mean']:
    for di in ['noise','restore']:
        for population in ['core20','additional69','all89']:
            ps=pop(population)
            for bg in ['intact','weakened']:
                vectors=[]
                for hs in [pp['g2']['residual23']]+pp['g2']['reference_cohorts']:
                    cfg=config(group_atoms(hs,'all')+(group_atoms(K) if bg=='weakened' else []))['id']
                    vectors.append(contrast([(cfg,1)]+([(kc,-1)] if bg=='weakened' else []),ps,mode,di))
                res=vectors[0];ref=np.stack(vectors[1:]);delta=np.abs(res)-np.abs(ref).mean(0)
                references.append(dict(mode=mode,direction=di,population=population,background=bg,residual_mean=float(res.mean()),residual_ma=float(np.abs(res).mean()),
                    references_mean_ma=float(np.abs(ref).mean()),ma_ratio=float(np.abs(res).mean()/np.abs(ref).mean()),**stats(delta)))
                if population=='all89':
                    for f,v in zip(family_order,res):residual_families.append(dict(family=f,mode=mode,direction=di,background=bg,effect=float(v)))
for population in ['core20','additional69','all89']:
    ps=pop(population);terms=[(cid(pp['g2']['residual23'],'all'),1)]+[(cid([h],'all'),-1) for h in pp['g2']['residual23']]
    residual_nonadd.append(dict(population=population,**stats(contrast(terms,ps))))
csvwrite('residual_vs_references.csv',references);csvwrite('residual_family_effects.csv',residual_families);csvwrite('residual_nonadditivity.csv',residual_nonadd)

# G3: conditional effect, actual changed attention/output, and preserved earlier-layer null.
g3=[];ps=pop('core20');intact=cid([])
for i,hs in enumerate(pp['g3']['backgrounds']+[pp['g3']['downstream_attention_null']]):
    bg=cid(hs);arr=[]
    for p in ps:
        a=index[('donor',p['id'],intact,'noise')]['diagnostic'];b=index[('donor',p['id'],bg,'noise')]['diagnostic']
        pa,pb=np.array(a['pattern']),np.array(b['pattern']);za,zb=np.array(a['z']),np.array(b['z'])
        arr.append(dict(tv=float(np.abs(pb-pa).sum()/2),relative_z=float(np.linalg.norm(zb-za)/max(np.linalg.norm(za),1e-12)),
            mother_mass=float(pb[p['masks']['query_mother']].sum()),mother_delta=float((pb-pa)[p['masks']['query_mother']].sum())))
    cond=next(r for r in summaries['initial'] if r['id']==f'G3:{i}:conditional')
    g3.append(dict(background=i,heads=','.join(hs) or 'intact',mean=cond['mean'],ma=cond['mean_absolute'],ci_low=cond['descriptive_ci_low'],ci_high=cond['descriptive_ci_high'],
        **{k:float(np.mean([a[k] for a in arr])) for k in arr[0]},max_tv=max(a['tv'] for a in arr)))
csvwrite('g3_backup_audit.csv',g3)

# Extension outcomes and sensitivity agreement remain stratified by claim family.
extmap={(x['id'],x['direction']):x for x in summaries['extension']};senmap={(x['id'],x['direction']):x for x in summaries['sensitivity']}
counts=[]
for panel in ['G1','blocking','G2','G2group']:
    for di in ['noise','restore']:
        keys=[k for k,r in extmap.items() if r['panel']==panel and k[1]==di]
        counts.append(dict(panel=panel,direction=di,comparisons=len(keys),donor_retained=sum(extmap[k]['retained'] for k in keys),
            mean_retained=sum(senmap[k]['retained'] for k in keys),both_retained=sum(extmap[k]['retained'] and senmap[k]['retained'] for k in keys),
            opposite_signed_means=sum(extmap[k]['mean']*senmap[k]['mean']<0 for k in keys)))
csvwrite('sensitivity_counts.csv',counts)
norms=[]
targets=[('O3',cid(next(g['members'] for g in pp['groups'] if g['id']=='O3')))]
targets +=[(label,cid(hs,'all')) for label,hs in [('RI23',pp['g2']['residual23'])]+[(f'reference{i+1}',x) for i,x in enumerate(pp['g2']['reference_cohorts'])]]
for label,c in targets:
    for mode in ['donor','mean']:
        vals=[float(np.sqrt(sum(v*v for v in index[(mode,p['id'],c,'noise')]['norms'].values()))) for p in pairs]
        norms.append(dict(group=label,mode=mode,mean_joint_head_output_delta_norm=float(np.mean(vals)),median=float(np.median(vals)),
            note='Unprojected concatenated head outputs across patched positions; descriptive, not a norm-matched causal control'))
csvwrite('replacement_norms.csv',norms)
full_order=[]
for before in [True,False]:
    ps=[p for p in pairs if bool(p['masks']['_query_first'])==before]
    full_order.append(dict(query_first=before,pairs=len(ps),clean_margin=float(np.mean([p['saved_baselines'][0] for p in ps])),
        corrupt_margin=float(np.mean([p['saved_baselines'][1] for p in ps])),gap=float(np.mean([p['saved_baselines'][0]-p['saved_baselines'][1] for p in ps]))))
csvwrite('full_model_order_baselines.csv',full_order)
for ph in summaries:
    for name in ['summary.json','additional_families.csv','candidate_accuracy.csv']:
        src=RUN/'analysis'/ph/name
        if src.exists():shutil.copyfile(src,HERE/f'{ph}_{name}')
facts=dict(verified_records=51904,new_endpoints=44754,reused_endpoints=7150,K=K,K_fraction=json_read(RUN/'K_selection.json')['candidates'][0]['fraction'],
    extended_RI=[r['head'] for r in ri if r['phase']=='extension' and r['direction']=='noise'],g3=g3,blocking=blocking,
    full_discovery_new_cpu_interactions=nfull,ri_audit=ri,reference_comparisons=references,residual_nonadditivity=residual_nonadd,full_model_order=full_order,replacement_norms=norms,
    source_archive_sha256=digest(ROOT/'results/stage4_s42_all_v1_results.tar.gz'),scope='Analysis only; no new model runs; no held-out data')
json_write(HERE/'facts.json',facts)
print('Derived evidence saved;',len(index),'unique endpoint records;',len(nfull),'G1 interaction/population summaries',flush=True)
