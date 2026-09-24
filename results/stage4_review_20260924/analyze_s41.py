"""Read-only audit of immutable S4.1 results; derived outputs stay in this folder."""
from pathlib import Path
import sys,json,hashlib,gzip,collections
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'results/stage4_all_v2'
PKG=ROOT/'results/stage4_s41_v2'
sys.path[:0]=[str(PKG),str(ROOT/'pilot_v2')]
from stage4_analyze import analyze,compare,next_schedule
from stage4_plan import retained
from stage4_run import verify_run

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item())+'\n',encoding='utf-8')

def main():
    audit={}
    archive=ROOT/'results/stage4_all_v2_results.tar.gz'
    expected=archive.with_name(archive.name+'.sha256').read_text().split()[0]
    assert sha(archive)==expected
    audit['archive_sha256']=expected
    source_checks={}
    for name,h in read(PKG/'bundle_hashes.json').items():
        candidates=[PKG/name,ROOT/'pilot_v2'/name]
        p=next((p for p in candidates if p.exists() and sha(p)==h),None)
        assert p is not None,(name,'missing or changed source')
        source_checks[name]=str(p.relative_to(ROOT))
    audit['bundle_verified']=source_checks
    manifest=read(RUN/'pipeline_manifest.json')
    for n,h in manifest['code'].items():
        assert any(p.exists() and sha(p)==h for p in [PKG/n,ROOT/'pilot_v2'/n]),n
    inputs=PKG/'inputs'
    phases={'coverage':inputs/'coverage.json','localize_roles':RUN/'schedules/after_coverage.json',
            'localize_slots':RUN/'schedules/after_localize_roles.json','seed':RUN/'schedules/after_localize_slots.json',
            'refine':RUN/'schedules/refine.json','extend_seed':RUN/'schedules/extend_seed.json',
            'extend_refine':RUN/'schedules/extend_refine.json'}
    gate_rows=[]
    for d in sorted((RUN/'runs').iterdir()):
        gate=read(d/'gate.json');tiny=read(d/'tiny_gate.json')
        assert gate['passed'] and tiny['passed'] and len(gate['pairs'])==6
        assert read(d/'manifest.json')['identity']['code']=={k:v for k,v in manifest['code'].items() if k!='stage4_pipeline.py'}
        for p in gate['pairs']:
            for k,v in p['errors'].items():gate_rows.append(dict(worker=d.name,pair_id=p['pair_id'],check=k,error=v))
    pd.DataFrame(gate_rows).to_csv(OUT/'gate_errors.csv',index=False)
    audit['gate_maxima']=pd.DataFrame(gate_rows).groupby('check').error.max().to_dict()
    allrows=[];checks=[];summary=[]
    for phase,schedule in phases.items():
        runs=[RUN/'runs'/f'{phase}_r{i}' for i in range(3)]
        rows=[]
        for d in runs:
            rr=verify_run(d,inputs,schedule);assert len(rr)==read(d/'done.json')['records']
            assert not list((d/'chunks').glob('*.tmp'))
            assert len(list((d/'chunks').glob('*.json')))==len(list((d/'chunks').glob('*.ok')))
            rows+=rr
        assert len({(r['pair_id'],r['config_id'],r['direction']) for r in rows})==len(rows)
        target=OUT/'reproduced'/phase
        if not target.exists():analyze(inputs,schedule,runs,target)
        for name in ['events.csv','family_effects.csv','route_summary.csv','context_summary.csv']:
            a=pd.read_csv(target/name);b=pd.read_csv(RUN/'analysis'/phase/name)
            pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-12,atol=1e-12)
            checks.append(dict(phase=phase,file=name,rows=len(a),equal=True))
        sc=read(schedule)
        summary.append(dict(phase=phase,configs=len(sc['configs']),records=len(rows),reused=sum(r['origin']!='stage4_measurement' for r in rows)))
        for r in rows:
            if r['result'] is not None:
                m=r['result']['clean_logit']-r['result']['corr_logit']
                assert abs(m-r['result']['margin'])<1e-6
                eff=r['baseline']['margin']-m if r['direction']=='noise' else m-r['baseline']['margin']
                assert abs(eff-r['effect'])<1e-6
                assert all(np.isfinite(v) for v in r['result'].values())
            r['phase']=phase
        allrows+=rows
        print('VERIFIED',phase,len(rows),flush=True)
    comp=OUT/'reproduced/interactions_core'
    if not comp.exists():compare([OUT/'reproduced/seed',OUT/'reproduced/refine'],comp)
    for name in ['interaction_family.csv','interaction_summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(comp/name),pd.read_csv(RUN/'analysis/interactions_core'/name),rtol=1e-12,atol=1e-12)
    # Reproduce all adaptive decisions, including which controls and extensions ran.
    ds=[('after_coverage','after-coverage','coverage',None),('after_localize_roles','after-localize','localize_roles',None),
        ('after_localize_slots','after-localize','localize_slots',None),('refine','refine','seed',None),
        ('extend_seed','extend','seed',None),('extend_refine','extend','refine',RUN/'schedules/extend_seed.json')]
    for label,mode,phase,prior in ds:
        dest=OUT/'decisions'/f'{label}.json'
        if not dest.exists():next_schedule(inputs,phases[phase],[RUN/'runs'/f'{phase}_r{i}' for i in range(3)],dest,mode,prior)
        a=read(dest);b=read(RUN/'schedules'/f'{label}.json')
        a.pop('parent_runs');b.pop('parent_runs');assert a==b,label
    audit.update(phases=summary,total_records=len(allrows),csv_reproduction=checks,adaptive_decisions_reproduced=True)
    dump(OUT/'verification.json',audit)
    flat=[]
    for r in allrows:
        c=r['config'];base=r['baseline'];end=r['result']
        flat.append(dict(phase=r['phase'],pair_id=r['pair_id'],family=r['family'],order=r['order'],query_first=r['query_first'],
            config_id=r['config_id'],direction=r['direction'],effect=r['effect'],source=c['source'],site=c['site'],kind=c['kind'],
            receiver=c['receiver'],channel=c['channel'],panel=c['panel'],control_of=c.get('control_of',''),norm=r['channel_norm'],
            prefix=bool(r['prefix']),self_error=r.get('self_endpoint_error'),origin=r['origin'],baseline=base['margin'],
            endpoint=end['margin'] if end else np.nan))
    df=pd.DataFrame(flat);df.to_csv(OUT/'all_events.csv.gz',index=False)
    derived()

def derived():
    df=pd.read_csv(OUT/'all_events.csv.gz',keep_default_na=False,low_memory=False)
    plan=read(PKG/'inputs/plan.json');core=set(plan['design']['core_families'])
    fam=df.groupby(['phase','config_id','direction','family'],as_index=False).effect.mean()
    cfg=df.drop_duplicates('config_id').set_index('config_id')[['source','site','kind','receiver','channel','panel','control_of']]
    ext=fam[fam.phase.str.startswith('extend')].copy()
    ext['population']=np.where(ext.family.isin(core),'core20','additional69')
    sub=[];rng=np.random.default_rng(20260923);boot={}
    for (cid,di),g in ext.groupby(['config_id','direction']):
        for pop in ['all89','core20','additional69']:
            gg=g if pop=='all89' else g[g.population==pop]
            v=gg.sort_values('family').effect.to_numpy();n=len(v)
            if n not in boot:boot[n]=rng.integers(n,size=(20000,n))
            lo,hi=np.quantile(v[boot[n]].mean(axis=1),[.025,.975])
            sub.append(dict(config_id=cid,direction=di,population=pop,n=n,**retained(v),ci_low=lo,ci_high=hi,**cfg.loc[cid].to_dict()))
    sub=pd.DataFrame(sub);sub.to_csv(OUT/'extension_population_summary.csv',index=False)
    # Paired control differences on the same core families, not comparisons of unrelated means.
    cf=fam[fam.phase.isin(['seed','refine'])].drop_duplicates(['config_id','direction','family'])
    controls=[]
    for cid,c in cfg[cfg.panel=='control'].iterrows():
        a=cf[cf.config_id==c.control_of].set_index('family').effect
        b=cf[cf.config_id==cid].set_index('family').effect
        assert set(a.index)==set(b.index)
        delta=a-b;absolute=a.abs()-b.abs()
        controls.append(dict(config_id=cid,parent=c.control_of,control_type='random_receiver' if c.receiver!=cfg.loc[c.control_of,'receiver'] else 'other_site',
            parent_mean=a.mean(),parent_ma=a.abs().mean(),control_ma=b.abs().mean(),
            signed_difference=delta.mean(),absolute_difference=absolute.mean(),parent_larger_fraction=(a.abs()>b.abs()).mean(),
            **{'control_'+k:v for k,v in retained(b).items()}))
    pd.DataFrame(controls).to_csv(OUT/'control_comparisons.csv',index=False)
    # Source/site map: all seed outcomes (not only extended or positive results).
    seed=pd.read_csv(OUT/'reproduced/seed/route_summary.csv')
    seed.groupby(['source_head','source_site','receiver_kind','channel'],dropna=False).agg(tested=('retained','size'),retained=('retained','sum'),
        max_absolute_mean=('mean',lambda x:x.abs().max()),max_mean_absolute=('mean_absolute','max')).reset_index().to_csv(OUT/'seed_map_coverage.csv',index=False)
    # Prefix summary from measured events; bypass nulls must remain visible.
    pre=df[df.phase.str.startswith('extend')].groupby(['config_id','direction','prefix','family'],as_index=False).effect.mean()
    pre.groupby(['config_id','direction','prefix']).effect.agg(['count','mean','min','max']).reset_index().to_csv(OUT/'prefix_summary.csv',index=False)
    # Unextended retained routes remain explicit unknowns on the larger population.
    combined=pd.concat([pd.read_csv(OUT/'reproduced'/p/'route_summary.csv').assign(phase=p) for p in ['seed','refine']])
    combined[(combined.retained)&(combined.panel!='control')&(~combined.config_id.isin(ext.config_id))].to_csv(OUT/'retained_not_extended.csv',index=False)
    print('DONE',len(df),'records',flush=True)

if __name__=='__main__':
    if '--derived' in sys.argv:derived()
    else:main()
