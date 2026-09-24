"""Integrity/reproducibility audit only; no scientific interpretation or report."""
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'results/stage4_s42_v1'
sys.path.insert(0,str(PACKAGE))
from stage3_common import json_read,json_write,digest,read_lines
from s42_run import code_identity,identity,mean_identity,reusable
from s42_plan import load,group_atoms,config
from s42_analyze import gather,analyze,finish

OUT=ROOT/'results/stage4_s42_all_v1'
INPUTS=PACKAGE/'inputs'
AUDIT=Path(__file__).resolve().parent

def equal(a,b,label):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),label
        for k in a:equal(a[k],b[k],label+'/'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),label
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,label+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert isinstance(b,(float,int)) and np.isclose(a,b,rtol=1e-10,atol=1e-10),label
    else:assert a==b,label

def compare_file(a,b):
    if a.suffix=='.json':equal(json_read(a),json_read(b),a.name);return
    with a.open(encoding='utf-8',newline='') as f:aa=list(csv.reader(f))
    with b.open(encoding='utf-8',newline='') as f:bb=list(csv.reader(f))
    assert len(aa)==len(bb),a.name
    for i,(ra,rb) in enumerate(zip(aa,bb)):
        assert len(ra)==len(rb),a.name
        for ca,cb in zip(ra,rb):
            if ca==cb:continue
            try:equal(float(ca),float(cb),f'{a.name}:{i}')
            except ValueError:raise AssertionError(f'{a.name}:{i}: nonnumeric mismatch')

def main():
    plan,pairs=load(INPUTS);pairmap={p['id']:p for p in pairs}
    assert len(pairs)==178 and len({p['family'] for p in pairs})==89
    assert all(p[s]['split']=='discovery' for p in pairs for s in ['clean','corr'])
    phases=json_read(OUT/'pipeline_done.json')['phases'];code=code_identity()
    audit=dict(scope='Readiness audit only; no scientific interpretation',archive_sha256=digest(ROOT/'results/stage4_s42_all_v1_results.tar.gz'),
               phases={},gates={},reproduced_files=[],limitations=[],new_endpoints=0,reused_endpoints=0)
    history={};gateids={p['id'] for p in pairs if p['family'] in plan['gate_families']}
    for run in sorted((OUT/'runs').iterdir()):
        ident=json_read(run/'manifest.json')['identity'];assert ident['code']==code and ident['plan_hash']==digest(INPUTS/'plan.json')
        if run.name.startswith('mean_collection'):
            assert ident==mean_identity(INPUTS,OUT/'schedules/mean_request.json',ident['shard'],ident['shards'])
            assert json_read(run/'done.json')['complete'];continue
        gate=json_read(run/'gate.json');assert gate['passed']
        assert {p['pair_id'] for p in gate['pairs']}==gateids
        assert {p['pair_id'] for p in gate['s42']}==gateids
        limits={'sdpa':.02,'saved':.05,'P_':.05,'F':.05,'qkv_output':.05}
        largest=0.
        for p in gate['pairs']+gate['s42']:
            for name,value in p['errors'].items():
                limit=next((v for k,v in limits.items() if k in name),.001)
                assert np.isfinite(value) and 0<=value<=limit,(run.name,name,value,limit)
                largest=max(largest,value)
        audit['gates'][run.name]=dict(passed=True,pairs=6,max_error=largest)
    for phase in phases:
        print('Reproducing',phase,flush=True)
        schedule=INPUTS/'initial.json' if phase=='initial' else OUT/'schedules'/f'{phase}.json'
        runs=[OUT/'runs'/f'{phase}_r{i}' for i in range(3)]
        _,sc,pp,rows,ix=gather(INPUTS,schedule,runs)
        for i,run in enumerate(runs):
            assert json_read(run/'manifest.json')['identity']==identity(INPUTS,schedule,i,3,OUT/'mean_bank' if sc['mode']=='mean' else None)
        cfgs={c['id']:c for c in sc['configs']};nnew=nreuse=0;missing_logits=0;diagnostics=0
        for r in rows:
            p=pairmap[r['pair_id']];cid=r['config_id'];k=(sc['mode'],p['id'],cid,r['direction'])
            if r['origin'].startswith('prior_phase:'):
                assert k in history,(phase,k)
                equal({q:v for q,v in r.items() if q!='origin'},{q:v for q,v in history[k].items() if q!='origin'},'reuse')
            elif r['origin'].startswith('saved_'):
                saved=reusable(p,cfgs[cid],r['direction'],sc['mode']);assert saved is not None
                equal(r['effect'],saved['effect'],'historical effect');equal(r['result'],saved['result'],'historical result')
            else:assert r['origin']=='new'
            if r['origin']=='new':
                nnew+=1;assert r['result'] is not None
                for field in ['clean_prob','corr_prob','top_token']:assert field in r['result']
                for field in ['clean_prob','corr_prob']:assert 0<=r['result'][field]<=1
            else:nreuse+=1
            missing_logits+=r['result'] is None
            if r['diagnostic']:
                d=r['diagnostic'];diagnostics+=1
                assert d['head']==863 and d['row']==len(p['original_ids']['clean'])-1 and len(d['z'])==128
                assert d['reconstruction_error']<=.02 and min(d['pattern'])>=0
            history[k]=r
        target=AUDIT/'reproduced/analysis'/phase
        analyze(INPUTS,schedule,runs,target)
        for name in json_read(OUT/'analysis'/phase/'verification.json')['files']:
            compare_file(OUT/'analysis'/phase/name,target/name);audit['reproduced_files'].append(phase+'/'+name)
        audit['phases'][phase]=dict(records=len(rows),pairs=len(pp),families=len({p['family'] for p in pp}),
            configs=len(cfgs),new=nnew,reused=nreuse,diagnostic_records=diagnostics,historical_effect_only_records=missing_logits)
        audit['new_endpoints']+=nnew;audit['reused_endpoints']+=nreuse
        if phase=='initial':
            for g in plan['proposal']['groups']:
                ms=g['members'];sets=[ms]+[[h] for h in ms]+[[x for x in ms if x!=h] for h in ms]
                assert all(config(group_atoms(hs))['id'] in cfgs for hs in sets)
            assert sum(c.get('kind')=='blocking' for c in sc['contrasts'])==6
            # Full downstream null and pre-own-intervention equality, not only gate pairs.
            intact=config([])['id'];later=config(group_atoms(['L27H6']))['id']
            for p in pp:
                a=ix[(p['id'],intact,'noise')]['diagnostic'];b=ix[(p['id'],later,'noise')]['diagnostic']
                assert max(np.max(np.abs(np.array(a[f])-b[f])) for f in ['pattern','z'])<=.001
            for hs in plan['proposal']['g3']['backgrounds']+[plan['proposal']['g3']['downstream_attention_null']]:
                bg=config(group_atoms(hs))['id'];own=config(group_atoms(hs+['L26H31']))['id']
                for p in pp:
                    a=ix[(p['id'],bg,'noise')]['diagnostic'];b=ix[(p['id'],own,'noise')]['diagnostic']
                    assert max(np.max(np.abs(np.array(a[f])-b[f])) for f in ['pattern','z'])<=.001
        if phase=='conditional':
            assert sum(c['panel']=='G2' and c['kind']=='conditional' for c in sc['contrasts'])==31
            assert len({c['cohort'] for c in sc['contrasts'] if c['panel']=='G2group'})==7
    finish(INPUTS,AUDIT/'reproduced',phases,json_read(OUT/'K_selection.json'))
    for name in ['results_summary.json','baseline_sensitivity.csv']:
        compare_file(OUT/name,AUDIT/'reproduced'/name);audit['reproduced_files'].append(name)
    assert json_read(OUT/'portable_verification.json')['passed']
    audit['limitations']=['Historical Scope-P reused measurements may have effects only, not individual patched logits; new measurements include both logits.',
        'Core-only G1 contrasts outside the declared extension cap remain provisional.',
        'Mean reverse is sensitivity in corrupted recipients, not literal clean restoration.',
        'Discovery data only; no held-out circuit validation is claimed.']
    audit['ready_for_full_analysis']=True;audit['additional_gpu_required_for_declared_S42']=False
    json_write(AUDIT/'readiness.json',audit)
    print('READY:',audit['new_endpoints']+audit['reused_endpoints'],'records;',len(audit['reproduced_files']),'reproduced files',flush=True)

if __name__=='__main__':main()
