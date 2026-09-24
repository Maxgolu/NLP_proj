"""S4.2 worker and portable verification. No torch imported by CPU subcommands."""
import argparse
import collections
import os
from pathlib import Path
import signal
import time
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,run_lock,npz_write
from stage4_run import gates,spec_checked,check,error,CODE_FILES as S41_FILES
from s42_plan import load,prepare,assigned,reusable,config,atom,group_atoms

ROOT=Path(__file__).resolve().parent
CODE_FILES=sorted(set(S41_FILES+['s42_plan.py','s42_engine.py','s42_means.py','s42_run.py','s42_analyze.py',
    's42_pipeline.py','stage4_pipeline.py','s42_review.py','s42_submit.py','test_s42.py','test_s42_pipeline.py']))

def code_identity():return {n:digest(ROOT/n) for n in CODE_FILES}
def identity(inputs,schedule,shard,shards,bank=None):
    return dict(plan_hash=digest(inputs/'plan.json'),schedule_hash=digest(schedule),code=code_identity(),
                shard=shard,shards=shards,bank_hash=digest(bank/'manifest.json') if bank else None)
def mean_identity(inputs,request,shard,shards):
    return dict(plan_hash=digest(inputs/'plan.json'),request_hash=digest(request),code=code_identity(),shard=shard,shards=shards)
def key(r):return (r['pair_id'],r['config_id'],r['direction'])
def expected(sc,pairs,shard,shards):return {(p['id'],c['id'],di) for p in assigned(pairs,shard,shards) for c in sc['configs'] for di in c['directions']}

def chunks(out,ident):
    rows=[]
    for path in sorted((out/'chunks').glob('*.json')):
        marker=path.with_suffix('.ok')
        if not marker.exists():continue
        mark=json_read(marker)
        if mark['identity']!=ident or mark['sha256']!=digest(path):raise ValueError('Checkpoint identity/hash: '+str(path))
        rows.extend(json_read(path))
    return rows

def verify(out,inputs,schedule):
    ident=json_read(out/'manifest.json')['identity'];plan,sc,pairs=load(inputs,schedule)
    if ident['plan_hash']!=digest(inputs/'plan.json') or ident['schedule_hash']!=digest(schedule):raise ValueError('Run/input mismatch')
    if (sc['mode']=='mean')!=(ident['bank_hash'] is not None):raise ValueError('Replacement baseline mismatch')
    rows=chunks(out,ident);keys=[key(r) for r in rows];want=expected(sc,pairs,ident['shard'],ident['shards'])
    if len(keys)!=len(set(keys)) or set(keys)!=want:raise ValueError(f'Incomplete/duplicate coverage {len(keys)}/{len(want)}')
    pairs_by_id={p['id']:p for p in pairs}
    configs={c['id']:c for c in sc['configs']}
    for r in rows:
        p=pairs_by_id[r['pair_id']]
        if r['family']!=p['family'] or r['order']!=p['order'] or r['mode']!=sc['mode']:raise ValueError('Record labels')
        if not np.isfinite(r['effect']):raise ValueError('Nonfinite effect')
        if configs[r['config_id']]['observe'] and r['diagnostic'] is None:raise ValueError('Missing G3 diagnostic')
        if r['diagnostic']:
            diag=r['diagnostic']
            if len(diag['pattern'])!=len(p['original_ids']['clean'])+len(p['shared_prefix']):
                # shared_prefix is stored as token IDs, not a text string.
                raise ValueError('G3 attention length mismatch')
            if not np.isfinite(diag['pattern']).all() or not np.isfinite(diag['z']).all():raise ValueError('Nonfinite diagnostic')
            check(abs(sum(diag['pattern'])-1),.001,'attention normalization')
        if r['result'] is not None:
            for v in ['margin','clean_logit','corr_logit']:
                if not np.isfinite(r['result'][v]):raise ValueError('Nonfinite readout')
            check(abs(r['result']['margin']-(r['result']['clean_logit']-r['result']['corr_logit'])),1e-4,'stored margin')
            base=r.get('measurement_baseline_margin',r['baseline']['margin'])
            value=base-r['result']['margin'] if r['direction']=='noise' else r['result']['margin']-base
            check(abs(value-r['effect']),1e-4,'stored intervention effect')
    if not json_read(out/'done.json')['complete'] or not json_read(out/'gate.json')['passed']:raise ValueError('Missing successful completion/gate')
    return rows

def write_chunk(out,ident,number,rows):
    path=out/'chunks'/f'{number:07d}.json'
    json_write(path,rows);json_write(path.with_suffix('.ok'),dict(identity=ident,sha256=digest(path)))

def prior_cache(inputs,prior,mode,bankhash):
    found={};sources=[]
    for item in prior:
        schedule=Path(item['schedule']);sc=json_read(schedule)
        if sc['mode']!=mode:continue
        observe={c['id']:c['observe'] for c in sc['configs']}
        for folder in item['runs']:
            run=Path(folder);meta=json_read(run/'manifest.json')['identity']
            if meta['bank_hash']!=bankhash:continue
            rows=verify(run,inputs,schedule);sources.append(digest(run/'manifest.json'))
            for r in rows:
                k=key(r)
                if k in found:
                    check(abs(found[k]['effect']-r['effect']),.001,'duplicate cached effect')
                if k not in found or observe[r['config_id']]:found[k]=r
    return found,sources

def new_gates(e,pairs,plan):
    report=gates(e,pairs,plan);report['s42']=[]
    hs=[513,533,595,863,870]
    for p in pairs:
        if p['family'] not in plan['gate_families']:continue
        sp=spec_checked(e,p);g,d,n=sp['g'],sp['d'],sp['n']
        rec=e.capture_z(sp['clean'],hs,g,d);don=e.capture_z(sp['corr'],hs,g,d)
        patch={h:([n-1],rec['z'][h][n-1:n]) for h in hs}
        out,_=e.evaluate(sp['clean'],patch,g,d)
        errs={'joint_self':check(error(out,rec['output']),.001,'S42 simultaneous self')}
        for h in [513,533,595,863]:
            out,_=e.evaluate(sp['clean'],{h:([n-1],don['z'][h][n-1:n])},g,d)
            errs[f'F{h}']=check(abs(rec['output']['margin']-out['margin']-p['saved_single'][f'{h}:colon']['effect']),.05,'saved F')
        _,a=e.evaluate(sp['clean'],{},g,d,863,n-1)
        _,b=e.evaluate(sp['clean'],{870:([n-1],don['z'][870][n-1:n])},g,d,863,n-1)
        errs['future_attention_null']=check(float(np.max(np.abs(np.array(a['pattern'])-b['pattern']))),.001,'future attention')
        errs['future_output_null']=check(float(np.max(np.abs(np.array(a['z'])-b['z']))),.001,'future head output')
        report['s42'].append(dict(pair_id=p['id'],errors=errs))
    report['scope']='S4.1 path gates plus S4.2 simultaneous hooks, saved F and downstream attention null'
    return report

def start_model():
    import unittest
    from test_s42 import TinyJoint
    from test_stage4 import TinyRoutes
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in [TinyRoutes,TinyJoint])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:raise ValueError('Required tiny model gate failed/skipped')
    from stage1_scan import load_model
    from s42_engine import S42Engine
    t,tok,model,lock=load_model();return S42Engine(t,tok,model)

def run(a):
    plan,sc,pairs=load(a.inputs,a.schedule);allpairs=list(read_lines(a.inputs/'pairs.jsonl.gz'))
    if json_read(ROOT/'model_lock_olmo2.json')!=plan['model']:raise ValueError('Model lock')
    if (sc['mode']=='mean')!=(a.bank is not None):raise ValueError('Mean bank required exactly for sensitivity')
    ident=identity(a.inputs,a.schedule,a.shard,a.shards,a.bank);out=a.out
    if out.exists() and not a.resume:raise FileExistsError('Use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mf=out/'manifest.json'
        if mf.exists() and json_read(mf)['identity']!=ident:raise ValueError('Resume identity changed')
        json_write(mf,dict(identity=ident))
        old=chunks(out,ident);done={key(r) for r in old};want=expected(sc,pairs,a.shard,a.shards)
        if len(done)!=len(old) or not done<=want:raise ValueError('Invalid partial coverage')
        cache,sources=prior_cache(a.inputs,json_read(a.prior) if a.prior else [],sc['mode'],ident['bank_hash'])
        json_write(out/'reuse_sources.json',sources)
        e=start_model();gate=new_gates(e,allpairs,plan);json_write(out/'gate.json',gate)
        import transformers,platform
        json_write(out/'runtime.json',dict(torch=e.t.__version__,transformers=transformers.__version__,python=platform.python_version(),
            job=os.environ.get('SLURM_JOB_ID'),host=platform.node(),devices=str(getattr(e.model,'hf_device_map',{}))))
        if a.gate_only:return
        from s42_means import MeanBank
        bank=MeanBank(a.bank,a.inputs) if a.bank else None
        heads=sorted({x['head'] for c in sc['configs'] for x in c['atoms']})
        if any(c['observe'] for c in sc['configs']):heads=sorted(set(heads)|{863})
        stop=[False]
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        number=max([int(p.stem) for p in (out/'chunks').glob('*.json')]+[-1])+1
        pending=[]
        for p in assigned(pairs,a.shard,a.shards):
            todo=[(c,di) for c in sc['configs'] for di in c['directions'] if (p['id'],c['id'],di) not in done]
            if not todo:continue
            sp=spec_checked(e,p);g,d,n=sp['g'],sp['d'],sp['n'];caps={}
            # Paired donors and both intact baselines are shared by every configuration.
            for side,j in [('clean',0),('corr',1)]:
                caps[side]=e.capture_z(sp[side],heads,g,d)
                check(abs(caps[side]['output']['margin']-p['saved_baselines'][j]),.05,'per-pair baseline')
            for c,di in todo:
                if stop[0]:
                    if pending:write_chunk(out,ident,number,pending)
                    raise SystemExit(75)
                recipient,donor=('clean','corr') if di=='noise' else ('corr','clean')
                rec,don=caps[recipient],caps[donor];base=rec['output'];k=(p['id'],c['id'],di)
                row=dict(pair_id=p['id'],family=p['family'],order=p['order'],prefix=p['shared_prefix'],
                    query_first=p['masks']['_query_first'],config_id=c['id'],direction=di,mode=sc['mode'],
                    baseline=base,origin='new',diagnostic=None,norms={},fallback={})
                cached=cache.get(k);saved=reusable(p,c,di,sc['mode'])
                if cached is not None and (not c['observe'] or cached['diagnostic'] is not None):
                    row=dict(cached);row['origin']='prior_phase:'+cached['origin']
                elif saved is not None:
                    row.update(effect=saved['effect'],result=saved['result'],origin=saved['origin'],saved_reference=saved)
                    row['measurement_baseline_margin']=saved['baseline_margin']
                else:
                    patches,norms,fb=e.patches(p,c,rec,don,sc['mode'],bank)
                    result,diag=e.evaluate(sp[recipient],patches,g,d,863 if c['observe'] else None,n-1)
                    if diag:check(diag['reconstruction_error'],.02,'G3 attention reconstruction')
                    effect=base['margin']-result['margin'] if di=='noise' else result['margin']-base['margin']
                    row.update(effect=effect,result=result,diagnostic=diag,norms=norms,fallback=fb)
                    if all(x['donor']=='self' for x in c['atoms']):check(error(result,base),.001,'self clamp control')
                pending.append(row);done.add(k)
                if len(pending)>=16:write_chunk(out,ident,number,pending);number+=1;pending=[]
            if pending:write_chunk(out,ident,number,pending);number+=1;pending=[]
            print('Completed',sc['phase'],p['id'],len(done),'/',len(want),flush=True)
        if done!=want:raise ValueError('Coverage incomplete')
        json_write(out/'done.json',dict(complete=True,records=len(done)));verify(out,a.inputs,a.schedule)

def collect_means(a):
    from s42_means import accumulate
    plan,pairs=load(a.inputs);req=json_read(a.request)
    if req['plan_hash']!=digest(a.inputs/'plan.json'):raise ValueError('Mean request plan')
    ident=mean_identity(a.inputs,a.request,a.shard,a.shards);out=a.out
    if out.exists() and not a.resume:raise FileExistsError('Use --resume')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mf=out/'manifest.json'
        if mf.exists() and json_read(mf)['identity']!=ident:raise ValueError('Mean resume identity')
        json_write(mf,dict(identity=ident));e=start_model()
        vocab={k:i for i,k in enumerate(plan['mean_keys'])};grouped=collections.defaultdict(list)
        for p in assigned(pairs,a.shard,a.shards):grouped[p['family']].append(p)
        for fam,ps in grouped.items():
            path=out/'families'/f'{fam:03d}.npz';marker=path.with_suffix('.ok')
            if marker.exists():
                m=json_read(marker)
                if m['identity']!=ident or m['sha256']!=digest(path):raise ValueError('Mean checkpoint mismatch')
                continue
            if len(ps)!=2 or {p['order'] for p in ps}!={0,1}:raise ValueError('Expected both family orders')
            sums=np.zeros((len(req['heads']),len(vocab),e.DH),dtype=np.float64);counts=np.zeros(len(vocab),dtype=np.int64)
            for p in ps:
                sp=spec_checked(e,p)
                for side,j in [('clean',0),('corr',1)]:
                    cap=e.capture_z(sp[side],req['heads'],sp['g'],sp['d'])
                    check(abs(cap['output']['margin']-p['saved_baselines'][j]),.05,'mean capture baseline')
                    values=np.stack([cap['z'][h][:sp['n']].float().numpy() for h in req['heads']])
                    accumulate(sums,counts,p['mean_keys'],values,vocab)
            npz_write(path,family=fam,sums=sums,counts=counts)
            json_write(marker,dict(identity=ident,sha256=digest(path)))
            print('Mean family',fam,flush=True)
        json_write(out/'done.json',dict(complete=True))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare')
    for n in ['stage4','stage3-inputs','stage3-runs','proposal','out']:q.add_argument('--'+n,type=Path,required=True)
    q=sub.add_parser('check');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--schedule',type=Path)
    for name in ['run','means']:
        q=sub.add_parser(name)
        for n in ['inputs','out']:q.add_argument('--'+n,type=Path,required=True)
        q.add_argument('--shard',type=int,default=0);q.add_argument('--shards',type=int,default=1);q.add_argument('--resume',action='store_true')
        if name=='run':
            q.add_argument('--schedule',type=Path,required=True);q.add_argument('--prior',type=Path);q.add_argument('--bank',type=Path);q.add_argument('--gate-only',action='store_true')
        else:q.add_argument('--request',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='prepare':prepare(a.stage4,a.stage3_inputs,a.stage3_runs,a.proposal,a.out);print('Frozen S4.2 inputs ready')
    elif a.cmd=='check':
        plan,sc,pairs=load(a.inputs,a.schedule or a.inputs/'initial.json')
        print(dict(pairs=len(pairs),configs=len(sc['configs']),endpoints=sum(len(c['directions']) for c in sc['configs'])*len(pairs)))
    elif a.cmd=='run':run(a)
    else:collect_means(a)

if __name__=='__main__':main()
