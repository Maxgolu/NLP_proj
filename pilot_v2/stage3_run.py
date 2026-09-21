"""Restartable Stage-3 discovery measurements. --check-inputs needs only NumPy."""
import argparse
import collections
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from stage3_common import *

ROOT=Path(__file__).resolve().parent
CODE_FILES=['stage3_run.py','stage3_common.py','stage3_engine.py','stage3_measure.py',
            'stage3_analyze.py','stage2_engine.py','stage1_scan.py','stage1_audit.py','model_lock_olmo2.json']
CHUNK_SIZE=5


def validate_inputs(inputs):
    plan=json_read(inputs/'plan.json')
    if plan['policy']!=POLICY:raise ValueError('Input policy and code differ')
    if json_read(ROOT/'model_lock_olmo2.json')!=plan['model']:raise ValueError('Model lock mismatch')
    for name,sha in plan['files'].items():
        if digest(inputs/name)!=sha:raise ValueError('Input changed: '+name)
    pairs=list(read_lines(inputs/'pairs.jsonl.gz'));prompts=list(read_lines(inputs/'prompts.jsonl.gz'))
    if len(pairs)!=178 or len(prompts)!=534 or len(plan['heads'])!=105:raise ValueError('Coverage mismatch')
    byid={p['id']:p for p in prompts};position_count=0
    for p in pairs:
        a,b=[byid[p[k]['id']]['token_ids'] for k in ['clean','corr']]
        if len(a)!=len(b):raise ValueError('Unaligned pair')
        if p['exact_all']:position_count+=len(a)-next(i for i in range(len(a)) if a[i]!=b[i])
    with np.load(inputs/'references.npz') as z:missing=int(np.isnan(z['scopeF_delta']).sum())
    # Colon results are reused in the per-position profile.
    counts=dict(missing_final_position=missing,position_scan_extra=(position_count-40)*105,
                attention_value=4*178*105,reverse=2*178*len(plan['reverse_heads']))
    return plan,pairs,prompts,dict(patched_forwards=sum(counts.values()),components=counts,
                                 note='Excludes gates, captures and diagnostics; wall time measured by GPU gate.')


def checkpoint_done(path,identity):
    marker=path.with_name(path.name+'.ok.json')
    if not marker.exists():return False
    info=json_read(marker)
    if info['identity']!=identity or not path.exists() or digest(path)!=info['sha256']:
        raise ValueError('Checkpoint identity/checksum mismatch: '+str(path))
    for name,sha in info.get('companions',{}).items():
        if digest(path.parent/name)!=sha:raise ValueError('Checkpoint companion changed: '+name)
    return True


def finish_checkpoint(path,identity,companions=()):
    json_write(path.with_name(path.name+'.ok.json'),dict(identity=identity,sha256=digest(path),
               companions={p.name:digest(p) for p in companions}))


def pair_filename(pair,chunk):return f'family_{pair["family"]:03d}_order{pair["order"]}_chunk{chunk:02d}.npz'


def gate(e,heads,pairs,plan,ref,inputs):
    from stage1_scan import runner_score
    from stage3_measure import checked_capture,matched_ri
    reports=e.gate([p for p in pairs if p['family'] in plan['gate_families']],json_read(inputs/'behavioral_reference.json'),runner_score)
    details=[]
    probes=sorted(set([heads[0],heads[len(heads)//2],heads[-1]]))
    for index,p in enumerate(pairs):
        if p['family'] not in plan['gate_families']:continue
        spec=e.encode(p);n=spec['n'];rows=list(range(n));base=e.readout(spec['clean'],spec['g'],spec['d'])
        clean=checked_capture(e,spec['clean'][:,:n],probes,rows)
        corr=checked_capture(e,spec['corr'][:,:n],probes,rows)
        first=next(i for i in rows if spec['clean'][0,i]!=spec['corr'][0,i]);errors=[]
        bare=e.readout(spec['clean'][:,:n],spec['g'],spec['d'])
        captured=clean['logits'];drift=abs(float(captured[spec['g']]-captured[spec['d']])-bare[0])
        if drift>POLICY['inert_tolerance']:raise ValueError('Capture altered model forward')
        for h in probes:
            z=clean['heads'][h]['z'];z1=corr['heads'][h]['z']
            inert=e.patched(spec,h,[n-1],z[n-1:n])
            prefix=e.patched(spec,h,list(range(first)),z1[:first])
            if max(np.max(abs(inert-base)),np.max(abs(prefix-base)))>POLICY['inert_tolerance']:
                raise ValueError('Self/identical-prefix patch failed')
            exact=e.patched(spec,h,rows,z1);saved=ref['scopeP_delta'][index,plan['heads'].index(h)]
            if abs(exact[0]-base[0]-saved)>POLICY['probe_tolerance']:raise ValueError('Saved Scope P replication failed')
            f=e.patched(spec,h,[n-1],z1[n-1:n])
            mix=e.mix_vectors(clean,corr,h,n-1)
            err=max(np.max(abs(e.patched(spec,h,[n-1],mix[0])-base)),np.max(abs(e.patched(spec,h,[n-1],mix[3])-f)))
            if err>POLICY['probe_tolerance']:raise ValueError('AV reconstruction readout failed')
            errors.append(float(err))
        details.append(dict(id=p['id'],capture_drift=drift,av_drift=max(errors)))
    # Fail early on the raw RI replication convention, before the expensive scan.
    evs=[x for x in read_lines(inputs/'events.jsonl.gz') if x['layer']*32+x['head'] in heads and x['scores']]
    if evs:
        selected=evs[:1];r=next(r for r in read_lines(inputs/'prompts.jsonl.gz') if r['id']==selected[0]['id'])
        hs=sorted({x['layer']*32+x['head'] for x in selected});rows=sorted({x['j'] for x in selected})
        cap=checked_capture(e,e.t.tensor([r['token_ids']],device=e.device),hs,rows,True)
        checked=matched_ri(e,r,cap,selected)
        reports['raw_ri_probe']=checked
    reports['stage3']=details
    reports['patched_forward_seconds']=float(np.median([x['patched_forward_s'] for x in reports['pairs']]))
    return reports


def worker(out,rank):
    from stage1_scan import load_model,runner_score
    from stage1_audit import first_divergence
    from stage3_engine import Stage3Engine
    from stage3_measure import (causal_pair,checked_capture,matched_ri,anatomy,token_pool,copying_weights,synthetic)
    import transformers
    m=json_read(out/'manifest.json');inputs=Path(m['inputs']);plan,pairs,prompts,_=validate_inputs(inputs)
    heads=m['shards'][rank];identity=m['identity']+f':{rank}';base=out/f'replica_{rank}'
    base.mkdir(exist_ok=True)
    def state(status,**kw):json_write(out/f'state_{rank}.json',dict(status=status,**kw))
    if rank:
        while not (out/f'loaded_{rank-1}.json').exists():time.sleep(1)
    state('loading model');start=time.monotonic();t,tok,model,lock=load_model()
    if lock!=plan['model']:raise ValueError('Loaded model lock differs')
    e=Stage3Engine(t,tok,model)
    if (e.L,e.H)!=(32,32):raise ValueError('Expected 32 layers and 32 heads')
    runtime=dict(torch=t.__version__,transformers=transformers.__version__,dtype=str(next(model.parameters()).dtype),
                 attention=model.config._attn_implementation,device_map={k:str(v) for k,v in model.hf_device_map.items()})
    previous=base/'runtime.json'
    if previous.exists():
        old=json_read(previous)
        for key in ['torch','transformers','dtype','attention']:
            if old[key]!=runtime[key]:raise ValueError('Resume runtime changed: '+key)
    else:json_write(previous,runtime)
    json_write(out/f'loaded_{rank}.json',dict(seconds=time.monotonic()-start))
    with np.load(inputs/'references.npz') as z:ref={k:z[k].copy() for k in z.files}
    state('gating');report=gate(e,heads,pairs,plan,ref,inputs)
    json_write(out/f'gate_{rank}.json',report)
    while not (out/'gate_passed.json').exists():time.sleep(1)
    if m['gate_only']:return
    # Independent batches of five heads cap work lost if a killable job is stopped.
    chunks=[heads[i:i+CHUNK_SIZE] for i in range(0,len(heads),CHUNK_SIZE)]
    for i,p in enumerate(pairs):
        for ci,hs in enumerate(chunks):
            path=base/'pairs'/pair_filename(p,ci);tag=identity+':'+p['id']+':'+str(ci)
            if checkpoint_done(path,tag):continue
            state('causal profiles',pair=i+1,total=len(pairs),chunk=ci+1,chunks=len(chunks))
            values,roles=causal_pair(e,p,hs,plan,ref,i)
            npz_write(path,**values);side=path.with_suffix('.roles.json');json_write(side,roles)
            finish_checkpoint(path,tag,[side])
    all_events=collections.defaultdict(list)
    for x in read_lines(inputs/'events.jsonl.gz'):
        if x['layer']*32+x['head'] in heads and x['scores']:all_events[x['id']].append(x)
    scan=list(prompts)+[changed_query(p['clean']) for p in pairs if p['exact_all']]
    for i,r in enumerate(scan):
        core=r['family'] in plan['common_families'];evs=all_events[r['id']]
        if not core and not evs:continue
        path=base/'prompts'/(slug(r['id'])+'.jsonl.gz');tag=identity+':'+r['id']
        if checkpoint_done(path,tag):continue
        state('attention/output and contextual RI',prompt=i+1,total=len(scan))
        ann=annotate(tok,r);ids=ann['ids']
        if 'token_ids' in r and ids!=r['token_ids']:raise ValueError('Tokenizer differs from saved RI')
        hs=heads if core else sorted({x['layer']*32+x['head'] for x in evs})
        rows=ann['test_positions'] if core else sorted({x['j'] for x in evs})
        cap=checked_capture(e,t.tensor([ids],device=e.device),hs,rows,bool(evs))
        ri=matched_ri(e,r,cap,evs) if evs else []
        records=anatomy(e,r,hs,cap,ann) if core else []
        # Full behavior uses the SAME candidate continuation convention as Stage 2.
        spec=first_divergence(tok,r);readout=e.readout(t.tensor([spec['input_ids']],device=e.device),spec['gold_token'],spec['other_token'])
        other=next(c for c in r['candidates'] if c!=r['gold'])
        gold_lp=runner_score(t,tok,model,e.device,r['prompt'],r['gold'])
        other_lp=runner_score(t,tok,model,e.device,r['prompt'],other)
        meta=dict(kind='metadata',id=r['id'],family=r['family'],order=r['order'],variant=r['variant'],
                  core=core,heads=hs,annotation=ann,shared_prefix=spec['shared_prefix'],
                  divergent_readout=readout.tolist(),candidate_margin=gold_lp-other_lp)
        write_lines(path,[meta]+[dict(kind='anatomy',**x) for x in records]+[dict(kind='contextual_ri',**x) for x in ri])
        companions=[]
        if core:
            attention=path.with_name(path.name.replace('.jsonl.gz','.attention.npz'))
            npz_write(attention,heads=hs,rows=rows,pattern=np.stack([cap['heads'][h]['pattern'].numpy() for h in hs]))
            companions.append(attention)
        finish_checkpoint(path,tag,companions)
    pool_path=base/'token_pool.json';pool=token_pool(tok,prompts)
    if pool_path.exists() and json_read(pool_path)!=pool:raise ValueError('Token reference pool changed')
    json_write(pool_path,pool)
    path=base/'copying_weights.npz'
    if not checkpoint_done(path,identity+':weights'):
        state('weight copying');values,stats=copying_weights(e,heads,pool);npz_write(path,**values)
        side=base/'copying_summary.json';json_write(side,stats);finish_checkpoint(path,identity+':weights',[side,pool_path])
    path=base/'synthetic.jsonl.gz'
    if not checkpoint_done(path,identity+':synthetic'):
        state('repeated-sequence and retrieval fingerprints');write_lines(path,synthetic(e,heads,pool))
        finish_checkpoint(path,identity+':synthetic',[pool_path])
    json_write(out/f'done_{rank}.json',dict(complete=True));state('complete')


def controller(a):
    inputs=a.inputs.resolve();plan,pairs,prompts,work=validate_inputs(inputs)
    print(json.dumps(work,indent=2),flush=True)
    if a.check_inputs:return
    import torch
    from stage1_scan import RUNS
    if a.gpus not in [2,4,6] or torch.cuda.device_count()!=a.gpus:raise ValueError('Allocate exactly 2, 4 or 6 GPUs')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(a.gpus)))).split(',')
    groups=[devices[i:i+2] for i in range(0,len(devices),2)]
    shards=[plan['heads'][i::len(groups)] for i in range(len(groups))]
    stable=dict(input_hash=digest(inputs/'plan.json'),code_hashes={n:digest(ROOT/n) for n in CODE_FILES},
                shards=shards,chunk_size=CHUNK_SIZE,gate_only=a.gate_only)
    identity=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    m=dict(**stable,identity=identity,inputs=str(inputs),workload=work,model=plan['model'])
    if not re.fullmatch('[A-Za-z0-9_-]+',a.name):raise ValueError('Invalid run name')
    out=RUNS/a.name
    if out.exists():
        if not a.resume or json_read(out/'manifest.json')!=m:raise ValueError('Resume requires identical inputs, code, GPU count, mode and name')
    else:out.mkdir(parents=True);json_write(out/'manifest.json',m)
    with run_lock(out):
        for name in ['gate_passed.json','summary.json',*[f'{k}_{r}.json' for r in range(len(groups)) for k in ['gate','loaded','done','state']]]:
            (out/name).unlink(missing_ok=True)
        children=[];logs=[]
        def stop(*_):raise InterruptedError('Interrupted. Resume the same run with the same settings.')
        signals={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
        def wait(kind):
            last=0
            while True:
                for r,c in enumerate(children):
                    if c.poll() not in (None,0) or (c.poll()==0 and not (out/f'{kind}_{r}.json').exists()):
                        raise RuntimeError(f'Replica {r} failed; read worker_{r}.log')
                if all((out/f'{kind}_{r}.json').exists() for r in range(len(children))):return
                if time.monotonic()-last>30:
                    print(kind,[json_read(out/f'state_{r}.json') if (out/f'state_{r}.json').exists() else 'starting' for r in range(len(children))],flush=True);last=time.monotonic()
                time.sleep(1)
        try:
            for r,g in enumerate(groups):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                f=(out/f'worker_{r}.log').open('a',encoding='utf-8');logs.append(f)
                children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(r),'--out',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT))
            wait('gate')
            rates=[json_read(out/f'gate_{r}.json')['patched_forward_seconds'] for r in range(len(groups))]
            estimate=dict(passed=True,seconds_per_patch=rates,patched_forwards=work['patched_forwards'],
                          estimated_patch_wall_hours=max(rates)*work['patched_forwards']/len(groups)/3600,
                          caveat='Approximate; add model load, diagnostics, captures, IO and imbalance. No scope reduction.')
            json_write(out/'cost_estimate.json',estimate);print(json.dumps(estimate,indent=2),flush=True)
            json_write(out/'gate_passed.json',dict(passed=True))
            if a.gate_only:
                for c in children:
                    if c.wait(timeout=60)!=0:raise RuntimeError('Gate worker failed')
                json_write(out/'summary.json',dict(complete=False,gate_only=True,gate_passed=True));return
            wait('done')
            for c in children:
                if c.wait(timeout=60)!=0:raise RuntimeError('Worker failed after completion')
            from stage3_analyze import analyze
            analyze(out,inputs)
        finally:
            for c in children:
                if c.poll() is None:c.terminate()
            for c in children:
                try:c.wait(timeout=10)
                except subprocess.TimeoutExpired:c.kill();c.wait()
            for f in logs:f.close()
            for s,old in signals.items():signal.signal(s,old)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=ROOT/'inputs')
    ap.add_argument('--name',default='stage3_v1');ap.add_argument('--gpus',type=int,default=6)
    ap.add_argument('--gate-only',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--check-inputs',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if a.worker is not None:worker(a.out,a.worker)
    else:controller(a)


if __name__=='__main__':main()
