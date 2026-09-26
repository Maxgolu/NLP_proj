"""Stage 4.1: gated, restartable single-worker measurements. CPU CLI imports no torch."""
import argparse
import collections
import contextlib
import json
import os
import platform
from pathlib import Path
import signal
import sys
import time
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,run_lock
from stage4_plan import POLICY,load,prepare

ROOT=Path(__file__).resolve().parent
CODE_FILES=['stage4_run.py','stage4_plan.py','stage4_engine.py','stage4_analyze.py',
            'stage3_common.py','stage3_engine.py','stage2_engine.py','stage1_scan.py',
            'stage1_audit.py','model_lock_olmo2.json','test_stage4.py']

def identity(inputs,schedule,shard,shards):
    return dict(plan_hash=digest(inputs/'plan.json'),schedule_hash=digest(schedule),
                code={n:digest(ROOT/n) for n in CODE_FILES},shard=shard,shards=shards)

def error(a,b):
    return max(abs(a[k]-b[k]) for k in ['clean_logit','corr_logit'])

def check(value,limit,label):
    if not np.isfinite(value) or value>limit:raise ValueError(f'Gate {label}: {value} > {limit}')
    return float(value)

def spec_checked(e,p):
    spec=e.encode(p)
    for side in ['clean','corr']:
        if spec[side][0,:spec['n']].tolist()!=p['original_ids'][side]:raise ValueError('Tokenizer/input IDs changed')
    if spec['shared_prefix']!=p['shared_prefix']:raise ValueError('Prefix mismatch')
    return spec

def gates(e,pairs,plan):
    """Six fixed pairs, individual logits plus saved metric replication; fail closed."""
    hs=[15*32+25,17*32+1,18*32+19,26*32+31]; reports=[]
    for p in pairs:
        if p['family'] not in plan['gate_families']:continue
        sp=spec_checked(e,p);g,d,n=sp['g'],sp['d'],sp['n']
        e.sync();start=time.monotonic()
        c=e.capture4(sp['clean'],hs,g,d);r=e.capture4(sp['corr'],hs,g,d)
        errs={}
        for side,cap,j in [('clean',c,0),('corr',r,1)]:
            errs[side+'_saved']=check(abs(cap['output']['margin']-p['saved_baselines'][j]),.05,'saved margin')
            errs[side+'_no_hook']=check(error(cap['output'],e.output(sp[side],g,d)),.001,'inert instrumentation')
            errs[side+'_sdpa']=check(cap['reconstruction_error'],.02,'SDPA reconstruction')
        errs['saved_clean_logits']=check(max(abs(c['output'][k]-v) for k,v in zip(['clean_logit','corr_logit'],p['clean_logits'])),.05,'saved individual logits')
        for h in hs:
            with e.vector_patch(h,list(range(n)),c['z'][h][:n]):out=e.output(sp['clean'],g,d)
            errs[f'self_{h}']=check(error(out,c['output']),.001,'self patch')
            with e.vector_patch(h,list(range(n)),r['z'][h][:n]):out=e.output(sp['clean'],g,d)
            errs[f'P_{h}']=check(abs(out['margin']-c['output']['margin']-p['saved_P'][str(h)]),.05,'saved Scope P')
        src=17*32+1;rec=18*32+19;pos=p['masks']['query_is_token']
        hy=e.hybrid(sp['clean'],src,pos,r,c,hs,g,d)
        selfhy=e.hybrid(sp['clean'],src,pos,c,c,hs,g,d)
        cfg=dict(kind='head',receiver=rec,channel='V')
        out=e.endpoint(sp['clean'],cfg,pos,selfhy,g,d,n)
        errs['self_path']=check(error(out,c['output']),.001,'self path')
        # Different-position Q cannot receive a direct residual update when intervening branches are frozen.
        errs['cross_position_Q']=check(float((hy['qkv'][rec]['Q'][n-1]-c['qkv'][rec]['Q'][n-1]).abs().max()),.001,'cross-position Q')
        # QKV at all source rows must equal replacing the actual recipient head output at the colon.
        inj=dict(head=rec,positions=list(range(sp['clean'].shape[1])),row=n-1,channel='QKV',qkv=hy['qkv'][rec])
        with e.instrument(injection=inj):qout=e.output(sp['clean'],g,d)
        with e.vector_patch(rec,[n-1],hy['z'][rec][n-1:n]):zout=e.output(sp['clean'],g,d)
        errs['qkv_output']=check(error(qout,zout),.05,'QKV vs output endpoint')
        # Same-layer and future-to-past captures must be unchanged.
        for receiver in [src,15*32+25]:
            delta=max(float((hy['qkv'][receiver][ch]-c['qkv'][receiver][ch]).abs().max()) for ch in 'QKV')
            errs[f'temporal_{receiver}']=check(delta,.001,'temporal null')
        # Freeze all post-source increments: only the original source-position residual delta survives.
        errs['cross_position_bypass']=check(error(hy['output'],c['output']),.001,'fact-to-final residual bypass')
        # Joint self replacement checks simultaneous hooks and all-retained identity without introducing means.
        with contextlib.ExitStack() as stack:
            for h in hs:stack.enter_context(e.vector_patch(h,list(range(n)),c['z'][h][:n]))
            out=e.output(sp['clean'],g,d)
        errs['joint_self']=check(error(out,c['output']),.001,'joint self identity')
        e.sync();reports.append(dict(pair_id=p['id'],errors=errs,seconds=time.monotonic()-start))
        print('Gate passed:',p['id'],flush=True)
    if len(reports)!=6:raise ValueError('Expected six gate pairs')
    return dict(passed=True,pairs=reports,scope='S4.1 path gates; mean-mask gates belong to later mean-ablation implementation')

def expected_keys(sc,pairs,shard,shards):
    return {(p['id'],c['id'],di) for i,p in enumerate(pairs) if i%shards==shard for c in sc['configs'] for di in sc['directions']}

def chunks(out,ident):
    rows=[]
    for p in sorted((out/'chunks').glob('*.json')):
        marker=p.with_suffix('.ok')
        if not marker.exists():continue
        meta=json_read(marker)
        if meta['identity']!=ident or meta['sha256']!=digest(p):raise ValueError('Invalid checkpoint: '+str(p))
        rows.extend(json_read(p))
    return rows

def verify_run(out,inputs,schedule,require_done=True):
    meta=json_read(out/'manifest.json'); ident=meta['identity']
    # Analysis remains portable: compare data identity, not local code byte line endings.
    if ident['plan_hash']!=digest(inputs/'plan.json') or ident['schedule_hash']!=digest(schedule):raise ValueError('Run/input identity mismatch')
    plan,sc,pairs=load(inputs,schedule)
    rows=chunks(out,ident);keys=[(x['pair_id'],x['config_id'],x['direction']) for x in rows]
    want=expected_keys(sc,pairs,ident['shard'],ident['shards'])
    if len(keys)!=len(set(keys)) or set(keys)!=want:raise ValueError(f'Incomplete/duplicate coverage {len(set(keys))}/{len(want)}')
    if require_done and not json_read(out/'done.json')['complete']:raise ValueError('Missing completion')
    if not json_read(out/'gate.json')['passed']:raise ValueError('Missing gate')
    return rows

def run(args):
    plan,sc,pairs=load(args.inputs,args.schedule)
    allpairs=list(read_lines(args.inputs/'pairs.jsonl.gz'))
    if not 0<=args.shard<args.shards:raise ValueError('Invalid shard')
    if sc.get('requires_coverage') and not args.gate_only:
        raise ValueError('Use next after coverage; seed.json is a preview, not approved execution input')
    if json_read(ROOT/'model_lock_olmo2.json')!=plan['model']:raise ValueError('Model lock mismatch')
    ident=identity(args.inputs,args.schedule,args.shard,args.shards)
    out=args.out
    if out.exists() and not args.resume:raise FileExistsError('Use --resume for exactly the same run')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mp=out/'manifest.json'
        if mp.exists():
            if json_read(mp)['identity']!=ident:raise ValueError('Cannot resume changed code/plan/sharding')
        else:json_write(mp,dict(identity=ident,inputs=str(args.inputs.resolve()),schedule=str(args.schedule.resolve()),created=time.time()))
        import unittest
        from test_stage4 import TinyRoutes
        tiny=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TinyRoutes))
        if not tiny.wasSuccessful() or tiny.skipped:raise ValueError('Mandatory tiny-model implementation gate failed/skipped')
        json_write(out/'tiny_gate.json',dict(passed=True,tests=tiny.testsRun))
        from stage1_scan import load_model
        from stage4_engine import Stage4Engine
        json_write(out/'state.json',dict(status='loading',time=time.time()))
        t,tok,model,lock=load_model();e=Stage4Engine(t,tok,model)
        import transformers
        json_write(out/'runtime.json',dict(torch=t.__version__,transformers=transformers.__version__,python=platform.python_version(),
                                          host=platform.node(),slurm_job=os.environ.get('SLURM_JOB_ID'),devices=str(getattr(model,'hf_device_map',{}))))
        gate=gates(e,allpairs,plan);json_write(out/'gate.json',gate)
        json_write(out/'gpu_memory_after_gate.json',dict(peak_allocated_bytes={str(i):t.cuda.max_memory_allocated(i) for i in range(t.cuda.device_count())}))
        if args.gate_only:
            json_write(out/'state.json',dict(status='gate_passed',time=time.time()));return
        oldrows=chunks(out,ident);done={(r['pair_id'],r['config_id'],r['direction']) for r in oldrows}
        stopped=[False]; self_checked=set()
        for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
        groups=collections.defaultdict(list)
        for cfg in sc['configs']:groups[(cfg['source'],cfg['site'])].append(cfg)
        heads=sorted({c['source'] for c in sc['configs']}|{c['receiver'] for c in sc['configs'] if c['kind']=='head'})
        (out/'chunks').mkdir(exist_ok=True)
        for pi,p in enumerate(pairs):
            if pi%args.shards!=args.shard:continue
            if all((p['id'],c['id'],di) in done for c in sc['configs'] for di in sc['directions']):continue
            sp=spec_checked(e,p);g,d,n=sp['g'],sp['d'],sp['n']
            profile_positions=dict(mother_last=p['masks']['query_mother'][-1:],is_token=p['masks']['query_is_token'],
                                   child_last=p['masks']['query_child_last'],period=p['masks']['query_period'],colon=[n-1])
            caps={k:e.capture4(sp[k],heads,g,d,profile_positions=profile_positions) for k in ['clean','corr']}
            for side,j in [('clean',0),('corr',1)]:
                check(abs(caps[side]['output']['margin']-p['saved_baselines'][j]),.05,'per-pair saved baseline')
                check(caps[side]['reconstruction_error'],.02,'per-pair SDPA reconstruction')
            profiles=[]
            for li in range(e.L):
                for site in profile_positions:
                    a=caps['clean']['residual_profiles'][li][site];b=caps['corr']['residual_profiles'][li][site]
                    profiles.append(dict(layer=li,site=site,positions=profile_positions[site],clean_norm=float(a.norm()),
                                         corrupt_norm=float(b.norm()),distance=float((a-b).norm())))
            profile_path=out/'profiles'/f'f{p["family"]:03d}_o{p["order"]}.json'
            json_write(profile_path,dict(pair_id=p['id'],identity=ident,profiles=profiles))
            for (src,site),configs in groups.items():
                positions=p['masks'][site]
                for di in sc['directions']:
                    todo=[c for c in configs if (p['id'],c['id'],di) not in done]
                    if not todo:continue
                    if stopped[0]:
                        json_write(out/'state.json',dict(status='interrupted',completed=len(done)))
                        raise SystemExit(75)
                    recipient,donor=('clean','corr') if di=='noise' else ('corr','clean')
                    ids=sp[recipient];rec=caps[recipient];don=caps[donor];base=rec['output']
                    e.sync();start=time.monotonic()
                    hy=None
                    if any(c['kind']!='single' for c in todo):hy=e.hybrid(ids,src,positions,don,rec,heads,g,d)
                    records=[]
                    for cfg in todo:
                        # Every channel/mask configuration gets a self-donor endpoint
                        # control once per worker, in addition to the six-pair hybrid gates.
                        self_error=None
                        if cfg['id'] not in self_checked and cfg['kind'] not in ['single','bypass']:
                            inert=e.endpoint(ids,cfg,positions,rec,g,d,n)
                            self_error=check(error(inert,base),.001,'configuration self-donor endpoint')
                            self_checked.add(cfg['id'])
                        saved=None
                        if cfg['kind']=='single' and site=='all' and di=='noise' and str(src) in p['saved_P']:
                            saved=-p['saved_P'][str(src)]
                        if saved is not None:
                            result=None;effect=saved;origin='saved_stage2_exact_P';norm=None
                        else:
                            if cfg['kind']=='single':
                                with e.vector_patch(src,positions,don['z'][src][positions]):result=e.output(ids,g,d)
                                norm=float((don['z'][src][positions].float()-rec['z'][src][positions].float()).norm())
                            else:
                                result=e.endpoint(ids,cfg,positions,hy,g,d,n)
                                norm=e.channel_norm(cfg,positions,hy,rec,n)
                            effect=base['margin']-result['margin'] if di=='noise' else result['margin']-base['margin'];origin='stage4_measurement'
                            if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite endpoint')
                        records.append(dict(pair_id=p['id'],family=p['family'],order=p['order'],query_first=p['masks']['_query_first'],
                            config_id=cfg['id'],direction=di,config=cfg,source_positions=positions,receiver_row=n-1,
                            prefix=p['shared_prefix'],baseline=base,result=result,effect=float(effect),origin=origin,channel_norm=norm,
                            self_endpoint_error=self_error,
                            frozen='postnorm branches >= source except source attention; endpoint downstream live'))
                    name=f'f{p["family"]:03d}_o{p["order"]}_h{src}_{site}_{di}.json';path=out/'chunks'/name
                    # One atomic chunk per pair/source/site/direction; resume never rewrites verified chunks.
                    json_write(path,records);json_write(path.with_suffix('.ok'),dict(identity=ident,sha256=digest(path)))
                    done.update((x['pair_id'],x['config_id'],x['direction']) for x in records)
                    e.sync();json_write(out/'state.json',dict(status='running',completed=len(done),expected=len(expected_keys(sc,pairs,args.shard,args.shards)),last_chunk=name,seconds=time.monotonic()-start))
                    print(f'{len(done)} records; {name}',flush=True)
                    del hy
        verify_run(out,args.inputs,args.schedule,require_done=False)
        json_write(out/'done.json',dict(complete=True,records=len(done),time=time.time()))
        json_write(out/'state.json',dict(status='complete',completed=len(done)))

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--stage3',type=Path,required=True);p.add_argument('--design',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('compare');p.add_argument('--analyses',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for name in ['check','run','analyze','next']:
        p=sub.add_parser(name);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
        if name in ['analyze','next']:p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
        if name=='next':
            p.add_argument('--mode',choices=['after-coverage','after-localize','refine','extend'],required=True)
            p.add_argument('--prior-extension',type=Path,help='Required for refinement extension: reserves the shared 96-route budget')
        if name=='run':
            p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--gate-only',action='store_true')
            p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=ap.parse_args()
    if a.command=='prepare':
        prepare(a.stage3,a.design,a.out);print('Prepared discovery-only inputs:',a.out)
    elif a.command=='compare':
        from stage4_analyze import compare
        compare(a.analyses,a.out)
    elif a.command=='check':
        plan,sc,pairs=load(a.inputs,a.schedule)
        print(json.dumps(dict(phase=sc.get('phase'),pairs=len(pairs),configs=len(sc['configs']),endpoint_evaluations=len(pairs)*len(sc['configs'])*len(sc['directions']),
                              excludes='hybrid captures, baseline captures, gates, conditional follow-ups',preview_only=sc.get('requires_coverage',False)),indent=2))
    elif a.command=='run':run(a)
    else:
        from stage4_analyze import analyze,next_schedule
        if a.command=='analyze':analyze(a.inputs,a.schedule,a.runs,a.out)
        else:next_schedule(a.inputs,a.schedule,a.runs,a.out,a.mode,a.prior_extension)

if __name__=='__main__':main()
