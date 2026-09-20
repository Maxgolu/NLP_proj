"""Persistent model replicas: gate ALL workers, screen, exact map, IG fallback,
then candidate extension. Stage-1 anatomy is deliberately not a dependency.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
from stage1_audit import file_hash
from stage2_common import atomic_json,groups,make_pairs,plan_work,spearman

ROOT=Path(__file__).resolve().parent
CODE_FILES=['stage2_run.py','stage2_engine.py','stage2_common.py','stage1_scan.py','stage1_audit.py']


def pair_file(out,phase,p):
    return out/phase/f'family_{p["family"]:03d}_order{p["order"]}.npz'


def save_npz(path,**data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**data)
    temp.replace(path)


def wait_command(out,phase):
    while True:
        p=out/'command.json'
        if p.exists() and json.loads(p.read_text())['phase']==phase:return json.loads(p.read_text())
        time.sleep(1)


def worker(out,rank):
    from stage2_engine import Engine
    from stage1_scan import load_model,runner_score
    m=json.loads((out/'manifest.json').read_text());pairs=m['pairs']
    def state(status,**kw):atomic_json(out/f'state_{rank}.json',dict(status=status,**kw))
    # Serial initial loading reduces simultaneous reads of large shared weights.
    if rank:
        state('waiting for previous replica load')
        while not (out/f'loaded_{rank-1}.json').exists():time.sleep(1)
    state('loading model');print(f'Replica {rank}: loading pinned model on allocated GPUs.',flush=True)
    start=time.monotonic();torch,tok,model,lock=load_model()
    if lock!=m['model']:raise ValueError('Model lock differs from manifest')
    engine=Engine(torch,tok,model)
    if (engine.L,engine.H)!=(32,32):raise ValueError('Expected 32 x 32 heads')
    atomic_json(out/f'loaded_{rank}.json',dict(seconds=time.monotonic()-start,
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'),device_map={k:str(v) for k,v in model.hf_device_map.items()}))
    # Token alignment is checked on every pair BEFORE allowing stage 2.
    for p in pairs:
        spec=engine.encode(p)
        if spec['shared_prefix']!=m['shared_prefixes'][p['id']]:raise ValueError('Stage-1 token specification mismatch')
    refs={json.loads(l)['id']:json.loads(l)['gold_minus_best_other'] for l in open(m['baseline'],encoding='utf-8')}
    state('running mandatory gate')
    gate_pairs=[p for p in pairs if p['family'] in m['gate_families']]
    gate=engine.gate(gate_pairs,refs,runner_score)
    atomic_json(out/f'gate_{rank}.json',gate);state('gate passed; waiting for all replicas')
    wait_command(out,'screen')
    assigned=[p for p in pairs if p['worker']==rank]
    all_heads=[(li,hi) for li in range(32) for hi in range(32)]
    def process(phase,head_indices=None):
        for i,p in enumerate(assigned):
            path=pair_file(out,phase,p)
            if path.exists():
                if phase=='extension':
                    with np.load(path) as z:
                        if not np.array_equal(z['head_indices'],head_indices):
                            raise ValueError('Cached extension shortlist changed')
                continue
            state(phase,pair=p['id'],index=i+1,total=len(assigned))
            print(f'Replica {rank}: {phase} {i+1}/{len(assigned)} {p["id"]}',flush=True)
            spec,clean,corr,m0,mc=engine.prepare(p)
            start=time.monotonic()
            data=dict(pair_id=p['id'],m_clean=m0,m_corr_fixed_sign=mc,
                      family=p['family'],order=p['order'])
            if phase=='screen':
                data['attribution']=engine.attribution(spec,clean,corr)
                if p['exact_all']:
                    exact,sec=engine.exact(spec,corr,all_heads)
                    data['exact_delta']=np.array([exact[h]-m0 for h in range(1024)])
                    data['exact_seconds']=sec
            elif phase=='ig':
                data['attribution']=engine.attribution(spec,clean,corr,steps=m['ig_steps'])
            elif phase=='extension':
                if p['exact_all']:
                    with np.load(pair_file(out,'screen',p)) as old:
                        data['exact_delta']=old['exact_delta'][head_indices]
                else:
                    exact,sec=engine.exact(spec,corr,[divmod(h,32) for h in head_indices])
                    data['exact_delta']=np.array([exact[h]-m0 for h in head_indices])
                    data['exact_seconds']=sec
                data['head_indices']=head_indices
            data['seconds']=time.monotonic()-start
            save_npz(path,**data)
            del clean,corr
        atomic_json(out/f'done_{phase}_{rank}.json',dict(complete=True))
        state(phase+' complete')
    process('screen')
    cmd=wait_command(out,'decision')
    if cmd['use_ig']:process('ig')
    else:atomic_json(out/f'done_ig_{rank}.json',dict(skipped=True))
    cmd=wait_command(out,'extension')
    process('extension',cmd['heads'])
    state('complete')


def load_vectors(out,phase,pairs,field,length=1024):
    vectors=[]
    for p in pairs:
        with np.load(pair_file(out,phase,p),allow_pickle=False) as z:
            if str(z['pair_id'])!=p['id']:raise ValueError('Pair identity mismatch')
            a=z[field]
            if a.shape!=(length,) or not np.isfinite(a).all():raise ValueError('Incomplete or nonfinite results')
            vectors.append(a.copy())
    return np.stack(vectors)


def ri_scores(stage1):
    sums=np.zeros((1024,3))
    with (stage1/'head_stats.csv').open() as f:
        for r in csv.DictReader(f):
            h=int(r['layer'])*32+int(r['head']);n=int(r['n_a'])
            sums[h]+=[n,float(r['strength_first'])*n,float(r['strength_last'])*n]
    return sums[:,1:]/np.maximum(sums[:,:1],1),sums[:,0]


def shortlist(scores,counts,estimate,seed):
    chosen={3*32+11,9*32+22,16*32+4}
    eligible=np.flatnonzero(counts>=50)
    for col in range(2):chosen.update(eligible[np.argsort(-scores[eligible,col],kind='stable')[:25]].tolist())
    chosen.update(np.argsort(-np.abs(estimate),kind='stable')[:25].tolist())
    random_controls=random.Random(seed).sample(sorted(set(range(1024))-chosen),25)
    return sorted(chosen|set(random_controls)),random_controls


def quality(exact,estimate):
    return dict(mean_head_signed_spearman=spearman(exact.mean(axis=0),estimate.mean(axis=0)),
        mean_head_absolute_spearman=spearman(np.abs(exact.mean(axis=0)),np.abs(estimate.mean(axis=0))),
        pair_head_spearman=spearman(exact.reshape(-1),estimate.reshape(-1)))


def controller(args):
    import torch
    from stage1_scan import RUNS
    stage1=Path(args.stage1).resolve();data=Path(args.data).resolve();baseline=Path(args.baseline).resolve()
    prov=json.loads((stage1/'provenance.json').read_text())
    if json.loads((ROOT/'model_lock_olmo2.json').read_text())!=json.loads((stage1/'config.json').read_text())['model']:
        raise ValueError('Model lock differs from stage 1')
    if file_hash(data)!=prov['data_sha256'] or file_hash(baseline)!=prov['baseline_sha256']:
        raise ValueError('Data or baseline differs from stage 1')
    if torch.cuda.device_count()!=args.gpus:raise ValueError('GPU count differs from Slurm allocation')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(args.gpus)))).split(',')
    if len(devices)!=args.gpus:raise ValueError('Unexpected GPU visibility')
    gpu_groups=groups(devices,args.gpus_per_replica)
    rows=[json.loads(l) for l in data.open(encoding='utf-8')]
    pairs=plan_work(make_pairs(rows,prov['prompt_ids']),len(gpu_groups),args.exact_families,args.seed)
    specs=json.loads((stage1/'metric_specs.json').read_text())
    shared={p['id']:specs[p['id']]['shared_prefix'] for p in pairs}
    fams=sorted({p['family'] for p in pairs});prefix_fams=sorted({p['family'] for p in pairs if shared[p['id']]})
    gate_fams=sorted(set(prefix_fams[:2]+[f for f in fams if f not in prefix_fams][:2]))
    manifest=dict(version=1,stage1=str(stage1),data=str(data),baseline=str(baseline),
        model=json.loads((ROOT/'model_lock_olmo2.json').read_text()),
        input_hashes={str(p):file_hash(p) for p in [data,baseline,stage1/'provenance.json',stage1/'metric_specs.json',stage1/'head_stats.csv']},
        code_hashes={n:file_hash(ROOT/n) for n in CODE_FILES},pairs=pairs,shared_prefixes=shared,
        gate_families=gate_fams,gpu_group_sizes=list(map(len,gpu_groups)),seed=args.seed,
        ig_steps=args.ig_steps,quality_threshold=.7,metric='first divergent token; fixed clean-gold sign',
        patch_scope='all original prompt positions; answer prefix recomputed',
        exact_subset='20 randomly sampled whole families by default; both orders',
        gate_only=args.gate_only)
    out=RUNS/args.name
    if out.exists():
        if not args.resume or json.loads((out/'manifest.json').read_text())!=manifest:
            raise ValueError('Run exists. Resume requires identical code, inputs and settings.')
        for name in ['command.json','gate_passed.json','summary.json',*[f'{kind}_{i}.json' for i in range(len(gpu_groups)) for kind in ['loaded','gate','state','done_screen','done_ig','done_extension']]]:
            (out/name).unlink(missing_ok=True)
    else:out.mkdir(parents=True);atomic_json(out/'manifest.json',manifest)
    children=[];logs=[]
    def wait_for(kind):
        last=0.
        while True:
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Worker failed; see worker logs.')
            if all((out/f'{kind}_{i}.json').exists() for i in range(len(children))):return
            if any(c.poll()==0 and not (out/f'{kind}_{i}.json').exists() for i,c in enumerate(children)):
                raise RuntimeError('Worker exited before completing its phase')
            if time.monotonic()-last>30:
                states=[json.loads((out/f'state_{i}.json').read_text()) if (out/f'state_{i}.json').exists() else {'status':'starting'} for i in range(len(children))]
                print(kind,states,flush=True);last=time.monotonic()
            time.sleep(1)
    def stop(*_):raise InterruptedError('Job interrupted; use --resume with identical settings.')
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for i,g in enumerate(gpu_groups):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g))
            env['OMP_NUM_THREADS']=str(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))//len(gpu_groups)))
            env['OPENBLAS_NUM_THREADS']=env['OMP_NUM_THREADS']
            f=(out/f'worker_{i}.log').open('a',encoding='utf-8');logs.append(f)
            children.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(i),'--out',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT))
        wait_for('gate')
        for i in range(len(children)):
            if not json.loads((out/f'gate_{i}.json').read_text()).get('passed'):raise RuntimeError('Gate failed')
        atomic_json(out/'gate_passed.json',dict(passed=True,replicas=len(children)))
        print('ALL REPLICAS PASSED THE GATE.',flush=True)
        if args.gate_only:return
        atomic_json(out/'command.json',dict(phase='screen'));wait_for('done_screen')
        exact_pairs=[p for p in pairs if p['exact_all']]
        exact=load_vectors(out,'screen',exact_pairs,'exact_delta')
        attr=load_vectors(out,'screen',pairs,'attribution')
        idx=[i for i,p in enumerate(pairs) if p['exact_all']]
        q=quality(exact,attr[idx]);rho=q['mean_head_signed_spearman']
        use_ig=rho is None or rho<.7
        atomic_json(out/'first_order_quality.json',q)
        atomic_json(out/'command.json',dict(phase='decision',use_ig=use_ig));wait_for('done_ig')
        estimate=load_vectors(out,'ig',pairs,'attribution') if use_ig else attr
        final_quality=quality(exact,estimate[idx])
        atomic_json(out/'final_estimate_quality.json',dict(method=f'input_embedding_IG_{args.ig_steps}' if use_ig else 'first_order',**final_quality))
        scores,counts=ri_scores(stage1);heads,controls=shortlist(scores,counts,estimate.mean(axis=0),args.seed)
        atomic_json(out/'shortlist.json',dict(heads=heads,random_controls=controls,
            rule='top25 supported RI first + last; top25 absolute mean estimate; L3H11,L9H22,L16H4; 25 random controls'))
        atomic_json(out/'command.json',dict(phase='extension',heads=heads));wait_for('done_extension')
        extension=load_vectors(out,'extension',pairs,'exact_delta',len(heads))
        final_rho=final_quality['mean_head_signed_spearman']
        summary=dict(complete=True,pairs=len(pairs),all_head_exact_pairs=len(exact_pairs),
            exact_extension_heads=len(heads),estimate_method='input_embedding_IG' if use_ig else 'first_order',
            estimate_validated=final_rho is not None and final_rho>=.7,
            ri_vs_exact_subset_spearman=spearman(scores[:,0],-exact.mean(axis=0)),
            comparison='RI versus negative signed exact mean: positive importance means movement toward corrupted answer',
            warning='Head exact effects do not establish circuit edges or semantic specificity.')
        with (out/'head_effects.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['layer','head','ri_first','ri_last','first_order_mean','final_estimate_mean','exact_subset_mean','exact_subset_mean_abs','exact_all_pairs_mean'])
            for h in range(1024):
                full=float(extension[:,heads.index(h)].mean()) if h in heads else ''
                w.writerow([h//32,h%32,*scores[h],attr[:,h].mean(),estimate[:,h].mean(),exact[:,h].mean(),np.abs(exact[:,h]).mean(),full])
        atomic_json(out/'summary.json',summary)
        print('STAGE 2 COMPLETE:',summary,flush=True)
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        for f in logs:f.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',type=int);ap.add_argument('--out',type=Path)
    ap.add_argument('--name',default='stage2_v1');ap.add_argument('--gpus',type=int,default=6)
    ap.add_argument('--gpus-per-replica',type=int,default=2)
    ap.add_argument('--stage1');ap.add_argument('--data');ap.add_argument('--baseline')
    ap.add_argument('--exact-families',type=int,default=20);ap.add_argument('--seed',type=int,default=20260914)
    ap.add_argument('--ig-steps',type=int,default=5);ap.add_argument('--resume',action='store_true');ap.add_argument('--gate-only',action='store_true')
    a=ap.parse_args()
    if a.worker is not None:worker(a.out,a.worker);return
    import re
    if not re.fullmatch('[A-Za-z0-9_-]+',a.name) or a.ig_steps<2:ap.error('Invalid name or IG steps')
    from stage1_scan import RUNS
    a.stage1=a.stage1 or str(RUNS/'stage1_v4_calibrated')
    a.data=a.data or str(ROOT/'data/singlehop_v1_4shot.jsonl')
    a.baseline=a.baseline or str(RUNS/'olmo2_singlehop_4shot/results.jsonl')
    controller(a)


if __name__=='__main__':main()
