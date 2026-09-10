"""Partition complete families among model replicas on allocated GPUs only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pilot

ROOT = Path(__file__).resolve().parent


def gpu_groups(devices, minimum=2, max_workers=100):
    if minimum < 1 or len(devices) < minimum or max_workers < 1:
        raise ValueError('Not enough GPUs or families for the requested replica size.')
    if len(set(devices)) != len(devices):
        raise ValueError('Duplicate GPU identifiers.')
    workers = min(len(devices)//minimum, max_workers)
    size, remainder = divmod(len(devices),workers)
    groups=[]; start=0
    for i in range(workers):
        end=start+size+(i<remainder)
        groups.append(devices[start:end]); start=end
    return groups


def merge_rows(expected, shards):
    by_id = {r['id']:r for r in expected}
    if len(by_id)!=len(expected):
        raise ValueError('Duplicate input IDs.')
    found={}
    for index,rows in enumerate(shards):
        for row in rows:
            key=row['id']
            if key in found:
                raise ValueError('Duplicate result: '+key)
            if key not in by_id or any(row.get(k)!=v for k,v in by_id[key].items()):
                raise ValueError('Result does not match the input dataset: '+key)
            if row['family'] % len(shards) != index:
                raise ValueError('Result belongs to another worker: '+key)
            found[key]=row
    missing=set(by_id)-set(found)
    if missing:
        raise ValueError(f'{len(missing)} results missing; refusing to report a complete run.')
    return [found[r['id']] for r in expected]


def run(args):
    import torch
    count=torch.cuda.device_count()
    if count != args.gpus:
        raise ValueError(f'Requested {args.gpus} GPUs but this process sees {count}. Check the Slurm allocation; do not override CUDA visibility manually.')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible is not None else [str(i) for i in range(count)]
    if len(devices)!=count or any(not d for d in devices):
        raise ValueError('Unexpected CUDA_VISIBLE_DEVICES mapping.')
    dataset=pilot.dataset_path(args.shots,args.prompt_version)
    rows=pilot.read_results(dataset)
    if args.limit_families:
        rows=[r for r in rows if r['family']<args.limit_families]
    families={r['family'] for r in rows}
    groups=gpu_groups(devices,args.gpus_per_replica,len(families))
    out=pilot.RUNS/args.name
    out.mkdir(parents=True,exist_ok=True)
    pilot.check_data_model_match(rows, args.model)
    lock_file = pilot.MODELS[args.model]['lock']
    config=dict(model=json.loads((ROOT/lock_file).read_text()), model_key=args.model,
                data_sha256=pilot.digest(dataset), pilot_sha256=pilot.digest(ROOT/'pilot.py'),
                launcher_sha256=pilot.digest(Path(__file__)), shots=args.shots,
                limit_families=args.limit_families, gpus=args.gpus,
                group_sizes=[len(g) for g in groups])
    if args.prompt_version.startswith(('completion','singlehop')):
        config['completion_code_sha256']=pilot.digest(ROOT/'completion_evaluation.py')
    manifest=out/'parallel_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Settings or GPU count changed. Choose a new run name.')
    pilot.dump(manifest,config)
    pilot.dump(out/f'allocation_{time.time_ns()}.json',dict(groups=groups,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    print(f'{len(rows)} prompts; {len(groups)} model replicas; GPU groups {groups}',flush=True)
    children=[]; handles=[]
    def stop_children(*_):
        for child in children:
            if child.poll() is None: child.terminate()
        raise InterruptedError('Launcher interrupted; completed worker rows remain available for resume.')
    previous={sig:signal.signal(sig,stop_children) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for i,group in enumerate(groups):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(group),PILOT_RUNS=str(out/'workers'))
            env['OMP_NUM_THREADS']=str(max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))//len(groups)))
            handle=(out/f'worker_{i:03d}.log').open('a',encoding='utf-8')
            handles.append(handle)
            command=[sys.executable,'-u',str(ROOT/'pilot.py'),'run','--name',f'{args.name}_worker_{i:03d}',
                     '--model',args.model,
                     '--shots',str(args.shots),'--prompt-version',args.prompt_version,'--limit-families',str(args.limit_families),
                     '--shard-index',str(i),'--shard-count',str(len(groups))]
            children.append(subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT))
        while any(c.poll() is None for c in children):
            failed=[i for i,c in enumerate(children) if c.poll() not in (None,0)]
            if failed:
                raise RuntimeError(f'Worker failure {failed}; read worker logs in {out}. Resume using the same command after resolving the error.')
            time.sleep(1)
        if any(c.returncode!=0 for c in children):
            raise RuntimeError('Worker failed; see individual worker logs.')
    finally:
        for c in children:
            if c.poll() is None: c.terminate()
        for c in children:
            try: c.wait(timeout=10)
            except subprocess.TimeoutExpired: c.kill(); c.wait()
        for h in handles: h.close()
        for sig,handler in previous.items(): signal.signal(sig,handler)
    shards=[pilot.read_results(out/'workers'/f'{args.name}_worker_{i:03d}'/'results.jsonl') for i in range(len(groups))]
    merged=merge_rows(rows,shards)
    temp=out/'results.jsonl.tmp'
    temp.write_text(''.join(json.dumps(r)+'\n' for r in merged),encoding='utf-8')
    temp.replace(out/'results.jsonl')
    pilot.summarize(out)
    print('ALL WORKERS COMPLETE; validated merged results:',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,required=True)
    parser.add_argument('--gpus-per-replica',type=int,default=2,
                        help='Minimum GPUs per model copy; default 2 for 11-12GB GPUs. Use 1 on sufficiently large GPUs.')
    parser.add_argument('--name',required=True)
    parser.add_argument('--model',choices=sorted(pilot.MODELS),default=pilot.DEFAULT_MODEL)
    parser.add_argument('--shots',type=int,choices=[0,4,12],default=12)
    parser.add_argument('--prompt-version',choices=pilot.PROMPT_VERSIONS,default='v2')
    parser.add_argument('--limit-families',type=int,default=0)
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.name) or args.limit_families<0:
        parser.error('Invalid run name or family count.')
    import fcntl
    locks=pilot.STATE/'locks'; locks.mkdir(parents=True,exist_ok=True)
    with (locks/(args.name+'.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__': main()
