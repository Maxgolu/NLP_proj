"""One allocation: gated coverage -> localization -> mapping -> refinement/extensions.

CPU decisions run only after exact shard coverage verification. GPUs are assigned
in disjoint pairs to subprocesses; a failed gate/worker stops the pipeline.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from stage3_common import digest,json_read,json_write,run_lock
from stage4_plan import load
from stage4_run import CODE_FILES,verify_run,identity
from stage4_analyze import analyze,compare,next_schedule,NoFollowup

ROOT=Path(__file__).resolve().parent

def gpu_groups(visible,gpus):
    ids=[x.strip() for x in visible.split(',') if x.strip()]
    if gpus not in [2,4,6] or len(ids)!=gpus or len(set(ids))!=len(ids):
        raise ValueError('CUDA_VISIBLE_DEVICES must contain exactly the requested 2/4/6 distinct GPUs')
    return [ids[i:i+2] for i in range(0,gpus,2)]

class Pipeline:
    def __init__(self,args):
        self.a=args;self.out=args.out;self.groups=gpu_groups(os.environ.get('CUDA_VISIBLE_DEVICES',''),args.gpus)
        self.children=[];self.stop=False
    def state(self,status,**kw):json_write(self.out/'pipeline_state.json',dict(status=status,time=time.time(),**kw))
    def interrupt(self,*_):
        self.stop=True
        for p,_,_ in self.children:
            if p.poll() is None:p.terminate()
    def preserve_partial(self,path):
        if not path.resolve().is_relative_to(self.out.resolve()):raise ValueError('Output outside pipeline directory')
        path.rename(path.with_name(path.name+f'.incomplete_{time.time_ns()}'))
    def work(self,label,schedule,gate_only=False):
        if self.stop:raise InterruptedError('Pipeline interrupted before next GPU phase')
        self.state('running',phase=label)
        runs=[];self.children=[]
        count=1 if gate_only else len(self.groups)
        try:
            for i in range(count):
                run=self.out/'runs'/f'{label}_r{i}';runs.append(run)
                if (run/'done.json').exists() and not gate_only:
                    if json_read(run/'manifest.json')['identity']!=identity(self.a.inputs,schedule,i,count):raise ValueError('Completed run identity changed')
                    verify_run(run,self.a.inputs,schedule);continue
                if gate_only and (run/'gate.json').exists() and json_read(run/'gate.json')['passed']:
                    if json_read(run/'manifest.json')['identity']!=identity(self.a.inputs,schedule,i,count):raise ValueError('Gate identity changed')
                    continue
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(self.groups[i]);env['PYTHONUNBUFFERED']='1'
                cmd=[sys.executable,str(ROOT/'stage4_run.py'),'run','--inputs',str(self.a.inputs),'--schedule',str(schedule),
                     '--out',str(run),'--shard',str(i),'--shards',str(count)]
                if gate_only:cmd+=['--gate-only']
                if run.exists():cmd+=['--resume']
                log=(self.out/'logs'/f'{label}_r{i}.log').open('a',encoding='utf-8')
                p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                self.children.append((p,log,run));print('Started',label,'replica',i,'GPUs',env['CUDA_VISIBLE_DEVICES'],flush=True)
            while any(p.poll() is None for p,_,_ in self.children):
                if self.stop:raise InterruptedError('Scheduler/user interruption')
                failed=[(p.returncode,str(run)) for p,_,run in self.children if p.poll() not in [None,0]]
                if failed:raise RuntimeError(f'Worker failed: {failed}; see worker logs')
                time.sleep(1)
            for p,_,run in self.children:
                if p.returncode:raise RuntimeError(f'{run}: exit {p.returncode}')
            if self.stop:raise InterruptedError('Interrupted')
        finally:
            for p,_,_ in self.children:
                if p.poll() is None:p.terminate()
            for p,log,run in self.children:
                try:p.wait(timeout=120)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                log.close()
                # These are this controller's own confirmed-dead children only.
                lock=run/'.running.lock'
                if lock.exists() and json_read(lock).get('pid')==p.pid:lock.unlink()
            self.children=[]
        if gate_only:
            if not json_read(runs[0]/'gate.json')['passed']:raise ValueError('Gate not passed')
        else:
            analysis=self.out/'analysis'/label
            if analysis.exists() and not (analysis/'verification.json').exists():self.preserve_partial(analysis)
            if not analysis.exists():analyze(self.a.inputs,schedule,runs,analysis)
            else:
                for run in runs:verify_run(run,self.a.inputs,schedule)
                meta=json_read(analysis/'verification.json')
                if meta['schedule_hash']!=digest(schedule) or meta['family_effects_hash']!=digest(analysis/'family_effects.csv'):
                    raise ValueError('Analysis identity/checksum changed')
        return runs
    def decision(self,label,mode,schedule,runs,prior=None):
        if self.stop:raise InterruptedError('Pipeline interrupted before selection')
        target=self.out/'schedules'/f'{label}.json';empty=target.with_suffix('.empty.json')
        if target.exists():load(self.a.inputs,target);return target
        if empty.exists():return None
        self.state('cpu_selection',phase=label)
        try:next_schedule(self.a.inputs,schedule,runs,target,mode,prior)
        except NoFollowup as exc:
            json_write(empty,dict(reason=str(exc),parent_schedule_hash=digest(schedule)))
            print('No eligible follow-up:',label,flush=True);return None
        return target
    def execute(self):
        coverage=self.a.inputs/'coverage.json'
        self.work('gate',coverage,True)
        runs=self.work('coverage',coverage)
        seed=self.decision('after_coverage','after-coverage',coverage,runs)
        for _ in range(2):
            phase=json_read(seed)['phase']
            if phase=='seed':break
            if phase not in ['localize_roles','localize_slots']:raise ValueError('Unexpected localization phase')
            runs=self.work(phase,seed)
            seed=self.decision('after_'+phase,'after-localize',seed,runs)
        if json_read(seed)['phase']!='seed':raise ValueError('Localization did not terminate')
        runs=self.work('seed',seed)
        refine=self.decision('refine','refine',seed,runs)
        extension=self.decision('extend_seed','extend',seed,runs)
        if refine is not None:
            rr=self.work('refine',refine)
            comp=self.out/'analysis'/'interactions_core'
            if comp.exists() and not (comp/'unavailable.json').exists():self.preserve_partial(comp)
            if not comp.exists():compare([self.out/'analysis'/'seed',self.out/'analysis'/'refine'],comp)
            if extension is None:raise ValueError('Refinement exists but no eligible seed extension; inspect selection')
            rex=self.decision('extend_refine','extend',refine,rr,extension)
        else:rex=None
        if extension is not None:self.work('extend_seed',extension)
        if rex is not None:self.work('extend_refine',rex)
        status='complete' if extension is not None else 'complete_no_retained_routes'
        json_write(self.out/'pipeline_done.json',dict(status=status,complete=True,time=time.time(),
                    note='S4.1 only. No retained direct route does not rule out mediation; G/S4.3 require review.'))
        self.state(status)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,choices=[2,4,6],default=6);p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.inputs=a.inputs.resolve();a.out=a.out.resolve()
    ident=dict(gpus=a.gpus,plan_hash=digest(a.inputs/'plan.json'),coverage_hash=digest(a.inputs/'coverage.json'),
               code={n:digest(ROOT/n) for n in CODE_FILES+['stage4_pipeline.py']})
    if a.out.exists() and not a.resume:raise FileExistsError('Use --resume with the same pipeline name')
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['logs','schedules','runs','analysis']:(a.out/name).mkdir(exist_ok=True)
    with run_lock(a.out):
        mf=a.out/'pipeline_manifest.json'
        if mf.exists():
            if json_read(mf)!=ident:raise ValueError('Pipeline identity changed; use a new run directory')
        else:json_write(mf,ident)
        pipe=Pipeline(a)
        signal.signal(signal.SIGTERM,pipe.interrupt);signal.signal(signal.SIGINT,pipe.interrupt)
        try:pipe.execute()
        except BaseException as exc:
            pipe.state('interrupted' if pipe.stop else 'failed',error=str(exc));raise

if __name__=='__main__':main()
