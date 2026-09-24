"""One allocation, three independent two-GPU replicas, adaptive CPU boundaries."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from stage3_common import digest,json_read,json_write,run_lock
from stage4_pipeline import gpu_groups
from s42_plan import load,conditional,write_schedule
from s42_run import identity,mean_identity,code_identity,verify
from s42_analyze import analyze,choose_k,refine,extension,sensitivity_registry,finish
from s42_means import reduce_bank,MeanBank

ROOT=Path(__file__).resolve().parent

class Pipeline:
    def __init__(self,a):
        self.a=a;self.out=a.out;self.groups=gpu_groups(os.environ.get('CUDA_VISIBLE_DEVICES',''),a.gpus)
        self.children=[];self.stop=False;self.history=[];self.completed=[]
    def state(self,status,**kw):json_write(self.out/'pipeline_state.json',dict(status=status,time=time.time(),**kw))
    def interrupt(self,*_):
        self.stop=True
        for p,_,_ in self.children:
            if p.poll() is None:p.terminate()
    def work(self,label,schedule=None,request=None,bank=None,gate=False):
        if self.stop:raise InterruptedError('Pipeline interrupted')
        self.state('running',phase=label);runs=[];count=1 if gate else len(self.groups)
        prior=self.out/'schedules'/f'{label}_reuse.json';json_write(prior,self.history)
        try:
            for i in range(count):
                run=self.out/'runs'/f'{label}_r{i}';runs.append(run)
                ident=mean_identity(self.a.inputs,request,i,count) if request else identity(self.a.inputs,schedule,i,count,bank)
                if (run/'manifest.json').exists() and json_read(run/'manifest.json')['identity']!=ident:raise ValueError('Changed worker identity')
                if (run/'done.json').exists():
                    if not json_read(run/'done.json')['complete']:raise ValueError('Failed completion marker')
                    if not request:verify(run,self.a.inputs,schedule)
                    continue
                if gate and (run/'gate.json').exists() and json_read(run/'gate.json')['passed']:continue
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(self.groups[i]);env['PYTHONUNBUFFERED']='1'
                cmd=[sys.executable,str(ROOT/'s42_run.py'),'means' if request else 'run','--inputs',str(self.a.inputs),
                    '--out',str(run),'--shard',str(i),'--shards',str(count)]
                if request:cmd+=['--request',str(request)]
                else:cmd+=['--schedule',str(schedule),'--prior',str(prior)]
                if bank:cmd+=['--bank',str(bank)]
                if gate:cmd+=['--gate-only']
                if run.exists():cmd+=['--resume']
                log=(self.out/'logs'/f'{label}_r{i}.log').open('a',encoding='utf-8')
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);self.children.append((child,log,run))
                print('Started',label,'replica',i,'GPUs',env['CUDA_VISIBLE_DEVICES'],flush=True)
            while any(p.poll() is None for p,_,_ in self.children):
                if self.stop:raise InterruptedError('Scheduler/user interruption')
                failed=[(p.returncode,str(r)) for p,_,r in self.children if p.poll() not in [None,0]]
                if failed:raise RuntimeError(f'Worker failed: {failed}; inspect worker log')
                time.sleep(1)
            if any(p.returncode for p,_,_ in self.children):raise RuntimeError('Worker failed')
            if self.stop:raise InterruptedError('Pipeline interrupted')
        finally:
            for p,_,_ in self.children:
                if p.poll() is None:p.terminate()
            for p,log,run in self.children:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                log.close();lock=run/'.running.lock'
                # Only our own, confirmed-dead subprocesses may have locks removed.
                if lock.exists() and json_read(lock).get('pid')==p.pid:lock.unlink()
            self.children=[]
        if gate:
            if not json_read(runs[0]/'gate.json')['passed']:raise ValueError('Gate failed')
            return runs,None
        if request:return runs,None
        self.state('cpu_analysis',phase=label)
        summ=analyze(self.a.inputs,schedule,runs,self.out/'analysis'/label)
        self.history.append(dict(schedule=str(schedule),runs=[str(r) for r in runs]));self.completed.append(label)
        return runs,summ
    def schedule(self,label,reg,population,mode,**kw):
        path=self.out/'schedules'/f'{label}.json'
        # Recompute decisions on resume; compare before overwriting frozen schedules.
        candidate=path.with_suffix('.candidate.json');write_schedule(candidate,self.a.inputs,reg,population,mode,label,**kw)
        if path.exists() and json_read(path)!=json_read(candidate):raise ValueError('Adaptive decision changed on resume')
        candidate.replace(path);return path
    def execute(self):
        plan,_=load(self.a.inputs);initial=self.a.inputs/'initial.json'
        self.work('gate',initial,gate=True)
        runs,summary=self.work('initial',initial)
        K=choose_k(self.a.inputs,initial,runs);json_write(self.out/'K_selection.json',K)
        cs=self.schedule('conditional',conditional(plan,K['K']),'core','donor',K_selection=K)
        cr,ct=self.work('conditional',cs)
        reg,decision=refine(self.a.inputs,initial,runs,summary);json_write(self.out/'refinement_decision.json',decision)
        phases=[(initial,summary),(cs,ct)]
        if reg.contrasts:
            rs=self.schedule('refinement',reg,'core','donor',decision=decision)
            rr,rt=self.work('refinement',rs);phases.append((rs,rt))
        ex,decision=extension(self.a.inputs,phases);json_write(self.out/'extension_decision.json',decision)
        if ex.contrasts:
            es=self.schedule('extension',ex,'all','donor',decision=decision)
            self.work('extension',es)
            ms=self.schedule('sensitivity',sensitivity_registry(es),'all','mean',donor_schedule_hash=digest(es))
            heads=sorted({a['head'] for c in ex.configs.values() for a in c['atoms'] if a['donor']=='other'})
            request=self.out/'schedules'/'mean_request.json'
            json_write(request,dict(heads=heads,plan_hash=digest(self.a.inputs/'plan.json'),sensitivity_hash=digest(ms)))
            mr,_=self.work('mean_collection',request=request)
            bank=self.out/'mean_bank';self.state('cpu_mean_reduction');reduce_bank(self.a.inputs,request,mr,bank)
            MeanBank(bank,self.a.inputs)
            self.work('sensitivity',ms,bank=bank)
        self.state('cpu_final_analysis');finish(self.a.inputs,self.out,self.completed,K)
        json_write(self.out/'phase_index.json',self.history)
        json_write(self.out/'pipeline_done.json',dict(complete=True,phases=self.completed,time=time.time(),
            results_hash=digest(self.out/'results_summary.json'),status='complete' if ex.contrasts else 'complete_no_retained_contrasts'))
        self.state('complete');print('S4.2 completed and verified:',self.out,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,choices=[2,4,6],default=6);p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.inputs=a.inputs.resolve();a.out=a.out.resolve();load(a.inputs,a.inputs/'initial.json')
    ident=dict(gpus=a.gpus,plan_hash=digest(a.inputs/'plan.json'),initial_hash=digest(a.inputs/'initial.json'),code=code_identity())
    if a.out.exists() and not a.resume:raise FileExistsError('Use --resume with identical code/inputs')
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['logs','schedules','runs','analysis']:(a.out/name).mkdir(exist_ok=True)
    with run_lock(a.out):
        mf=a.out/'pipeline_manifest.json'
        if mf.exists() and json_read(mf)!=ident:raise ValueError('Pipeline identity changed')
        json_write(mf,ident);pipe=Pipeline(a)
        signal.signal(signal.SIGINT,pipe.interrupt);signal.signal(signal.SIGTERM,pipe.interrupt)
        try:pipe.execute()
        except BaseException as exc:pipe.state('interrupted' if pipe.stop else 'failed',error=str(exc));raise

if __name__=='__main__':main()
