"""One Slurm allocation, three independent two-GPU replicas, frozen CPU decisions between phases.

Discovery mode: means -> gate -> Stage A (+C50 fallback) -> Stage B -> Stage C -> Stage D ->
full-discovery behaviour -> freeze manifest. Held-out mode: the frozen validation suite only,
opened by a validating freeze manifest, with the fixed full mean bank.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from stage3_common import digest,json_read,json_write,run_lock
from s45_plan import load,stage_a,stage_b,stage_c,stage_d,full_discovery,C33,C50
from s45_run import identity,mean_identity,code_identity,verify
from s45_means import reduce_bank,MeanBank
import s45_analyze as A
import s45_freeze as F

ROOT=Path(__file__).resolve().parent

def gpu_groups(visible,gpus):
    ids=[x.strip() for x in visible.split(',') if x.strip()]
    if gpus not in [2,4,6] or len(ids)!=gpus or len(set(ids))!=len(ids):
        raise ValueError('CUDA_VISIBLE_DEVICES must contain exactly the requested 2/4/6 distinct GPUs')
    return [ids[i:i+2] for i in range(0,gpus,2)]

class Pipeline:
    def __init__(self,a):
        self.a=a;self.out=a.out;self.groups=gpu_groups(os.environ.get('CUDA_VISIBLE_DEVICES',''),a.gpus)
        self.children=[];self.stop=False;self.history=[];self.stages={}
    def state(self,status,**kw):json_write(self.out/'pipeline_state.json',dict(status=status,time=time.time(),**kw))
    def interrupt(self,*_):
        self.stop=True
        for p,_,_ in self.children:
            if p.poll() is None:p.terminate()
    def work(self,label,schedule=None,bank=None,gate=False,means=False):
        if self.stop:raise InterruptedError('Pipeline interrupted')
        self.state('running',phase=label);runs=[];count=1 if gate else len(self.groups)
        prior=self.out/'schedules'/f'{label}_reuse.json';json_write(prior,self.history)
        try:
            for i in range(count):
                run=self.out/'runs'/f'{label}_r{i}';runs.append(run)
                ident=mean_identity(self.a.inputs,i,count) if means else identity(self.a.inputs,schedule,i,count,bank,self.mode if bank else None)
                if (run/'manifest.json').exists() and json_read(run/'manifest.json')['identity']!=ident:raise ValueError('Changed worker identity: '+label)
                if (run/'done.json').exists():
                    if not json_read(run/'done.json')['complete']:raise ValueError('Failed completion marker')
                    if not means:verify(run,self.a.inputs,schedule)
                    continue
                if gate and (run/'gate.json').exists() and json_read(run/'gate.json')['passed']:continue
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(self.groups[i]);env['PYTHONUNBUFFERED']='1'
                cmd=[sys.executable,str(ROOT/'s45_run.py'),'means' if means else 'run','--inputs',str(self.a.inputs),'--out',str(run),'--shard',str(i),'--shards',str(count)]
                if not means:cmd+=['--schedule',str(schedule),'--prior',str(prior)]
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
                if lock.exists() and json_read(lock).get('pid')==p.pid:lock.unlink()
            self.children=[]
        if gate:
            if not json_read(runs[0]/'gate.json')['passed']:raise ValueError('Gate failed')
            return runs
        if not means:self.history.append(dict(schedule=str(schedule),runs=[str(r) for r in runs]))
        self.stages[label]=dict(schedule=str(schedule) if schedule else None,runs=[str(r) for r in runs],analysis=[])
        return runs
    def schedule(self,label,sch,**kw):
        path=self.out/'schedules'/f'{label}.json';candidate=path.with_suffix('.candidate.json');sch.write(candidate,**kw)
        if path.exists() and json_read(path)!=json_read(candidate):raise ValueError('Adaptive decision changed on resume: '+label)
        candidate.replace(path);return path
    def rows(self,label):
        st=self.stages[label];return A.gather(self.a.inputs,Path(st['schedule']),[Path(r) for r in st['runs']])
    def record(self,label,*paths):self.stages[label]['analysis']+=[str(p) for p in paths]

    def discovery(self):
        plan,_=load(self.a.inputs);inputs=self.a.inputs;out=self.out
        mr=self.work('means',means=True);bank=out/'mean_bank';self.state('cpu_mean_reduction')
        reduce_bank(inputs,mr,bank);MeanBank(bank,inputs,'lofo');self.stages['means']=dict(schedule=None,runs=[str(r) for r in mr],analysis=[str(bank/'mean_bank_manifest.json')])
        self.work('gate',inputs/'stage_a.json',bank=bank,gate=True)
        ra=self.work('stage_a',inputs/'stage_a.json',bank=bank);an=out/'analysis'
        decision=A.analyze_stage_a(inputs,inputs/'stage_a.json',ra,an/'stage_a');extra=[]
        if decision['status']=='need_C50':
            rc=self.work('stage_a_c50',inputs/'stage_a_c50.json',bank=bank)
            decision=A.analyze_stage_a(inputs,inputs/'stage_a.json',ra,an/'stage_a',extra=[(inputs/'stage_a_c50.json',rc)])
        json_write(out/'B_decision.json',decision);B=C33 if decision['B']=='C33' else C50
        self.record('stage_a',an/'stage_a'/'B_decision.json',an/'stage_a'/'stage_a_summary.json')
        sb=self.schedule('stage_b',stage_b(inputs,plan,B),B=B,B_status=decision['status'])
        rb_=self.work('stage_b',sb,bank=bank);prior=self.rows('stage_a')
        table=A.analyze_stage_b(inputs,sb,rb_,an/'stage_b',B,prior_rows=prior);self.record('stage_b',an/'stage_b'/'stage_b_summary.json')
        sc=self.schedule('stage_c',stage_c(inputs,plan,B),B=B)
        # Full-background anchors ride along as a regression against historical means (five anchors x 40 pairs).
        rc=self.work('stage_c',sc,bank=bank)
        selection=A.analyze_stage_c(inputs,sc,rc,an/'stage_c',B,table);self.record('stage_c',an/'stage_c'/'frozen_selection.json',an/'stage_c'/'stage_c_summary.json')
        drift=[r for r in selection['full_background_regression'] if not r['within_0_05']]
        if drift and not os.environ.get('S45_ALLOW_ANCHOR_DRIFT'):
            raise ValueError('Full-background anchors drift from historical means by >0.05: '+json.dumps(drift)+'; inspect analysis/stage_c before setting S45_ALLOW_ANCHOR_DRIFT=1 and resuming')
        json_write(out/'frozen_selection.json',selection)
        sd=self.schedule('stage_d',stage_d(inputs,plan,B,selection),B=B,selection=selection)
        rd=self.work('stage_d',sd,bank=bank) if json_read(sd)['jobs'] else []
        stage_d_summary=A.analyze_stage_d(inputs,sd,rd,an/'stage_d',B,selection,self.rows('stage_c')) if rd else dict(gamma=[],attachments=[])
        if not rd:self.stages['stage_d']=dict(schedule=str(sd),runs=[],analysis=[]);json_write(an/'stage_d'/'stage_d_summary.json',stage_d_summary)
        self.record('stage_d',an/'stage_d'/'stage_d_summary.json')
        fd=self.schedule('full_discovery',full_discovery(inputs,plan,B,selection),B=B,selection=selection)
        rf=self.work('full_discovery',fd,bank=bank)
        A.analyze_behavior_suite(inputs,fd,rf,an/'full_discovery',B,'extension',prior_rows=self.rows('stage_a')+self.rows('stage_b'))
        self.record('full_discovery',an/'full_discovery'/'extension_summary.json')
        signs=A.discovery_signs(table,selection,stage_d_summary);json_write(out/'discovery_signs.json',dict(signs=signs))
        A.membership_ledger(plan,B,selection,out)
        self.state('freezing');self.stages['means']['runs']=[str(r) for r in mr]
        freeze=F.build(inputs,out/'freeze_manifest.json',self.stages,B,decision['status'],selection,signs,bank/'mean_bank_manifest.json',code_identity())
        F.validate_freeze(inputs,freeze,code_identity())
        json_write(out/'pipeline_done.json',dict(complete=True,mode='discovery',time=time.time(),B=B,B_status=decision['status'],selected=selection['selected'],
            freeze_signature=freeze['signature'],next='prepare-heldout with this freeze manifest, then the heldout pipeline'))
        self.state('complete');print('S4.5 discovery complete; freeze manifest written:',out/'freeze_manifest.json',flush=True)

    def heldout(self):
        plan,pairs=load(self.a.inputs);inputs=self.a.inputs;out=self.out;freeze=json_read(self.a.freeze)
        F.validate_freeze(inputs,freeze,code_identity())
        if any(p['split']!='heldout' for p in pairs):raise ValueError('Held-out pipeline needs sealed held-out inputs')
        sch=inputs/'validation.json'
        if json_read(sch)['freeze_hash']!=digest(self.a.freeze):raise ValueError('Validation schedule was sealed with a different freeze')
        bank=self.a.bank;MeanBank(bank,inputs,'full')
        if digest(bank/'mean_bank_manifest.json')!=freeze['mean_bank']['manifest_hash']:raise ValueError('Mean bank differs from the frozen one')
        self.work('gate',sch,bank=bank,gate=True)
        rv=self.work('validation',sch,bank=bank);an=out/'analysis'
        discovery=json_read(self.a.discovery/'discovery_signs.json')
        A.analyze_validation(inputs,sch,rv,an/'validation',freeze,discovery)
        json_write(out/'pipeline_done.json',dict(complete=True,mode='heldout',time=time.time(),freeze_signature=freeze['signature']))
        self.state('complete');print('S4.5 held-out validation complete:',out,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['discovery','heldout'],required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,choices=[2,4,6],default=6);p.add_argument('--resume',action='store_true')
    p.add_argument('--freeze',type=Path);p.add_argument('--bank',type=Path);p.add_argument('--discovery',type=Path);a=p.parse_args()
    a.inputs=a.inputs.resolve();a.out=a.out.resolve()
    if a.mode=='heldout' and not (a.freeze and a.bank and a.discovery):raise SystemExit('heldout mode needs --freeze, --bank and --discovery')
    ident=dict(mode=a.mode,gpus=a.gpus,plan_hash=digest(a.inputs/'plan.json'),code=code_identity(),freeze=digest(a.freeze) if a.freeze else None)
    if a.out.exists() and not a.resume:raise FileExistsError('Use --resume with identical code/inputs')
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['logs','schedules','runs','analysis']:(a.out/name).mkdir(exist_ok=True)
    with run_lock(a.out):
        mf=a.out/'pipeline_manifest.json'
        if mf.exists() and json_read(mf)!=ident:raise ValueError('Pipeline identity changed')
        json_write(mf,ident);pipe=Pipeline(a);pipe.mode='full' if a.mode=='heldout' else 'lofo'
        signal.signal(signal.SIGINT,pipe.interrupt);signal.signal(signal.SIGTERM,pipe.interrupt)
        try:pipe.discovery() if a.mode=='discovery' else pipe.heldout()
        except BaseException as exc:pipe.state('interrupted' if pipe.stop else 'failed',error=str(exc));raise

if __name__=='__main__':main()
