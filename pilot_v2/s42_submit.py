"""Read Slurm state before submission; recover locks only from confirmed terminal jobs."""
import argparse
import os
from pathlib import Path
import subprocess
from stage3_common import json_read

FINAL={'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def output(args):return subprocess.check_output(args,text=True).strip()

def check_submission(out,name,resume):
    active=output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%.128j']).splitlines()
    active=[tuple(x.strip() for x in line.split('|',1)) for line in active if line.strip()]
    if any(jobname==name for jobid,jobname in active):raise ValueError('An existing job with this name is still pending/running; inspect it instead of resubmitting')
    if out.exists() and not resume:raise FileExistsError('Output exists; use --resume after checking the existing job')
    if not out.exists():return
    root=out.resolve();locks=[root/'.running.lock']+list((root/'runs').glob('*/.running.lock'))
    for path in locks:
        if not path.exists():continue
        if not path.resolve().is_relative_to(root):raise ValueError('Lock path outside run')
        data=json_read(path);job=str(data.get('job',''))
        if not job.isdigit():raise ValueError('Cannot safely recover a lock without a recorded Slurm job ID')
        if any(jobid==job or jobid.startswith(job+'_') for jobid,jobname in active):raise ValueError('Recorded job still active: '+job)
        states=output(['sacct','-j',job,'-X','--noheader','--parsable2','--format=State']).splitlines()
        if not states or any(x.split('|')[0].split()[0].rstrip('+') not in FINAL for x in states):
            raise ValueError('Slurm has not confirmed that the old job ended; lock preserved')
        path.rename(path.with_name('.recovered_job_'+job+'_'+str(data['pid'])+'.lock'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--job-name',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    check_submission(a.out,a.job_name,a.resume)

if __name__=='__main__':main()
