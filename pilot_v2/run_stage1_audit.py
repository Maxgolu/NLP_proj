"""One Slurm job: GPU scan, candidate audit, then CPU-only calibration."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--draws',type=int,default=100000)
    args,scan_args=ap.parse_known_args()
    state=Path(os.environ.get('PILOT_STORAGE',str(ROOT/'storage')))
    run=Path(os.environ.get('PILOT_RUNS',str(state/'runs')))/args.name
    commands=[
        ['stage1_scan.py','--name',args.name,*scan_args],
        ['analyze_stage1_audit.py',str(run)],
        ['calibrate_stage1_null.py',str(run),'--draws',str(args.draws)],
    ]
    for cmd in commands:
        subprocess.run([sys.executable,'-u',str(ROOT/cmd[0]),*cmd[1:]],check=True)
    print(f'Scan, candidate audit and null calibration complete: {run}',flush=True)


if __name__=='__main__': main()
