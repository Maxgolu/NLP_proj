"""Portable bundle and completed-results checks. No model loading."""
import argparse
from pathlib import Path
from stage3_common import json_read,json_write,digest
from s42_plan import load,conditional
from s42_run import code_identity
from s42_analyze import gather,analyze,choose_k,refine,extension,sensitivity_registry,finish

def bundle(path):
    for name,sha in json_read(path/'bundle_hashes.json').items():
        f=(path/name).resolve()
        if not f.is_relative_to(path.resolve()) or digest(f)!=sha:raise ValueError('Bundle file changed: '+name)
    load(path/'inputs',path/'inputs/initial.json');print('Bundle and frozen inputs verified')

def results(inputs,out):
    done=json_read(out/'pipeline_done.json')
    if not done['complete'] or done['results_hash']!=digest(out/'results_summary.json'):raise ValueError('Pipeline not complete or result hash changed')
    mf=json_read(out/'pipeline_manifest.json');n=mf['gpus']//2
    if mf['plan_hash']!=digest(inputs/'plan.json') or mf['code']!=code_identity():raise ValueError('Code/input identity mismatch')
    plan,_=load(inputs);phases=[];total=0
    for label in done['phases']:
        s=inputs/'initial.json' if label=='initial' else out/'schedules'/f'{label}.json'
        runs=[out/'runs'/f'{label}_r{i}' for i in range(n)]
        _,sc,_,rows,_=gather(inputs,s,runs);total+=len(rows)
        if label=='initial':
            K=choose_k(inputs,s,runs)
            if K!=json_read(out/'K_selection.json'):raise ValueError('K decision does not reproduce')
        elif label=='conditional':
            reg=conditional(plan,K['K'])
            if sorted(reg.configs.values(),key=lambda c:c['id'])!=sc['configs'] or reg.contrasts!=sc['contrasts']:raise ValueError('Conditional schedule mismatch')
        elif label=='extension':
            reg,decision=extension(inputs,phases)
            if decision!=json_read(out/'extension_decision.json') or sorted(reg.configs.values(),key=lambda c:c['id'])!=sc['configs'] or reg.contrasts!=sc['contrasts']:raise ValueError('Extension selection mismatch')
        if label=='refinement':
            reg,decision=refine(inputs,inputs/'initial.json',[out/'runs'/f'initial_r{i}' for i in range(n)],phases[0][1])
            if decision!=json_read(out/'refinement_decision.json') or sorted(reg.configs.values(),key=lambda c:c['id'])!=sc['configs'] or reg.contrasts!=sc['contrasts']:raise ValueError('Refinement mismatch')
        if label=='sensitivity':
            from s42_means import MeanBank,reduce_bank
            meanruns=[out/'runs'/f'mean_collection_r{i}' for i in range(n)]
            reduce_bank(inputs,out/'schedules/mean_request.json',meanruns,out/'mean_bank')
            MeanBank(out/'mean_bank',inputs)
            for run in runs:
                if json_read(run/'manifest.json')['identity']['bank_hash']!=digest(out/'mean_bank/manifest.json'):raise ValueError('Worker mean bank differs')
            reg=sensitivity_registry(out/'schedules/extension.json')
            if sorted(reg.configs.values(),key=lambda c:c['id'])!=sc['configs'] or reg.contrasts!=sc['contrasts']:raise ValueError('Sensitivity closure mismatch')
        meta=json_read(out/'analysis'/label/'verification.json')
        for name,sha in meta['files'].items():
            if digest(out/'analysis'/label/name)!=sha:raise ValueError('Analysis file changed')
        summary=json_read(out/'analysis'/label/'summary.json')
        if label in ['initial','conditional','refinement']:phases.append((s,summary))
    json_write(out/'portable_verification.json',dict(passed=True,records=total,phases=done['phases']))
    print('Verified completed S4.2:',total,'records')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('bundle');q.add_argument('--package',type=Path,required=True)
    q=sub.add_parser('results');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='bundle':bundle(a.package)
    else:results(a.inputs,a.out)

if __name__=='__main__':main()
