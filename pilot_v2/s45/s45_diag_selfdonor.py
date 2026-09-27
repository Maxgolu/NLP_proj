"""Diagnose the self-donor identity gate on the real model: which background, anchor and component drift.

Run on a GPU node from pilot_v2/:  python3 s45/s45_diag_selfdonor.py --inputs s45/inputs_s45 --bank $PILOT_RUNS/s45_discovery_v1/mean_bank
No measurement is written; this is a read-only diagnostic.
"""
import argparse
from pathlib import Path
import numpy as np
from stage3_common import json_read,read_lines
from s45_plan import load,ids,hid,STRUCTURES,C33,CANDIDATES,attachment,CELLS
from s45_run import spec_checked,replacement
from stage4_run import error

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--pairs',type=int,default=2);a=p.parse_args()
    plan,pairs=load(a.inputs)
    from stage1_scan import load_model
    from s45_engine import S45Engine
    from s45_means import MeanBank
    t,tok,model,lock=load_model();e=S45Engine(t,tok,model);bank=MeanBank(a.bank,a.inputs,'lofo')
    print('devices',getattr(model,'hf_device_map',{}),flush=True)
    gate=[q for q in pairs if q['family'] in plan['gate_families']][:a.pairs]
    cand=hid(CANDIDATES[0]);ctrl=hid(plan['control_rosters'][attachment('T1',CANDIDATES[0])['key']]['control'])
    bgs={'full':None,'all_live':list(range(1024)),'C33':ids(C33),'C33+h':sorted(set(ids(C33))|{cand}),'diag':sorted(set(ids(C33))|{cand,ctrl})}
    for q in gate:
        sp=spec_checked(e,q,'x00');rep,info=replacement(e,q,'x00','mean',bank,{});n=sp['n']
        base=e.output(sp['ids'],sp['g'],sp['d']);base2=e.output(sp['ids'],sp['g'],sp['d'])
        print(f"\n=== {q['id']} n={n} prefix={q['shared_prefix']}  plain-forward repeat drift={error(base,base2):.6f}",flush=True)
        for name,live in bgs.items():
            o1,_=e.behave(sp,live,rep);o2,_=e.behave(sp,live,rep)
            print(f"[{name}] behave repeat drift={error(o1,o2):.6f} margin={o1['margin']:.4f}",flush=True)
            for tname in ['T4','T5','T1','T2']:
                an=STRUCTURES[tname]['anchor'];job=dict(source=hid(an['source']),site=an['site'],live=ids(an['live']),receiver=hid(an['receiver']),channel=an['channel'],direction='noise')
                if live is not None and any(h not in live for h in [job['source'],job['receiver'],*job['live']]):continue
                cache={};r=e.route(sp,sp,live,rep,rep,job,cache)
                rec=[v for k,v in cache.items() if k[0]=='x00'][0]
                a_=error(rec['output'],o1);b_=error(r['hybrid_output'],rec['output']);c_=error(r['endpoint'],rec['output'])
                # receiver channel drift between hybrid and recipient captures is r['channel_norm'] (already relative to rec)
                print(f"   {tname:2s} effect={r['effect']:+.5f} | capture-vs-behave={a_:.5f} | hybrid(bypass)-vs-intact={b_:.5f} | endpoint-vs-intact={c_:.5f} | receiver channel norm={r['channel_norm']:.5f} | source_norm={r['source_norm']:.5f} | recon={r['reconstruction_error']:.5f}",flush=True)
                # frozen-branch consistency: recapture recipient and compare branch tensors
                if b_>1e-3:
                    with e.background(live,sp['positions'],rep):rec2=e.capture_s43(sp['ids'],[job['source'],job['receiver']],sp['g'],sp['d'])
                    diffs={k:float((rec2['branches'][k].float()-rec['branches'][k].float()).abs().max()) for k in rec['branches']}
                    worst=sorted(diffs.items(),key=lambda x:-x[1])[:3];print('      recapture branch drift (worst 3):',worst,flush=True)
                    print('      recapture output drift:',error(rec2['output'],rec['output']),flush=True)

if __name__=='__main__':main()
