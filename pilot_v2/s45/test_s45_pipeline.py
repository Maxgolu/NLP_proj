"""End-to-end CPU exercise of the S4.5 worker/analysis paths that a null discovery skips.

Uses the random 32x32-head tiny model (S45_TINY_MODEL) and synthetic inputs: Stage D with a
forced selection (routes in both directions, attachments and controls in the diagnostic
background), the freeze manifest, sealed held-out inputs, the validation worker in full-bank
mode and the final labels. Slow (minutes); run by hand, not inside the worker gate.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
from stage3_common import json_read,json_write,digest,write_lines

ROOT=Path(__file__).resolve().parent

class Pipeline(unittest.TestCase):
    def test_stage_d_freeze_and_validation(self):
        import test_s45 as T
        from s45_plan import stage_d,validation,build_pair,annotate_row,freeze_inputs,load,C33,ids,state,hid,attachment,states_for,CELLS
        import s45_analyze as A
        import s45_freeze as F
        from s45_run import code_identity,mean_identity
        env=dict(os.environ,S45_TINY_MODEL='7',CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1')
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);inputs=td/'inputs';tok,pairs,plan=T.synthetic_inputs(inputs,families=range(0,24,2))
            def worker(args):
                r=subprocess.run([sys.executable,str(ROOT/'s45_run.py')]+args,env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stdout[-3000:]+r.stderr[-3000:])
            worker(['means','--inputs',str(inputs),'--out',str(td/'means_r0')])
            from s45_means import reduce_bank,MeanBank
            bank=td/'bank';reduce_bank(inputs,[td/'means_r0'],bank);MeanBank(bank,inputs,'lofo')
            sel=dict(selected=[dict(candidate='L17H5',structure='T2',record='route_pair',gamma_mean=.3,gamma_class='coherent',stage_b_functional=True),
                               dict(candidate='L23H10',structure='T5',record='route_pair',gamma_mean=-.2,gamma_class='heterogeneous',stage_b_functional=False),
                               dict(candidate='L1H27',structure='B',record='functional_only',d_b_mean=.15)],B=C33)
            sd=stage_d(inputs,plan,C33,sel).write(td/'stage_d.json');worker(['run','--inputs',str(inputs),'--schedule',str(sd),'--out',str(td/'sd_r0'),'--bank',str(bank)])
            rows=A.gather(inputs,sd,[td/'sd_r0']);self.assertEqual(len(rows),16*8)
            self.assertTrue(all(r['kind'] in ['route','attachment'] for r in rows));self.assertTrue(any(r['role']=='control' for r in rows))
            ctrl=plan['control_rosters'][attachment('T2','L17H5')['key']]['control']
            for r in rows:
                if r['kind']=='attachment':self.assertIn(hid(ctrl) if r['candidate']=='L17H5' else hid(plan['control_rosters'][attachment('T5','L23H10')['key']]['control']),json_read(sd)['states'][r['state']]['live'])
            d=A.analyze_stage_d(inputs,sd,[td/'sd_r0'],td/'an_d',C33,sel,[])
            self.assertEqual(len(d['gamma']),2);self.assertEqual(len(d['attachments']),2)
            for g in d['gamma']:self.assertIn(g['bidirectional'],['coherent','heterogeneous','asymmetric','unreplicated'])
            self.assertTrue((td/'an_d'/'attachment_controls.csv').exists() and (td/'an_d'/'core_bidirectional_gamma.csv').exists())
            # Freeze with every required stage marked complete (the other stages are stubs here).
            stages={}
            for name in F.REQUIRED_STAGES:
                r=td/f'{name}_stub';r.mkdir();json_write(r/'manifest.json',dict(identity=name));json_write(r/'done.json',dict(complete=True))
                stages[name]=dict(schedule=None,runs=[str(r)],analysis=[])
            stages['stage_d']=dict(schedule=str(sd),runs=[str(td/'sd_r0')],analysis=[])
            signs=A.discovery_signs([dict(candidate='L17H5',d_b=dict(retained=True,mean=.2))],sel,d)
            freeze=F.build(inputs,td/'freeze.json',stages,C33,'pass',sel,signs,bank/'mean_bank_manifest.json',code_identity())
            # Sealed held-out inputs: odd synthetic families, split 'heldout', full-bank mode.
            hp=[build_pair(tok,annotate_row(tok,T.synthetic_rows(f,o)),annotate_row(tok,T.synthetic_rows(f,o,True)),'heldout') for f in range(1,7,2) for o in (0,1)]
            hout=td/'heldout';hout.mkdir();write_lines(hout/'pairs.jsonl.gz',hp)
            hplan=dict(plan,heldout=dict(plan['heldout'],sealed=False,opened_with_freeze=digest(td/'freeze.json')),families=[1,3,5],core_families=[],gate_families=[1],
                       pair_hash=digest(hout/'pairs.jsonl.gz'),discovery_plan_hash=digest(inputs/'plan.json'))
            json_write(hout/'plan.json',hplan);vs=validation(hout,hplan,freeze).write(hout/'validation.json',freeze_hash=digest(td/'freeze.json'))
            _,sc,vp=load(hout,vs);self.assertEqual(len(vp),6);self.assertTrue(all(p['split']=='heldout' for p in vp))
            kinds=[(j['kind'],j.get('baseline'),len(j.get('cells',[]))) for j in sc['jobs']]
            self.assertEqual(sum(1 for k in kinds if k[0]=='behavior' and k[1]=='mean'),7);self.assertEqual(sum(1 for k in kinds if k[0]=='behavior' and k[1]=='donor'),5)
            self.assertEqual(sum(1 for k in kinds if k[0]=='route'),8);self.assertEqual(sum(1 for k in kinds if k[0]=='attachment'),8)
            worker(['run','--inputs',str(hout),'--schedule',str(vs),'--out',str(td/'val_r0'),'--bank',str(bank)])
            self.assertEqual(json_read(td/'val_r0'/'manifest.json')['identity']['bank_mode'],'full')
            labels=A.analyze_validation(hout,vs,[td/'val_r0'],td/'an_v',freeze,dict(signs=signs))
            self.assertEqual(len(labels),3);self.assertTrue(all(l['final_label'][:3] in ['(a)','(b)','(c)','(d)','(e)'] for l in labels))
            self.assertTrue((td/'an_v'/'validation_labels.csv').exists())
            # The mean bank refuses LOFO evaluation of a held-out family and full-bank evaluation of a discovery family.
            with self.assertRaises(ValueError):MeanBank(bank,hout,'lofo').cell_vectors(1,[['0','mother','last']])
            with self.assertRaises(ValueError):MeanBank(bank,inputs,'full').cell_vectors(0,[['0','mother','last']])

if __name__=='__main__':unittest.main()
