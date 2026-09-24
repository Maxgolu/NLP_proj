"""CPU decision and corruption tests on synthetic complete-family data."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from unittest.mock import patch
from stage3_common import json_write,json_read,digest
from s42_plan import Registry,config,group_atoms,POLICY
from s42_analyze import summaries,extension,choose_k,contrast_records

class Decisions(unittest.TestCase):
    def test_family_cancellation_and_missing_order(self):
        rows=[dict(id='x',direction='noise',family=f,order=o,effect=.3 if f%2 else -.3) for f in range(20) for o in [0,1]]
        fam,stats=summaries(rows);self.assertEqual(len(fam),20)
        self.assertTrue(stats[0]['retained']);self.assertEqual(stats[0]['classification'],'heterogeneous')
        with self.assertRaises(ValueError):summaries(rows[:-1])
    def test_K_measured_joint_and_symmetric_gap(self):
        names=['L21H18','L22H5','L21H6','L27H6'];plan=dict(proposal=dict(g2=dict(K_prefix_order=names)))
        pairs=[dict(id='a',saved_baselines=[5.,-5.])];ix={}
        for i,(a,b) in enumerate([(1,0),(2,2),(4,4),(5,5)],1):
            cid=config(group_atoms(names[:i]))['id']
            ix[('a',cid,'noise')]=dict(effect=a);ix[('a',cid,'restore')]=dict(effect=b)
        with patch('s42_analyze.gather',return_value=(plan,None,pairs,None,ix)):
            k=choose_k(None,None,None);self.assertEqual(k['K'],names[:2]);self.assertFalse(k['floor_warning'])
        for k in ix:ix[k]['effect']=8
        with patch('s42_analyze.gather',return_value=(plan,None,pairs,None,ix)):
            self.assertTrue(choose_k(None,None,None)['floor_warning'])
    def test_extension_cap_closure_both_directions(self):
        with tempfile.TemporaryDirectory() as td:
            s=Path(td)/'s.json';r=Registry();summary=[]
            for i in range(20):
                a=r.add(group_atoms([i]));b=r.add(group_atoms([i+30]));name=f'G1:{i}:interaction'
                r.contrast(name,[(a,1),(b,-1)],'G1',kind='interaction')
                summary.append(dict(id=name,retained=True,mean_absolute=i/10))
            json_write(s,dict(configs=list(r.configs.values()),contrasts=r.contrasts))
            out,decision=extension(None,[(s,summary)])
            self.assertEqual(len(out.contrasts),16);self.assertEqual(len(out.configs),32)
            self.assertTrue(all(c['directions']==['noise','restore'] for c in out.configs.values()))
            for c in out.contrasts:self.assertTrue(all(cid in out.configs for cid,w in c['terms']))
    def test_contrast_is_paired_before_averaging(self):
        sc=dict(configs=[dict(id='a',directions=['noise']),dict(id='b',directions=['noise'])],
            contrasts=[dict(id='delta',terms=[['a',1],['b',-1]])])
        p=dict(id='p',family=1,order=0,masks={'_query_first':True},shared_prefix=[],corr={'corruption':'answer_swap'})
        index={('p','a','noise'):dict(effect=7),('p','b','noise'):dict(effect=5)}
        self.assertEqual(contrast_records(sc,[p],index)[0]['effect'],2)
        del index[('p','b','noise')]
        with self.assertRaises(KeyError):contrast_records(sc,[p],index)

class Integrity(unittest.TestCase):
    def inputs(self):
        path=Path(__file__).resolve().parents[1]/'results/stage4_s42_inputs_v1'
        if not path.exists():path=Path(__file__).resolve().parent/'inputs'
        return path
    def make_runs(self,base):
        from s42_plan import load,assigned
        from s42_run import identity,write_chunk
        inputs=self.inputs();schedule=inputs/'initial.json';plan,sc,pairs=load(inputs,schedule);runs=[]
        for rank in range(3):
            run=base/f'r{rank}';runs.append(run);ident=identity(inputs,schedule,rank,3)
            json_write(run/'manifest.json',dict(identity=ident));json_write(run/'gate.json',dict(passed=True));rows=[]
            for p in assigned(pairs,rank,3):
                n=len(p['original_ids']['clean'])+len(p['shared_prefix'])
                for c in sc['configs']:
                    for di in c['directions']:
                        effect=0 if all(a['donor']=='self' for a in c['atoms']) else .03*len(c['atoms'])**2
                        margin=5-effect if di=='noise' else -5+effect
                        rows.append(dict(pair_id=p['id'],family=p['family'],order=p['order'],prefix=p['shared_prefix'],
                            query_first=p['masks']['_query_first'],config_id=c['id'],direction=di,mode='donor',effect=effect,
                            result=dict(margin=margin,clean_logit=margin,corr_logit=0.),baseline=dict(margin=5 if di=='noise' else -5),
                            diagnostic=dict(pattern=[1/n]*n,z=[0]*128) if c['observe'] else None,origin='synthetic',norms={},fallback={}))
            write_chunk(run,ident,0,rows);json_write(run/'done.json',dict(complete=True))
        return inputs,schedule,runs
    def test_complete_analysis_selection_and_tamper(self):
        from s42_run import verify
        from s42_analyze import analyze,refine,choose_k,gather,summarize
        from stage4_plan import retained
        # Bootstrap itself is exercised above; reduce draws only by replacing the statistical
        # helper here, to focus this test on disk integrity and scheduling, not Monte Carlo cost.
        def stats(x):return dict(families=len(x),**retained(x),descriptive_ci_low=float(np.mean(x)),descriptive_ci_high=float(np.mean(x)))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs,schedule,runs=self.make_runs(root)
            with patch('s42_analyze.summarize',side_effect=stats):summ=analyze(inputs,schedule,runs,root/'analysis')
            K=choose_k(inputs,schedule,runs);self.assertTrue(K['K'])
            reg,decision=refine(inputs,schedule,runs,summ);self.assertLessEqual(decision['common_configs'],128)
            ext,dec=extension(inputs,[(schedule,summ)]);self.assertTrue(ext.contrasts)
            with self.assertRaises(ValueError):gather(inputs,schedule,runs[:2])
            path=runs[0]/'chunks/0000000.json';rows=json_read(path);rows[0]['effect']=999;json_write(path,rows)
            with self.assertRaises(ValueError):verify(runs[0],inputs,schedule)
    def test_duplicate_records_even_with_valid_hash_rejected(self):
        from s42_run import verify
        with tempfile.TemporaryDirectory() as td:
            inputs,schedule,runs=self.make_runs(Path(td));path=runs[0]/'chunks/0000000.json';rows=json_read(path);rows.append(rows[0]);json_write(path,rows)
            marker=path.with_suffix('.ok');meta=json_read(marker);meta['sha256']=digest(path);json_write(marker,meta)
            with self.assertRaises(ValueError):verify(runs[0],inputs,schedule)
    def test_submission_refuses_active_job_and_uncertain_lock(self):
        import s42_submit
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run'
            with patch.dict('os.environ',{'USER':'test'}),patch('s42_submit.output',return_value='123|s42_test'):
                with self.assertRaises(ValueError):s42_submit.check_submission(path,'s42_test',False)
            path.mkdir();json_write(path/'.running.lock',dict(pid=123,job='321'))
            with patch.dict('os.environ',{'USER':'test'}),patch('s42_submit.output',side_effect=['','']):
                with self.assertRaises(ValueError):s42_submit.check_submission(path,'s42_test',True)
            self.assertTrue((path/'.running.lock').exists())
            with patch.dict('os.environ',{'USER':'test'}),patch('s42_submit.output',side_effect=['','TIMEOUT|']):
                s42_submit.check_submission(path,'s42_test',True)
            self.assertFalse((path/'.running.lock').exists())

    def test_worker_reuse_resume_and_replacement_paths(self):
        import argparse,contextlib,io
        import torch
        from s42_plan import Registry,atom,write_schedule,load
        from s42_engine import S42Engine
        from s42_run import run,verify
        class FakeEngine:
            # Tests persistence/orchestration independently of transformer math (TinyJoint).
            t=torch;DH=2;model=object()
            patches=S42Engine.patches
            def capture_z(self,ids,heads,g,d):
                p,side=ids;n=len(p['original_ids'][side])+len(p['shared_prefix']);v=1 if side=='clean' else -1
                base=p['saved_baselines'][0 if side=='clean' else 1]
                return dict(z={h:torch.full((n,2),float(v)) for h in heads},output=dict(margin=base,clean_logit=base,corr_logit=0.))
            def evaluate(self,ids,patches,g,d,observe_head=None,row=None):
                p,side=ids;v=1 if side=='clean' else -1;base=p['saved_baselines'][0 if side=='clean' else 1]
                delta=sum(float((x-v).sum()) for ix,x in patches.values())*.02
                return dict(margin=base+delta,clean_logit=base+delta,corr_logit=0.),None
        def spec(e,p):return dict(clean=(p,'clean'),corr=(p,'corr'),n=len(p['original_ids']['clean']),g=0,d=1)
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);inputs=self.inputs();reg=Registry()
            reg.add([atom(513)],('noise',)) # saved F
            reg.add([atom(513),atom(533)],('noise','restore')) # fresh joint
            reg.add([atom(513,donor='self')],('noise','restore')) # zero control
            schedule=base/'s.json';write_schedule(schedule,inputs,reg,'core','donor','worker_test')
            args=argparse.Namespace(inputs=inputs,schedule=schedule,out=base/'run',resume=False,shard=0,shards=3,bank=None,prior=None,gate_only=False)
            with patch('s42_run.start_model',return_value=FakeEngine()),patch('s42_run.new_gates',return_value=dict(passed=True)),patch('s42_run.spec_checked',side_effect=spec),contextlib.redirect_stdout(io.StringIO()):
                run(args);rows=verify(args.out,inputs,schedule)
                self.assertTrue(any(r['origin']=='saved_stage3_F' for r in rows));self.assertTrue(any(r['origin']=='new' for r in rows))
                before={p.name:digest(p) for p in (args.out/'chunks').glob('*.json')}
                args.resume=True;run(args)
                self.assertEqual(before,{p.name:digest(p) for p in (args.out/'chunks').glob('*.json')})
                # Identical donor configurations from a prior verified phase are reused.
                prior=base/'prior.json';json_write(prior,[dict(schedule=str(schedule),runs=[str(args.out)])])
                args.out=base/'reused';args.resume=False;args.prior=prior;run(args)
                self.assertTrue(all(r['origin'].startswith('prior_phase:') for r in verify(args.out,inputs,schedule)))

if __name__=='__main__':unittest.main()
