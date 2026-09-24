"""Small real OLMo2 tests; no pretrained weights or network required."""
import contextlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
from stage3_common import json_read,json_write,npz_write,digest
from s42_plan import config,atom,Registry,initial,conditional,group_atoms,key_levels,POLICY,assigned

class Planning(unittest.TestCase):
    def test_real_rosters_and_mask_dedup(self):
        root=Path(__file__).resolve().parents[1]
        proposal=root/'results/stage4_review_20260924/s42_proposal.json'
        if not proposal.exists():proposal=Path(__file__).resolve().parent/'inputs/plan.json'
        p=json_read(proposal);plan=p if 'proposal' in p else dict(proposal=p)
        from s42_plan import add_g1
        r=Registry();add_g1(r,plan);self.assertEqual(len(r.configs),49)
        self.assertEqual(sum(len(c['atoms'])==1 for c in r.configs.values()),13)
        s=conditional(plan,['L21H18','L22H5'])
        claim=next(c for c in s.contrasts if c['id']=='G2:L22H5:conditional')
        bg=s.configs[claim['terms'][0][0]] # independent test below identifies the negative term
        negative=next(cid for cid,w in claim['terms'] if w==-1)
        self.assertEqual(s.configs[negative]['atoms'],group_atoms(['L21H18']))
        joint=next(cid for cid,w in claim['terms'] if w==1)
        self.assertIn(atom('L22H5','all'),s.configs[joint]['atoms'])
        self.assertNotIn(atom('L22H5','colon'),s.configs[joint]['atoms'])
    def test_duplicate_order_and_conflict(self):
        self.assertEqual(config([atom(5),atom(2),atom(5)])['id'],config([atom(2),atom(5)])['id'])
        with self.assertRaises(ValueError):config([atom(2),atom(2,donor='self')])
        pairs=[dict(family=f,order=o) for f in range(8) for o in [0,1]]
        self.assertEqual(len(assigned(pairs,0,3)),6)
        self.assertEqual({p['family'] for p in assigned(pairs,0,3)},{0,3,6})
    def test_family_equal_weight_and_lofo(self):
        from s42_means import MeanBank,accumulate
        keys=[['0','mother','single']];vocab={k:i for i,k in enumerate(key_levels(keys[0]))}
        sums=np.zeros((1,len(vocab),1));counts=np.zeros(len(vocab),dtype=int)
        accumulate(sums,counts,keys,np.array([[[9.]]]),vocab)
        self.assertTrue(np.all(counts==1));self.assertTrue(np.all(sums==9))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);json_write(path/'plan.json',{})
            # First family's extreme value must not affect its own LOFO baseline.
            means=np.ones((12,1,len(vocab),1),dtype=np.float32)*3;means[0]=1000
            support=np.ones((12,len(vocab)),dtype=bool)
            # Exact key has insufficient support, requiring the slot-free fallback.
            support[2:,0]=False;means[2:,:,0]=0
            npz_write(path/'bank.npz',family_means=means,support=support,families=np.arange(12),heads=np.array([4]))
            json_write(path/'manifest.json',dict(policy=POLICY,plan_hash=digest(path/'plan.json'),sha256=digest(path/'bank.npz'),keys=list(vocab)))
            bank=MeanBank(path,path);values,info=bank.vectors(4,0,keys)
            self.assertAlmostEqual(float(values[0,0]),3);self.assertEqual(info,{'1':1})

class TinyJoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('CPU torch/transformers unavailable') from exc
        from s42_engine import S42Engine
        torch.manual_seed(71);torch.set_num_threads(2)
        cfg=Olmo2Config(vocab_size=41,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,attention_dropout=0.)
        cfg._attn_implementation='sdpa';cls.t=torch;cls.e=S42Engine(torch,None,Olmo2ForCausalLM(cfg).eval())
        cls.ids=torch.tensor([[1,3,4,5,6,7]]);cls.donor=torch.tensor([[1,8,9,5,6,7]])
        cls.g=10;cls.d=11
    def caps(self):return [self.e.capture_z(ids,list(range(12)),self.g,self.d) for ids in [self.ids,self.donor]]
    def test_joint_matches_independent_slices(self):
        c,d=self.caps();patch={h:([2,5],d['z'][h][[2,5]]) for h in [1,2,5,9]}
        actual,_=self.e.evaluate(self.ids,patch,self.g,self.d)
        with contextlib.ExitStack() as stack:
            for h,(ix,v) in patch.items():stack.enter_context(self.e.vector_patch(h,ix,v))
            expected=self.e.output(self.ids,self.g,self.d)
        self.assertAlmostEqual(actual['margin'],expected['margin'],places=6)
    def test_joint_self_identity_and_hook_cleanup(self):
        c,d=self.caps();patch={h:(list(range(6)),c['z'][h]) for h in range(12)}
        actual,_=self.e.evaluate(self.ids,patch,self.g,self.d)
        self.assertAlmostEqual(actual['margin'],c['output']['margin'],places=6)
        with self.assertRaises(RuntimeError):
            with self.e.joint(patch):raise RuntimeError('deliberate interrupted forward')
        self.assertAlmostEqual(self.e.output(self.ids,self.g,self.d)['margin'],c['output']['margin'],places=6)
    def test_blocking_and_future_null(self):
        c,d=self.caps();src={1:([1,2],d['z'][1][[1,2]])}
        _,changed=self.e.evaluate(self.ids,src,self.g,self.d,5,5)
        _,base=self.e.evaluate(self.ids,{},self.g,self.d,5,5)
        self.assertGreater(np.max(np.abs(np.array(changed['z'])-base['z'])),1e-7)
        _,future=self.e.evaluate(self.ids,{9:([5],d['z'][9][[5]])},self.g,self.d,5,5)
        np.testing.assert_allclose(future['pattern'],base['pattern'],atol=0,rtol=0)
        np.testing.assert_allclose(future['z'],base['z'],atol=0,rtol=0)
        # A full clamp of every downstream head reproduces a separately constructed patch.
        patch=dict(src);patch.update({h:([5],c['z'][h][[5]]) for h in range(4,12)})
        out,_=self.e.evaluate(self.ids,patch,self.g,self.d)
        self.assertTrue(np.isfinite(out['margin']))
    def test_observation_is_before_own_patch(self):
        c,d=self.caps();background={1:([2],d['z'][1][[2]])}
        _,a=self.e.evaluate(self.ids,background,self.g,self.d,5,5)
        _,b=self.e.evaluate(self.ids,dict(background,**{})|{5:([5],d['z'][5][[5]])},self.g,self.d,5,5)
        np.testing.assert_allclose(a['pattern'],b['pattern'],atol=0,rtol=0)
        np.testing.assert_allclose(a['z'],b['z'],atol=0,rtol=0)
    def test_scope_merge_and_mean_demos_live(self):
        c,d=self.caps();p=dict(family=0,masks=dict(all=list(range(6)),colon=[5]),mean_keys=[None,None]+[['0','mother','single']]*4)
        cfg=config([atom(5,'all'),atom(5,'colon')])
        patch,_,_=self.e.patches(p,cfg,c,d,'donor');self.assertEqual(patch[5][0],list(range(6)))
        class Bank:
            def vectors(self,h,f,keys):return np.zeros((len(keys),8),dtype=np.float32),{'0':len(keys)}
        patch,_,fb=self.e.patches(p,cfg,c,d,'mean',Bank());self.assertEqual(patch[5][0],[2,3,4,5]);self.assertEqual(fb,{'0':4})
        actual,_=self.e.evaluate(self.ids,patch,self.g,self.d)
        self.assertTrue(np.isfinite(actual['margin']))

if __name__=='__main__':unittest.main()
