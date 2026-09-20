import unittest
from types import SimpleNamespace
import numpy as np
from stage2_common import groups,plan_work,spearman
from stage1_anatomy import distance_bucket


class PlanningTests(unittest.TestCase):
    def test_gpu_groups(self):
        self.assertEqual(list(map(len,groups(list('abcdef')))),[2,2,2])
        self.assertEqual(list(map(len,groups(list('abcde')))),[3,2])
        self.assertEqual(list(map(len,groups(list('abcd')))),[2,2])
        with self.assertRaises(ValueError):groups(['0'])

    def test_balanced_complete_family_assignment(self):
        pairs=[dict(family=f,id=f'{f}/{o}',order=o,clean={'prompt':'x'*50}) for f in range(89) for o in range(2)]
        planned=plan_work(pairs,3)
        self.assertEqual(sum(p['exact_all'] for p in planned),40)
        for f in range(89):self.assertEqual(len({p['worker'] for p in planned if p['family']==f}),1)
        counts=[sum(p['exact_all'] and p['worker']==w for p in planned) for w in range(3)]
        self.assertLessEqual(max(counts)-min(counts),2)

    def test_tied_ranks_and_distances(self):
        self.assertAlmostEqual(spearman([1,1,3],[4,4,9]),1)
        self.assertIsNone(spearman([1,1],[2,3]))
        self.assertEqual([distance_bucket(j,2) for j in [2,3,5]],['self','previous','distance_ge_2'])


try:
    import torch
    from stage2_engine import Engine
except ImportError:
    torch=None


if torch is not None:
    class TinyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.v_proj=torch.nn.Linear(8,8,bias=False);self.o_proj=torch.nn.Linear(8,8,bias=False)
        def forward(self,x):
            n=x.shape[1];a=torch.ones(n,n,device=x.device).tril()
            a=a/a.sum(dim=-1,keepdim=True)
            return self.o_proj(a@self.v_proj(x))

    class TinyLayer(torch.nn.Module):
        def __init__(self,nonlinear):
            super().__init__();self.self_attn=TinyAttention();self.nonlinear=nonlinear
        def forward(self,x):
            y=x+self.self_attn(x)
            return y+0.1*torch.tanh(y) if self.nonlinear else y

    class TinyModel(torch.nn.Module):
        def __init__(self,nonlinear=False):
            super().__init__();torch.manual_seed(42)
            self.embedding=torch.nn.Embedding(12,8);self.model=torch.nn.Module()
            self.model.layers=torch.nn.ModuleList([TinyLayer(nonlinear) for _ in range(2)])
            self.unembed=torch.nn.Linear(8,12,bias=False)
            self.config=SimpleNamespace(num_attention_heads=2,hidden_size=8)
        def get_input_embeddings(self):return self.embedding
        def forward(self,ids,use_cache=False):
            x=self.embedding(ids)
            for l in self.model.layers:x=l(x)
            return SimpleNamespace(logits=self.unembed(x))


@unittest.skipIf(torch is None,'CPU PyTorch required for numerical tests')
class EngineTests(unittest.TestCase):
    def setup_engine(self,nonlinear=False):
        e=Engine(torch,None,TinyModel(nonlinear))
        # Token 7 is a common answer prefix and must NEVER be donor-patched.
        spec=dict(clean=torch.tensor([[1,2,3,7]]),corr=torch.tensor([[1,4,3,7]]),n=3,g=8,d=9)
        clean=e.cache(spec['clean'][:,:3]);corr=e.cache(spec['corr'][:,:3])
        return e,spec,clean,corr

    def test_linear_exact_first_order_and_ig_agree(self):
        e,s,a,b=self.setup_engine();m=e.metric(s['clean'],s['g'],s['d'])
        exact,_=e.exact(s,b,[(0,0),(0,1),(1,0),(1,1)])
        expected=np.array([exact[h]-m for h in range(4)])
        np.testing.assert_allclose(e.attribution(s,a,b),expected,atol=2e-7)
        np.testing.assert_allclose(e.attribution(s,a,b,steps=5),expected,atol=2e-7)

    def test_nonlinear_gradient_matches_local_finite_difference(self):
        e,s,a,b=self.setup_engine(True);attr=e.attribution(s,a,b);eps=.01
        for h in range(4):
            li,hi=divmod(h,2)
            with e.patch(li,hi,b[li],eps):plus=e.metric(s['clean'],s['g'],s['d'])
            with e.patch(li,hi,b[li],-eps):minus=e.metric(s['clean'],s['g'],s['d'])
            self.assertAlmostEqual(attr[h],(plus-minus)/(2*eps),delta=2e-5)

    def test_patch_does_not_replace_answer_prefix(self):
        e,s,a,b=self.setup_engine();full=e.cache(s['clean']);seen={}
        with e.patch(1,0,b[1]):
            hook=e.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda mod,inp:seen.update(z=inp[0].detach().clone()))
            try:e.metric(s['clean'],s['g'],s['d'])
            finally:hook.remove()
        torch.testing.assert_close(seen['z'][:,3:],full[1][:,3:],rtol=0,atol=0)
        torch.testing.assert_close(seen['z'][:,:3,:4],b[1][:,:,:4],rtol=0,atol=0)

    def test_self_patch_and_self_attribution_zero(self):
        e,s,a,b=self.setup_engine(True);m=e.metric(s['clean'],s['g'],s['d'])
        exact,_=e.exact(s,a,[(0,0),(1,1)])
        for v in exact.values():self.assertEqual(v,m)
        self.assertTrue(np.all(e.attribution(s,a,a)==0))


if __name__=='__main__':unittest.main()
