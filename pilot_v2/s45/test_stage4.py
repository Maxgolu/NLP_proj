"""CPU tests: real tiny random OLMo2, explicit residual route, masks and decisions."""
import contextlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from stage4_plan import masks,retained,seed_configs,prepare,load,hid
from stage3_common import json_read,json_write,read_lines,digest


class Decisions(unittest.TestCase):
    def test_cancellation_is_retained(self):
        self.assertEqual(retained([.3,-.3]*10)['classification'],'heterogeneous')
        self.assertFalse(retained([.01]*20)['retained'])
        self.assertTrue(retained([-.2]*20)['retained'])

    def test_real_inputs(self):
        root=Path(__file__).resolve().parents[1];stage3=root/'results/stage3_inputs_v1';design=root/'results/stage4_design_v2'
        if not stage3.exists():self.skipTest('Source inputs not present in isolated bundle')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'inputs';prepare(stage3,design,out)
            _,sc,pairs=load(out,out/'coverage.json');self.assertEqual(len(pairs),178)
            _,sc,core=load(out,out/'seed.json');self.assertEqual(len(core),40)
            self.assertEqual(len(sc['configs']),len({c['id'] for c in sc['configs']}))
            for p in pairs:
                self.assertTrue(p['masks']['query_is_token']);self.assertEqual(len(p['masks']['colon']),1)
                self.assertFalse(set(p['masks']['query_mother']) & set(p['masks']['other_mother']))

    def test_analysis_shards_and_tamper_detection(self):
        from stage4_run import identity,verify_run
        from stage4_analyze import gather,next_schedule,analyze
        root=Path(__file__).resolve().parents[1]
        if not (root/'results/stage3_inputs_v1').exists():self.skipTest('Saved inputs absent')
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);inputs=base/'inputs'
            prepare(root/'results/stage3_inputs_v1',root/'results/stage4_design_v2',inputs)
            _,sc,pairs=load(inputs,inputs/'coverage.json');runs=[]
            for shard in range(2):
                run=base/f'run{shard}';(run/'chunks').mkdir(parents=True);runs.append(run)
                ident=identity(inputs,inputs/'coverage.json',shard,2)
                json_write(run/'manifest.json',dict(identity=ident));json_write(run/'gate.json',dict(passed=True))
                rows=[]
                for i,p in enumerate(pairs):
                    if i%2!=shard:continue
                    for c in sc['configs']:
                        rows.append(dict(pair_id=p['id'],family=p['family'],order=p['order'],config_id=c['id'],direction='noise',
                                         effect=.01,prefix=p['shared_prefix'],origin='synthetic_test',channel_norm=.1,
                                         query_first=p['masks']['_query_first'],baseline=dict(clean_logit=1.,corr_logit=0.),
                                         result=dict(clean_logit=.99,corr_logit=0.)))
                fp=run/'chunks/data.json';json_write(fp,rows);json_write(fp.with_suffix('.ok'),dict(identity=ident,sha256=digest(fp)))
                json_write(run/'done.json',dict(complete=True))
            _,_,rr,fam,summ=gather(inputs,inputs/'coverage.json',runs)
            self.assertEqual(len(rr),712);self.assertEqual(len(fam),356)
            self.assertTrue(all(abs(s['mean']-.01)<1e-10 for s in summ))
            analyze(inputs,inputs/'coverage.json',runs,base/'analysis')
            self.assertTrue(json_read(base/'analysis/verification.json')['complete'])
            with self.assertRaises(ValueError):gather(inputs,inputs/'coverage.json',runs[:1])
            next_schedule(inputs,inputs/'coverage.json',runs,base/'seed.json','after-coverage')
            _,seed,_=load(inputs,base/'seed.json');self.assertFalse(seed.get('requires_coverage',False))
            fp=runs[0]/'chunks/data.json';fp.write_text('[]')
            with self.assertRaises(ValueError):verify_run(runs[0],inputs,inputs/'coverage.json')

    def test_schedule_rejects_noncausal_Q(self):
        from stage4_plan import write_schedule,route
        root=Path(__file__).resolve().parents[1]
        if not (root/'results/stage3_inputs_v1').exists():self.skipTest('Saved inputs absent')
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);inputs=base/'inputs'
            prepare(root/'results/stage3_inputs_v1',root/'results/stage4_design_v2',inputs)
            cfg=route(hid('L17H1'),'query_mother','head',hid('L18H19'),'Q','bad')
            write_schedule(base/'bad.json',inputs,[cfg],'core',['noise'])
            with self.assertRaises(ValueError):load(inputs,base/'bad.json')


class TinyRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('CPU torch/transformers unavailable') from exc
        from stage4_engine import Stage4Engine
        torch.manual_seed(43);torch.set_num_threads(2)
        cfg=Olmo2Config(vocab_size=41,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
                        num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        cls.t=torch;cls.e=Stage4Engine(torch,None,Olmo2ForCausalLM(cfg).eval())
        cls.ids=torch.tensor([[1,3,4,5,6,7]]);cls.other=torch.tensor([[1,8,9,5,6,7]])
        cls.heads=list(range(12));cls.g=10;cls.d=11

    def captures(self):
        e=self.e
        return e.capture4(self.ids,self.heads,self.g,self.d),e.capture4(self.other,self.heads,self.g,self.d)

    def test_identity_channel_and_cleanup(self):
        e=self.e;c,r=self.captures()
        for ch,pos in [('Q',[5]),('K',[1,2]),('V',[1,2]),('KV',[1,2])]:
            cfg=dict(kind='head',receiver=9,channel=ch)
            a=e.endpoint(self.ids,cfg,pos,c,self.g,self.d,6)
            self.assertAlmostEqual(a['clean_logit'],c['output']['clean_logit'],places=6)
        self.assertLess(c['reconstruction_error'],1e-5)
        for layer in e.layers:
            self.assertFalse(layer.self_attn._forward_pre_hooks)
            self.assertFalse(layer.mlp._forward_pre_hooks)

    def test_explicit_residual_route_matches_manual_construction(self):
        e=self.e;t=self.t;c,r=self.captures();source=1;pos=[1,2]
        hy=e.hybrid(self.ids,source,pos,r,c,self.heads,self.g,self.d)
        # Independent oracle: add the exact shared-normalization increment directly
        # at the receiver MLP input; earlier normalized branches are clean frozen.
        delta=hy['branches'][(0,'attn')]-c['branches'][(0,'attn')]
        expected=c['mlp_inputs'][2]+delta
        np.testing.assert_allclose(hy['mlp_inputs'][2].numpy(),expected.numpy(),atol=1e-6)
        cfg=dict(kind='mlp',receiver=2,channel='')
        got=e.endpoint(self.ids,cfg,pos,hy,self.g,self.d,6)
        def inject(mod,args):
            x=args[0].clone();x[0,pos]=expected[0,pos];return (x,)
        hook=e.layers[2].mlp.register_forward_pre_hook(inject)
        try:want=e.output(self.ids,self.g,self.d)
        finally:hook.remove()
        self.assertAlmostEqual(got['clean_logit'],want['clean_logit'],places=6)
        self.assertGreater(float(delta.abs().max()),1e-5)
        # Earlier positions and direct final Q cannot change from a fact-position source.
        np.testing.assert_allclose(hy['qkv'][9]['Q'][5].numpy(),c['qkv'][9]['Q'][5].numpy(),atol=1e-6)
        self.assertAlmostEqual(hy['output']['margin'],c['output']['margin'],places=6)

    def test_qkv_matches_output_and_other_rows_are_clean(self):
        e=self.e;c,r=self.captures();rec=9
        inj=dict(head=rec,positions=list(range(6)),row=5,channel='QKV',qkv=r['qkv'][rec])
        with e.instrument(heads=self.heads,collect=True,injection=inj):q=e.output(self.ids,self.g,self.d)
        with e.vector_patch(rec,[5],r['z'][rec][5:6]):z=e.output(self.ids,self.g,self.d)
        self.assertAlmostEqual(q['clean_logit'],z['clean_logit'],places=6)
        # Hook the actual o_proj input after SDPA to assert only the target row changes.
        saved=[]
        hook=e.layers[2].self_attn.o_proj.register_forward_pre_hook(lambda m,a:saved.append(a[0].detach().clone()))
        try:
            with e.instrument(injection=inj):e.output(self.ids,self.g,self.d)
        finally:hook.remove()
        actual=saved[0][0,:,:].reshape(6,4,8)
        np.testing.assert_allclose(actual[:5,1].numpy(),c['z'][rec][:5].numpy(),atol=1e-6)

    def test_joint_patching_is_one_shared_normalization(self):
        e=self.e;c,r=self.captures();pos=[2]
        with contextlib.ExitStack() as stack:
            for h in [0,1]:stack.enter_context(e.vector_patch(h,pos,r['z'][h][pos]))
            together=e.capture4(self.ids,self.heads,self.g,self.d)
        # Independently reconstruct the normalized attention sum at the source layer.
        z=self.t.stack([c['z'][h] for h in range(4)],dim=1).reshape(1,6,32)
        for h in [0,1]:z[0,pos,h*8:(h+1)*8]=r['z'][h][pos]
        with self.t.no_grad():expected=e.layers[0].post_attention_layernorm(e.layers[0].self_attn.o_proj(z))
        np.testing.assert_allclose(together['branches'][(0,'attn')].numpy(),expected.numpy(),atol=1e-6)

    def test_shared_prefix_direct_bypass_null(self):
        e=self.e;t=self.t;ids=t.cat([self.ids,t.tensor([[12]])],dim=1);other=t.cat([self.other,t.tensor([[12]])],dim=1)
        c=e.capture4(ids,self.heads,self.g,self.d);r=e.capture4(other,self.heads,self.g,self.d)
        hy=e.hybrid(ids,1,[5],r,c,self.heads,self.g,self.d)
        self.assertAlmostEqual(hy['output']['margin'],c['output']['margin'],places=6)
        # Donor replacement at the colon can still act through later live attention.
        cfg=dict(kind='head',receiver=9,channel='V')
        out=e.endpoint(ids,cfg,[5],hy,self.g,self.d,6)
        self.assertTrue(np.isfinite(out['margin']))

    def test_exception_restores_global_sdpa(self):
        e=self.e;original=self.t.nn.functional.scaled_dot_product_attention
        with self.assertRaises(RuntimeError):
            with e.instrument():raise RuntimeError('injected failure')
        self.assertIs(original,self.t.nn.functional.scaled_dot_product_attention)


if __name__=='__main__':unittest.main()
