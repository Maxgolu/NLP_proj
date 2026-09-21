"""Protocol regressions and real tiny-OLMo2 hook tests; no pretrained weights/network."""
import collections
import tempfile
import unittest
from pathlib import Path
import numpy as np
from stage3_common import *
from stage3_run import checkpoint_done,finish_checkpoint,validate_inputs


class ProtocolTests(unittest.TestCase):
    def test_bootstrap_uses_all_families(self):
        a=np.zeros((89,2));a[20:]=1;b=np.zeros((89,3))
        r=bootstrap_median_difference(a,b,draws=300)
        self.assertEqual(r['families'],89);self.assertGreater(r['interval'][0],.5)

    def test_undefined_ri_not_zero(self):
        self.assertIsNone(normalized_ri([.2,.2,.2]))
        np.testing.assert_allclose(normalized_ri([.1,.3,.2]),[0,1,0],atol=1e-12)

    def test_resume_detects_mutation_and_missing_companion(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'output.json';s=Path(folder)/'roles.json'
            json_write(p,{'a':1});json_write(s,{'b':2});finish_checkpoint(p,'run1',[s])
            self.assertTrue(checkpoint_done(p,'run1'))
            with self.assertRaises(ValueError):checkpoint_done(p,'run2')
            json_write(p,{'a':3})
            with self.assertRaises(ValueError):checkpoint_done(p,'run1')

    def test_query_swap_changes_only_question(self):
        row=dict(id='a',prompt='A is the mother of B.\nC is the mother of D.\nQuestion: Who is the mother of B?\nAnswer:',
                 question_entity='B',gold='A',candidates=['A','C'],facts=[dict(block='test',head='B',tail='A'),dict(block='test',head='D',tail='C')])
        new=changed_query(row)
        self.assertEqual(new['gold'],'C');self.assertEqual(new['facts'],row['facts'])
        self.assertEqual(new['prompt'],row['prompt'].replace('mother of B?','mother of D?'))
        self.assertEqual(row['question_entity'],'B')

    def test_inventory_and_counts(self):
        inputs=Path(__file__).resolve().parents[1]/'results/stage3_inputs_v1'
        if not inputs.exists():self.skipTest('Local prepared inputs not beside source tree')
        p,pairs,_,work=validate_inputs(inputs)
        self.assertEqual(work['patched_forwards'],263328)
        self.assertEqual(work['components']['missing_final_position'],46*178)
        with np.load(inputs/'references.npz') as z:means=-z['scopeP_delta'].mean(0)
        inv=json_read(inputs/'inventory.json');lookup=dict(zip(map(head_name,p['heads']),means))
        for name in inv['groups']['RI-selected_small-effect']:self.assertLess(abs(lookup[name]),.1)
        for name in inv['groups']['strong-negative']:self.assertLessEqual(lookup[name],-.3)
        for group in ['strong-positive_outside-RI','strong-positive_RI-selected']:
            for name in inv['groups'][group]:self.assertGreaterEqual(lookup[name],.3)
        for name in inv['groups']['moderate-effect_supplement']:self.assertTrue(.1<=abs(lookup[name])<.3)
        self.assertEqual(len(set(p['heads'])),105)

    def test_real_saved_prompt_annotations_and_query_controls(self):
        root=Path(__file__).resolve().parents[1];tokenizer=root/'pilot_v3/olmo2_tokenizer/tokenizer.json'
        if not tokenizer.exists():self.skipTest('Local tokenizer artifact not in portable cluster bundle')
        from transformers import PreTrainedTokenizerFast
        tok=PreTrainedTokenizerFast(tokenizer_file=str(tokenizer))
        prompts=list(read_lines(root/'results/stage3_inputs_v1/prompts.jsonl.gz'))
        for r in prompts:
            ann=annotate(tok,r);self.assertEqual(ann['ids'],r['token_ids'])
            self.assertEqual(ann['test_positions'][0],next(i for i,(a,b) in enumerate(ann['offsets']) if b>r['test_start']))
        for p in read_lines(root/'results/stage3_inputs_v1/pairs.jsonl.gz'):
            if p['exact_all']:
                r=changed_query(p['clean']);ann=annotate(tok,r)
                self.assertEqual(r['order'],p['order']);self.assertEqual(len(ann['facts']),4)
                self.assertNotEqual(r['gold'],p['clean']['gold'])


class OlmoHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('Install CPU torch and transformers for architecture tests') from exc
        from stage3_engine import Stage3Engine
        torch.manual_seed(7);torch.set_num_threads(2)
        config=Olmo2Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=128,
                           pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.)
        config._attn_implementation='sdpa'
        cls.t=torch;cls.model=Olmo2ForCausalLM(config).eval();cls.e=Stage3Engine(torch,None,cls.model)

    def test_actual_sdpa_rope_qknorm_reconstruction(self):
        t=self.t;e=self.e;ids=t.tensor([[1,8,4,6,9,7]])
        before=e.readout(ids,10,11);cap=e.capture(ids,list(range(8)),[2,5],True)
        after=e.readout(ids,10,11)
        np.testing.assert_array_equal(before,after)
        for h,c in cap['heads'].items():
            self.assertLess(c['reconstruction_error'],1e-6)
            self.assertEqual(tuple(c['pattern'].shape),(2,6))
            self.assertTrue(t.equal(c['pattern'][0,3:],t.zeros(3)))
        self.assertEqual(tuple(cap['value_inputs'][1].shape),(6,32))

    def test_final_patch_keeps_shared_prefix_live(self):
        t=self.t;e=self.e;a=t.tensor([[1,8,4,6,9,7,12]]);b=t.tensor([[1,8,5,6,9,7,12]])
        n=6;head=4;rows=list(range(n));ca=e.capture(a[:,:n],[head],rows);cb=e.capture(b[:,:n],[head],rows)
        spec=dict(clean=a,corr=b,g=10,d=11,n=n)
        base=e.readout(a,10,11)
        np.testing.assert_allclose(e.patched(spec,head,[n-1],ca['heads'][head]['z'][-1:]),base,atol=1e-7)
        seen=[]
        handle=e.layers[1].self_attn.o_proj.register_forward_pre_hook(lambda mod,args:seen.append(args[0].detach().clone()))
        try:
            direct=e.patched(spec,head,[5],cb['heads'][head]['z'][-1:])
        finally:handle.remove()
        self.assertEqual(seen[0].shape[1],7)
        # The inserted answer-prefix position is recomputed, never copied from the bare donor.
        self.assertEqual(cb['heads'][head]['z'].shape[0],6)
        mix=e.mix_vectors(ca,cb,head,5)
        np.testing.assert_allclose(e.patched(spec,head,[5],mix[3]),direct,atol=1e-6)
        np.testing.assert_allclose(e.patched(spec,head,[5],mix[0]),base,atol=1e-6)
        full=e.patched(spec,head,rows,cb['heads'][head]['z'])
        caches=e.cache(b[:,:n])
        with e.patch(1,0,caches[1]):ordinary=e.readout(a,10,11)
        np.testing.assert_allclose(full,ordinary,atol=1e-7)
        np.testing.assert_allclose(e.patched(spec,head,[0,1],cb['heads'][head]['z'][:2]),base,atol=1e-7)

    def test_projection_uses_exact_value_input_and_weight_slice(self):
        t=self.t;e=self.e;ids=t.tensor([[1,8,4,6]])
        cap=e.capture(ids,[6],[3],True);x=cap['value_inputs'][1][3:4];att=e.layers[1].self_attn
        actual=e.projected_logits(6,x,True,[10,11,12])
        expected=x@att.v_proj.weight[16:24].T@att.o_proj.weight[:,16:24].T@e.model.lm_head.weight[[10,11,12]].T
        np.testing.assert_allclose(actual,expected.numpy(),atol=1e-7)
        self.assertFalse(np.allclose(x.numpy(),e.model.get_input_embeddings().weight[ids[0,3]].numpy()))

    def test_hooks_restored_after_capture_error(self):
        t=self.t;original=t.nn.functional.scaled_dot_product_attention
        with self.assertRaises(IndexError):self.e.capture(t.tensor([[1,3,4]]),[0],[99])
        self.assertIs(t.nn.functional.scaled_dot_product_attention,original)
        self.assertTrue(np.isfinite(self.e.readout(t.tensor([[1,3,4]]),10,11)).all())

    def test_diagnostics_on_real_tokenized_prompt(self):
        from transformers import PreTrainedTokenizerFast,Olmo2Config,Olmo2ForCausalLM
        from stage3_engine import Stage3Engine
        from stage3_measure import checked_capture,anatomy,matched_ri,causal_pair
        root=Path(__file__).resolve().parents[1];path=root/'pilot_v3/olmo2_tokenizer/tokenizer.json'
        if not path.exists():self.skipTest('Local tokenizer artifact not in cluster bundle')
        tok=PreTrainedTokenizerFast(tokenizer_file=str(path));t=self.t
        config=Olmo2Config(vocab_size=len(tok),hidden_size=8,intermediate_size=16,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=512)
        config._attn_implementation='sdpa';model=Olmo2ForCausalLM(config).eval();e=Stage3Engine(t,tok,model)
        r=next(read_lines(root/'results/stage3_inputs_v1/prompts.jsonl.gz'));ann=annotate(tok,r)
        cap=checked_capture(e,t.tensor([ann['ids']]),[0],ann['test_positions'],True)
        records=anatomy(e,r,[0],cap,ann)
        self.assertEqual(len(records),len(ann['test_positions']));self.assertTrue(records[-1]['final'])
        ev=next(read_lines(root/'results/stage3_inputs_v1/events.jsonl.gz'));ev['layer']=0;ev['head']=0
        x=model.get_input_embeddings().weight[ev['current_id']:ev['current_id']+1]
        prob=t.from_numpy(e.projected_logits(0,x,True)).softmax(-1).numpy()[0]
        visible=sorted(set(r['token_ids'][:ev['j']+1]));ri=normalized_ri(prob[visible]);lookup=dict(zip(visible,ri))
        for s in ev['scores'].values():
            s['target']=float(lookup[s['target_id']])
            for kind in ['names','words']:
                for item in s[kind]:item['score']=float(lookup[item['token_id']])
        scored=matched_ri(e,r,cap,[ev]);self.assertEqual(len(scored),2)
        # At layer zero in OLMo2, the actual value input is the input embedding.
        for s in scored:self.assertAlmostEqual(s['scores']['raw']['target'],s['scores']['contextual']['target'],places=6)
        p=next(read_lines(root/'results/stage3_inputs_v1/pairs.jsonl.gz'));p['exact_all']=False
        spec=e.encode(p);base=e.readout(spec['clean'],spec['g'],spec['d']);corr=e.readout(spec['corr'],spec['g'],spec['d'])
        ref=dict(baselines=np.array([[base[0],corr[0]]]),scopeF_delta=np.array([[np.nan]]),scopeF_readout=np.full((1,1,3),np.nan))
        result,roles=causal_pair(e,p,[0],dict(heads=[0],reverse_heads=[0]),ref,0)
        self.assertEqual(result['position_readout'].shape,(1,0,3));self.assertEqual(result['reverse_readout'].shape,(1,2,3))
        self.assertEqual(np.shape(result['av_readout']),(1,4,3));self.assertFalse(result['scopeF_reused'][0])


if __name__=='__main__':unittest.main()
