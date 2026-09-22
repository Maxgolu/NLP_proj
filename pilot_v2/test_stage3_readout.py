"""Implementation tests for the Section-1.5 readout: real tiny random OLMo2, no pretrained weights."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from stage3_common import *
import stage3_readout as R


class PlanTests(unittest.TestCase):
    def test_prepare_on_saved_tables(self):
        root=Path(__file__).resolve().parents[1]
        inputs=root/'results/stage3_inputs_v1';run=root/'results/stage3_v1'
        if not (inputs/'pairs.jsonl.gz').exists() or not (run/'analysis/position_profiles.csv').exists():
            self.skipTest('Saved Stage-3 tables not beside the source tree')
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'plan';R.prepare(inputs,run,out);plan=json_read(out/'readout_plan.json')
            self.assertEqual(len(plan['items']),160);self.assertEqual(plan['policy'],R.READOUT_POLICY)
            for it in plan['items']:
                self.assertGreater(it['j'],it['first_diff']-1);self.assertNotEqual(it['j'],it['control_j'])
                names=it['names'];self.assertNotIn(names['query_child'],names['other_mothers'])
                self.assertNotEqual(names['query_mother'],names['other_candidate'])
            main=[it for it in plan['items'] if it['writer']=='L17H1' and it['site']=='query_is']
            # The chosen site is where L17H1's single-position effect is large in most families.
            self.assertGreaterEqual(sum(it['saved_site_importance']>0.5 for it in main),30)
            with self.assertRaises(FileExistsError):R.prepare(inputs,run,out)


class TinyOlmoReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('Install CPU torch and transformers') from exc
        from stage3_engine import Stage3Engine
        torch.manual_seed(11);torch.set_num_threads(2)
        config=Olmo2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,
                           num_key_value_heads=4,max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.)
        config._attn_implementation='sdpa'
        class Tok:
            """Deterministic stand-in: every 'word' is one id; the placeholder is id 7."""
            def encode(self,text,add_special_tokens=False):
                return [min(63,3+sum(map(ord,w))%60) if w!='x' else 7 for w in text.replace('\n',' \\n ').split()]
            def decode(self,ids):return ' '.join(f't{i}' for i in ids)
        cls.t=torch;cls.model=Olmo2ForCausalLM(config).eval();cls.e=Stage3Engine(torch,Tok(),cls.model)

    def test_identity_injection_and_change(self):
        t=self.t;e=self.e
        # Placeholder must be one token and readout prompts compositional under this tokenizer.
        ro=R.Readout(e)
        for name,p in ro.prompts.items():
            for layer in [0,1]:  # an injection at the LAST block's output cannot reach the final logits
                own=ro.hidden(p['ids'],layer,[p['pos']])[0]
                base=ro.inject(name,layer,None,[3,4,5]);back=ro.inject(name,layer,own,[3,4,5])
                np.testing.assert_allclose(base['candidates'],back['candidates'],atol=1e-5)
                self.assertEqual(base['top_tokens'],back['top_tokens'])
                other=ro.inject(name,layer,own+1.0,[3,4,5])
                self.assertGreater(max(abs(a-b) for a,b in zip(base['candidates'],other['candidates'])),1e-4)
        # Hooks are removed after use.
        self.assertEqual(len(e.layers[1]._forward_hooks),0)

    def test_hidden_under_head_patch_matches_manual_patch(self):
        t=self.t;e=self.e;ids=t.tensor([[1,8,4,6,9,7,12]]);n=7;head=5;j=4;layer=1
        cap=e.capture(ids,[head],[j]);z=cap['heads'][head]['z']
        ro=R.Readout(e)
        intact=ro.hidden(ids,layer,[j])[0];selfp=ro.hidden(ids,layer,[j],e.vector_patch(head,[j],z))[0]
        np.testing.assert_allclose(intact.numpy(),selfp.numpy(),atol=1e-5)
        changed=ro.hidden(ids,layer,[j],e.vector_patch(head,[j],z+0.5))[0]
        self.assertGreater(float((intact-changed).abs().max()),1e-4)
        # A patch at position j must not change the residual at an earlier position.
        earlier=ro.hidden(ids,layer,[2],e.vector_patch(head,[j],z+0.5))[0]
        np.testing.assert_allclose(ro.hidden(ids,layer,[2])[0].numpy(),earlier.numpy(),atol=1e-6)


class AnalysisTests(unittest.TestCase):
    def test_analyze_synthetic_records(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'run';out.mkdir();plan_dir=Path(folder)/'plan';plan_dir.mkdir()
            items=[];recs=[]
            for fam in [0,16]:
                for order in [0,1]:
                    pid=f'{fam:03d}/direct/base/0/order{order}'
                    for w,spec in R.READOUT_POLICY['writers'].items():
                        for site in spec['sites']:
                            items.append(dict(pair_id=pid,family=fam,order=order,writer=w,head=spec['head'],layer=spec['layer'],site=site,j=5,control_j=9,
                                              first_diff=3,names=dict(query_mother='A',other_candidate='B',other_mothers=['C'],query_child='D'),saved_site_importance=1.))
                            for cond in R.READOUT_POLICY['conditions']+['none']:
                                for ro in R.READOUT_POLICY['readouts']:
                                    lp=[-1.,-2.,-3.,-4.] if cond in ('intact_clean','irrelevant','none') else [-2.,-1.,-3.,-4.]
                                    recs.append(dict(pair_id=pid,family=fam,order=order,writer=w,site=site,j=5,condition=cond,readout=ro,
                                                     candidate_names=['A','B','C','D'],candidate_logprobs=lp,top_tokens=[' A',' B',' C',' D',' E'],top_logprobs=[-1,-2,-3,-4,-5]))
            json_write(plan_dir/'readout_plan.json',dict(policy=R.READOUT_POLICY,items=items,model={}))
            json_write(out/'manifest.json',dict(plan=str(plan_dir/'readout_plan.json')));json_write(out/'gate.json',dict(passed=True))
            write_lines(out/'readout_records.jsonl.gz',recs);json_write(out/'done.json',dict(complete=True))
            R.analyze(out)
            import csv
            with (out/'analysis/readout_family_summary.csv').open() as f:agg=list(csv.DictReader(f))
            self.assertEqual(len(agg),8)
            for a in agg:
                self.assertAlmostEqual(float(a['entity_swap_shift']),-2.);self.assertAlmostEqual(float(a['head_site_shift']),-2.)
                self.assertAlmostEqual(float(a['irrelevant_shift']),0.);self.assertEqual(float(a['head_span_over_swap']),1.)
            # A missing record is not accepted as completion.
            write_lines(out/'readout_records.jsonl.gz',recs[:-1])
            with self.assertRaisesRegex(ValueError,'Expected'):R.analyze(out)


if __name__=='__main__':unittest.main()


class DryRunTests(unittest.TestCase):
    """End-to-end worker on the tiny model: gate, measurements, resume and analysis, without GPUs."""
    def test_worker_end_to_end(self):
        try:
            import torch
            from transformers import Olmo2Config,Olmo2ForCausalLM
        except ImportError as exc:raise unittest.SkipTest('Install CPU torch and transformers') from exc
        from unittest.mock import patch
        import stage1_scan
        from stage3_engine import Stage3Engine
        torch.manual_seed(3)
        config=Olmo2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,num_attention_heads=4,
                           num_key_value_heads=4,max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.)
        config._attn_implementation='sdpa';model=Olmo2ForCausalLM(config).eval()
        vocab={}
        class Tok:
            def encode(self,text,add_special_tokens=False):
                out=[]
                for w in text.replace('\n',' \\n ').replace('.',' .').replace('?',' ?').replace(':',' :').split():
                    if w=='x':out.append(7);continue
                    vocab.setdefault(w,10+len(vocab)%50);out.append(vocab[w])
                return out
            def decode(self,ids):return ' '.join(f't{i}' for i in ids)
        tok=Tok()
        def row(mothers,children,q,gold,other):
            lines=[f'{m} is the mother of {c} .' for m,c in zip(mothers,children)]
            prompt='Demo line here .\n'+'\n'.join(lines)+f'\nQuestion : Who is the mother of {q} ?\nAnswer :'
            facts=[dict(block='demo',head='d',tail='e')]+[dict(block='test',head=c,tail=m) for m,c in zip(mothers,children)]
            return dict(prompt=prompt,gold=gold,candidates=[other,gold],question_entity=q,facts=facts)
        pairs=[]
        for fam in [0,16]:
            for order in [0,1]:
                m=['Ama','Bea','Cla','Dor'];c=['Kid','Ama','Lou','Cla']   # chain: Ama is child of Bea and mother of Kid
                q='Ama';gold='Bea';other='Dor'
                clean=row(m,c,q,gold,other);corr=row(['Ama','Dor','Cla','Bea'],c,q,other,gold)
                pairs.append(dict(id=f'{fam:03d}/direct/base/0/order{order}',family=fam,order=order,exact_all=True,clean=clean,corr=corr))
        with patch.object(stage1_scan,'load_model',lambda:(torch,tok,model,R.READOUT_POLICY and {'m':1})), tempfile.TemporaryDirectory() as folder:
            e=Stage3Engine(torch,tok,model);folder=Path(folder);plan_dir=folder/'plan';plan_dir.mkdir()
            items=[]
            for p in pairs:
                spec=e.encode(p);ids=spec['clean'][0].tolist()
                # positions: tokens of clean prompt; locate 'is' after 'Bea' and last token of child 'Ama' of the query fact
                words=tok.encode(p['clean']['prompt']);bea=vocab['Bea'];is_=vocab['is'];ama=vocab['Ama'];dot=vocab['.']
                j_is=[i for i in range(1,len(words)) if words[i]==is_ and words[i-1]==bea][0]
                j_child=[i for i in range(len(words)) if words[i]==ama and words[i-1]==vocab['of']][0]
                j_dot=[i for i in range(len(words)) if words[i]==dot and i>j_child][0]
                first=next(i for i in range(len(ids)) if spec['clean'][0,i]!=spec['corr'][0,i])
                sites={'query_is':j_is,'query_period':j_dot,'query_child_last':j_child}
                ctrl={'query_is':j_is+12,'query_period':j_dot+12,'query_child_last':j_child+12}
                for w,sp in R.READOUT_POLICY['writers'].items():
                    layer=min(sp['layer'],2);head=sp['head']%16  # tiny model: 4 layers x 4 heads
                    for site in sp['sites']:
                        j=sites[site];cap=e.capture(spec['corr'][:,:spec['n']],[head],[j]);z1=cap['heads'][head]['z']
                        imp=float(e.readout(spec['clean'],spec['g'],spec['d'])[0]-e.patched(spec,head,[j],z1)[0])
                        items.append(dict(pair_id=p['id'],family=p['family'],order=p['order'],writer=w,head=head,layer=layer,site=site,j=j,control_j=ctrl[site],
                                          first_diff=first,names=dict(query_mother='Bea',other_candidate='Dor',other_mothers=['Cla'],query_child='Ama'),saved_site_importance=imp))
            json_write(plan_dir/'readout_plan.json',dict(policy=R.READOUT_POLICY,items=items,model={'m':1}))
            write_lines(plan_dir/'pairs.jsonl.gz',pairs)
            out=folder/'run';out.mkdir();json_write(out/'manifest.json',dict(plan=str(plan_dir/'readout_plan.json'),gate_only=False))
            R.run_worker(out)
            self.assertTrue(json_read(out/'gate.json')['passed']);self.assertTrue(json_read(out/'done.json')['complete'])
            recs=list(read_lines(out/'readout_records.jsonl.gz'))
            self.assertEqual(len(recs),len(items)*(len(R.READOUT_POLICY['conditions'])+1)*len(R.READOUT_POLICY['readouts']))
            # Resume: a second call adds nothing.
            R.run_worker(out);self.assertEqual(len(list(read_lines(out/'readout_records.jsonl.gz'))),len(recs))
            R.analyze(out)
            self.assertTrue((out/'analysis/readout_family_summary.csv').exists())
