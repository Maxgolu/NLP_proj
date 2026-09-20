"""CPU tests for scientific scope, controls, aggregation, pairing and resume gates."""
import collections
import csv
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import ri_test_audit as audit


def prompt(family=0, variant='base', order=0):
    edges = [('Alpha','Beta'),('Beta','Gamma'),('Delta','Epsilon'),('Epsilon','Zeta')]
    if variant == 'corrupted':
        edges[1] = ('Beta','Zeta')
        edges[3] = ('Epsilon','Gamma')
    if order:
        edges = edges[2:] + edges[:2]
    if variant == 'reorder':
        edges = list(reversed(edges))
    text = 'Example.\n\n'
    facts = []
    for source,target in edges:
        line = f'{target} is the mother of {source}.'
        start = len(text)
        si,oi = line.index(source),line.index(target)
        facts.append(dict(block='test', head=source, tail=target,
                          head_span=[start+si,start+si+len(source)],
                          tail_span=[start+oi,start+oi+len(target)]))
        text += line+'\n'
    text += 'Question: Who is the mother of Beta?\nAnswer:'
    ms = list(re.finditer(r'\w+|[^\w\s]', text))
    vocabulary = {t:i for i,t in enumerate(sorted({m.group() for m in ms}))}
    return audit.enrich_prompt(dict(id=f'{family:03d}/direct/{variant}/0/order{order}',
         family=family, variant=variant, gold='Zeta' if variant=='corrupted' else 'Gamma',
         prompt=text, facts=facts, offsets=[[m.start(),m.end()] for m in ms],
         token_ids=[vocabulary[m.group()] for m in ms]))


class AuditTests(unittest.TestCase):
    def test_query_fact_tracks_corruption_and_reorder(self):
        for v in ('base','corrupted','reorder'):
            for order in (0,1):
                p = prompt(variant=v, order=order)
                q = next(f for f in p['test_facts'] if f['query'])
                self.assertEqual(q['source'],'Beta')
                self.assertEqual(q['target'],p['gold'])

    def test_controls_visible_distinct_and_reproducible(self):
        p = prompt()
        f = p['test_facts'][1]
        early = f['s']
        names, words = audit.controls(p,f,early)
        self.assertTrue(all(c['last'] <= early for c in names+words))
        self.assertNotIn('Gamma',[c['text'] for c in names])
        end = len(p['token_ids'])-1
        a,b = audit.controls(p,f,end)
        self.assertEqual(len(a),3)
        self.assertEqual(len(b),3)
        self.assertEqual((a,b),audit.controls(p,f,end))
        self.assertEqual(b,audit.controls(p,p['test_facts'][3],end)[1])
        self.assertFalse({w['text'] for w in b} & {'Alpha','Beta','Gamma','Delta','Epsilon','Zeta'})

    def test_full_name_visibility_required(self):
        p = prompt()
        p['test_facts'][0]['ol'] = p['test_facts'][0]['of']+2
        f = p['test_facts'][1]
        names,_ = audit.controls(p,f,p['test_facts'][0]['of'])
        self.assertNotIn(p['test_facts'][0]['target'],[c['text'] for c in names])

    def test_denominators_include_zero_pass_families(self):
        ps = {p['id']:p for p in (prompt(0),prompt(2))}
        den = audit.denominators(ps)
        self.assertEqual(den[(('query_fact','final','all','pooled'),0)],1)
        self.assertEqual(den[(('all_facts','final','all','pooled'),2)],4)
        self.assertGreater(den[(('all_facts','all_test','all','pooled'),0)],4)

    def test_ties_are_not_wins(self):
        s = audit.compare_scores(.2,[dict(score=.2,collision=True),dict(score=.1,collision=False)])
        self.assertEqual((s['rank_min'],s['rank_max']),(1,2))
        self.assertEqual(s['wins'],.5)
        self.assertEqual(s['strict_top'],0)
        self.assertEqual(s['collisions'],1)
        self.assertIsNone(audit.compare_scores(.2,[])['gap'])

    def test_normalization_and_degenerate_denominator(self):
        q = audit.normalized_scores(np.array([.1,.3,.6],dtype=np.float32))
        np.testing.assert_allclose(q,[0,0,1])
        self.assertIsNone(audit.normalized_scores(np.array([.2,.2])))

    def test_ov_slice_matches_full_linear_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional CPU torch not installed')
        torch.manual_seed(17)
        E=torch.randn(13,4,dtype=torch.float16)
        U=torch.randn(13,4,dtype=torch.float16)
        V=torch.randn(4,4,dtype=torch.float16)
        O=torch.randn(4,4,dtype=torch.float16)
        attn=SimpleNamespace(v_proj=SimpleNamespace(weight=V),o_proj=SimpleNamespace(weight=O))
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=4,num_attention_heads=2),
                              model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
                              get_input_embeddings=lambda:SimpleNamespace(weight=E),
                              lm_head=SimpleNamespace(weight=U))
        for hi in (0,1):
            actual=audit.ov_probabilities(torch,model,0,hi,[2,5,8],[0,3,9],batch=2)
            values=torch.nn.functional.linear(E[[2,5,8]],V)
            masked=torch.zeros_like(values)
            masked[:,hi*2:(hi+1)*2]=values[:,hi*2:(hi+1)*2]
            z=torch.nn.functional.linear(masked,O)
            expected=torch.nn.functional.linear(z,U).float().softmax(-1)[:,[0,3,9]].numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-3)

    def test_lock_blocks_parallel_run_and_releases(self):
        with tempfile.TemporaryDirectory() as d:
            with audit.run_lock(Path(d)):
                with self.assertRaises(RuntimeError):
                    with audit.run_lock(Path(d)):
                        pass
            self.assertFalse((Path(d)/'.running.lock').exists())

    def test_manifest_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            audit.write_lines(out/'events.jsonl.gz',[])
            m = dict(version=audit.VERSION,policy=audit.POLICY,code=audit.code_hashes(),
                     prepared={'events.jsonl.gz':audit.digest(out/'events.jsonl.gz')})
            audit.write_json(out/'manifest.json',m)
            audit.check_manifest(out)
            audit.write_lines(out/'events.jsonl.gz',[dict(changed=True)])
            with self.assertRaises(ValueError):
                audit.check_manifest(out)

    def test_end_to_end_family_weighting_zero_heads_and_pairing(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            ps = [prompt(f,v,o) for f in (0,2) for v in ('base','corrupted','reorder') for o in (0,1)]
            # Family 0: two .8 events; family 2: one .2 event => equal-family mean .5, not .6.
            events=[]
            for pi,value in ((0,.8),(2,.8),(6,.2)):
                p=ps[pi]; f=next(x for x in p['test_facts'] if x['query'])
                score=dict(target=value,target_id=p['token_ids'][f['of']], names=[], words=[])
                for kind in ('names','words'):
                    score[kind+'_stats']=audit.compare_scores(value,[dict(score=.1,collision=False)])
                events.append(dict(event_id=len(events),layer=0,head=0,id=p['id'],family=p['family'],
                                   source=f['source'],target=f['target'],order=p['order'],variant=p['variant'],
                                   query_fact=True,position='final',distance='long',
                                   scores={'first':score,'last':score}))
            audit.write_lines(out/'prompts.jsonl.gz',ps)
            audit.write_lines(out/'events.jsonl.gz',events)
            m=dict(version=audit.VERSION,policy=audit.POLICY,code=audit.code_hashes(),
                   families=[0,2],events=3,heads=[[0,0],[0,1]],
                   prepared={n:audit.digest(out/n) for n in ('prompts.jsonl.gz','events.jsonl.gz')})
            audit.write_json(out/'manifest.json',m)
            (out/'heads').mkdir()
            data=out/'heads/L0H0.jsonl.gz'
            audit.write_lines(data,events)
            audit.write_json(out/'heads/L0H0.json',dict(manifest=audit.digest(out/'manifest.json'),
                             sha256=audit.digest(data)))
            audit.analyze(out)
            with (out/'head_summary.csv').open() as f:
                rows=list(csv.DictReader(f))
            filt=lambda r:r['scope']=='query_fact' and r['position']=='final' and r['distance']=='all' and r['variant']=='pooled' and r['anchor']=='first'
            a=next(r for r in rows if r['head']=='0' and filt(r))
            b=next(r for r in rows if r['head']=='1' and filt(r))
            self.assertAlmostEqual(float(a['target_mean']),.5)
            self.assertAlmostEqual(float(a['names_gap_mean']),.4)
            self.assertEqual(int(a['target_events']),3)
            self.assertEqual(int(a['target_families']),2)
            self.assertAlmostEqual(float(a['frequency']),.25)
            self.assertEqual(float(b['frequency']),0)
            self.assertEqual(b['target_mean'],'')
            with (out/'paired_variants.csv').open() as f:
                pairs=list(csv.DictReader(f))
            pair=next(r for r in pairs if r['family']=='0' and r['order']=='0' and r['comparison']=='corrupted')
            self.assertEqual(pair['base_target_name'],'Gamma')
            self.assertEqual(pair['twin_target_name'],'Zeta')
            self.assertEqual(pair['both_pass'],'True')
            self.assertTrue(audit.read_json(out/'summary.json')['complete'])
            self.assertEqual(audit.read_json(out/'candidates.json')['candidates'],[])


if __name__ == '__main__':
    unittest.main()
