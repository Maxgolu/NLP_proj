import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from stage1_audit import first_divergence, DominanceStore
from analyze_stage1_audit import summarize


class Tokenizer:
    def encode(self, text, **kwargs):
        return {'Q': [1,2], 'Q Alice.': [1,2,7,8,10],
                'Q Alina.': [1,2,7,9,10], 'Q Bob.': [1,2,6,10]}[text]


class AuditTests(unittest.TestCase):
    def test_shared_prefix_and_reversed_gold(self):
        row = dict(id='x', prompt='Q', gold='Alice', candidates=['Alice','Alina'])
        s = first_divergence(Tokenizer(), row)
        self.assertEqual(s['input_ids'], [1,2,7])
        self.assertEqual((s['gold_token'],s['other_token']), (8,9))
        row['gold'] = 'Alina'
        t = first_divergence(Tokenizer(), row)
        self.assertEqual(s['input_ids'], t['input_ids'])
        self.assertEqual((t['gold_token'],t['other_token']), (9,8))

    def test_no_shared_prefix(self):
        s = first_divergence(Tokenizer(), dict(id='x', prompt='Q', gold='Alice',
                                              candidates=['Alice','Bob']))
        self.assertEqual(s['input_ids'], [1,2])

    def test_full_distribution_includes_late_layer_and_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            store = DominanceStore(d, 3)
            store.add(0, [1,2]); store.add(2, [100,200]); store.add(0, [3])
            r = store.finish()
            self.assertEqual(r['n'], 5)
            self.assertEqual(r['p95'], np.percentile([1,2,3,100,200],95))
            self.assertEqual(r['by_layer']['2']['n'],2)
            self.assertIsNone(r['by_layer']['1']['p95'])

    def test_repeated_twins_are_one_prefix_condition(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            def write(name, rows):
                with gzip.open(p/name, 'wt') as f:
                    for r in rows: f.write(json.dumps(r)+'\n')
            common = dict(family=0, variant='base', layer=9, fact_block='demo')
            write('audit_prompts.jsonl.gz', [dict(id=i, token_ids=[1,2], tokens=['a','b'],
                  facts=[dict(line='example',tail='B')]) for i in ('a','b')])
            write('ri_opportunities.jsonl.gz', [dict(common,id=i,heads=32,positions_by_block={'demo':2})
                                                for i in ('a','b')])
            write('ri_events.jsonl.gz', [dict(common,id=i,head=22,position_block='demo',j=1,
                  fact=0, scored=True,strength_first=.4,strength_last=.2,target_first_id=2,
                  target_last_id=2,entity_controls={}) for i in ('a','b')])
            result = summarize(p, heads=((9,22),))['L9H22']
            self.assertEqual(result['raw_scored_events'],2)
            self.assertEqual(result['unique_prefix_fact_conditions'],1)
            self.assertEqual(result['families_with_scored_events'],1)


if __name__ == '__main__': unittest.main()
