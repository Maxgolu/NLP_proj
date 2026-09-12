import collections
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from calibrate_stage1_null import calibrate, collect, holm_adjust


class NullTests(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm_adjust([.04,.01,.5]),[.08,.03,.5])

    def test_ties_unsupported_and_family_weight(self):
        totals={(0,0):np.array([40.,40.,100.]),(0,1):np.array([.4,.4,1.])}
        rows,_=calibrate(totals,2,1000,min_families=2,min_events=1)
        self.assertAlmostEqual(rows[0]['true_mean'],.4)
        self.assertEqual(rows[0]['p_randomization'],1.)
        self.assertEqual(rows[1]['status'],'insufficient_support')
        self.assertIsNone(rows[1]['p_randomization'])

    def test_family_swap_matches_enumerated_null(self):
        totals={(0,i):np.array([1.,0.,1.]) for i in range(3)}
        rows,_=calibrate(totals,1,50000,seed=10,min_families=1,min_events=1)
        self.assertAlmostEqual(rows[0]['p_randomization'],1/8,delta=.006)

    def test_duplicates_do_not_create_independent_families(self):
        totals={(0,i):np.array([.7,.2,1.]) for i in range(12)}
        duplicated={k:v*6 for k,v in totals.items()}
        a,_=calibrate(totals,1,1000,min_events=1)
        b,_=calibrate(duplicated,1,1000,min_events=1)
        self.assertEqual(a[0]['families'],b[0]['families'])
        self.assertEqual(a[0]['p_randomization'],b[0]['p_randomization'])
        self.assertAlmostEqual(a[0]['null_p99'],b[0]['null_p99'])

    def test_matched_events_exclude_invisible_collision_and_demo(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            p=dict(id='x',candidates=['A','B'],offsets=[[0,1],[2,3]],token_ids=[5,6],
                   facts=[dict(block='test',tail='A',tail_span=[0,1]),
                          dict(block='test',tail='B',tail_span=[2,3])])
            event=dict(id='x',family=0,layer=0,head=0,fact=0,j=1,scored=True,
                fact_block='test',position_block='test',target_first_id=5,strength_first=.6,
                entity_controls={'B':dict(score=.2,token_id=6,token_collision=False)})
            evs=[event,dict(event,j=0),dict(event,fact_block='demo'),
                 dict(event,entity_controls={'B':dict(score=.2,token_id=6,token_collision=True)})]
            for name,items in [('audit_prompts.jsonl.gz',[p]),('ri_events.jsonl.gz',evs)]:
                with gzip.open(run/name,'wt') as f:
                    for r in items:f.write(json.dumps(r)+'\n')
            totals,coverage=collect(run,1,1)
            np.testing.assert_allclose(totals[(0,0)],[.6,.2,1])
            self.assertEqual(coverage[0]['not_fully_visible'],1)
            self.assertEqual(coverage[0]['first_token_collision'],1)
            self.assertEqual(coverage[0]['outside_test_scope'],1)


if __name__=='__main__':unittest.main()
