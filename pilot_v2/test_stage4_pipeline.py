"""Orchestration tests without GPUs; failures must not unlock scientific stages."""
import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from stage3_common import json_write
from stage4_pipeline import Pipeline,gpu_groups

class PipelineTests(unittest.TestCase):
    def test_disjoint_gpu_assignments(self):
        self.assertEqual(gpu_groups('GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f',6),
                         [['GPU-a','GPU-b'],['GPU-c','GPU-d'],['GPU-e','GPU-f']])
        for text,num in [('0,0',2),('0,1',6),('0,1,2',3)]:
            with self.assertRaises(ValueError):gpu_groups(text,num)

    def scenario(self,folder,fail_gate=False,no_routes=False,localize=False):
        out=Path(folder);(out/'analysis').mkdir();events=[]
        args=argparse.Namespace(out=out,inputs=out/'inputs',gpus=6)
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'):p=Pipeline(args)
        def work(label,schedule,gate_only=False):
            events.append(label)
            if fail_gate and gate_only:raise ValueError('gate failed')
            return [out/label]
        def decision(label,mode,schedule,runs,prior=None):
            if no_routes and label in ['refine','extend_seed']:return None
            phase={'after_coverage':'localize_roles' if localize else 'seed',
                   'after_localize_roles':'localize_slots','after_localize_slots':'seed'}.get(label,'refine' if label=='refine' else 'extension')
            file=out/(label+'.json');json_write(file,dict(phase=phase));return file
        p.work=work;p.decision=decision
        with patch('stage4_pipeline.compare'):
            if fail_gate:
                with self.assertRaises(ValueError):p.execute()
            else:p.execute()
        return events

    def test_gate_failure_stops_before_coverage(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(self.scenario(d,fail_gate=True),['gate'])

    def test_full_adaptive_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.scenario(d,localize=True),['gate','coverage','localize_roles','localize_slots','seed','refine','extend_seed','extend_refine'])

    def test_no_retained_routes_is_not_a_computational_failure(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.scenario(d,no_routes=True),['gate','coverage','seed'])
            from stage3_common import json_read
            self.assertEqual(json_read(Path(d)/'pipeline_done.json')['status'],'complete_no_retained_routes')

if __name__=='__main__':unittest.main()
