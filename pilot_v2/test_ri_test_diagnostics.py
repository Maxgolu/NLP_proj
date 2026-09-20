import tempfile
from pathlib import Path
import unittest
import numpy as np

from ri_test_audit import write_json, read_json
from ri_test_diagnostics import array_stats, capture_failure, probe_weights


class DiagnosticTests(unittest.TestCase):
    def test_probe_reads_only_bounded_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'one.safetensors').write_bytes(b'x'*200)
            (root/'two.safetensors').write_bytes(b'y'*20)
            report={}
            probe_weights(root,report,lambda r:write_json(root/'probe.json',r),bytes_per_shard=64)
            self.assertEqual([r['bytes_read'] for r in report['read_probe']['shards']],[64,20])
            self.assertEqual((root/'one.safetensors').stat().st_size,200)
            self.assertEqual(read_json(root/'probe.json'),report)

    def test_failure_saved_even_if_replay_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            event=dict(layer=0,head=2,current_id=627,event_id=17,
                       j=0,strength_first=.2,strength_last=.3)
            capture_failure(None,None,out,event,dict(prompt='x',offsets=[[0,1]]),
                            [627],[1,2],np.zeros(2,dtype=np.float32),'zero denominator',write_json)
            r=read_json(out/'gate_failure.json')
            self.assertEqual(r['event']['event_id'],17)
            self.assertEqual(r['denominator'],0)
            self.assertIn('replay_error',r)
            with np.load(out/'gate_failure_arrays.npz') as a:
                np.testing.assert_array_equal(a['cached_probabilities'],[0,0])
                self.assertAlmostEqual(float(a['historical_first']),.2)

    def test_nonfinite_stats_json_safe(self):
        s=array_stats(np.array([np.nan,np.inf,0,2]))
        self.assertEqual(s['nonfinite'],2)
        self.assertEqual(s['zeros'],1)
        self.assertEqual(s['maximum'],2)
        self.assertIsNone(array_stats(np.array([np.nan]))['minimum'])


if __name__=='__main__':
    unittest.main()
