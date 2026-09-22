"""Read source checkpoints; write all regenerated outputs only into this review folder."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pilot_v2'))
import numpy as np
import pandas as pd
import stage3_analyze

RUN=ROOT/'results/stage3_v1'
DEST=Path(__file__).resolve().parent/'reproduced'
DEST.mkdir(exist_ok=True)

class ReadOnlyRunView:
    def __truediv__(self,name):
        return (DEST if name in ('analysis','summary.json') else RUN)/name

before={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [RUN/'summary.json',*sorted((RUN/'analysis').glob('*'))] if p.is_file()}
print('Regenerating from source checkpoints with redirected outputs.',flush=True)
stage3_analyze.analyze(ReadOnlyRunView(),ROOT/'results/stage3_inputs_v1')
comparisons={}
keys={
 'attended_name_events.csv':['head','id','j'],
 'attended_name_movers.csv':['head'],
 'head_profiles.csv':['head'],
 'position_profiles.csv':['head','family','order','j'],
 'attention_profiles.csv':['head','variant','site','metric'],
 'output_profiles.csv':['head','variant','site','role','anchor','mode'],
 'synthetic_profiles.csv':['head','benchmark','condition','metric'],
 'synthetic_support.csv':['head','benchmark'],
 'copying_weights.csv':['head','population'],
 'contextual_ri.csv':['head','anchor','scope','site','metric'],
 'contextual_ri_correlations.csv':['anchor','scope','site','metric'],
 'paired_attention_controls.csv':['head','family','order'],
}
for name,key in keys.items():
    a=pd.read_csv(RUN/'analysis'/name).sort_values(key).reset_index(drop=True)
    b=pd.read_csv(DEST/'analysis'/name).sort_values(key).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-12,rtol=1e-12)
        comparisons[name]={'reproduced':True,'rows':len(a)}
    except AssertionError as e:
        comparisons[name]={'reproduced':False,'detail':str(e)[:2000]}
after={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in [RUN/'summary.json',*sorted((RUN/'analysis').glob('*'))] if p.is_file()}
assert before==after,'Original analysis changed!'
result={'source_outputs_unchanged':True,'table_comparisons':comparisons}
(DEST/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
