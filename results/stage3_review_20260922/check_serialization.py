"""Inspect the sole strict table-comparison mismatch instead of treating it as exact."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parent
A=P.parent/'stage3_v1/analysis';B=P/'reproduced/analysis'
x=pd.read_csv(A/'attended_name_events.csv').sort_values(['head','id','j']).reset_index(drop=True)
y=pd.read_csv(B/'attended_name_events.csv').sort_values(['head','id','j']).reset_index(drop=True)
result={}
for c in x:
    if pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c]):
        diff=(x[c]-y[c]).abs()
        result[c]={'max_abs_difference':float(diff.max()),'missingness_equal':bool(x[c].isna().equals(y[c].isna()))}
        if c.startswith('contrast_'):
            result[c]['sign_mismatches']=int(((np.sign(x[c])!=np.sign(y[c]))&x[c].notna()).sum())
    else:result[c]={'exactly_equal':bool(x[c].equals(y[c]))}
assert all(r.get('exactly_equal',True) and r.get('missingness_equal',True) for r in result.values())
pd.testing.assert_frame_equal(x,y,check_exact=False,atol=3e-7,rtol=0)
with np.load(A/'family_effects.npz') as u,np.load(B/'family_effects.npz') as v:
    assert u.files==v.files
    maxdiff={k:float(np.abs(u[k]-v[k]).max()) for k in u.files}
    summary=[]
    for name in ['L27H6','L18H18','L18H19']:
        l,h=map(int,name[1:].split('H'));j=list(u['heads']).index(l*32+h);t=u['interaction'][:,j]
        summary.append(dict(head=name,signed_mean=float(t.mean()),mean_absolute_family=float(abs(t).mean()),max_absolute_family=float(abs(t).max())))
out={'event_columns':result,'event_table_matches_at_absolute_tolerance_3e_7':True,
     'family_arrays_max_abs_difference':maxdiff,'av_interactions':summary}
(P/'serialization_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'differing_columns':{c:r for c,r in result.items() if r.get('max_abs_difference',0)>1e-12},
    'family_arrays_max_abs_difference':maxdiff,'av_interactions':summary},indent=2))
