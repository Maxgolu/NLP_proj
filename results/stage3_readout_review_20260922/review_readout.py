"""Read-only analysis of saved Stage-3 readouts, with outputs in this review folder."""
from pathlib import Path
import sys,json,gzip,hashlib,collections
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'results/stage3_readout_v1';INPUT=ROOT/'results/stage3_readout_inputs_v1'
sys.path.insert(0,str(ROOT/'pilot_v2'))
import stage3_readout as R
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((RUN/'manifest.json').read_text());plan=json.loads((INPUT/'readout_plan.json').read_text())
assert digest(INPUT/'readout_plan.json')==manifest['plan_hash']
assert plan['policy']==R.READOUT_POLICY
hash_matches={n:digest(ROOT/'pilot_v2'/n)==h for n,h in manifest['code_hashes'].items()}
with gzip.open(RUN/'readout_records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(l) for l in f]
key=lambda r:(r['pair_id'],r['writer'],r['site'],r['readout'],r['condition'])
expected={(i['pair_id'],i['writer'],i['site'],ro,c) for i in plan['items'] for ro in plan['policy']['readouts'] for c in [*plan['policy']['conditions'],'none']}
actual=collections.Counter(map(key,records))
assert set(actual)==expected and set(actual.values())=={1}
assert len(records)==1920
items={(i['pair_id'],i['writer'],i['site']):i for i in plan['items']}
for r in records:
    it=items[(r['pair_id'],r['writer'],r['site'])];n=it['names']
    assert r['candidate_names']==[n['query_mother'],n['other_candidate'],*n['other_mothers'],n['query_child']]
    assert (r['family'],r['order'],r['j'])==(it['family'],it['order'],it['j'])
    assert np.isfinite(r['candidate_logprobs']).all()

DEST=OUT/'reproduced';DEST.mkdir(exist_ok=True)
class Redirect:
    def __truediv__(self,p):return (DEST if p in ['analysis','summary.json'] else RUN)/p
before={p.name:digest(p) for p in (RUN/'analysis').glob('*.csv')}
R.analyze(Redirect())
comparison={p.name:p.read_bytes()==(DEST/'analysis'/p.name).read_bytes() for p in (RUN/'analysis').glob('*.csv')}
assert all(comparison.values())
assert before=={p.name:digest(p) for p in (RUN/'analysis').glob('*.csv')}

rows=[]
for r in records:
    lp=r['candidate_logprobs']
    rows.append({**{k:r[k] for k in ['pair_id','writer','site','family','order','readout','condition']},
                 'contrast':lp[0]-lp[1],'p_mother':float(np.exp(lp[0])),
                 'p_other':float(np.exp(lp[1])),'p_child':float(np.exp(lp[-1])),
                 'mother_top':float(np.argmax(lp)==0),'child_top':float(np.argmax(lp)==len(lp)-1),
                 'top_token':r['top_tokens'][0],'norm':r.get('head_site_vs_intact_norm')})
d=pd.DataFrame(rows)
group=['writer','site','readout'];index=group+['family','order']
wide=d.pivot(index=index,columns='condition',values='contrast')
paired=pd.DataFrame(index=wide.index)
for c in ['head_site','head_span','intact_corrupted','irrelevant','none']:
    paired[c+'_shift']=wide[c]-wide.intact_clean
paired['intact']=wide.intact_clean
fam=paired.groupby(group+['family']).mean()
fam.to_csv(OUT/'family_details.csv')
stats=[]
for k,g in fam.groupby(group):
    row=dict(zip(group,k))
    for col in fam.columns:
        v=g[col]
        row.update({col+'_mean':v.mean(),col+'_median':v.median(),col+'_mean_absolute':v.abs().mean(),col+'_min':v.min(),col+'_max':v.max(),col+'_negative_families':int((v<0).sum()),col+'_zero_families':int((v==0).sum())})
    site=g.head_site_shift;swap=g.intact_corrupted_shift
    row['head_swap_pearson']=site.corr(swap)
    row['site_minus_swap_mean_absolute']=(site-swap).abs().mean()
    stats.append(row)
stats=pd.DataFrame(stats);stats.to_csv(OUT/'descriptive_checks.csv',index=False)

probs=d.groupby(group+['condition','family'])[['p_mother','p_other','p_child','mother_top','child_top']].mean().groupby(group+['condition']).mean()
probs.to_csv(OUT/'absolute_probabilities.csv')
relevance=[]
for k,g in d.groupby(group):
    w=g.pivot(index=['family','order'],columns='condition',values='p_mother')
    f=w.groupby('family').mean()
    relevance.append(dict(zip(group,k))|{'p_intact':f.intact_clean.mean(),'p_none':f.none.mean(),'p_irrelevant':f.irrelevant.mean(),'p_head_site':f.head_site.mean(),'p_corrupted':f.intact_corrupted.mean(),
        'intact_minus_none':(f.intact_clean-f.none).mean(),'intact_minus_irrelevant':(f.intact_clean-f.irrelevant).mean()})
pd.DataFrame(relevance).to_csv(OUT/'injection_relevance.csv',index=False)
lookup={key(r):r for r in records}
span_equal=all(lookup[(*k[:-1],'head_site')]['candidate_logprobs']==lookup[(*k[:-1],'head_span')]['candidate_logprobs'] and lookup[(*k[:-1],'head_site')]['top_tokens']==lookup[(*k[:-1],'head_span')]['top_tokens'] and lookup[(*k[:-1],'head_site')]['top_logprobs']==lookup[(*k[:-1],'head_span')]['top_logprobs'] for k in lookup if k[-1]=='head_site')

site_df=pd.DataFrame(plan['items']).groupby(['writer','site','family']).saved_site_importance.mean()
site_df.to_csv(OUT/'site_importance_by_family.csv')
site_stats=site_df.groupby(['writer','site']).agg(['mean','min','max'])
print('SITE IMPORTANCE',site_stats.to_dict('index'))
print('DESCRIPTIVES',stats[group+['head_site_shift_mean','head_site_shift_median','head_site_shift_min','head_site_shift_max','head_site_shift_negative_families','intact_corrupted_shift_mean','head_swap_pearson']].to_dict('records'))
print('RELEVANCE',relevance)
print('TOP TOKENS',d.groupby(group+['condition']).top_token.agg(lambda v:collections.Counter(v).most_common(3)).to_string())
print('NORM',d[d.condition=='head_site'].groupby(['writer','site']).norm.agg(['mean','min','max']).to_string())
print('SPAN EQUAL',span_equal)
status={'records':len(records),'unique_expected_coverage':True,'plan_hash_matches':True,'code_hash_matches':hash_matches,'four_csvs_byte_identical':comparison,'head_site_head_span_all_saved_outputs_equal':span_equal,'original_csvs_unchanged':True}
(OUT/'verification.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
