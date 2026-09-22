"""CPU-only evidence and roster preparation for a proposed Stage-4 protocol."""
from pathlib import Path
import hashlib,json,gzip,collections
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def order(h):return tuple(map(int,h[1:].split('H')))
inv=read('results/stage3_plan/inventory_policy.json')
ri=sorted({f'L{x["layer"]}H{x["head"]}' for x in read('results/ri_test_v2/candidates.json')['candidates']},key=order)
historical=['L3H11','L9H22'];R=sorted(set(ri+historical),key=order)
h=pd.read_csv(ROOT/'results/stage3_v1/analysis/head_profiles.csv')
h['ri_test_only']=h['head'].isin(ri);h['ri_historical_extra']=h['head'].isin(historical)
h['ri_union']=h['head'].isin(R)
strong=h[h.stage2_P.abs()>=.3]
positive=strong[strong.stage2_P>0]['head'].tolist();negative=strong[strong.stage2_P<0]['head'].tolist()
rp=[x for x in positive if x in R];rn=[x for x in negative if x in R];low=[x for x in R if x not in rp+rn]
rosters=dict(ri_test_only=ri,ri_historical_extra=historical,ri_union=R,ri_strong_positive=rp,ri_strong_negative=rn,ri_other=low,
    strong_all=strong['head'].tolist(),strong_positive=positive,strong_negative=negative,
    writer_primary=['L17H1','L15H25'],writer_secondary=['L13H10','L14H26','L17H3'],
    mother_readers=['L18H18','L18H19','L19H22','L21H6','L21H18','L22H5','L27H6'],
    child_readers=['L16H1','L16H21'],same_layer_reference=['L17H24'],
    positive_group=['L27H6','L21H18','L22H5','L21H6'],negative_group=['L20H1','L19H16','L26H23'],
    mediation_sources=['L16H1','L16H21','L17H24','L18H18','L18H19','L19H16','L20H1','L23H15','L25H17'],
    backup_candidate=['L26H31'],early_minor=['L6H24','L7H1','L8H15','L9H16','L12H17'],
    original_random=inv['groups']['random-controls'])
assert len(ri)==59 and len(R)==61 and len(strong)==25 and len(rp)==5 and len(rn)==3 and len(low)==53
h.to_csv(OUT/'head_evidence.csv',index=False)
plan=read('results/stage3_readout_inputs_v1/readout_plan.json')
pd.DataFrame(plan['items']).groupby(['writer','site','family']).saved_site_importance.mean().groupby(['writer','site']).agg(['mean','min','max']).to_csv(OUT/'writer_site_evidence.csv')
ref=np.load(ROOT/'results/stage3_inputs_v1/references.npz');gap=ref['baselines'][:,0]-ref['baselines'][:,1]
allmeans=-ref['exact'][ref['common']].mean(0)
allheads=pd.DataFrame([dict(head=f'L{k//32}H{k%32}',common40_exact_P=float(allmeans[k]),in_stage3=k in set(ref['heads'])) for k in range(1024)])
allheads.sort_values('common40_exact_P',key=abs,ascending=False).to_csv(OUT/'stage2_all_head_priority.csv',index=False)
rosters['outside_inventory_reserve']=allheads[(~allheads.in_stage3)&(allheads.common40_exact_P.abs()>=.3)]['head'].tolist()
with gzip.open(ROOT/'results/stage3_inputs_v1/pairs.jsonl.gz','rt',encoding='utf-8') as f:pairs=[json.loads(s) for s in f]
# Role-localization of secondary writers from existing saved position annotations.
roles={}
for p in pairs:
    if not p['exact_all']:continue
    q=next(i for i,x in enumerate(p['clean']['facts']) if x['block']=='test' and x['head']==p['clean']['question_entity'])
    file=ROOT/f'results/stage3_v1/replica_0/pairs/family_{p["family"]:03d}_order{p["order"]}_chunk00.roles.json'
    for x in json.loads(file.read_text()):
        if x['final']:label='colon'
        elif x['question']:label='question'
        elif x['entities']:
            e=x['entities'][0];label=('query_' if e['query'] else 'other_fact_')+e['role']
        else:label=('query_' if q in x['fact_indices'] else 'other_fact_')+('punctuation' if x['punctuation'] else x['text'].strip())
        roles[(p['family'],p['order'],x['j'])]=label
pp=pd.read_csv(ROOT/'results/stage3_v1/analysis/position_profiles.csv')
pp=pp[pp['head'].isin(rosters['writer_secondary'])].copy()
pp['role']=[roles[(f,o,j)] for f,o,j in zip(pp.family,pp.order,pp.j)]
corepairs=[(p['family'],p['order']) for p in pairs if p['exact_all']]
site=[]
for (head,role),d in pp.groupby(['head','role']):
    per=d.groupby(['family','order']).importance.sum().reindex(pd.MultiIndex.from_tuples(corepairs),fill_value=0)
    fm=per.groupby(level=0).mean()
    site.append(dict(head=head,role=role,mean=fm.mean(),mean_absolute_family=fm.abs().mean(),families=20))
pd.DataFrame(site).sort_values(['head','mean'],key=lambda x:x.abs() if pd.api.types.is_numeric_dtype(x) else x,ascending=False).to_csv(OUT/'secondary_writer_sites.csv',index=False)
# Reference cohorts match the layer histogram; they are controls, not a significance cutoff.
rng=np.random.default_rng(20260923);forbidden=set(R+rosters['strong_all']);cohorts=[]
for _ in range(16):
    chosen=[]
    for layer,count in sorted(collections.Counter(order(x)[0] for x in low).items()):
        pool=[f'L{layer}H{i}' for i in range(32) if f'L{layer}H{i}' not in forbidden]
        chosen.extend(rng.choice(pool,count,replace=False).tolist())
    cohorts.append(sorted(chosen,key=order))
sources=['results/stage3_v1/analysis/head_profiles.csv','results/stage3_v1/analysis/position_profiles.csv','results/ri_test_v2/candidates.json','results/stage3_plan/inventory_policy.json','results/stage3_inputs_v1/references.npz','results/stage3_readout_v1/analysis/readout_family_summary.csv']
design=dict(status='PROPOSED protocol; CPU source inspection only; no Stage-4 model measurements',seed=20260923,
    model=read('pilot_v2/model_lock_olmo2.json'),source_hashes={p:sha(p) for p in sources},rosters=rosters,
    reference_cohorts_for_ri_other=cohorts,discovery_families=sorted({p['family'] for p in pairs}),
    core_families=sorted({p['family'] for p in pairs if p['exact_all']}),heldout_family_count=87,
    saved_gap_mean=float(gap.mean()),saved_common_gap_mean=float(gap[ref['common']].mean()),
    shared_prefix_pair_ids=[p['id'] for p in pairs if p['shared_prefix']],
    decisions=dict(primary_discovery='exact group and path interventions',edge_attribution='conditional expansion only',
        mean_effect_threshold_logits=.1,heterogeneous_family_threshold_logits=.1,discovery_family_sign_fraction=.7,
        faithful_gap_threshold=.8,upper_gap_guard=1.2,gap_L1_error_guard=.2,accuracy_drop_guard=.05,
        bootstrap_draws=20000,max_expansion_rounds=2,max_new_heads_per_round=12,max_new_mlps_per_round=4,
        retain_routes_for_full_discovery_cap=96,addback_head_cap=16,max_new_exact_configurations_per_expansion_round=256,
        attention_circuit_scope='all original test-block token positions; demonstrations, MLPs and answer-prefix computation live; explicitly conditional'))
(OUT/'manifest_proposal.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
print(json.dumps({'RI59':len(ri),'RI61':len(R),'strong_RI_positive':rp,'strong_RI_negative':rn,'other_RI':len(low),'all_strong':len(strong),'gap':float(gap.mean()),'cohorts':len(cohorts)},indent=2))
print(pd.DataFrame(site).query("head == 'L17H3'").sort_values('mean',key=abs,ascending=False).head(7).to_string(index=False))
