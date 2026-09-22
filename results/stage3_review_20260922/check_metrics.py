import sys,json,gzip
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
A=ROOT/'results/stage3_v1/analysis';INP=ROOT/'results/stage3_inputs_v1'
hp=pd.read_csv(A/'head_profiles.csv').set_index('head')
pp=pd.read_csv(A/'position_profiles.csv')
with gzip.open(INP/'pairs.jsonl.gz','rt',encoding='utf-8') as f:pairs=[json.loads(l) for l in f]
ref=np.load(INP/'references.npz');heads=ref['heads'] if 'heads' in ref.files else json.loads((INP/'plan.json').read_text())['heads']
names=[f'L{h//32}H{h%32}' for h in heads]
commonpairs=[(p['family'],p['order']) for p in pairs if p['exact_all']]
summed=pp.groupby(['head','family','order']).importance.sum()
rows=[]
for hi,h in enumerate(names):
    for pi,p in enumerate(pairs):
        if not p['exact_all']:continue
        exact=-float(ref['scopeP_delta'][pi,hi]);s=float(summed.loc[(h,p['family'],p['order'])])
        rows.append(dict(head=h,family=p['family'],order=p['order'],exact=exact,sum_positions=s,residual=s-exact))
add=pd.DataFrame(rows);add.to_csv(OUT/'additivity_per_pair.csv',index=False)
stats=add.groupby('head').agg(mean_residual=('residual','mean'),mean_absolute_residual=('residual',lambda x:x.abs().mean()),max_absolute_residual=('residual',lambda x:x.abs().max()))
stats.to_csv(OUT/'additivity_by_head.csv')
print('ADDITIVITY',stats.loc[['L17H1','L15H25','L18H19','L27H6','L23H15']].round(5).to_dict('index'))
print('WORST PAIRS',add.loc[add.residual.abs().nlargest(6).index].to_dict('records'))

# Independent annotation lookup from saved role sidecars, not forward-filled CSV roles.
lookup={}
for p in pairs:
    if not p['exact_all']:continue
    path=ROOT/f'results/stage3_v1/replica_0/pairs/family_{p["family"]:03d}_order{p["order"]}_chunk00.roles.json'
    roles=json.loads(path.read_text())
    query_index=next(i for i,f in enumerate(p['clean']['facts']) if f['block']=='test' and f['head']==p['clean']['question_entity'])
    for r in roles:
        key=(p['family'],p['order'],r['j']);ents=r['entities']
        role=None
        if ents and ents[0]['query']:
            e=ents[0];role=('mother' if e['role']=='target' else 'child')+' '+('first' if e['first'] else 'last' if e['last'] else 'middle')
        elif query_index in r['fact_indices']:
            role='period' if r['punctuation'] else r['text'].strip()
        lookup[key]=role
posout=[]
for h in ['L17H1','L15H25','L13H10']:
    d=pp[pp['head']==h].copy();d['role']=[lookup[(f,o,j)] for f,o,j in zip(d.family,d.order,d.j)]
    d=d.dropna(subset=['role'])
    groups=d.groupby(['family','order','role']).importance.sum()
    for role in sorted(d.role.unique()):
        v=groups.xs(role,level='role');full=v.reindex(pd.MultiIndex.from_tuples(commonpairs,names=['family','order']),fill_value=0)
        posout.append(dict(head=h,role=role,observed_pairs=len(v),conditional_report_mean=v.groupby('family').mean().mean(),all40_mean=full.groupby('family').mean().mean()))
pd.DataFrame(posout).to_csv(OUT/'position_role_denominators.csv',index=False)
print('L17 TOKEN DENOMINATORS',[r for r in posout if r['head']=='L17H1'])

# Reordering/corruption magnitudes: signed cancellation versus absolute changes.
pc=pd.read_csv(A/'paired_attention_controls.csv')
q=[]
for h,d in pc.groupby('head'):
    for role in ['target','source']:
        for cond in ['reorder','corruption']:
            x=d[f'{role}_{cond}_change'];fam=d.assign(v=x).groupby('family').v.mean()
            q.append(dict(head=h,role=role,condition=cond,signed_mean=x.mean(),mean_absolute_pair=x.abs().mean(),max_absolute_pair=x.abs().max(),mean_absolute_family=fam.abs().mean()))
q=pd.DataFrame(q);q.to_csv(OUT/'attention_change_magnitudes.csv',index=False)
print('REORDER STRONG',q[(q['head'].isin(hp.index[hp.stage3_F.abs()>=.3]))&(q.role=='target')&(q.condition=='reorder')].sort_values('mean_absolute_pair',ascending=False).head(8).round(4).to_dict('records'))

# Synthetic conditional/unconditional and colon-specific attended labels.
am=pd.read_csv(A/'attended_name_movers.csv').set_index('head');ae=pd.read_csv(A/'attended_name_events.csv')
final=[]
def positive_valid(d):
    return (d.contrast_first_all.dropna()>0).mean()
for h,d in ae.groupby('head'):
    b=d[(d.site=='final')&(d.variant=='base')]
    f=d[d.site=='final']
    final.append(dict(head=h,all_positive=positive_valid(d),base_final_events=len(b),base_final_positive=positive_valid(b),final_events=len(f),final_positive=positive_valid(f),all_answer_fraction=d.attended_is_answer.mean(),base_final_answer_fraction=b.attended_is_answer.mean(),final_answer_fraction=f.attended_is_answer.mean()))
final=pd.DataFrame(final);final.to_csv(OUT/'attended_site_comparison.csv',index=False)
print('ATTENDED SITE',final[final['head'].isin(['L26H31','L30H18','L23H15','L18H19','L27H6','L20H1'])].round(4).to_dict('records'))
print('STRONG RATIOS',hp[hp.stage2_P.abs()>=.3].assign(ratio=hp.stage3_F/hp.stage2_P)[['stage2_P','stage3_F','ratio']].round(4).to_dict('index'))

# Missing event-level diagnostic already saved before the GPU run.
cd=pd.DataFrame(json.loads((INP/'current_name_diagnostics.json').read_text()))
cd.to_csv(OUT/'current_name_diagnostics.csv',index=False)
print('OVERLAP DIAGNOSTIC',cd[(cd['head'].isin(['L18H19','L21H6','L17H24','L27H6','L19H16','L26H23','L30H18']))&(cd.anchor=='first')].round(6).to_dict('records'))
print('DONE',flush=True)
