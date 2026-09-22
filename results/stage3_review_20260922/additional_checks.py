"""Independent checks from exported numeric tables; does not alter original data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
R=O.parent/'stage3_v1'
A=R/'analysis'
hp=pd.read_csv(A/'head_profiles.csv').set_index('head')
add=pd.read_csv(O/'additivity_per_pair.csv')
af=add.groupby(['head','family']).residual.mean().groupby('head').agg(
    signed_mean='mean',mean_absolute_family=lambda v:v.abs().mean(),
    max_absolute_family=lambda v:v.abs().max())
af.to_csv(O/'additivity_by_family.csv')
print('L17H1 family residuals:',af.loc['L17H1'].to_dict())

events=pd.read_csv(A/'attended_name_events.csv')
rows=[]
for h,d in events.groupby('head'):
    for site,part in [('all',d),('final',d[d.site=='final']),('base_final',d[(d.site=='final')&(d.variant=='base')])]:
        valid=part.dropna(subset=['contrast_first_all'])
        rows.append(dict(head=h,site=site,raw_events=len(part),valid_events=len(valid),
            families=valid.family.nunique(),positive_fraction=(valid.contrast_first_all>0).mean(),
            answer_fraction_all=part.attended_is_answer.mean(),
            mean_contrast=valid.contrast_first_all.mean()))
df=pd.DataFrame(rows);df.to_csv(O/'attended_site_comparison_valid.csv',index=False)
print('Attended sites:',df[df['head'].isin(['L26H31','L30H18','L23H15','L18H19'])].round(5).to_dict('records'))

fin=pd.read_csv(A/'attention_profiles.csv')
fin=fin[(fin.site=='final')&(fin.variant=='base')].pivot(index='head',columns='metric',values='mean')
sel=fin.loc[['L25H17','L21H23','L30H13','L17H17'],['self','relevant_target_mass','relevant_source_mass','distractor_target_mass','distractor_source_mass']].copy()
sel['all_fact_name_mass']=sel.relevant_target_mass+sel.relevant_source_mass+3*(sel.distractor_target_mass+sel.distractor_source_mass)
sel.to_csv(O/'self_attender_fact_mass.csv')
print('Self/fact mass:',sel.round(5).to_dict('index'))

with np.load(A/'family_effects.npz') as z:
    error=z['scopeF']-z['promotion']-z['suppression']
    av_error=z['scopeF']-z['routing']-z['values']-z['interaction']
    print('Max F decomposition residual:',float(abs(error).max()))
    print('Max family F versus AV factorial residual:',float(abs(av_error).max()))
    out=[]
    for j,h in enumerate(z['heads']):
        out.append(dict(head=f'L{h//32}H{h%32}',
            routing_signed=z['routing'][:,j].mean(),routing_mean_absolute=np.abs(z['routing'][:,j]).mean(),
            values_mean_absolute=np.abs(z['values'][:,j]).mean(),
            interaction_mean_absolute=np.abs(z['interaction'][:,j]).mean()))
    pd.DataFrame(out).to_csv(O/'av_magnitudes_by_head.csv',index=False)

labels=pd.read_csv(A/'attended_name_movers.csv')
print('Primary labels:',labels[labels.label.isin(['name_mover','negative_name_mover'])][['head','events','families','positive_fraction','label','label_final_site_only','label_last_anchor']].to_dict('records'))
print('Synthetic:',pd.read_csv(A/'synthetic_profiles.csv').query("head == 'L23H15' and condition == 'all_probes' and metric in ['source_argmax','source_attention']")[['benchmark','metric','mean']].to_dict('records'))
print('Done')
