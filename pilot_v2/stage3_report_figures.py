"""Stage-3 report figures and numbers (CPU only, reads the saved analysis tables)."""
import json, collections
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

A=Path('/mnt/user-data/uploads/project/results/stage3_v1/analysis')
OUT=Path('/home/claude/report/figs'); OUT.mkdir(parents=True,exist_ok=True)
PROV=Path('/home/claude/run/stage3_v1/analysis/attended_name_movers.csv')
C=dict(blue='#2a78d6',orange='#eb6834',aqua='#1baf7a',yellow='#eda100',magenta='#e87ba4',green='#008300',violet='#4a3aa7',red='#e34948',
       ink='#0b0b0b',ink2='#52514e',muted='#898781',grid='#e1e0d9',axis='#c3c2b7')
GROUPC={'strong-positive_outside-RI':C['blue'],'strong-positive_RI-selected':C['aqua'],'strong-negative':C['red'],
        'moderate-effect_supplement':C['yellow'],'RI-selected_small-effect':C['violet'],'random-controls':C['muted']}
GROUPL={'strong-positive_outside-RI':'strong +, outside RI','strong-positive_RI-selected':'strong +, RI-selected','strong-negative':'strong −',
        'moderate-effect_supplement':'moderate','RI-selected_small-effect':'RI-selected, small','random-controls':'random controls'}
plt.rcParams.update({'font.size':9,'axes.edgecolor':C['axis'],'axes.labelcolor':C['ink2'],'xtick.color':C['ink2'],'ytick.color':C['ink2'],
                     'axes.grid':True,'grid.color':C['grid'],'grid.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
def symlog(ax,which='both'):
    for a in (['x','y'] if which=='both' else [which]):
        getattr(ax,f'set_{a}scale')('symlog',linthresh=0.05)
hp=pd.read_csv(A/'head_profiles.csv'); hp['layer']=hp['head'].str.extract(r'L(\d+)').astype(int); hp['color']=hp.groups.map(GROUPC)
H=hp.set_index('head')
N={}  # numbers for the text

# ---------- Fig 1: Scope P vs Scope F ----------
fig,ax=plt.subplots(figsize=(6.2,5))
for g,d in hp.groupby('groups'):
    ax.scatter(d.stage2_P,d.stage3_F,s=22,color=GROUPC[g],label=GROUPL[g],alpha=.9,edgecolor='white',linewidth=.5)
lim=[-2,6.5]; ax.plot(lim,lim,color=C['axis'],lw=1,ls='--'); ax.axhline(0,color=C['axis'],lw=.8); ax.axvline(0,color=C['axis'],lw=.8)
symlog(ax); ax.set_xlim(-2,6.5); ax.set_ylim(-2,6.5)
for h in ['L17H1','L18H19','L27H6','L18H18','L21H18','L25H17','L15H25','L14H26','L13H10','L20H1','L30H18','L23H15','L19H16','L8H15','L16H21','L16H1']:
    r=H.loc[h]; ax.annotate(h,(r.stage2_P,r.stage3_F),fontsize=7,color=C['ink2'],xytext=(4,3),textcoords='offset points')
ax.set_xlabel('Scope P importance $I_h$ (all prompt positions, 178 pairs)'); ax.set_ylabel('Scope F importance $I_h^F$ (final colon only, 178 pairs)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax.set_title('Where does the effect enter? Scope P vs Scope F (symlog axes)',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig1_scopeP_vs_scopeF.png'); plt.close(fig)
hp['ratio']=hp.stage3_F/hp.stage2_P
strong=hp[hp.stage2_P.abs()>=0.3]
N['ratio_table']=strong.sort_values('stage2_P',key=abs,ascending=False)[['head','layer','groups','stage2_P','stage3_F','ratio','F_positive_family_fraction','F_promotion','F_suppression']].round(3).to_dict('records')

# ---------- Fig 2: position profiles ----------
pp=pd.read_csv(A/'position_profiles.csv')
def role(r):
    ents=json.loads(r['entities'])
    if r['final']: return 'final colon'
    if ents:
        e=ents[0]; q='query fact' if e['query'] else 'distractor'
        role='mother' if e['role']=='target' else 'child'
        return f'{q} {role}'
    if r['question']: return 'question tokens'
    if r['punctuation']: return 'punctuation'
    return 'template word'
pp['role']=pp.apply(role,axis=1)
# assign template words / punctuation to the fact they belong to (the sentence of the last entity token seen)
pp=pp.sort_values(['head','family','order','j']).reset_index(drop=True)
cur=None;last=None;fine=[]
for r in pp.itertuples():
    key=(r.head,r.family,r.order)
    if key!=last: cur='distractor'; last=key
    ents=json.loads(r.entities)
    if ents: cur='query fact' if ents[0]['query'] else 'distractor'
    if r.role in ('template word','punctuation'): fine.append(f"{cur} {'template' if r.role=='template word' else 'punct.'}")
    else: fine.append(r.role)
pp['role']=fine
ORDER=['query fact mother','query fact child','query fact template','query fact punct.','distractor mother','distractor child','distractor template','distractor punct.','question tokens','final colon']
def rolesum(h):
    d=pp[pp['head']==h]
    per=d.groupby(['family','order','role'])['importance'].sum().reset_index()   # sum over positions in role, per prompt pair
    fam=per.groupby(['family','role'])['importance'].mean().reset_index()       # orders averaged
    return fam.groupby('role')['importance'].agg(['mean','std']).reindex(ORDER).fillna(0)
SEL=['L17H1','L15H25','L14H26','L13H10','L8H15','L7H1','L18H19','L27H6','L20H1','L23H15']
fig,axes=plt.subplots(2,5,figsize=(12,5),sharex=True)
for ax,h in zip(axes.flat,SEL):
    s=rolesum(h); cols=[C['red'] if v<0 else C['blue'] for v in s['mean']]
    ax.barh(range(len(ORDER)),s['mean'],color=cols,height=.7); ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER,fontsize=7)
    ax.invert_yaxis(); ax.axvline(0,color=C['axis'],lw=.8); ax.set_title(f"{h}  (P={H.loc[h,'P_common40']:+.2f}, F={H.loc[h,'F_common40']:+.2f})",fontsize=8,color=C['ink'])
    ax.grid(axis='y',visible=False)
for ax in axes[1]: ax.set_xlabel('summed importance at role positions (logits)',fontsize=7)
fig.suptitle('Causal position profile on the common 40 pairs: single-position patches summed by position role (20 families, orders averaged)',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig2_position_profiles.png'); plt.close(fig)
# additivity check
add=[]
for h in hp['head']:
    s=pp[pp['head']==h].groupby(['family','order'])['importance'].sum().mean()
    add.append(dict(head=h,sum_positions=s,P40=H.loc[h,'P_common40'],F40=H.loc[h,'F_common40']))
add=pd.DataFrame(add); add['resid']=add.sum_positions-add.P40
N['additivity']=dict(spearman=float(spearmanr(add.sum_positions,add.P40)[0]),max_abs_resid=float(add.resid.abs().max()),
                     median_abs_resid=float(add.resid.abs().median()),
                     strong=add[add.P40.abs()>0.3].round(3).to_dict('records'))
N['L17H1_roles']=rolesum('L17H1').round(3)['mean'].to_dict()
N['L15H25_roles']=rolesum('L15H25').round(3)['mean'].to_dict()
N['L13H10_roles']=rolesum('L13H10').round(3)['mean'].to_dict()
N['L14H26_roles']=rolesum('L14H26').round(3)['mean'].to_dict()
N['L8H15_roles']=rolesum('L8H15').round(3)['mean'].to_dict()
# per-token profile of L17H1 within the query fact
d=pp[(pp['head']=='L17H1')]
def tokrole(r):
    ents=json.loads(r['entities'])
    if ents and ents[0]['query'] and ents[0]['role']=='target':
        return 'mother first' if ents[0]['first'] else ('mother last' if ents[0]['last'] else 'mother middle')
    return None
d=d.assign(tr=d.apply(tokrole,axis=1)).dropna(subset=['tr'])
N['L17H1_mother_tokens']=d.groupby(['family','tr'])['importance'].mean().reset_index().groupby('tr')['importance'].mean().round(3).to_dict()
# per-token profile of L17H1 inside the query fact sentence (token classes)
def qtok(r):
    ents=json.loads(r['entities'])
    if ents and ents[0]['query']:
        role='mother' if ents[0]['role']=='target' else 'child'
        return f"{role} "+('first' if ents[0]['first'] else 'last' if ents[0]['last'] else 'middle')
    if r['role']=='query fact template': return str(r['text']).strip()
    if r['role']=='query fact punct.': return 'period'
    return None
QORDER=['mother first','mother middle','mother last','is','the','mother','of','child first','child middle','child last','period']
fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=False)
for ax,h in zip(axes,['L17H1','L13H10','L15H25']):
    d=pp[pp['head']==h].copy(); d['q']=d.apply(qtok,axis=1); d=d.dropna(subset=['q'])
    per=d.groupby(['family','order','q'])['importance'].sum().reset_index().groupby(['family','q'])['importance'].mean().reset_index()
    s_=per.groupby('q')['importance'].agg(['mean','std','count']).reindex(QORDER)
    cols=[C['red'] if v<0 else C['blue'] for v in s_['mean'].fillna(0)]
    ax.bar(range(len(QORDER)),s_['mean'].fillna(0),color=cols,yerr=s_['std'].fillna(0),error_kw=dict(ecolor=C['axis'],lw=.8,capsize=2))
    ax.set_xticks(range(len(QORDER))); ax.set_xticklabels(QORDER,rotation=60,fontsize=7,ha='right'); ax.axhline(0,color=C['axis'],lw=.8); ax.grid(axis='x',visible=False)
    ax.set_title(f'{h}: query-fact sentence "<mother> is the mother of <child>."',fontsize=8,color=C['ink'])
    N[f'{h}_qtokens']=s_['mean'].round(3).to_dict()
axes[0].set_ylabel('single-position importance (logits; family mean ± SD)')
fig.tight_layout(); fig.savefig(OUT/'fig2b_query_fact_tokens.png'); plt.close(fig)
N['reverse']=hp.dropna(subset=['reverse_P_recovery'])[['head','stage2_P','stage3_F','reverse_P_recovery','reverse_F_recovery']].round(3).to_dict('records')

# ---------- Fig 3: final-colon attention ----------
ap=pd.read_csv(A/'attention_profiles.csv')
fin=ap[(ap.site=='final')&(ap.variant=='base')].pivot(index='head',columns='metric',values='mean')
top=hp[hp.stage3_F.abs()>=0.3].sort_values('stage3_F',ascending=False)['head'].tolist()
cols=[('relevant_target_mass','query-fact mother',C['blue']),('relevant_source_mass','query-fact child',C['aqua']),
      ('distractor_target_mass','distractor mother (mean of 3)',C['orange']),('distractor_source_mass','distractor child (mean of 3)',C['yellow']),('self','self (colon)',C['violet'])]
fig,ax=plt.subplots(figsize=(10,4.2)); x=np.arange(len(top)); w=.16
for i,(m,l,c) in enumerate(cols): ax.bar(x+(i-2)*w,fin.loc[top,m],width=w,color=c,label=l)
ax.set_xticks(x); ax.set_xticklabels([f"{h}\n{H.loc[h,'stage3_F']:+.2f}" for h in top],fontsize=7); ax.set_ylabel('attention probability at the final colon (base prompts, 20 families)')
ax.legend(frameon=False,fontsize=7,ncol=5,loc='upper right'); ax.grid(axis='x',visible=False)
ax.set_title('Relation-tracking attention profile at the answer position, heads with $|I^F_h|\\geq0.3$ (label: Scope-F importance)',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig3_final_attention.png'); plt.close(fig)
rc=fin.loc[hp[hp.groups=='random-controls']['head']]
N['attention_controls']={m:dict(mean=float(rc[m].mean()),max=float(rc[m].max())) for m,_,_ in cols}
N['attention_top']=fin.loc[top,[m for m,_,_ in cols]].round(3).to_dict('index')

# ---------- Fig 4: query control / reorder / corruption ----------
pc=pd.read_csv(A/'paired_attention_controls.csv')
fam=pc.groupby(['head','family']).mean(numeric_only=True).reset_index()
g=fam.groupby('head').agg(tq=('target_query_preference_shift','mean'),sq=('source_query_preference_shift','mean'),
    tr=('target_reorder_change','mean'),sr=('source_reorder_change','mean'),tc=('target_corruption_change','mean'),sc=('source_corruption_change','mean'),
    tq_pos=('target_query_preference_shift',lambda v:(v>0).mean()),sq_pos=('source_query_preference_shift',lambda v:(v>0).mean()))
g=g.join(H[['stage3_F','groups','color']])
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
ax=axes[0]
for grp,d in g.groupby('groups'): ax.scatter(d.tq,d.sq,s=22,color=GROUPC[grp],label=GROUPL[grp],edgecolor='white',linewidth=.5)
for h in ['L21H18','L27H6','L23H15','L22H5','L21H6','L19H22','L18H19','L20H1','L19H16','L16H21','L16H1','L16H24','L16H31','L30H18']:
    ax.annotate(h,(g.loc[h,'tq'],g.loc[h,'sq']),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
ax.axhline(0,color=C['axis'],lw=.8); ax.axvline(0,color=C['axis'],lw=.8)
ax.set_xlabel('query-control shift toward the newly queried MOTHER (attention, paired)'); ax.set_ylabel('shift toward the newly queried CHILD')
ax.set_title('Attention follows the question: changed-query control',fontsize=9,color=C['ink']); ax.legend(frameon=False,fontsize=7)
ax=axes[1]
sub=g.loc[top]; x=np.arange(len(top)); w=.27
ax.bar(x-w,sub.tq,width=w,color=C['blue'],label='query change (mother)'); ax.bar(x,sub.tr,width=w,color=C['orange'],label='reorder (mother, same fact)')
ax.bar(x+w,sub.tc,width=w,color=C['yellow'],label='corruption (mother, same fact)')
ax.set_xticks(x); ax.set_xticklabels(top,rotation=60,fontsize=7); ax.axhline(0,color=C['axis'],lw=.8); ax.grid(axis='x',visible=False)
ax.set_ylabel('paired change in attention mass'); ax.legend(frameon=False,fontsize=7); ax.set_title('Same heads: relocation and corruption barely move attention',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig4_query_control.png'); plt.close(fig)
N['query_control']=g.loc[top].round(3).to_dict('index')
N['query_control_random']=g[g.groups=='random-controls'][['tq','sq','tr','tc']].abs().max().round(3).to_dict()
N['margins']=dict(base_mean=float(pc.base_margin.mean()),qc_mean=float(pc.query_change_margin.mean()),qc_negative_fraction=float((pc.query_change_margin<0).mean()),
                  qc_negative_pairs=int(((pc.groupby(['family','order']).query_change_margin.first())<0).sum()))

# ---------- Fig 5: contextual output at colon ----------
op=pd.read_csv(A/'output_profiles.csv')
f=op[(op.site=='final')&(op.variant=='base')&(op.anchor=='first')]
piv=f.pivot_table(index='head',columns=['role','mode'],values='mean'); piv.columns=[f'{r}_{m}' for r,m in piv.columns]
piv=piv.join(H[['stage3_F','groups','color']])
fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
for ax,mode,title in [(axes[0],'target_output','contextual output $o^h_j W_U$ (actual head output at the colon)'),(axes[1],'target_raw','raw-embedding projection $e_j W^h_{OV} W_U$ (RI-style, colon token)')]:
    for grp,d in piv.groupby('groups'): ax.scatter(d[mode],d.stage3_F,s=22,color=GROUPC[grp],label=GROUPL[grp],edgecolor='white',linewidth=.5)
    ax.axhline(0,color=C['axis'],lw=.8); ax.axvline(0,color=C['axis'],lw=.8); symlog(ax,'y'); ax.set_ylim(-2,6.5)
    ax.set_title(title,fontsize=9,color=C['ink']); ax.set_xlabel('query-mother minus mean distractor-mother logit (first token)')
    rho=spearmanr(piv[mode],piv.stage3_F)[0]; ax.text(.02,.95,f'Spearman ρ = {rho:+.2f} (n=105)',transform=ax.transAxes,fontsize=8,color=C['ink2'])
    N[f'rho_{mode}']=float(rho)
axes[0].set_ylabel('Scope-F importance (symlog)'); axes[0].legend(frameon=False,fontsize=7,loc='lower right')
for h in ['L27H6','L30H18','L30H13','L22H5','L21H6','L21H18','L18H19','L18H18','L20H1','L23H15','L19H16','L25H17']:
    axes[0].annotate(h,(piv.loc[h,'target_output'],piv.loc[h,'stage3_F']),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
axes[1].set_xlim(-.02,.02)
fig.tight_layout(); fig.savefig(OUT/'fig5_contextual_output.png'); plt.close(fig)
N['output_top']=piv.loc[top,['target_output','source_output','target_raw']].round(3).to_dict('index')
# last anchor too
fl=op[(op.site=='final')&(op.variant=='base')&(op.anchor=='last')&(op.role=='target')&(op['mode']=='output')].set_index('head')['mean']
N['output_top_last']=fl.loc[top].round(3).to_dict()

# ---------- Fig 6: copying weights ----------
cw=pd.read_csv(A/'copying_weights.csv')
w=cw.pivot(index='head',columns='population',values=['self_minus_others','self_top_fraction']); w.columns=[f'{a}_{b}' for a,b in w.columns]
w=w.join(H[['stage2_P','stage3_F','groups']])
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
ax=axes[0]
for grp,d in w.groupby('groups'): ax.scatter(d.self_minus_others_name,d.stage2_P,s=22,color=GROUPC[grp],label=GROUPL[grp],edgecolor='white',linewidth=.5)
ax.axhline(0,color=C['axis'],lw=.8); ax.axvline(0,color=C['axis'],lw=.8); symlog(ax,'y'); ax.set_ylim(-2,6.5)
for h in ['L30H18','L21H6','L17H24','L19H22','L18H19','L21H18','L20H1','L26H23','L19H16','L18H18','L27H6','L17H1','L23H15']:
    ax.annotate(h,(w.loc[h,'self_minus_others_name'],w.loc[h,'stage2_P']),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
ax.set_xlabel('weight copying score on name tokens: $e_t W^h_{OV} W_U$, self minus mean of other names'); ax.set_ylabel('Scope-P importance (symlog)')
rho_s=spearmanr(w.self_minus_others_name,w.stage2_P)[0]; rho_a=spearmanr(w.self_minus_others_name,w.stage2_P.abs())[0]
ax.text(.02,.95,f'ρ(score, signed $I_h$) = {rho_s:+.2f};  ρ(score, |$I_h$|) = {rho_a:+.2f}',transform=ax.transAxes,fontsize=8,color=C['ink2'])
ax.legend(frameon=False,fontsize=7,loc='lower right'); ax.set_title('Static copying weights vs causal importance',fontsize=9,color=C['ink'])
N['rho_copy_signed']=float(rho_s); N['rho_copy_abs']=float(rho_a)
ax=axes[1]
for grp,d in w.groupby('groups'): ax.scatter(d.self_minus_others_nonname,d.self_minus_others_name,s=22,color=GROUPC[grp],edgecolor='white',linewidth=.5)
lim=[-.25,.65]; ax.plot(lim,lim,color=C['axis'],ls='--',lw=1); ax.set_xlabel('copying score, 128 ordinary-word reference tokens'); ax.set_ylabel('copying score, test-name tokens')
for h in ['L30H18','L21H6','L17H24','L18H19','L21H18','L20H1','L26H23']:
    ax.annotate(h,(w.loc[h,'self_minus_others_nonname'],w.loc[h,'self_minus_others_name']),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
ax.set_title('Name-token vs ordinary-word copying',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig6_copying_weights.png'); plt.close(fig)
N['copy_top']=w.loc[top,['self_minus_others_name','self_top_fraction_name','self_minus_others_nonname']].round(3).to_dict('index')
N['copy_random']=w[w.groups=='random-controls'][['self_minus_others_name','self_minus_others_nonname']].agg(['mean','std','min','max']).round(3).to_dict()

# ---------- Fig 7: synthetic fingerprints ----------
sp=pd.read_csv(A/'synthetic_profiles.csv')
s=sp[sp.condition=='all_probes'].pivot_table(index='head',columns=['benchmark','metric'],values='mean'); s.columns=[f'{a}|{b}' for a,b in s.columns]
s=s.join(H[['stage2_P','stage3_F','groups']])
fig,axes=plt.subplots(1,2,figsize=(10,4.2))
for ax,bm,title in [(axes[0],'key_value_retrieval','Key–value retrieval (16 records, 32 trials)'),(axes[1],'repeated_sequence','Repeated random sequence (length 32, 32 trials × 31 probes)')]:
    xs=s[f'{bm}|source_attention']; ys=s[f'{bm}|output_gold_minus_others']
    for grp,d in s.groupby('groups'): ax.scatter(d[f'{bm}|source_attention'],d[f'{bm}|output_gold_minus_others'],s=22,color=GROUPC[grp],label=GROUPL[grp],edgecolor='white',linewidth=.5)
    ax.axhline(0,color=C['axis'],lw=.8); ax.set_xlabel('attention to the copy source (mean over probes)'); ax.set_ylabel('$o_j W_U$: gold token minus mean of other candidates')
    ax.set_title(title,fontsize=9,color=C['ink'])
    for h in ['L19H16','L21H18','L23H15','L18H18','L16H1','L21H6','L22H5','L26H23','L25H17','L18H19','L27H6','L17H1','L30H13']:
        ax.annotate(h,(xs[h],ys[h]),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
axes[0].legend(frameon=False,fontsize=7,loc='upper left')
fig.tight_layout(); fig.savefig(OUT/'fig7_synthetic.png'); plt.close(fig)
N['synthetic_top']=s.loc[top,[c for c in s.columns if '|' in c and ('source_attention' in c or 'source_argmax' in c or 'output_gold' in c or 'self_attention' in c)]].round(3).to_dict('index')
N['synthetic_model_correct']={bm:float(s[f'{bm}|model_correct'].iloc[0]) for bm in ['key_value_retrieval','repeated_sequence']}
N['synthetic_random_max']=s[s.groups=='random-controls'][[c for c in s.columns if '|' in c]].max().round(3).to_dict()

# ---------- Fig 8: attention/value factorial ----------
fig,ax=plt.subplots(figsize=(10,3.8)); x=np.arange(len(top)); w=.2
ax.bar(x-1.5*w,H.loc[top,'stage3_F'],width=w,color=C['ink2'],label='ordinary head-output patch (Scope F)')
ax.bar(x-.5*w,H.loc[top,'routing_importance'],width=w,color=C['orange'],label='corrupted attention pattern only')
ax.bar(x+.5*w,H.loc[top,'value_importance'],width=w,color=C['blue'],label='corrupted values only')
ax.bar(x+1.5*w,H.loc[top,'AV_interaction_importance'],width=w,color=C['magenta'],label='interaction  −(M11−M10−M01+M00)')
ax.set_xticks(x); ax.set_xticklabels(top,rotation=60,fontsize=7); ax.axhline(0,color=C['axis'],lw=.8); ax.grid(axis='x',visible=False)
ax.set_ylabel('importance (logits, 178 pairs)'); ax.legend(frameon=False,fontsize=7,ncol=2); ax.set_title('Attention–value factorial at the final colon',fontsize=9,color=C['ink'])
fig.tight_layout(); fig.savefig(OUT/'fig8_attention_value.png'); plt.close(fig)
N['av_top']=H.loc[top,['stage3_F','routing_importance','value_importance','AV_interaction_importance']].round(3).to_dict('index')

# ---------- Fig 9: attended-name classification (provisional) ----------
if PROV.exists():
    at=pd.read_csv(PROV).set_index('head').join(H[['groups']])
    fig,ax=plt.subplots(figsize=(6.4,4.6))
    d=at[at.label!='insufficient']
    for grp,dd in d.groupby('groups'): ax.scatter(dd.positive_fraction,dd.scopeF_importance_178,s=22,color=GROUPC[grp],label=GROUPL[grp],edgecolor='white',linewidth=.5)
    ax.axvline(.8,color=C['axis'],ls='--',lw=1); ax.axvline(.2,color=C['axis'],ls='--',lw=1); ax.axhline(0,color=C['axis'],lw=.8); symlog(ax,'y'); ax.set_ylim(-2,6.5)
    for h in ['L27H6','L18H19','L22H5','L21H6','L17H24','L21H18','L18H18','L20H1','L19H16','L26H23','L23H15','L30H18','L25H17','L16H21','L16H1','L17H1']:
        if h in d.index: ax.annotate(h,(d.loc[h,'positive_fraction'],d.loc[h,'scopeF_importance_178']),fontsize=7,color=C['ink2'],xytext=(3,3),textcoords='offset points')
    ax.set_xlabel('fraction of attended-name events with positive contrast (attended name vs other visible names, first token)')
    ax.set_ylabel('Scope-F importance (symlog)'); ax.legend(frameon=False,fontsize=7,loc='upper left')
    ax.set_title('Attended-name mover classification (rule: ≥80% positive / ≥80% negative)',fontsize=9,color=C['ink'])
    fig.tight_layout(); fig.savefig(OUT/'fig9_attended_name.png'); plt.close(fig)
    N['attended']=at.loc[[h for h in top if h in at.index],['label','events','families','positive_fraction','mean_contrast','scopeF_cross_check','label_last_anchor','label_excluding_self','label_final_site_only','label_same_role_controls','self_event_fraction']].round(3).to_dict('index')
    N['attended_counts']=at.label.value_counts().to_dict()

# ---------- contextual RI ----------
cc=pd.read_csv(A/'contextual_ri_correlations.csv'); N['ctx_ri']=cc[(cc.site=='all_test')].round(3).to_dict('records')
cr=pd.read_csv(A/'contextual_ri.csv'); N['ctx_status']=H['contextual_RI_status'].value_counts().to_dict()
# groups summary
N['group_summary']={g:d[['stage2_P','stage3_F']].agg(['mean','median']).round(4).to_dict() for g,d in hp.groupby('groups')}
N['F_new_46']=None
json.dump(N,open('/home/claude/report/numbers.json','w'),indent=1,default=str)
print('done'); print(json.dumps({k:N[k] for k in ['additivity','L17H1_roles','L17H1_mother_tokens','reverse','margins','query_control_random','rho_target_output','rho_target_raw','rho_copy_signed','rho_copy_abs','synthetic_model_correct','attended_counts']},indent=1,default=str)[:6000])
