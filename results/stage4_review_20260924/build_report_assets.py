"""Derived tables, figures and a declarative S4.2 proposal. No model execution."""
from pathlib import Path
import json,sys,itertools,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp/stage4_analysis_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
O=ROOT/'חומר כתוב/Stage4_Overleaf';(O/'figures').mkdir(parents=True,exist_ok=True);(O/'tables').mkdir(exist_ok=True);(O/'data').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':220})
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def head(h):return f'L{int(h)//32}H{int(h)%32}'
def savefig(name):
    plt.savefig(O/'figures'/f'{name}.png',bbox_inches='tight',facecolor='white');plt.close()
def escape(x):
    return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(name,header,rows,cols=None):
    cols=cols or 'l'+'r'*(len(header)-1)
    text=r'\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(header)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (O/'tables'/f'{name}.tex').write_text(text,encoding='utf-8')

s=pd.read_csv(R/'extension_population_summary.csv');alln=s[(s.population=='all89')&(s.direction=='noise')].set_index('config_id')
allj=s[(s.population=='all89')&(s.direction=='restore')].set_index('config_id')
new=s[(s.population=='additional69')&(s.direction=='noise')].set_index('config_id')
core=s[(s.population=='core20')&(s.direction=='noise')].set_index('config_id')
seed=pd.read_csv(R/'reproduced/seed/route_summary.csv');refine=pd.read_csv(R/'reproduced/refine/route_summary.csv')
e=pd.read_csv(R/'all_events.csv.gz',low_memory=False);audit=read(R/'verification.json')
for name in ['extension_population_summary.csv','control_comparisons.csv','retained_not_extended.csv','seed_map_coverage.csv','prefix_summary.csv']:
    (O/'data'/name).write_bytes((R/name).read_bytes())
table('execution',['Phase','Configs','Records','Reused'],[(escape(x['phase']),x['configs'],f"{x['records']:,}",x['reused']) for x in audit['phases']])
cov=pd.read_csv(R/'reproduced/coverage/route_summary.csv')
table('coverage',['Head','Scope','Mean $I$','Mean $|I_f|$','Sign fraction'],[(r.source_head,r.source_site,f'{r.mean:.3f}',f'{r.mean_absolute:.3f}',f'{r.sign_fraction:.2f}') for r in cov.itertuples()],cols='llrrr')
picked=['545:query_writer_union:head:594:V','545:query_writer_union:head:595:V','545:query_writer_union:head:600:V','545:query_writer_union:head:709:V',
        '545:query_writer_union:head:641:V','545:query_writer_union:head:624:V','505:query_child_last:head:513:V','505:query_child_last:head:533:V',
        '513:colon:head:594:Q','533:colon:head:594:Q','594:colon:head:870:Q','595:colon:head:870:Q','426:query_sentence:head:513:V',
        '426:query_sentence:head:547:V','474:query_sentence:head:513:V','547:colon:head:594:Q','595:colon:mlp:19:',
        '595:colon:bypass:31:','594:colon:bypass:31:','787:colon:bypass:31:']
short={'query_writer_union':'union','query_child_last':'child-last','query_sentence':'sentence','query_mother':'mother','colon':'colon'}
rows=[]
for cid in picked:
    r=alln.loc[cid];j=allj.loc[cid];v=new.loc[cid]
    target=head(r.receiver)+'/'+r.channel if r.kind=='head' else ('MLP '+str(int(r.receiver)) if r.kind=='mlp' else 'bypass')
    rows.append((head(r.source)+'/'+short[r.site],target,f'{r["mean"]:+.3f}',f'{j["mean"]:+.3f}',f'{v["mean"]:+.3f}'))
table('key_routes',['Source/site','Receiver/channel','$I_{89}$','$J_{89}$','$I_{69}$'],rows,'llrrr')

# Full ledger makes selection and omissions inspectable without a 96-row body table.
ledger=alln.reset_index().merge(allj.reset_index()[['config_id','mean','retained']],on='config_id',suffixes=('','_restore'))
ledger=ledger.merge(new.reset_index()[['config_id','mean','retained']],on='config_id',suffixes=('','_additional69'))
base=e[e.phase=='extend_seed'].drop_duplicates(['pair_id','direction'])
gap=base.groupby('direction').baseline.mean().loc['noise']-base.groupby('direction').baseline.mean().loc['restore']
ledger['I_over_full_gap']=ledger['mean']/gap;ledger['J_over_full_gap']=ledger['mean_restore']/gap
ledger.to_csv(O/'data/route_ledger_96.csv',index=False)

# Fact-site V heatmap: all tested seed routes, not just retained ones.
sources=[('L15H25','query_child_last'),('L17H1','query_mother'),('L17H1','query_is_token'),('L17H1','query_period'),('L13H10','query_sentence'),('L14H26','query_sentence')]
receivers=['L16H1','L16H21','L17H3','L18H18','L18H19','L18H24','L19H16','L20H1','L21H6','L21H18','L22H5','L27H6']
arr=np.full((len(sources),len(receivers)),np.nan)
for i,(src,site) in enumerate(sources):
    for j,rec in enumerate(receivers):
        q=seed[(seed.source_head==src)&(seed.source_site==site)&(seed.receiver==rec)&(seed.channel=='V')]
        if len(q):arr[i,j]=q.iloc[0]['mean']
fig,ax=plt.subplots(figsize=(11.3,4));im=ax.imshow(arr,cmap='RdBu',vmin=-1.25,vmax=1.25,aspect='auto')
ax.set_xticks(range(len(receivers)),receivers,rotation=45,ha='right');ax.set_yticks(range(len(sources)),[a+' / '+{'query_child_last':'child-last','query_mother':'mother','query_is_token':'is','query_period':'period','query_sentence':'sentence'}[b] for a,b in sources])
for (i,j),v in np.ndenumerate(arr):
    if np.isfinite(v):ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.75 else 'black')
ax.set_title('Fact-site source → receiver V → colon output (20 families)');fig.colorbar(im,ax=ax,label='Mean noising effect I (logits)',shrink=.8)
savefig('fact_routes')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.5))
x=core.loc[alln.index,'mean'];y=new.loc[alln.index,'mean'];colors=np.where(alln['mean']>=0,'#176b87','#b34a3c')
axs[0].scatter(x,y,c=colors,s=24,alpha=.7);axs[0].plot([-1.2,3.6],[-1.2,3.6],color='gray',linestyle='--');axs[0].set(xlabel='Selection families: mean I (20)',ylabel='Additional families: mean I (69)',title='Expansion consistency: 96 selected routes')
axs[1].scatter(alln['mean'],allj.loc[alln.index,'mean'],c=colors,s=24,alpha=.7);axs[1].plot([-1.2,3.6],[-1.2,3.6],color='gray',linestyle='--');axs[1].set(xlabel='Noising: mean I (89)',ylabel='Restoration: mean J (89)',title='Both directions; fixed clean-answer sign')
fig.tight_layout();savefig('replication')

c=pd.read_csv(R/'control_comparisons.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.2))
for ax,typ,title in zip(axs,['other_site','random_receiver'],['Other-site controls (17)','Random-receiver controls (38)']):
    g=c[c.control_type==typ];ax.scatter(g.parent_ma,g.control_ma,color='#176b87');mx=max(g.parent_ma.max(),g.control_ma.max())*1.08
    ax.plot([0,mx],[0,mx],'--',color='gray');ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='Target route: mean |I family|',ylabel='Control: mean |I family|',title=title)
fig.tight_layout();savefig('controls')

inter=pd.read_csv(R/'reproduced/interactions_core/interaction_summary.csv');h=inter[inter.retained].copy()
labels=[]
for r in h.itertuples():
    bits=r.config_id.split(':');labels.append(('union' if r.contrast.startswith('union') else 'KV')+' / '+bits[1].replace('query_','').replace('writer_union','sites')+' → '+head(int(bits[3])))
fig,ax=plt.subplots(figsize=(10,4.4));y=np.arange(len(h));ax.barh(y-.17,h['mean'],height=.32,label='Signed mean',color='#176b87');ax.barh(y+.17,h.mean_absolute,height=.32,label='Mean absolute family contrast',color='#d7a34a')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(xlabel='Interaction contrast (logits)',title='Nonadditivity can be hidden by signed averaging (20 families)');ax.legend(loc='lower right');fig.tight_layout();savefig('interactions')

# A deliberately partial graph: every drawn edge is measured; their composition is not claimed.
fig,ax=plt.subplots(figsize=(11.5,5.2));ax.set(xlim=(-.2,11.5),ylim=(0,5.2));ax.axis('off')
nodes={'w15':(1,4.1,'L15H25\nchild-last'),'w17':(1,1.5,'L17H1\nwriter sites'),
       'c':(3.8,4.1,'L16H1 / L16H21\ncolon'),'r':(6.6,2.8,'L18H18 / L18H19\ncolon'),
       'neg':(6.6,.65,'L18H24 / L20H1\ncolon'),'late':(10,4.1,'L27H6\ncolon'),'bypass':(10,1.5,'Residual bypass\nand MLP 19')}
for x,y,label in nodes.values():
    ax.add_patch(FancyBboxPatch((x-.98,y-.38),1.96,.76,boxstyle='round,pad=.08',facecolor='#eef3f6',edgecolor='#69808d'));ax.text(x,y,label,ha='center',va='center',fontsize=10)
def edge(a,b,label,negative=False,offset=(0,0)):
    x,y,_=nodes[a];u,v,_=nodes[b];color='#b34a3c' if negative else '#176b87'
    ax.add_patch(FancyArrowPatch((x+1.06,y),(u-1.06,v),arrowstyle='-|>',mutation_scale=15,color=color,lw=2))
    ax.text((x+u)/2+offset[0],(y+v)/2+offset[1],label,color=color,ha='center',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',pad=1))
edge('w15','c','V: +0.18 / +0.21',offset=(0,.22));edge('c','r','Q: +0.23 to +0.37',offset=(0,.27));edge('w17','r','V union: +3.40 / +2.07',offset=(0,-.25));edge('w17','neg','V union: -1.00 / -0.58',True,offset=(0,-.27));edge('r','late','Q: +0.66 / +0.73',offset=(0,.25));edge('r','bypass','Measured parallel exits',offset=(0,-.3))
ax.text(5.75,5.0,'Selected measured routes; grouped boxes are NOT joint interventions',ha='center',fontsize=12,weight='bold');savefig('route_schematic')

groups=[
 dict(id='O1',type='outgoing',source='L13H10',site='query_sentence',members=['L16H1','L17H3','L18H18','L18H19'],reason='V and K routes; mixed signs'),
 dict(id='O2',type='outgoing',source='L15H25',site='query_child_last',members=['L16H1','L16H21'],reason='Both V routes extend in both directions'),
 dict(id='O3',type='outgoing',source='L17H1',site='query_mother',members=['L18H18','L18H19','L18H24','L19H16','L20H1','L22H5','L27H6'],reason='One common source mask with all seven V routes extended'),
 dict(id='O4',type='outgoing',source='L17H3',site='colon',members=['L18H18','L18H19'],reason='Two opposing Q routes'),
 dict(id='I1',type='incoming',receiver='L27H6',channel='Q',site='colon',members=['L18H18','L18H19','L19H16','L20H1','L23H15','L25H17'],reason='Highest ranked incoming group'),
 dict(id='I2',type='incoming',receiver='L21H23',channel='KV',site='colon',members=['L17H24','L18H18','L18H19','L19H16','L20H1'],reason='KV/V identical node roster deduplicated; L17H24 remains core-only for KV'),
 dict(id='I3',type='incoming',receiver='L30H18',channel='Q',site='colon',members=['L18H18','L18H19','L20H1','L23H15','L25H17'],reason='Next distinct incoming group'),
 dict(id='I4',type='incoming',receiver='L18H18',channel='Q',site='colon',members=['L16H1','L16H21','L17H3'],reason='Same source roster also reaches L18H19 Q; one node-group measurement')]
configsets=set();group_configsets={}
for g in groups:
    ms=g['members'];subsets={tuple(sorted([h])) for h in ms}|{tuple(sorted(ms))}|{tuple(sorted(set(ms)-{h})) for h in ms}
    subsets.discard(());group_configsets[g['id']]=subsets;configsets|=subsets
    g['unique_within_group']=len(subsets);g['intervention_site']='colon'
design=read(ROOT/'results/stage4_design_v2/manifest_proposal.json')
proposal=dict(status='Data-informed S4.2 proposal; not GPU implementation or submitted schedule',groups=groups,
    g1_unique_colon_node_configurations=len(configsets),g1_endpoint_evaluations_core=len(configsets)*40,
    group_measurements=[list(x) for x in sorted(configsets,key=lambda x:(len(x),x))],
    g1_max_common_configurations=128,full_discovery_group_contrast_cap=16,
    g2=dict(ri31=design['ri31'],positive=['L16H1','L16H21','L17H24','L18H19','L22H5'],negative=['L19H16','L23H15','L26H23'],residual23=design['ri23'],
        reference_cohorts=design['reference_cohorts'],K_prefix_order=['L21H18','L22H5','L21H6','L27H6'],K_site='colon',candidate_site='all_original_prompt',
        K_target_gap_fraction=[.3,.8],K_fallback_closest_to=.55),
    g3=dict(head='L26H31',site='colon',backgrounds=[[],['L21H18'],['L21H18','L22H5','L21H6'],['L21H18','L22H5','L21H6','L27H6']],downstream_attention_null=['L27H6']),
    blocking=[dict(source=g['source'],source_masks=({'O1':['query_sentence'],'O2':['query_child_last','query_sentence'],'O3':['query_writer_union','query_sentence'],'O4':['colon']}[g['id']]),receivers=g['members'],receiver_site='colon',both_directions=True) for g in groups if g['type']=='outgoing'],
    separately_targeted=[dict(source='L14H26',site='query_sentence',receiver='L16H1',reason='Singleton; do not invent an outgoing coalition')],
    unresolved=['L13H18: local mediated-path expansion warranted','L21H18 and L21H6: weak/missing mapped incoming support does not erase prior causal effects','L26H23 and L26H31: conditional RI/backup audit remains necessary'],
    boundaries=['L17H24 to L21H23 KV is core-only; extended V falls below selection rule','No group, membership or faithfulness result has yet been measured'])
(R/'s42_proposal.json').write_text(json.dumps(proposal,indent=2)+'\n',encoding='utf-8');(O/'data/s42_proposal.json').write_bytes((R/'s42_proposal.json').read_bytes())
table('groups',['ID','Pattern','Members','Configs'],[(g['id'],(g.get('source','')+' / '+short.get(g['site'],g['site']) if g['type']=='outgoing' else g['receiver']+' / '+g['channel']),', '.join(g['members']),g['unique_within_group']) for g in groups],cols=r'lp{.23\linewidth}p{.49\linewidth}r')
print('G1 config count',len(configsets),'within-group sum',sum(g['unique_within_group'] for g in groups))

# Independent conditional order comparisons; paired family, not unpaired order means.
ex=e[e.phase.str.startswith('extend')]
order=[]
for (cid,di),g in ex.groupby(['config_id','direction']):
    p=g.pivot(index='family',columns='query_first',values='effect');delta=p[True]-p[False]
    order.append(dict(config_id=cid,direction=di,before_mean=p[True].mean(),after_mean=p[False].mean(),
        signed_order_difference=delta.mean(),mean_absolute_order_difference=delta.abs().mean(),max_absolute_order_difference=delta.abs().max()))
pd.DataFrame(order).to_csv(O/'data/order_sensitivity.csv',index=False)
pd.DataFrame(order).to_csv(R/'order_sensitivity.csv',index=False)

fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for ax,cid,title in zip(axs,['545:query_writer_union:head:594:V','545:query_writer_union:head:595:V'],['L17H1 union → L18H18 V','L17H1 union → L18H19 V']):
    p=ex[(ex.config_id==cid)&(ex.direction=='noise')].pivot(index='family',columns='query_first',values='effect')
    ax.scatter(p[False],p[True],s=17,alpha=.65,color='#176b87');mx=max(p.max())*1.06
    ax.plot([0,mx],[0,mx],'--',color='gray');ax.set(xlabel='Query fact after competitor: I',ylabel='Query fact before competitor: I',title=title,xlim=(0,mx),ylim=(0,mx))
fig.tight_layout();savefig('order_sensitivity')

# RI participation ledger: observed routes are nominations, not membership.
both=pd.concat([seed,refine]);nom=both[(both.panel!='control')&both.retained]
rirows=[]
for h in design['ri31']:
    out=nom[nom.source_head==h];inc=nom[(nom.receiver_kind=='head')&(nom.receiver==h)]
    extout=alln[alln.source.apply(head)==h];extin=alln[(alln.kind=='head')&(alln.receiver.apply(head)==h)]
    rirows.append(dict(head=h,retained_core_outgoing=len(out),retained_core_incoming=len(inc),extended_outgoing=len(extout),extended_incoming=len(extin),membership='not yet established; S4.2 pending'))
pd.DataFrame(rirows).to_csv(O/'data/ri31_route_ledger.csv',index=False)

# Gate and report summary facts.
stats=dict(total_records=audit['total_records'],new_records=sum(x['records']-x['reused'] for x in audit['phases']),
    seed_retained=int(seed.retained.sum()),extended_routes=len(alln),retained_noise=int(alln.retained.sum()),retained_restore=int(allj.retained.sum()),
    retained_additional_noise=int(new.retained.sum()),controls=len(c),controls_retained=int(c.control_retained.sum()),
    G1_unique_configurations=len(configsets),G1_core_endpoints=len(configsets)*40,
    self_endpoint_checks=int(e.self_error.notna().sum()),self_endpoint_max=float(e.self_error.max()),
    gap=float(ex.drop_duplicates(['pair_id','direction']).groupby('direction').baseline.mean().loc['noise']-ex.drop_duplicates(['pair_id','direction']).groupby('direction').baseline.mean().loc['restore']))
(R/'report_facts.json').write_text(json.dumps(stats,indent=2)+'\n');print(stats)
