"""Report figures, tables and explicit next-stage manifest from verified saved data."""
from pathlib import Path
import sys,json,csv,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;O=ROOT/'חומר כתוב/Stage4_Overleaf'
sys.path.insert(0,str(ROOT/'tmp/stage4_analysis_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':210})
BLUE='#246087';ORANGE='#bc672a';GREEN='#287767';GRAY='#747d89'
def read(name):return list(csv.DictReader((HERE/name).open(encoding='utf-8')))
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def f(r,k='mean'):return float(r[k])
def sf(name,fig):fig.savefig(O/'figures'/('s42_'+name+'.png'),bbox_inches='tight',facecolor='white');plt.close(fig)
def table(name,headers,rows,cols):
    text='\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (O/'tables'/('s42_'+name+'.tex')).write_text(text,encoding='utf-8')
def num(x):return f'{float(x):+.3f}'
def pick(rows,**kw):return next(r for r in rows if all(str(r[k])==str(v) for k,v in kw.items()))
plan=js(ROOT/'results/stage4_s42_v1/inputs/plan.json');pp=plan['proposal'];facts=js(HERE/'facts.json')
for name in ['figures','tables','data']:(O/name).mkdir(exist_ok=True)
for p in HERE.glob('*.csv'):shutil.copyfile(p,O/'data'/('s42_'+p.name))
shutil.copyfile(HERE/'facts.json',O/'data/s42_facts.json')
shutil.copyfile(ROOT/'results/stage4_s42_readiness_20260924/readiness.json',O/'data/s42_readiness.json')

groups=[g['id'] for g in pp['groups']];heads=sorted({h for g in pp['groups'] for h in g['members']},key=lambda h:tuple(map(int,h[1:].split('H'))))
rr=read('g1_conditional_core.csv');matrix=np.full((8,len(heads)),np.nan)
for r in rr:matrix[groups.index(r['group']),heads.index(r['head'])]=f(r)
fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(matrix,cmap='RdBu',vmin=-4.4,vmax=4.4,aspect='auto')
ax.set_xticks(range(len(heads)),heads,rotation=55,ha='right');ax.set_yticks(range(8),groups);ax.set_title('Conditional head contribution inside each colon group | 20 families')
for i in range(8):
 for j in range(len(heads)):
  if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>2.6 else 'black')
fig.colorbar(im,ax=ax,pad=.02,label='I(group) - I(group minus head), logits');sf('conditional_heatmap',fig)

rows=read('g1_task_performance.csv');orders=read('g1_order_effects.csv');gs=['O1','O3','O4','I1','I2','I3']
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.3));x=np.arange(6)
for offset,before,label,color in [(-.18,'True','Query fact first',BLUE),(.18,'False','Query fact second',ORANGE)]:
 vals=[100*f(pick(orders,group=g,population='all89',direction='noise',mode='donor',query_first=before),'normalized_mean') for g in gs]
 a.bar(x+offset,vals,.35,label=label,color=color)
a.set_xticks(x,gs);a.set_ylabel('Effect / full-model gap in the same order (%)');a.set_title('Order dependence after scale normalization');a.legend(fontsize=8)
for offset,mode,label,color in [(-.18,'donor','Paired donor',BLUE),(.18,'mean','Role mean',ORANGE)]:
 vals=[100*f(pick(rows,group=g,population='all89',direction='noise',mode=mode),'recipient_accuracy') for g in gs]
 b.bar(x+offset,vals,.35,label=label,color=color)
b.axhline(100,color=GRAY,linestyle=':',linewidth=1);b.set_ylim(0,108);b.set_xticks(x,gs);b.set_ylabel('Clean candidate accuracy after replacement (%)');b.set_title('Effect magnitude is baseline-dependent');b.legend(fontsize=8)
fig.tight_layout();sf('order_accuracy',fig)

br=read('blocking_paired.csv');labels=[];src=[];left=[];brtable=[]
for source,site,population,label in [('L13H10','query_sentence','core20','L13H10 sentence [20]'),('L15H25','query_child_last','core20','L15H25 child-last [20]'),('L15H25','query_sentence','core20','L15H25 sentence [20]'),('L17H3','colon','core20','L17H3 colon [20]'),('L17H1','query_writer_union','all89','L17H1 union [89]'),('L17H1','query_sentence','all89','L17H1 sentence [89]')]:
 r=pick(br,source=source,site=site,population=population,mode='donor',direction='restore');labels.append(label);src.append(f(r,'source_mean'));left.append(f(r,'blocked_residual_mean'))
 brtable.append((source,site.replace('query_','').replace('_','-'),20 if population=='core20' else 89,num(r['source_mean']),num(r['mean']),f"{100*f(r,'ratio'):.1f}\\%"))
fig,ax=plt.subplots(figsize=(10,4.4));y=np.arange(6)
ax.barh(y-.17,src,.32,color=BLUE,label='Source restoration alone');ax.barh(y+.17,left,.32,color=ORANGE,label='With receivers clamped')
ax.axvline(0,color='black',lw=.7);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Restoration effect, clean-answer margin (logits)');ax.set_title('Receiver blocking in the otherwise live model');ax.legend(fontsize=9);sf('blocking',fig)
table('blocking',['Source','Mask','$n_f$','$J_W$','$B$','$B/J_W$'],brtable,'llrrrr')

ri=read('ri31_audit.csv');names=facts['extended_RI'];names=sorted(names,key=lambda h:tuple(map(int,h[1:].split('H'))))
fig,ax=plt.subplots(figsize=(10,6));y=np.arange(len(names))
for off,phase,label,color in [(-.12,'extension','Paired donor',BLUE),(.12,'sensitivity','Role mean',ORANGE)]:
 data=[pick(ri,head=h,phase=phase,direction='noise') for h in names];v=np.array([f(r,'conditional_mean') for r in data]);lo=np.array([f(r,'ci_low') for r in data]);hi=np.array([f(r,'ci_high') for r in data])
 ax.errorbar(v,y+off,xerr=[v-lo,hi-v],fmt='o',ms=4,capsize=2,color=color,label=label)
ax.axvspan(-.1,.1,color=GRAY,alpha=.12);ax.axvline(0,color=GRAY,lw=.7);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('Conditional full-prompt contribution with L21H18 replaced (logits)');ax.set_title('14 extended RI heads | 89 families, descriptive 95% intervals');ax.legend();sf('ri_heads',fig)
rtable=[]
for h in names:
 d=pick(ri,head=h,phase='extension',direction='noise');j=pick(ri,head=h,phase='extension',direction='restore');m=pick(ri,head=h,phase='sensitivity',direction='noise')
 rtable.append((h,num(d['conditional_mean']),num(j['conditional_mean']),num(m['conditional_mean']),f"{f(d,'conditional_ma'):.3f}",num(d['change_mean'])))
table('ri_heads',['Head','$I_{K}$','$J_{K}$',r'$I_{K,\mu}$',r'$\E|I_{K,f}|$','$I_K-I_0$'],rtable,'lrrrrr')

rf=read('residual_family_effects.csv');d={int(r['family']):f(r,'effect') for r in rf if r['mode']=='donor' and r['direction']=='noise' and r['background']=='intact'};m={int(r['family']):f(r,'effect') for r in rf if r['mode']=='mean' and r['direction']=='noise' and r['background']=='intact'}
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.4));a.scatter(list(d.values()),[m[k] for k in d],s=20,alpha=.65,color=BLUE)
a.axhline(0,color=GRAY,lw=.7);a.axvline(0,color=GRAY,lw=.7);a.set_xlabel('Paired-donor family effect');a.set_ylabel('Role-mean family effect');a.set_title('Residual RI23: baseline changes the pattern')
cohorts=['residual23','reference1','reference2','reference3','reference4'];x=np.arange(5)
for off,phase,label,color in [(-.18,'extension','Paired donor',BLUE),(.18,'sensitivity','Role mean',ORANGE)]:
 ss=js(HERE/(phase+'_summary.json'));vals=[next(r['mean_absolute'] for r in ss if r['id']==f'G2group:{c}:intact' and r['direction']=='noise') for c in cohorts]
 b.bar(x+off,vals,.35,color=color,label=label)
b.set_xticks(x,['RI23','Ref 1','Ref 2','Ref 3','Ref 4']);b.set_ylabel('Mean absolute family effect (logits)');b.set_title('Four fixed layer-matched comparison groups');b.legend(fontsize=8);fig.tight_layout();sf('residual',fig)

g3=read('g3_backup_audit.csv');fig,(a,b)=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(5);v=np.array([f(r) for r in g3]);lo=np.array([f(r,'ci_low') for r in g3]);hi=np.array([f(r,'ci_high') for r in g3])
a.axhspan(-.1,.1,color=GREEN,alpha=.10);a.axhline(.1,color=GRAY,linestyle='--');a.axhline(-.1,color=GRAY,linestyle='--');a.errorbar(x,v,yerr=[v-lo,hi-v],fmt='o',color=BLUE,capsize=3);a.set_ylim(-.115,.115);a.set_ylabel('Conditional L26H31 effect (logits)');a.set_title('No retained backup contrast on the core panel')
b.bar(x,[f(r,'tv') for r in g3],color=[GRAY,BLUE,BLUE,BLUE,GRAY]);b.set_ylabel('Mean attention total-variation distance');b.set_title('Upstream change does not imply useful backup')
for ax in [a,b]:ax.set_xticks(x,['Intact','K1','K3','K4','L27H6'],rotation=25)
fig.tight_layout();sf('g3',fig)

gtable=[];nrows=read('g1_interactions_cpu_extended.csv');wt=read('g1_whole_populations.csv')
for g in groups:
 population='all89' if g in gs else 'core20';n=pick(nrows,group=g,population=population);w=pick(wt,group=g,population=population)
 gtable.append((g,89 if population=='all89' else 20,num(w['mean']),num(n['mean']),f"{f(n,'mean_absolute'):.3f}",n['classification']))
table('groups',['Group','$n_f$','$I(S)$','$N(S)$',r'$\E|N_f|$','Interaction label'],gtable,'lrrrrl')

# Frozen proposed head sets; none of these follow-up experiments have been run.
base=js(ROOT/'results/stage4_design_v2/manifest_proposal.json')['initial_route_pool']
R6=['L6H24','L8H15','L14H23','L15H3','L16H31','L20H7'];U=['L13H18','L24H19']
sort=lambda hs:sorted(set(hs),key=lambda h:tuple(map(int,h[1:].split('H'))))
C0=sort(base+R6+U);R14=sort(pp['g2']['positive']+pp['g2']['negative']+R6);R17=sort(set(pp['g2']['ri31'])-set(R14));NR=sort(set(C0)-set(R14))
blocks=dict(W=['L15H25','L17H1'],D=['L16H1','L16H21'],P=['L18H18','L18H19'],A=['L21H6','L21H18','L22H5','L27H6'],
    N=['L13H10','L17H3','L18H24','L19H16','L20H1','L23H15','L26H23','L30H18'],R6=R6,
    T=['L14H26','L17H17','L17H24','L19H22','L21H23','L25H17','L30H13'],U=U)
assert len(C0)==33 and len(R14)==14 and len(NR)==19 and len(R17)==17
assert sort([h for b in blocks.values() for h in b])==C0 and sum(map(len,blocks.values()))==33
future=dict(status='Prospective plan from S4.1/S4.2; not implemented or executed',seed=20260923,removed='broad all-head/MLP attribution search and fallback scan',
    C0=C0,C50_if_C0_fails=sort(C0+R17),RI14=R14,nonRI19=NR,R6=R6,R17=R17,RI23=pp['g2']['residual23'],reference_cohorts=pp['g2']['reference_cohorts'],reduction_blocks=blocks,
    local_S43=[dict(source='L13H18',sites=['fact1_mother','fact3_mother','fact3_template'],receivers=blocks['D']+blocks['P'],channels=['K','V'],mediators=[['MLP13'],['MLP14'],['MLP13','MLP14']],configs=72),
        dict(source='L8H15',sites=['query_sentence','other_answer_fact_sentence'],receivers=blocks['D']+blocks['P'],channels=['K','V'],mediators=[[],['MLP8'],['MLP9'],['MLP8','MLP9']],configs=64),
        dict(source='L16H31',sites=['colon'],receivers=['L19H16','L21H6','L21H18','L22H5','residual_to_logits'],channels=['Q_or_bypass'],mediators=[[],['MLP16'],['block17'],['block18'],['MLP16','block17','block18']],configs=25),
        dict(source='L20H7',sites=['colon'],receivers=['L23H15','L25H17','L27H6','L30H18','residual_to_logits'],channels=['Q_or_bypass'],mediators=[[],['MLP20'],['block21'],['block22'],['MLP20','block21','block22']],configs=25),
        dict(source='L26H23',sites=['colon'],receivers=['L30H13','L30H18','residual_to_logits'],channels=['Q_or_bypass'],mediators=[[],['MLP26'],['block27'],['block28'],['MLP26','block27','block28']],configs=15)],
    composed_chains=[dict(source='L15H25',sites=['query_child_last','query_sentence'],live_head_groups=[blocks['D'],blocks['D'][:1],blocks['D'][1:]],receivers=blocks['P'],channel='Q',configs=12),
        dict(source='L17H1',sites=['query_writer_union'],live_head_groups=[blocks['P'],blocks['P'][:1],blocks['P'][1:]],receivers=blocks['A'],channel='Q',configs=12)],
    controls=dict(L13H18_alternate_sites=['fact0_mother','fact2_mother','fact2_template'],mediators=['MLP13','MLP14'],configs=24),
    S43_config_cap=249,S43_additional_direct_comparator_cap=56,S43_scientific_registry_cap=305,S43_rounds=1,S43_extension_claim_cap=16,heldout='sealed',
    S44=dict(unit='groups/mechanisms only',enable_four_condition_panel=True,conditions=['x00','x10','x01','x11'],
        final_configs=['full','Cstar','Cstar_minus_RI_members','Cstar_minus_nonRI_members','full_minus_O3_at_colon','full_minus_RI23_at_all_test'],
        independent_configs=['full','full_minus_O3_at_colon','full_minus_RI23_at_all_test'],orders=2,core_families=20,max_configs=6,primary_baseline='LOFO role mean',
        extension='same six configurations to 89 discovery families only if a C-vs-full condition accuracy discrepancy exceeds 5 points or B reverses sign; no new search'),
    S45=dict(initial_configs=['full','all_test_heads_mean','C0'],fallback_config='C0_plus_R17_if_C0_fails',scope='all test positions; MLPs/demos/prefix live',
        faithfulness=dict(F_min=.8,F_max=1.2,L_max=.2,condition_accuracy_loss_max=.05),reduction='group-first greedy, recompute after accepted removal, at most two passes',
        reduction_trial_cap=96,completeness_groups=['W','D','P','A','N','RI_current','nonRI_current','R6'],group_intersection_with_C=True,
        RI_current='C intersect RI31 (initially RI14)',nonRI_current='C minus RI31 (initially nonRI19)',
        final_baseline_sensitivity='paired donors on identical masks, both conditions',single_head_tests='not part of S44; only a later explicitly targeted C-background membership question, no exhaustive sweep'),
    dependencies=[['S43_local','final_route_interpretation'],['S45_C0_pilot','S45_reduction'],['S45_reduction','S44_final_Cstar_cells'],
        ['S44_final_Cstar_cells','S45_final_joint_acceptance'],['S45_final_joint_acceptance','freeze_then_heldout']],
    may_run_in_parallel=['S43_local','S45_C0_pilot','S44_three_fixed_full_model_configs'])
assert sum(x['configs'] for x in future['local_S43']+future['composed_chains'])+24==249
(HERE/'next_stage_plan.json').write_text(json.dumps(future,indent=2)+'\n',encoding='utf-8');shutil.copyfile(HERE/'next_stage_plan.json',O/'data/s42_next_stage_plan.json')

fig,ax=plt.subplots(figsize=(11,5.8));ax.set_xlim(0,11);ax.set_ylim(0,6);ax.axis('off')
boxes=[(.15,4.35,3.2,1.15,'S4.3 local routes\n249 configurations maximum\nNo broad search'),(3.9,4.35,3.2,1.15,'S4.5 fixed pilot\nFull / all-head mean / C0\nC0: 33 named heads'),(7.65,4.35,3.2,1.15,'S4.4 fixed group panel\nFull / minus O3 / minus RI23\nFact x query, both orders'),(3.9,2.25,3.2,1.1,'S4.5 bounded reduction\nGroup removals; candidate C*\nCore then 89 families'),(7.65,2.25,3.2,1.1,'S4.4 final C* panel\nC*, minus RI, minus non-RI\nSame cached measurements'),(3.9,.15,6.95,1.1,'Joint final decision: faithfulness + group behavior + sensitivity\nFreeze all choices before held-out validation')]
for i,(x,y,w,h,label) in enumerate(boxes):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#eef4f7' if i<3 else '#edf5f0',edgecolor=BLUE,lw=1.2));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
def arrow(a,b,dashed=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=GRAY,linestyle='--' if dashed else '-',connectionstyle='arc3,rad=0'))
arrow((5.5,4.3),(5.5,3.4));arrow((7.2,2.8),(7.6,2.8));arrow((9.2,4.3),(9.2,3.4));arrow((5.5,2.2),(5.5,1.3));arrow((9.2,2.2),(9.2,1.3));arrow((1.8,4.3),(4.0,1.25),True)
ax.text(5.5,5.83,'Top row can start in parallel. Solid arrows: data dependencies.',ha='center',fontsize=11)
ax.text(.1,.5,'Dashed: route evidence informs\ninterpretation; it does not\nblock the fixed node pilot.',fontsize=9,color=GRAY)
sf('dependencies',fig)
print('Prepared seven figures, three tables and exact follow-up manifest')
