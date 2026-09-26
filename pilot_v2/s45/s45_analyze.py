"""S4.5 analysis: family aggregation, guards, Gamma matrix, frozen selection, freeze, validation.

All rules are the inherited descriptive ones (tau=0.10 coherent/heterogeneous); bootstrap
intervals are paired family resamples (20,000 draws, seed 20260926) and are descriptive,
never selection-adjusted significance tests. Primary aggregation averages the two orders
within family, then families; per-order values and within-family order differences are
reported alongside.
"""
import collections
import csv
import json
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write
from stage4_plan import retained
from s45_plan import POLICY,STRUCTURES,C33,C50,CANDIDATES,REFERENCES,hid,hname,ids,rb,states_for,state,attachment,load,CELLS
from s45_run import verify,code_identity

def gather(inputs,schedule,runs):
    rows=[]
    for r in runs:rows+=verify(Path(r),inputs,Path(schedule))
    return rows

def bootstrap(values,seed=POLICY['seed'],draws=POLICY['bootstrap_draws']):
    a=np.asarray(values,float);rng=np.random.default_rng(seed);n=len(a)
    if n==0:return [None,None]
    ix=rng.integers(0,n,(draws,n));m=a[ix].mean(1)
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def summarize(values,paired=None):
    """Retention statistics + descriptive interval for a family vector."""
    a=np.asarray(values,float);s=retained(a) if len(a) else dict(mean=None,retained=False,classification='empty')
    s['families']=int(len(a));s['interval']=bootstrap(a) if len(a) else [None,None];return s

def by_family(rows,value,keyfn):
    """keyfn(row)->group key. Returns group -> dict(families -> dict(order -> value))."""
    out=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in rows:out[keyfn(r)][r['family']][r['order']]=r[value]
    return out

def orders_mean(d):
    """Family mean over the two orders; reports each order and the absolute order difference."""
    fams=sorted(f for f,o in d.items() if set(o)=={0,1})
    mean=np.array([(d[f][0]+d[f][1])/2 for f in fams]);o0=np.array([d[f][0] for f in fams]);o1=np.array([d[f][1] for f in fams])
    return fams,mean,o0,o1

def write_csv(path,rows,fields=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not rows:Path(path).write_text('');return
    fields=fields or sorted({k for r in rows for k in r})
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()})

# ---------------------------------------------------------------- behaviour ----
def cell_table(rows):
    """(state,baseline) -> cell -> family -> order -> margin, plus accuracy flags."""
    margins=by_family([r for r in rows if r['kind']=='behavior'],'margin',lambda r:(r['state'],r['baseline'],r['cell']))
    correct=by_family([r for r in rows if r['kind']=='behavior'],'candidate_correct',lambda r:(r['state'],r['baseline'],r['cell']))
    top=by_family([r for r in rows if r['kind']=='behavior'],'top_is_gold',lambda r:(r['state'],r['baseline'],r['cell']))
    return margins,correct,top

def behavior_summary(rows,states,full_state='full'):
    """Fact-axis gap g, query-axis contrasts, interaction b, fidelity F/L and accuracy guards per (state,baseline)."""
    margins,correct,top=cell_table(rows);labels={sid:st['label'] for sid,st in states.items()};keys=sorted({(s,b) for s,b,c in margins})
    def fam_dict(s,b,c):return margins.get((s,b,c),{})
    summaries=[];family_rows=[]
    def g_vec(s,b):
        x00,x10=fam_dict(s,b,'x00'),fam_dict(s,b,'x10');fams=sorted(set(x00)&set(x10))
        d={f:{o:x00[f][o]-x10[f][o] for o in (0,1) if o in x00[f] and o in x10[f]} for f in fams}
        return orders_mean(d)
    gfull=None
    for s,b in keys:
        if s==full_state:gfull=g_vec(s,b)
    if gfull is None:
        # a full-model row set may be missing (e.g. held-out donor schedules); guards are then not computed
        gfull=([],np.array([]),np.array([]),np.array([]))
    for s,b in keys:
        cells={c:fam_dict(s,b,c) for c in CELLS if (s,b,c) in margins}
        fams,g,g0,g1=g_vec(s,b);rec=dict(state=s,label=labels.get(s,s),baseline=b,families=len(fams),cells=sorted(cells))
        if len(fams):
            rec.update(g_mean=float(g.mean()),g_interval=bootstrap(g),g_order0=float(g0.mean()),g_order1=float(g1.mean()),
                       g_abs_order_difference=float(np.abs(g0-g1).mean()),g_abs_family_mean=float(np.abs(g).mean()))
            common=[i for i,f in enumerate(fams) if f in gfull[0]]
            if common and len(gfull[0]):
                ref=np.array([gfull[1][gfull[0].index(fams[i])] for i in common]);mine=g[common]
                rec.update(F=float(mine.mean()/ref.mean()) if ref.mean() else None,L=float(np.abs(mine-ref).mean()/np.abs(ref).mean()) if np.abs(ref).mean() else None)
                ref0=np.array([gfull[2][gfull[0].index(fams[i])] for i in common]);ref1=np.array([gfull[3][gfull[0].index(fams[i])] for i in common])
                rec.update(F_order0=float(g0[common].mean()/ref0.mean()) if ref0.mean() else None,F_order1=float(g1[common].mean()/ref1.mean()) if ref1.mean() else None)
        if all(c in cells for c in CELLS):
            fam4=sorted(set.intersection(*[set(cells[c]) for c in CELLS]))
            def contrast(fn):
                d={f:{o:fn({c:cells[c][f][o] for c in CELLS}) for o in (0,1) if all(o in cells[c][f] for c in CELLS)} for f in fam4}
                return orders_mean(d)
            _,bq,b0,b1=contrast(lambda m:(m['x00']-m['x10']-m['x01']+m['x11'])/4)
            _,q0,_,_=contrast(lambda m:m['x00']-m['x01']);_,q1,_,_=contrast(lambda m:m['x10']-m['x11']);_,f1,_,_=contrast(lambda m:m['x01']-m['x11'])
            rec.update(b_mean=float(bq.mean()),b_interval=bootstrap(bq),b_order0=float(b0.mean()),b_order1=float(b1.mean()),
                       query_contrast_original_facts=float(q0.mean()),query_contrast_swapped_facts=float(q1.mean()),fact_contrast_alt_query=float(f1.mean()),
                       b_retention=summarize(bq)['classification'])
            for i,f in enumerate(fam4):family_rows.append(dict(state=s,label=labels.get(s,s),baseline=b,family=f,b=float(bq[i]),
                **{f'M_{c}_order{o}':cells[c][f][o] for c in CELLS for o in (0,1)}))
        acc={}
        for c in cells:
            for o in (0,1):
                vals=[correct[(s,b,c)][f][o] for f in correct[(s,b,c)] if o in correct[(s,b,c)][f]]
                acc[f'{c}_order{o}']=float(np.mean(vals)) if vals else None
                tv=[top[(s,b,c)][f][o] for f in top[(s,b,c)] if o in top[(s,b,c)][f]]
                acc[f'top_{c}_order{o}']=float(np.mean(tv)) if tv else None
        rec['accuracy']=acc;summaries.append(rec)
    return summaries,family_rows

def guards(summary,full_state='full'):
    """F in [0.8,1.2], L<=0.2, candidate accuracy within five points of full in every cell x order."""
    full=next((s for s in summary if s['state']==full_state and s['baseline']=='mean'),None)
    for s in summary:
        if full is None or s.get('F') is None:s['passes']=None;s['guard']='not evaluable';continue
        lo,hi=POLICY['fidelity_F'];ok=lo<=s['F']<=hi and s['L']<=POLICY['fidelity_L'];reasons=[]
        if not lo<=s['F']<=hi:reasons.append('F')
        if s['L']>POLICY['fidelity_L']:reasons.append('L')
        for k,v in s['accuracy'].items():
            if k.startswith('top_') or v is None or full['accuracy'].get(k) is None:continue
            if v<full['accuracy'][k]-POLICY['accuracy_drop']-1e-9:ok=False;reasons.append('accuracy:'+k)
        s['passes']=bool(ok);s['guard']='pass' if ok else 'fail:'+','.join(reasons)
    return summary

def analyze_stage_a(inputs,schedule,runs,out,extra=()):
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs);states=dict(sc['states'])
    for sch,rr in extra:
        _,sc2,_=load(inputs,sch);rows+=gather(inputs,sch,rr);states.update(sc2['states'])
    summary,fam=behavior_summary(rows,states);guards(summary)
    out.mkdir(parents=True,exist_ok=True);write_csv(out/'core_behavior.csv',fam);json_write(out/'stage_a_summary.json',summary)
    bylabel={s['label']:s for s in summary}
    if bylabel['C33']['passes']:B,status='C33','pass'
    elif 'C50' in bylabel and bylabel['C50']['passes']:B,status='C50','pass'
    elif 'C50' in bylabel:B,status='C50','partial'
    else:B,status=None,'need_C50'
    decision=dict(B=B,status=status,C33=bylabel['C33'].get('guard'),C50=bylabel.get('C50',{}).get('guard'),empty=bylabel['empty'].get('guard'),
        note='Stage A selects B only; T rosters are never tuned. If the empty mask passes, no isolated head-mechanism claim is made.',
        empty_passes=bylabel['empty'].get('passes'),T_masks={t:bylabel[t].get('guard') for t in STRUCTURES})
    json_write(out/'B_decision.json',decision);return decision

def analyze_stage_b(inputs,schedule,runs,out,B,prior_rows=()):
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs)+list(prior_rows);st=states_for(B)
    summary,fam=behavior_summary(rows,sc['states']|{k:v for k,v in st.items()});guards(summary)
    bys={s['state']:s for s in summary if s['baseline']=='mean'}
    margins,correct,_=cell_table(rows);table=[]
    def cellfam(sid,c):return margins.get((sid,'mean',c),{})
    for h in CANDIDATES+REFERENCES+['R']:
        minus=st[f'B-{h}'] if h!='R' else state(set(ids(B))-set(rb(B,plan['original59'])),'B-R');plus=st[f'B+{h}'] if h in CANDIDATES else st['B']
        rec=dict(candidate=h,kind='candidate' if h in CANDIDATES else 'reference' if h!='R' else 'collective_RI',in_B=hid(h) in set(ids(B)) if h!='R' else None,
                 state_minus=minus['id'],state_plus=plus['id'])
        fams=None
        for c in CELLS:
            a,b=cellfam(plus['id'],c),cellfam(minus['id'],c);ff=sorted(set(a)&set(b))
            fams=ff if fams is None else [f for f in fams if f in ff]
        if not fams:table.append(dict(rec,status='missing cells'));continue
        for c in CELLS:
            a,b=cellfam(plus['id'],c),cellfam(minus['id'],c);sign=1 if c in ['x00','x11'] else -1
            _,fixed,_,_=orders_mean({f:{o:a[f][o]-b[f][o] for o in (0,1)} for f in fams})
            rec[f'fixed_sign_diff_{c}']=summarize(fixed);rec[f'gold_diff_{c}']=summarize(sign*fixed)
        bp={c:cellfam(plus['id'],c) for c in CELLS};bm={c:cellfam(minus['id'],c) for c in CELLS}
        def bf(cells,f,o):return (cells['x00'][f][o]-cells['x10'][f][o]-cells['x01'][f][o]+cells['x11'][f][o])/4
        _,db,db0,db1=orders_mean({f:{o:bf(bp,f,o)-bf(bm,f,o) for o in (0,1)} for f in fams})
        rec['d_b']=summarize(db);rec['d_b_order0']=float(db0.mean());rec['d_b_order1']=float(db1.mean());rec['d_b_abs_order_difference']=float(np.abs(db0-db1).mean())
        rec['delta_F']=(bys.get(plus['id'],{}).get('F') or 0)-(bys.get(minus['id'],{}).get('F') or 0) if plus['id'] in bys and minus['id'] in bys else None
        rec['delta_L']=(bys.get(plus['id'],{}).get('L') or 0)-(bys.get(minus['id'],{}).get('L') or 0) if plus['id'] in bys and minus['id'] in bys else None
        rec['delta_accuracy']={k:(bys[plus['id']]['accuracy'][k] or 0)-(bys[minus['id']]['accuracy'][k] or 0) for k in bys.get(plus['id'],{}).get('accuracy',{})} if plus['id'] in bys and minus['id'] in bys else None
        rec['functional']=rec['d_b']['retained'] or any(rec[f'gold_diff_{c}']['retained'] for c in ['x00','x10'])
        rec['functional_rule']='d_b' if rec['d_b']['retained'] else 'original_cell_gold_diff' if rec['functional'] else 'below_rule'
        table.append(rec)
    out.mkdir(parents=True,exist_ok=True);json_write(out/'stage_b_summary.json',dict(states=summary,candidates=table));write_csv(out/'core_behavior_stage_b.csv',fam)
    return table

# ---------------------------------------------------------------- routes -------
def route_family(rows,keyfn):
    """key -> (families, mean over orders, order0, order1) of route effects."""
    d=by_family([r for r in rows if r['kind'] in ['route','attachment']],'effect',keyfn);return {k:orders_mean(v) for k,v in d.items()}

def analyze_stage_c(inputs,schedule,runs,out,B,stage_b_table,full_rows=()):
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs);st=states_for(B)
    eff=route_family(rows,lambda r:(r['state'],r['anchor'],r['direction']));records=[];matrix=[]
    for (sid,t,d),(fams,m,o0,o1) in sorted(eff.items()):
        label=next((k for k,v in st.items() if v['id']==sid),sc['states'].get(sid,{}).get('label',sid))
        records.append(dict(state=sid,label=label,anchor=t,direction=d,**{k:v for k,v in summarize(m).items()},order0=float(o0.mean()),order1=float(o1.mean()),
                            abs_order_difference=float(np.abs(o0-o1).mean()),historical=STRUCTURES[t]['historical']))
    for h in CANDIDATES:
        for t in STRUCTURES:
            p,mn=st[f'B+{h}'],st[f'B-{h}']
            if (p['id'],t,'noise') not in eff or (mn['id'],t,'noise') not in eff:matrix.append(dict(candidate=h,structure=t,status='missing'));continue
            fp,mp,_,_=eff[(p['id'],t,'noise')];fm,mm,_,_=eff[(mn['id'],t,'noise')];fams=[f for f in fp if f in fm]
            gp=np.array([mp[fp.index(f)] for f in fams]);gm=np.array([mm[fm.index(f)] for f in fams]);gamma=gp-gm
            rp,rm=summarize(gp),summarize(gm);gs=summarize(gamma)
            eligible=rp['retained'] or rm['retained']
            fb=next((x for x in stage_b_table if x['candidate']==h),{})
            matrix.append(dict(candidate=h,structure=t,candidate_in_B=hid(h) in set(ids(B)),families=len(fams),route_plus=rp,route_minus=rm,gamma=gs,
                route_eligible=eligible,status='eligible' if eligible else 'route unavailable/weak in this background',
                stage_b_functional=fb.get('functional'),stage_b_rule=fb.get('functional_rule'),
                gamma_class=gs['classification'] if eligible else 'ineligible',
                interpretation='interaction: h changes the efficacy of this measured communication in B; not proof of serial membership'))
    # Deterministic selection: coherent first, then heterogeneous; decreasing mean|Gamma|; then layer/head; then T index.
    cand=[m for m in matrix if m.get('route_eligible') and m['gamma']['retained']]
    order=lambda m:(0 if m['gamma_class']=='coherent' else 1,-m['gamma']['mean_absolute'],hid(m['candidate']),int(m['structure'][1]))
    cand.sort(key=order);selected=[];per=collections.Counter()
    for m in cand:  # pass 1: at most one pair per candidate, covering different candidates
        if len(selected)<POLICY['max_selected'] and per[m['candidate']]==0:selected.append(m);per[m['candidate']]+=1
    for m in cand:  # pass 2: remaining slots, at most two pairs per candidate
        if len(selected)>=POLICY['max_selected']:break
        if m not in selected and per[m['candidate']]<POLICY['max_pairs_per_candidate']:selected.append(m);per[m['candidate']]+=1
    if len({m['candidate'] for m in selected})>POLICY['max_selected']:raise ValueError('Selection cap violated')
    chosen=[dict(candidate=m['candidate'],structure=m['structure'],gamma_class=m['gamma_class'],gamma_mean=m['gamma']['mean'],gamma_mean_absolute=m['gamma']['mean_absolute'],
                 stage_b_functional=m['stage_b_functional'],record='route_pair',label='conditional interaction candidate' if not m['stage_b_functional'] else 'functional and interaction') for m in selected]
    if len(chosen)<POLICY['max_selected']:
        fill=[x for x in stage_b_table if x['candidate'] in CANDIDATES and x['functional'] and x['candidate'] not in per]
        fill.sort(key=lambda x:(0 if x['d_b']['classification']=='coherent' else 1,-x['d_b']['mean_absolute'],hid(x['candidate'])))
        for x in fill:
            if len(chosen)>=POLICY['max_selected'] or len({c['candidate'] for c in chosen})>=POLICY['max_selected']:break
            chosen.append(dict(candidate=x['candidate'],structure='B',record='functional_only',d_b_class=x['d_b']['classification'],d_b_mean=x['d_b']['mean'],
                               label='functional_only: behavioural extension/validation only; no Gamma or attachment'))
    regression=[]
    for r in records:
        if r['label']=='full':
            hist=STRUCTURES[r['anchor']]['historical']['I' if r['direction']=='noise' else 'J'];drift=abs(r['mean']-hist)
            regression.append(dict(anchor=r['anchor'],direction=r['direction'],measured=r['mean'],historical=hist,drift=drift,within_0_05=drift<=.05,families=r['families'],
                                   note='S4.3 family-034 accommodation: report per-family drift explicitly; tolerances are not relaxed automatically'))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'core_route_effects.csv',[dict(pair_id=r['pair_id'],family=r['family'],order=r['order'],state=r['state'],label=r.get('label'),anchor=r['anchor'],direction=r['direction'],
        effect=r['effect'],intact_margin=r['intact_margin'],endpoint_margin=r['endpoint_margin'],hybrid_margin=r['hybrid_margin'],channel_norm=r['channel_norm'],source_norm=r['source_norm']) for r in rows])
    write_csv(out/'candidate_structure_matrix.csv',[dict(candidate=m['candidate'],structure=m['structure'],status=m['status'],gamma_class=m.get('gamma_class'),
        gamma_mean=m.get('gamma',{}).get('mean'),gamma_mean_absolute=m.get('gamma',{}).get('mean_absolute'),gamma_sign_fraction=m.get('gamma',{}).get('sign_fraction'),
        gamma_interval=m.get('gamma',{}).get('interval'),route_plus_mean=m.get('route_plus',{}).get('mean'),route_plus_class=m.get('route_plus',{}).get('classification'),
        route_minus_mean=m.get('route_minus',{}).get('mean'),route_minus_class=m.get('route_minus',{}).get('classification'),stage_b_functional=m.get('stage_b_functional')) for m in matrix])
    sel=dict(selected=chosen,B=B,rule='coherent first, then heterogeneous; decreasing mean|Gamma|; then layer/head; then T index; pass 1 one pair per candidate, pass 2 at most two per candidate; functional-only fill; at most three slots and three candidates',
             matrix_pairs=len(matrix),eligible_pairs=sum(1 for m in matrix if m.get('route_eligible')),retained_gamma_pairs=len(cand),full_background_regression=regression,
             discovery_only='unselected discoveries remain discovery-only')
    json_write(out/'frozen_selection.json',sel);json_write(out/'stage_c_summary.json',dict(routes=records,matrix=matrix));return sel

def analyze_stage_d(inputs,schedule,runs,out,B,selection,stage_c_rows):
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs)+list(stage_c_rows);st=states_for(B)
    eff=route_family(rows,lambda r:(r['state'],r['key'],r['direction'],r.get('role','anchor')));gamma=[];att=[]
    for s in selection['selected']:
        if s['record']!='route_pair':continue
        h,t=s['candidate'],s['structure'];p,m=st[f'B+{h}'],st[f'B-{h}'];rec=dict(candidate=h,structure=t)
        for d in ['noise','restore']:
            kp,km=(p['id'],t,d,'anchor'),(m['id'],t,d,'anchor')
            if kp not in eff or km not in eff:rec[d]='missing';continue
            fp,mp,p0,p1=eff[kp];fm,mm,m0,m1=eff[km];fams=[f for f in fp if f in fm]
            g=np.array([mp[fp.index(f)]-mm[fm.index(f)] for f in fams]);g0=np.array([p0[fp.index(f)]-m0[fm.index(f)] for f in fams]);g1=np.array([p1[fp.index(f)]-m1[fm.index(f)] for f in fams])
            rec[d]=dict(gamma=summarize(g),route_plus=summarize(np.array([mp[fp.index(f)] for f in fams])),route_minus=summarize(np.array([mm[fm.index(f)] for f in fams])),
                        order0=float(g0.mean()),order1=float(g1.mean()),abs_order_difference=float(np.abs(g0-g1).mean()),family_values=dict(zip(map(str,fams),map(float,g))))
        if isinstance(rec.get('noise'),dict) and isinstance(rec.get('restore'),dict):
            a,b=rec['noise'],rec['restore'];fa=a['family_values'];fb=b['family_values'];common=[f for f in fa if f in fb]
            agree=float(np.mean([np.sign(fa[f])==np.sign(fb[f]) for f in common])) if common else None
            same=np.sign(a['gamma']['mean'])==np.sign(b['gamma']['mean'])
            rec['between_direction_sign_agreement']=agree;rec['same_mean_sign']=bool(same)
            rec['bidirectional']='coherent' if a['gamma']['classification']=='coherent' and b['gamma']['classification']=='coherent' and same and (agree or 0)>=POLICY['coherent_fraction'] else \
                'heterogeneous' if a['gamma']['retained'] and b['gamma']['retained'] else 'asymmetric' if a['gamma']['retained']!=b['gamma']['retained'] else 'unreplicated'
        gamma.append(rec)
        a=attachment(t,h);ctrl=plan['control_rosters'][a['key']]['control'];diag=state(set(ids(B))|{hid(h),hid(ctrl)},'diag')
        arec=dict(candidate=h,structure=t,attachment=a['key'],control_receiver=ctrl,diagnostic_state=diag['id'],
                  note='local diagnostic background B+h+control, distinct from the primary B_h context; shared L18H18/L27H6 endpoints are shared, not structure-specific')
        for d in ['noise','restore']:
            kt,kc=(diag['id'],a['key'],d,'target'),(diag['id'],a['key']+'|control='+ctrl,d,'control')
            if kt not in eff or kc not in eff:arec[d]='missing';continue
            ft,mt,_,_=eff[kt];fc,mc,_,_=eff[kc];fams=[f for f in ft if f in fc]
            tv=np.array([mt[ft.index(f)] for f in fams]);cv=np.array([mc[fc.index(f)] for f in fams]);diff=np.abs(tv)-np.abs(cv)
            arec[d]=dict(target=summarize(tv),control=summarize(cv),abs_paired_difference=dict(mean=float(diff.mean()),interval=bootstrap(diff),families=len(fams)),
                         receiver_selectivity=bool(diff.mean()>0),bootstrap_note='descriptive; not a post-selection significance test')
        if isinstance(arec.get('noise'),dict) and isinstance(arec.get('restore'),dict):
            a_,b_=arec['noise']['target'],arec['restore']['target']
            arec['bidirectional']='coherent' if a_['classification']=='coherent' and b_['classification']=='coherent' and np.sign(a_['mean'])==np.sign(b_['mean']) else \
                'heterogeneous' if a_['retained'] and b_['retained'] else 'asymmetric' if a_['retained']!=b_['retained'] else 'unreplicated'
        att.append(arec)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'core_bidirectional_gamma.csv',[dict(candidate=r['candidate'],structure=r['structure'],direction=d,gamma_mean=r[d]['gamma']['mean'],gamma_class=r[d]['gamma']['classification'],
        gamma_sign_fraction=r[d]['gamma']['sign_fraction'],gamma_interval=r[d]['gamma']['interval'],route_plus=r[d]['route_plus']['mean'],route_minus=r[d]['route_minus']['mean'],
        order0=r[d]['order0'],order1=r[d]['order1'],bidirectional=r.get('bidirectional')) for r in gamma for d in ['noise','restore'] if isinstance(r.get(d),dict)])
    write_csv(out/'attachment_controls.csv',[dict(candidate=r['candidate'],structure=r['structure'],attachment=r['attachment'],control=r['control_receiver'],direction=d,
        target_mean=r[d]['target']['mean'],target_class=r[d]['target']['classification'],control_mean=r[d]['control']['mean'],control_class=r[d]['control']['classification'],
        abs_paired_difference=r[d]['abs_paired_difference']['mean'],abs_paired_interval=r[d]['abs_paired_difference']['interval'],receiver_selectivity=r[d]['receiver_selectivity'],
        bidirectional=r.get('bidirectional')) for r in att for d in ['noise','restore'] if isinstance(r.get(d),dict)])
    json_write(out/'stage_d_summary.json',dict(gamma=gamma,attachments=att));return dict(gamma=gamma,attachments=att)

def analyze_behavior_suite(inputs,schedule,runs,out,B,name,prior_rows=()):
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs)+list(prior_rows)
    summary,fam=behavior_summary(rows,sc['states']);guards(summary)
    mean={s['label']:s for s in summary if s['baseline']=='mean'};donor={s['label']:s for s in summary if s['baseline']=='donor'}
    discord=[]
    for label,d in donor.items():
        m=mean.get(label)
        if m and m.get('g_mean') is not None and d.get('g_mean') is not None:
            discord.append(dict(label=label,mean_g=m['g_mean'],donor_g=d['g_mean'],mean_F=m.get('F'),donor_F=d.get('F'),same_sign=bool(np.sign(m['g_mean'])==np.sign(d['g_mean'])),
                                note='baseline sensitivity: report discordance; no favourable baseline is chosen'))
    out.mkdir(parents=True,exist_ok=True);write_csv(out/f'{name}_behavior.csv',fam)
    json_write(out/f'{name}_summary.json',dict(states=summary,baseline_discordance=discord,B_status='pass' if mean.get('B',{}).get('passes') else 'partial'))
    return summary,discord

def membership_ledger(plan,B,selection,out):
    o59=set(plan['original59'])
    heads=set(ids(C50))|{hid(h) for s in STRUCTURES.values() for h in s['retained']}|set(ids(CANDIDATES))|{hid(v['control']) for v in plan['control_rosters'].values()}
    rows=[]
    for h in sorted(heads):
        rows.append(dict(head=hname(h),id=h,original59=h in o59,C33=hname(h) in C33,C50=hname(h) in C50,B=h in set(ids(B)),R_B=h in set(rb(B,plan['original59'])),
            candidate=hname(h) in CANDIDATES,reference=hname(h) in REFERENCES,structures=[t for t,s in STRUCTURES.items() if hname(h) in s['retained']],
            control_for=[k for k,v in plan['control_rosters'].items() if v['control']==hname(h)],selected=any(s['candidate']==hname(h) for s in selection['selected'])))
    write_csv(out/'ri_membership_ledger.csv',rows);return rows

# ---------------------------------------------------------------- validation ---
def analyze_validation(inputs,schedule,runs,out,freeze,discovery):
    """Frozen suite on the 87 sealed families. Signs/rules are frozen; labels (a)-(e)."""
    plan,sc,pairs=load(inputs,schedule);rows=gather(inputs,schedule,runs);B=freeze['B']['members'];st=states_for(B);sel=freeze['selection']
    summary,fam=behavior_summary(rows,sc['states']);guards(summary)
    out.mkdir(parents=True,exist_ok=True);write_csv(out/'validation_behavior.csv',fam)
    margins,_,_=cell_table(rows);eff=route_family(rows,lambda r:(r['state'],r['key'],r['direction'],r.get('role','anchor')));labels=[]
    for s in sel['selected']:
        h=s['candidate'];p,m=st[f'B+{h}'],st[f'B-{h}'];rec=dict(candidate=h,structure=s['structure'],record=s['record'])
        fams=None;cells={}
        for c in CELLS:
            a,b=margins.get((p['id'],'mean',c),{}),margins.get((m['id'],'mean',c),{});ff=sorted(set(a)&set(b));fams=ff if fams is None else [f for f in fams if f in ff];cells[c]=(a,b)
        if fams:
            def bf(side,f,o):return (cells['x00'][side][f][o]-cells['x10'][side][f][o]-cells['x01'][side][f][o]+cells['x11'][side][f][o])/4
            _,db,d0,d1=orders_mean({f:{o:bf(0,f,o)-bf(1,f,o) for o in (0,1)} for f in fams});rec['d_b']=summarize(db)
            rec['d_b_order_signs_agree']=bool(np.sign(d0.mean())==np.sign(d1.mean()))
            disc=discovery['signs'].get(h,{}).get('d_b')
            rec['functional_reproduced']=bool(rec['d_b']['retained'] and disc is not None and np.sign(rec['d_b']['mean'])==disc)
        if s['record']=='route_pair':
            t=s['structure']
            for d in ['noise','restore']:
                kp,km=(p['id'],t,d,'anchor'),(m['id'],t,d,'anchor')
                if kp in eff and km in eff:
                    fp,mp,_,_=eff[kp];fm,mm,_,_=eff[km];ff=[f for f in fp if f in fm];g=np.array([mp[fp.index(f)]-mm[fm.index(f)] for f in ff])
                    rec[f'gamma_{d}']=summarize(g);rec[f'route_plus_{d}']=summarize(np.array([mp[fp.index(f)] for f in ff]));rec[f'route_minus_{d}']=summarize(np.array([mm[fm.index(f)] for f in ff]))
            disc=discovery['signs'].get(h,{}).get('gamma')
            rec['gamma_reproduced']=bool(rec.get('gamma_noise',{}).get('retained') and disc is not None and np.sign(rec['gamma_noise']['mean'])==disc)
            rec['gamma_bidirectional']=bool(rec.get('gamma_noise',{}).get('retained') and rec.get('gamma_restore',{}).get('retained') and np.sign(rec['gamma_noise']['mean'])==np.sign(rec['gamma_restore']['mean']))
            a=attachment(t,h);ctrl=plan['control_rosters'][a['key']]['control'];diag=state(set(ids(B))|{hid(h),hid(ctrl)},'diag')
            for d in ['noise','restore']:
                kt,kc=(diag['id'],a['key'],d,'target'),(diag['id'],a['key']+'|control='+ctrl,d,'control')
                if kt in eff and kc in eff:
                    ft,mt,_,_=eff[kt];fc,mc,_,_=eff[kc];ff=[f for f in ft if f in fc];tv=np.array([mt[ft.index(f)] for f in ff]);cv=np.array([mc[fc.index(f)] for f in ff]);diff=np.abs(tv)-np.abs(cv)
                    rec[f'attachment_{d}']=dict(target=summarize(tv),control=summarize(cv),abs_paired_difference=dict(mean=float(diff.mean()),interval=bootstrap(diff)))
            at=rec.get('attachment_noise')
            rec['attachment_resolved']=bool(at and at['target']['retained'] and at['abs_paired_difference']['mean']>0 and (at['abs_paired_difference']['interval'][0] or 0)>0)
        f_,g_,a_=rec.get('functional_reproduced',False),rec.get('gamma_reproduced',False),rec.get('attachment_resolved',False)
        sensitive=(rec.get('d_b_order_signs_agree') is False)
        disc_any=discovery['signs'].get(h,{}).get('d_b') is not None or discovery['signs'].get(h,{}).get('gamma') is not None
        if f_ and g_ and a_:lab='(a) reproduced functional and communication participation'
        elif f_:lab='(b) functional participant, attachment unresolved'
        elif g_:lab='(c) route/interaction modulation only, overall functional contribution unresolved'
        elif sensitive or disc_any:lab='(e) order/baseline sensitive or not reproduced'
        else:lab='(d) no effect detected in this bounded scope'
        rec['final_label']=lab;rec['scope']='single-hop contextual task; live background '+('B='+freeze['B']['status']);labels.append(rec)
    json_write(out/'validation_summary.json',dict(states=summary,selected=labels,B=freeze['B'],
        discovery_only='T1..T5 standalone behaviour and unselected candidates remain discovery-only'))
    write_csv(out/'validation_labels.csv',[dict(candidate=r['candidate'],structure=r['structure'],record=r['record'],final_label=r['final_label'],
        d_b_mean=r.get('d_b',{}).get('mean'),d_b_class=r.get('d_b',{}).get('classification'),gamma_noise=r.get('gamma_noise',{}).get('mean'),gamma_restore=r.get('gamma_restore',{}).get('mean'),
        gamma_bidirectional=r.get('gamma_bidirectional'),attachment_resolved=r.get('attachment_resolved')) for r in labels])
    return labels

def discovery_signs(stage_b_table,selection,stage_d):
    signs={}
    for x in stage_b_table:
        if x['candidate'] in CANDIDATES and x['d_b']['retained']:signs.setdefault(x['candidate'],{})['d_b']=int(np.sign(x['d_b']['mean']))
    for s in selection['selected']:
        if s['record']=='route_pair':signs.setdefault(s['candidate'],{})['gamma']=int(np.sign(s['gamma_mean']))
    for r in stage_d.get('attachments',[]):
        if isinstance(r.get('noise'),dict):signs.setdefault(r['candidate'],{})['attachment']=int(np.sign(r['noise']['target']['mean']))
    return signs
