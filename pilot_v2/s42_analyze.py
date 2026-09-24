"""Complete-family CPU analysis and deterministic, auditable adaptive decisions."""
import collections
import csv
from pathlib import Path
import numpy as np
from stage3_common import digest,json_read,json_write,read_lines,head_name
from stage4_plan import retained,hid
from s42_plan import POLICY,Registry,config,group_atoms,load,conditional,write_schedule
from s42_run import verify,key


def gather(inputs,schedule,runs):
    plan,sc,pairs=load(inputs,schedule);rows=[];ranks=set()
    for folder in runs:
        run=Path(folder);meta=json_read(run/'manifest.json')['identity'];ranks.add(meta['shard'])
        if meta['shards']!=len(runs):raise ValueError('Missing/extra shards')
        rows+=verify(run,inputs,schedule)
    if ranks!=set(range(len(runs))) or len({key(r) for r in rows})!=len(rows):raise ValueError('Shard coverage')
    index={key(r):r for r in rows}
    want={(p['id'],c['id'],d) for p in pairs for c in sc['configs'] for d in c['directions']}
    if set(index)!=want:raise ValueError('Schedule coverage')
    return plan,sc,pairs,rows,index


def summarize(values):
    vals=np.asarray(values,dtype=float)
    if not len(vals) or not np.isfinite(vals).all():raise ValueError('Invalid summary population')
    stats=retained(vals)
    rng=np.random.default_rng(POLICY['seed']);ix=rng.integers(0,len(vals),size=(POLICY['bootstrap_draws'],len(vals)))
    lo,hi=np.quantile(vals[ix].mean(1),[.025,.975])
    return dict(families=len(vals),**stats,descriptive_ci_low=float(lo),descriptive_ci_high=float(hi))


def summaries(records):
    groups=collections.defaultdict(list);family=[];out=[]
    for r in records:groups[(r['id'],r['direction'],r['family'])].append(r)
    for (cid,di,f),rr in sorted(groups.items()):
        if len(rr)!=2 or {x['order'] for x in rr}!={0,1}:raise ValueError('Both orders required before decisions')
        family.append(dict(id=cid,direction=di,family=f,effect=float(np.mean([x['effect'] for x in rr])),
            order_delta=next(x['effect'] for x in rr if x['order']==0)-next(x['effect'] for x in rr if x['order']==1)))
    gg=collections.defaultdict(list)
    for r in family:gg[(r['id'],r['direction'])].append(r)
    for (cid,di),rr in sorted(gg.items()):
        out.append(dict(id=cid,direction=di,**summarize([x['effect'] for x in rr]),
            mean_absolute_order_difference=float(np.mean([abs(x['order_delta']) for x in rr]))))
    return family,out


def contrast_records(sc,pairs,index):
    configs={c['id']:c for c in sc['configs']};records=[]
    for c in sc['contrasts']:
        directions=set(['noise','restore'])
        for cid,w in c['terms']:directions &= set(configs[cid]['directions'])
        if not c['terms']:directions={'noise'}
        for di in sorted(directions):
            for p in pairs:
                value=sum(w*index[(p['id'],cid,di)]['effect'] for cid,w in c['terms'])
                records.append(dict(id=c['id'],direction=di,pair_id=p['id'],family=p['family'],order=p['order'],
                    query_first=p['masks']['_query_first'],prefix_length=len(p['shared_prefix']),axis=p['corr']['corruption'],effect=float(value)))
    return records


def write_csv(path,rows):
    if not rows:path.write_text('',encoding='utf-8');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def analyze(inputs,schedule,runs,out):
    plan,sc,pairs,rows,index=gather(inputs,schedule,runs)
    endpoint=[dict(id=r['config_id'],direction=r['direction'],family=r['family'],order=r['order'],effect=r['effect']) for r in rows]
    ef,es=summaries(endpoint);cr=contrast_records(sc,pairs,index);cf,cs=summaries(cr)
    gap=float(np.mean([p['saved_baselines'][0]-p['saved_baselines'][1] for p in pairs]))
    if gap<=0:raise ValueError('Nonpositive normalization gap')
    for st in es+cs:st.update(original_gap=gap,normalized_mean=st['mean']/gap,normalized_mean_absolute=st['mean_absolute']/gap)
    cm={c['id']:c for c in sc['contrasts']}
    for s in cs:s.update({k:v for k,v in cm[s['id']].items() if k not in ['id','terms']})
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'endpoint_families.csv',ef);write_csv(out/'endpoint_summary.csv',es)
    write_csv(out/'contrast_families.csv',cf);write_csv(out/'contrast_summary.csv',cs)
    json_write(out/'summary.json',cs)
    strata=[]
    # Prefix is a technical compatibility diagnostic, not a selected semantic stratum.
    for field in ['query_first','axis','prefix_length']:
        grouped=collections.defaultdict(list)
        for r in cr:grouped[(r['id'],r['direction'],r[field],r['family'])].append(r['effect'])
        gr=collections.defaultdict(list)
        for (cid,di,value,f),vv in grouped.items():gr[(cid,di,value)].append(float(np.mean(vv)))
        for (cid,di,value),vv in sorted(gr.items()):
            if len(vv)<10:continue
            strata.append(dict(id=cid,direction=di,stratum=field,value=value,**summarize(vv)))
    write_csv(out/'strata.csv',strata)
    noncore=[r for r in cr if r['family'] not in plan['core_families']]
    if noncore:
        _,ncs=summaries(noncore);write_csv(out/'additional_families.csv',ncs)
    # Endpoint accuracy is first divergent-token candidate accuracy, not generation accuracy.
    accuracies=[]
    for c in sc['configs']:
        for di in c['directions']:
            rr=[index[(p['id'],c['id'],di)] for p in pairs]
            margins=[r['result']['margin'] if r['result'] is not None else r.get('measurement_baseline_margin',r['baseline']['margin'])-r['effect'] for r in rr]
            accuracies.append(dict(id=c['id'],direction=di,pairs=len(rr),clean_candidate_preferred=float(np.mean(np.array(margins)>0)),
                recipient_candidate_accuracy=float(np.mean(np.array(margins)*(1 if di=='noise' else -1)>0)),
                mean_margin=float(np.mean(margins)),
                individual_logits_available=sum(r['result'] is not None for r in rr)))
    write_csv(out/'candidate_accuracy.csv',accuracies)
    # Blocking ratios only when the source denominator is operationally substantial.
    sm={(s['id'],s['direction']):s for s in cs};ratios=[]
    for c in sc['contrasts']:
        if c.get('kind')!='blocking':continue
        for di in ['noise','restore']:
            b=sm.get((c['id'],di));w=sm.get((c['id']+':source',di))
            if b and w:ratios.append(dict(id=c['id'],direction=di,blocking=b['mean'],source=w['mean'],
                ratio=b['mean']/w['mean'] if abs(w['mean'])>=POLICY['tau'] else None,
                interpretation='intervention ratio, not unique mediated fraction'))
    json_write(out/'blocking_ratios.json',ratios)
    diagnostics=[]
    for r in rows:
        if not r['diagnostic']:continue
        p=next(p for p in pairs if p['id']==r['pair_id']);diag=r['diagnostic'];pat=np.array(diag['pattern'])
        diagnostics.append(dict(pair_id=r['pair_id'],config_id=r['config_id'],direction=r['direction'],
            query_mother_mass=float(pat[p['masks']['query_mother']].sum()),child_last_mass=float(pat[p['masks']['query_child_last']].sum()),
            entropy=float(-(pat[pat>0]*np.log(pat[pat>0])).sum()),head_output_norm=float(np.linalg.norm(diag['z']))))
    write_csv(out/'g3_diagnostics.csv',diagnostics)
    files={f.name:digest(f) for f in out.iterdir() if f.is_file() and f.name!='verification.json'}
    json_write(out/'verification.json',dict(complete=True,schedule_hash=digest(schedule),files=files,records=len(rows),
        reused=sum(r['origin']!='new' for r in rows),new=sum(r['origin']=='new' for r in rows),
        interpretation='Adaptive discovery summaries and bootstrap intervals are descriptive, not held-out validation.'))
    return cs


def choose_k(inputs,schedule,runs):
    plan,sc,pairs,rows,ix=gather(inputs,schedule,runs);result=[]
    gap=float(np.mean([p['saved_baselines'][0]-p['saved_baselines'][1] for p in pairs]))
    if gap<=0:raise ValueError('Nonpositive original clean-corrupt gap')
    for i in range(1,5):
        hs=plan['proposal']['g2']['K_prefix_order'][:i];cid=config(group_atoms(hs))['id']
        I=float(np.mean([ix[(p['id'],cid,'noise')]['effect'] for p in pairs]));J=float(np.mean([ix[(p['id'],cid,'restore')]['effect'] for p in pairs]))
        result.append(dict(K=hs,fraction=(gap-I-J)/gap,noise=I,restore=J))
    eligible=[r for r in result if .3<=r['fraction']<=.8]
    chosen=eligible[0] if eligible else min(result,key=lambda r:(abs(r['fraction']-.55),len(r['K'])))
    return dict(K=chosen['K'],floor_warning=not bool(eligible),original_gap=gap,candidates=result,rule=POLICY['K_rule'])


def refine(inputs,schedule,runs,summ):
    plan,sc,pairs,rows,ix=gather(inputs,schedule,runs);r=Registry();skipped=[]
    configs={c['id']:c for c in sc['configs']};sm={s['id']:s for s in summ if s['direction']=='noise'}
    used={cid for c in sc['contrasts'] if c['panel'] in ['G1','blocking','control'] for cid,w in c['terms']}
    def add(atoms):
        cid=config(atoms)['id']
        if cid not in used and len(used)>=POLICY['g1_common_cap']:return None
        used.add(cid);return r.add(atoms)
    for g in plan['proposal']['groups']:
        tag='G1:'+g['id'];ms=g['members']
        if not any(sm[c['id']]['retained'] for c in sc['contrasts'] if c.get('group')==g['id'] and c['kind'] in ['interaction','conditional']):continue
        pos=[];neg=[]
        for h in ms:
            cid=config(group_atoms([h]))['id'];value=np.mean([ix[(p['id'],cid,'noise')]['effect'] for p in pairs])
            (pos if value>=0 else neg).append(h)
        if pos and neg:
            aa=add(group_atoms(pos));bb=add(group_atoms(neg));whole=add(group_atoms(ms))
            if aa and bb and whole:r.contrast(tag+':sign_split',[(whole,1),(aa,-1),(bb,-1)],'G1',group=g['id'],kind='sign_split')
            else:skipped.append(tag+':sign_split')
        top=sorted(ms,key=lambda h:(-sm[tag+':conditional:'+h]['mean_absolute'],h))[:2]
        pair=add(group_atoms(top));a=add(group_atoms(top[:1]));b=add(group_atoms(top[1:]))
        if pair and a and b:r.contrast(tag+':top_pair_interaction',[(pair,1),(a,-1),(b,-1)],'G1',group=g['id'],kind='pair_interaction',members=top)
        else:skipped.append(tag+':top_pair_interaction')
    return r,dict(common_configs=len(used),cap=POLICY['g1_common_cap'],skipped=skipped)


def extension(inputs,phases):
    """Selection counts claims, closure supplies all matched constituent measurements."""
    allconfigs={};contrasts={};scores={};order=[]
    for schedule,summary in phases:
        sc=json_read(schedule)
        for c in sc['configs']:
            if c['id'] not in allconfigs:allconfigs[c['id']]=c
            elif c['observe']:allconfigs[c['id']]['observe']=True
        for c in sc['contrasts']:contrasts[c['id']]=c
        for s in summary:
            if s['retained']:scores[s['id']]=max(scores.get(s['id'],0),s['mean_absolute'])
    g1=[cid for cid in scores if contrasts[cid]['panel']=='G1' or contrasts[cid].get('kind')=='blocking']
    selected=sorted(g1,key=lambda cid:(-scores[cid],cid))[:POLICY['g1_extension_contrasts']]
    # For an informative RI head/cohort or backup background, preserve its paired comparisons.
    units={cid.rsplit(':',1)[0] for cid in scores if contrasts[cid]['panel'] in ['G2','G2group','G3']}
    selected +=[cid for cid,c in contrasts.items() if c['panel'] in ['G2','G2group','G3'] and cid.rsplit(':',1)[0] in units]
    ancillary=[]
    for cid in selected:
        if contrasts[cid].get('kind')=='blocking':ancillary +=[cid+':source',cid+':self']
    selected=sorted(set(selected+ancillary));r=Registry()
    for cid in selected:
        c=contrasts[cid]
        for component,w in c['terms']:
            cfg=allconfigs[component];r.add(cfg['atoms'],('noise','restore'),cfg['observe'])
        r.contrasts.append(c)
    return r,dict(selected=selected,ranked_g1=sorted(g1,key=lambda cid:(-scores[cid],cid)),
        g1_selected_cap=POLICY['g1_extension_contrasts'],selection_scores=scores,
        caveat='Selected discovery extension; unextended common-panel claims remain provisional')


def sensitivity_registry(extension_schedule):
    s=json_read(extension_schedule);r=Registry()
    for c in s['configs']:r.add(c['atoms'],c['directions'],c['observe'])
    r.contrasts=s['contrasts'];return r


def finish(inputs,out,phases,K):
    plan,_=load(inputs);summary={}
    for label in phases:summary[label]=json_read(out/'analysis'/label/'summary.json')
    comparison=[]
    if 'sensitivity' in summary:
        donor={(s['id'],s['direction']):s for s in summary['extension']}
        for s in summary['sensitivity']:
            d=donor[(s['id'],s['direction'])]
            comparison.append(dict(id=s['id'],direction=s['direction'],donor_mean=d['mean'],mean_replacement_mean=s['mean'],
                donor_retained=d['retained'],mean_retained=s['retained'],same_mean_sign=bool(np.sign(d['mean'])==np.sign(s['mean'])),
                donor_classification=d['classification'],mean_classification=s['classification']))
    write_csv(out/'baseline_sensitivity.csv',comparison)
    json_write(out/'results_summary.json',dict(K=K,phases=summary,strong_RI_route_followup=plan['proposal']['g2']['positive']+plan['proposal']['g2']['negative'],
        boundaries=['S4.2 groups are not a validated sufficient circuit','No S4.3/S4.4/S4.5 or heldout evaluation executed',
        'Mean reverse is a corrupted-recipient sensitivity test, not literal clean restoration',
        'Means preserve demonstrations and shared answer prefix; donor Scope P uses all original prompt tokens',
        'K is selected once with paired donors; sensitivity recomputes the same fixed background using means']))
