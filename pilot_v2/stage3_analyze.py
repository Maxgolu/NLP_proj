"""CPU-only integrity audit and descriptive family-level Stage-3 summaries."""
import argparse
import collections
import csv
from pathlib import Path
import numpy as np
from stage3_common import *


def table(path,rows):
    if not rows:raise ValueError('Empty mandatory table: '+str(path))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def describe(values,families):
    values=np.asarray(values,float);f=family_means(values,families)
    return dict(mean=float(f.mean()),family_sd=float(f.std(ddof=1)) if len(f)>1 else 0.,
                family_min=float(f.min()),family_max=float(f.max()),families=len(f),events=len(values),
                positive_family_fraction=float(np.mean(f>0)))


def rankcorr(a,b):
    def rank(v):return np.array([np.sum(v<x)+(np.sum(v==x)-1)/2 for x in v])
    a,b=rank(np.asarray(a)),rank(np.asarray(b))
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and a.std()>0 and b.std()>0 else None


def analyze(out,inputs):
    from stage3_run import validate_inputs,checkpoint_done,pair_filename,CHUNK_SIZE
    plan,pairs,prompts,_=validate_inputs(inputs);m=json_read(out/'manifest.json')
    if digest(inputs/'plan.json')!=m['input_hash'] or m['gate_only']:raise ValueError('Wrong inputs or gate-only run')
    if not json_read(out/'gate_passed.json').get('passed'):raise ValueError('Global gate did not pass')
    if sorted(h for shard in m['shards'] for h in shard)!=plan['heads']:raise ValueError('Head shards are incomplete/duplicated')
    with np.load(inputs/'references.npz') as z:ref={k:z[k].copy() for k in z.files}
    H=plan['heads'];families=ref['families'];common=ref['common'];P=-ref['scopeP_delta']
    F=np.full_like(P,np.nan);promote=F.copy();suppress=F.copy();routing=F.copy();values=F.copy();interaction=F.copy()
    reverse={};position=[];position_counts=collections.Counter();attention=collections.defaultdict(list)
    contextual=collections.defaultdict(list);output=collections.defaultdict(list);synthetic=collections.defaultdict(list)
    weights=[];final_attention={};ri_counts=collections.Counter();expected_ri=collections.Counter()
    all_events=list(read_lines(inputs/'events.jsonl.gz'))
    for ev in all_events:
        for anchor in ev['scores']:expected_ri[(ev['layer']*32+ev['head'],ev['id'],ev['event_id'],anchor)]+=1
    by_prompt={r['id']:r for r in prompts};original_f=np.array(ref['scopeF_readout'])
    for rank,heads in enumerate(m['shards']):
        if not json_read(out/f'gate_{rank}.json').get('passed'):raise ValueError('Replica gate did not pass')
        identity=m['identity']+f':{rank}';base=out/f'replica_{rank}'
        if not json_read(out/f'done_{rank}.json')['complete']:raise ValueError('Worker incomplete')
        chunks=[heads[j:j+CHUNK_SIZE] for j in range(0,len(heads),CHUNK_SIZE)]
        for pi,p in enumerate(pairs):
            for ci,hs in enumerate(chunks):
                path=base/'pairs'/pair_filename(p,ci);tag=identity+':'+p['id']+':'+str(ci)
                if not checkpoint_done(path,tag):raise ValueError('Missing causal checkpoint: '+str(path))
                with np.load(path,allow_pickle=False) as z:
                    if str(z['pair_id'])!=p['id'] or list(z['heads'])!=hs:raise ValueError('Causal identity mismatch')
                    positions=z['positions'].astype(int).tolist();roles=json_read(path.with_suffix('.roles.json'))
                    if len(roles)!=len(positions) or [r['j'] for r in roles]!=positions:raise ValueError('Position role mismatch')
                    n=int(z['n']);first=int(z['first_diff'])
                    expected=list(range(first,n)) if p['exact_all'] else []
                    if positions!=expected:raise ValueError('Incomplete position profile')
                    for key in ['scopeF_readout','av_readout','position_readout','reverse_readout','baseline','corrupt_baseline']:
                        if not np.isfinite(z[key]).all():raise ValueError('Nonfinite output '+key)
                    if z['av_readout'].shape!=(len(hs),4,3) or z['position_readout'].shape!=(len(hs),len(positions),3):raise ValueError('Readout shape mismatch')
                    for j,h in enumerate(hs):
                        hi=H.index(h);f=z['scopeF_readout'][j];b=z['baseline'];av=z['av_readout'][j,:,0]
                        reused=bool(z['scopeF_reused'][j])
                        if reused!=bool(np.isfinite(ref['scopeF_delta'][pi,hi])):raise ValueError('Incorrect reuse flag')
                        fb=np.array([ref['baselines'][pi,0],*p['clean_logits']]) if reused else b
                        F[pi,hi]=fb[0]-f[0];promote[pi,hi]=f[2]-fb[2];suppress[pi,hi]=fb[1]-f[1]
                        if abs(F[pi,hi]-promote[pi,hi]-suppress[pi,hi])>1e-4:raise ValueError('Logit decomposition failed')
                        if reused and not np.array_equal(f,original_f[pi,hi]):raise ValueError('Saved F readout was replaced')
                        routing[pi,hi]=av[0]-av[1];values[pi,hi]=av[0]-av[2]
                        interaction[pi,hi]=-(av[3]-av[1]-av[2]+av[0])
                        for k,pos in enumerate(positions):
                            role=roles[k];delta=z['position_readout'][j,k];imp=float(b[0]-delta[0])
                            position.append(dict(head=head_name(h),family=p['family'],order=p['order'],j=pos,
                                                 text=role['text'],final=role['final'],question=role['question'],
                                                 punctuation=role['punctuation'],entities=json.dumps(role['entities']),importance=imp,
                                                 promotion=float(delta[2]-b[2]),suppression=float(b[1]-delta[1])))
                            position_counts[h]+=1
                    if list(z['reverse_heads'])!=[h for h in hs if h in plan['reverse_heads']]:raise ValueError('Reverse head coverage mismatch')
                    for j,h in enumerate(z['reverse_heads']):
                        reverse[(pi,int(h))]=z['reverse_readout'][j,:,0]-z['corrupt_baseline'][0]
        event_ids={x['id'] for x in all_events if x['layer']*32+x['head'] in heads and x['scores']}
        scan=list(prompts)+[changed_query(p['clean']) for p in pairs if p['exact_all']]
        for r in scan:
            core=r['family'] in plan['common_families']
            if not core and r['id'] not in event_ids:continue
            path=base/'prompts'/(slug(r['id'])+'.jsonl.gz')
            if not checkpoint_done(path,identity+':'+r['id']):raise ValueError('Missing prompt checkpoint: '+str(path))
            iterator=iter(read_lines(path));meta=next(iterator);ann=meta['annotation']
            if meta['id']!=r['id'] or meta['core']!=core:raise ValueError('Prompt identity mismatch')
            anatomy_count=collections.Counter()
            for x in iterator:
                h=x['head'];family=x['family']
                if h not in heads:raise ValueError('Unexpected head in prompt record')
                if x['kind']=='contextual_ri':
                    ri_counts[(h,x['id'],x['event_id'],x['anchor'])]+=1
                    for scope in ['all_facts']+(['query_fact'] if x['query_fact'] else []):
                        for site in ['all_test']+(['final'] if x['position']=='final' else []):
                            for metric in ['target','names_gap','words_gap']:
                                raw=x['scores']['raw'][metric];ctx=x['scores']['contextual']
                                ctx=ctx[metric] if ctx is not None else None
                                contextual[(h,x['anchor'],scope,site,metric)].append((family,x.get('order',r['order']),raw,ctx))
                    continue
                if x['kind']!='anatomy':raise ValueError('Unknown measurement kind')
                anatomy_count[(h,x['j'])]+=1
                site='final' if x['final'] else 'earlier';key=(h,x['variant'],site)
                attention[key+('self',)].append((family,x['self_attention']))
                attention[key+('previous',)].append((family,x['previous_attention']))
                for f in x['facts']:
                    for role in ['source','target']:
                        if f[role+'_visible']:
                            label=('relevant_' if f['query'] else 'distractor_')+role
                            for anchor in ['mass','first','last']:
                                attention[key+(label+'_'+anchor,)].append((family,f[role+'_'+anchor]))
                if x['final']:
                    mapping={f['index']:f for f in ann['facts']}
                    final_attention[(h,family,x['order'],x['variant'])]=dict(margin=meta['candidate_margin'],
                        facts={mapping[f['fact']]['source']:dict(query=f['query'],source=f['source_mass'],target=f['target_mass']) for f in x['facts']})
                # Target/source token contrasts against other facts of the SAME role.
                tokens=x['projection_tokens'];lookup={v:i for i,v in enumerate(tokens)};ids=ann['ids']
                query=next(f for f in ann['facts'] if f['query'])
                for role in ['source','target']:
                    if query[role+'_positions'][-1]>x['j']:continue
                    for anchor,ai in [('first',0),('last',-1)]:
                        target=ids[query[role+'_positions'][ai]]
                        controls=[ids[f[role+'_positions'][ai]] for f in ann['facts'] if not f['query'] and f[role+'_positions'][-1]<=x['j']]
                        if not controls:continue
                        for mode in ['raw','output']:
                            z=x['projections'][mode]['logits'];gap=z[lookup[target]]-np.mean([z[lookup[c]] for c in controls])
                            output[(h,x['variant'],site,role,anchor,mode)].append((family,float(gap)))
            if core:
                expected=collections.Counter({(h,j):1 for h in heads for j in ann['test_positions']})
                if anatomy_count!=expected:raise ValueError('Incomplete unconditional attention/output rows')
                attn=path.with_name(path.name.replace('.jsonl.gz','.attention.npz'))
                with np.load(attn) as z:
                    if list(z['heads'])!=heads or list(z['rows'])!=ann['test_positions']:raise ValueError('Attention cache coverage')
                    if z['pattern'].shape!=(len(heads),len(ann['test_positions']),len(ann['ids'])):raise ValueError('Attention cache shape')
                    if not np.isfinite(z['pattern']).all() or not np.allclose(z['pattern'].sum(-1),1,atol=1e-5):raise ValueError('Invalid attention probabilities')
        for name,label in [('copying_weights.npz','weights'),('synthetic.jsonl.gz','synthetic')]:
            if not checkpoint_done(base/name,identity+':'+label):raise ValueError('Missing diagnostic '+name)
        w=json_read(base/'copying_summary.json')
        if collections.Counter((x['head'],x['population']) for x in w)!=collections.Counter({(h,k):1 for h in heads for k in ['name','nonname']}):
            raise ValueError('Weight diagnostic coverage mismatch')
        with np.load(base/'copying_weights.npz') as z:
            if list(z['heads'])!=heads or z['logits'].shape!=(len(heads),len(z['tokens']),len(z['tokens'])) or not np.isfinite(z['logits']).all():
                raise ValueError('Invalid weight diagnostic matrix')
        weights+=[dict(x,head=head_name(x['head'])) for x in w]
        synth_count=collections.Counter()
        for x in read_lines(base/'synthetic.jsonl.gz'):
            h=x['head'];kind=x['kind'];synth_count[(h,kind)]+=1
            for condition in ['all_probes']+(['correct_copy_events'] if x['model_correct'] else []):
                for metric in ['source_attention','source_argmax','self_attention','previous_attention','output_gold_minus_others','model_correct']:
                    synthetic[(h,kind,condition,metric)].append((x['trial'],float(x[metric])))
        for h in heads:
            for kind,n in [('repeated_sequence',POLICY['repeat_length']-1),('key_value_retrieval',1)]:
                if synth_count[(h,kind)]!=POLICY['synthetic_trials']*n:raise ValueError('Synthetic trial/probe coverage mismatch')
    if ri_counts!=expected_ri:raise ValueError('Matched contextual RI event coverage differs from saved events')
    if not np.isfinite(F).all():raise ValueError('Missing final-position measurements')
    result=out/'analysis';result.mkdir(exist_ok=True)
    rows=[]
    for hi,h in enumerate(H):
        group=[k for k,vs in json_read(inputs/'inventory.json')['groups'].items() if head_name(h) in vs]
        supported=False
        for key,obs in contextual.items():
            if key[0]!=h:continue
            valid=[f for f,o,r,c in obs if r is not None and c is not None]
            if len(valid)>=POLICY['min_events'] and len(set(valid))>=POLICY['min_families']:supported=True;break
        row=dict(head=head_name(h),groups=';'.join(group),stage2_P=describe(P[:,hi],families)['mean'],
                 stage3_F=describe(F[:,hi],families)['mean'],F_family_sd=describe(F[:,hi],families)['family_sd'],
                 F_positive_family_fraction=describe(F[:,hi],families)['positive_family_fraction'],
                 F_promotion=describe(promote[:,hi],families)['mean'],F_suppression=describe(suppress[:,hi],families)['mean'],
                 P_common40=describe(P[common,hi],families[common])['mean'],F_common40=describe(F[common,hi],families[common])['mean'],
                 routing_importance=describe(routing[:,hi],families)['mean'],value_importance=describe(values[:,hi],families)['mean'],
                 AV_interaction_importance=describe(interaction[:,hi],families)['mean'],position_comparisons=position_counts[h],
                 contextual_RI_status='completed; eligibility varies by metric (see support table)' if supported else
                    ('insufficient matched support: see contextual_ri.csv' if any(k[0]==h for k in ri_counts) else 'insufficient support: zero saved scored events'))
        if h in plan['reverse_heads']:
            r=np.stack([reverse[(i,h)] for i in range(len(pairs))]);row['reverse_P_recovery']=describe(r[:,0],families)['mean'];row['reverse_F_recovery']=describe(r[:,1],families)['mean']
        rows.append(row)
    table(result/'head_profiles.csv',rows);table(result/'position_profiles.csv',position)
    npz_write(result/'family_effects.npz',heads=H,families=sorted(set(families.tolist())),
              **{k:family_means(v,families) for k,v in dict(scopeP=P,scopeF=F,promotion=promote,suppression=suppress,
                                                          routing=routing,values=values,interaction=interaction).items()})
    for name,data,keys in [('attention_profiles',attention,['head','variant','site','metric']),
                           ('output_profiles',output,['head','variant','site','role','anchor','mode']),
                           ('synthetic_profiles',synthetic,['head','benchmark','condition','metric'])]:
        summaries=[]
        for key,measurements in data.items():
            ds=describe([v for f,v in measurements],[f for f,v in measurements])
            if name=='synthetic_profiles':
                ds['trials']=ds.pop('families');ds['trial_sd']=ds.pop('family_sd');ds['positive_trial_fraction']=ds.pop('positive_family_fraction')
                ds['trial_min']=ds.pop('family_min');ds['trial_max']=ds.pop('family_max')
            summaries.append(dict(zip(keys,[head_name(key[0]),*key[1:]]),**ds))
        table(result/(name+'.csv'),summaries)
    synth_support=[]
    for h in H:
        for kind in ['repeated_sequence','key_value_retrieval']:
            count=len(synthetic.get((h,kind,'correct_copy_events','model_correct'),[]))
            synth_support.append(dict(head=head_name(h),benchmark=kind,correct_copy_events=count,
                                      conditional_status='completed' if count else 'insufficient support: no correct copy events'))
    table(result/'synthetic_support.csv',synth_support);table(result/'copying_weights.csv',weights)
    ri=[];correlations=collections.defaultdict(list)
    for key,obs in contextual.items():
        h,anchor,scope,site,metric=key;matched=[(f,o,r,c) for f,o,r,c in obs if r is not None and c is not None]
        support=len({f for f,o,r,c in matched});eligible=len(matched)>=POLICY['min_events'] and support>=POLICY['min_families']
        row=dict(head=head_name(h),anchor=anchor,scope=scope,site=site,metric=metric,total_events=len(obs),
                 matched_events=len(matched),families=support,eligible=eligible,status='completed' if eligible else 'insufficient support')
        if matched:
            f=[x[0] for x in matched];raw=describe([x[2] for x in matched],f);ctx=describe([x[3] for x in matched],f)
            active={(x[0],x[1]) for x in matched};mask=np.array([(p['family'],p['order']) in active for p in pairs])
            matched_causal=describe(P[mask,H.index(h)],families[mask])['mean']
            row.update(raw_mean=raw['mean'],contextual_mean=ctx['mean'],change=ctx['mean']-raw['mean'],contextual_family_sd=ctx['family_sd'])
            row.update(matched_causal_pairs=int(mask.sum()),matched_causal_mean=matched_causal)
            if eligible:correlations[(anchor,scope,site,metric)].append((h,raw['mean'],ctx['mean'],float(P[:,H.index(h)].mean()),matched_causal))
        ri.append(row)
    table(result/'contextual_ri.csv',ri)
    corr=[]
    for key,obs in correlations.items():
        _,raw,ctx,causal,matched_causal=zip(*obs)
        corr.append(dict(anchor=key[0],scope=key[1],site=key[2],metric=key[3],heads=len(obs),
                         raw_vs_causal=rankcorr(raw,causal),contextual_vs_causal=rankcorr(ctx,causal),
                         raw_vs_matched_causal=rankcorr(raw,matched_causal),contextual_vs_matched_causal=rankcorr(ctx,matched_causal),
                         population='same supported inventory heads; report both fixed 178-pair and RI-supported family/order causal means'))
    if corr:table(result/'contextual_ri_correlations.csv',corr)
    paired=[]
    for h in H:
        for p in pairs:
            if not p['exact_all']:continue
            key=(h,p['family'],p['order']);a=final_attention[key+('base',)];q=final_attention[key+('query_change',)]
            r=final_attention[key+('reorder',)];c=final_attention[key+('corrupted',)]
            old=next(k for k,v in a['facts'].items() if v['query']);new=next(k for k,v in q['facts'].items() if v['query'])
            row=dict(head=head_name(h),family=p['family'],order=p['order'],base_margin=a['margin'],query_change_margin=q['margin'])
            for role in ['source','target']:
                gap0=a['facts'][new][role]-a['facts'][old][role];gap1=q['facts'][new][role]-q['facts'][old][role]
                row[role+'_query_preference_shift']=gap1-gap0
                row[role+'_reorder_change']=r['facts'][old][role]-a['facts'][old][role]
                row[role+'_corruption_change']=c['facts'][old][role]-a['facts'][old][role]
            paired.append(row)
    table(result/'paired_attention_controls.csv',paired)
    cards=[]
    for r in rows:
        h=r['head'];cards.append(f"## {h}\n\nSelection: {r['groups']}. Scope P: {r['stage2_P']:+.5f}; F: {r['stage3_F']:+.5f}. "
            f"Routing: {r['routing_importance']:+.5f}; values: {r['value_importance']:+.5f}; interaction: {r['AV_interaction_importance']:+.5f}. "
            f"Contextual RI: {r['contextual_RI_status']}.\n\n"
            "Inspect this head's rows in position_profiles, attention_profiles, paired_attention_controls, output_profiles, "
            "copying_weights, contextual_ri and synthetic_profiles before assigning a mechanism. All projections are pre-shared-normalization diagnostics. "
            "Optional Patchscopes/spectral/mediation extensions: not triggered. No circuit or semantic label is inferred automatically.\n")
    (result/'head_cards.md').write_text('# Stage-3 evidence index\n\nDescriptive discovery analysis; orders averaged within families. Synthetic averages use trials, not kinship families.\n\n'+'\n'.join(cards),encoding='utf-8')
    json_write(out/'summary.json',dict(complete=True,heads=len(H),pairs=len(pairs),families=len(set(families)),
               common_position_pairs=int(sum(common)),core_prompts=160,matched_RI_anchor_comparisons=sum(ri_counts.values()),
               missing_support='See contextual_ri.csv and synthetic_support.csv; never encoded as zero.',
               interpretation='Descriptive discovery results, fixed selected inventory; no significance/type/circuit claims',
               source_corrections='inputs/matched_approximation_error.csv and corrected_median_comparison.json'))
    print('STAGE 3 COMPLETE: integrity and coverage checks passed; CPU summaries written.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--inputs',type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.inputs)
