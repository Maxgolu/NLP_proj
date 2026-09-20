"""Independent, read-only raw-event audit and post-selection diagnostics.

No model calls, no new candidate selection, no significance claims. Outputs go
to a separate directory; the frozen GPU results are never modified.
"""
import argparse
import collections as C
import csv
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def family_values(es, value):
    groups = C.defaultdict(list)
    for e in es:
        groups[e['family']].append(value(e))
    return {f: mean(v) for f, v in groups.items()}


def fmean(es, value):
    return mean(family_values(es, value).values())


def diagnostic(es, anchor):
    es = [e for e in es if e['scores'] and e['scores'][anchor]['names']]
    if not es:
        return None
    score = lambda e: e['scores'][anchor]
    gap = lambda e: score(e)['names_stats']['gap']
    vals = family_values(es, gap)
    counts = C.Counter(e['family'] for e in es)
    def concentration(key):
        # Contributions to the equal-family target mean (nonnegative).
        mass = C.defaultdict(float)
        for e in es:
            mass[key(e)] += score(e)['target'] / counts[e['family']] / len(counts)
        dominant = max(mass, key=mass.get)
        remaining = [e for e in es if key(e) != dominant]
        total = sum(mass.values())
        return dict(id=dominant, target_share=mass[dominant]/total if total else None,
                    unique=len(mass), removal_events=len(remaining),
                    removal_families=len({e['family'] for e in remaining}),
                    removal_gap=fmean(remaining, gap))
    arr = list(vals.values())
    loo = [(sum(arr)-x)/(len(arr)-1) for x in arr] if len(arr)>1 else []
    # Descriptive resampling only; no post-selection confidence/significance claim.
    rng = np.random.default_rng(20260918)
    boot = np.mean(rng.choice(arr, (5000,len(arr))),axis=1)
    return dict(events=len(es), families=len(vals), target=fmean(es,lambda e:score(e)['target']),
                control=fmean(es,lambda e:score(e)['names_stats']['mean']), gap=mean(arr),
                positive_families=sum(x>1e-7 for x in arr), zero_families=sum(abs(x)<=1e-7 for x in arr),
                loo_min=min(loo) if loo else None, loo_max=max(loo) if loo else None,
                bootstrap_descriptive_95=np.quantile(boot,[.025,.975]).tolist(),
                words_gap=fmean(es,lambda e:score(e)['words_stats']['gap']),
                strict_top=fmean(es,lambda e:score(e)['names_stats']['strict_top']),
                ties=fmean(es,lambda e:score(e)['names_stats']['ties']),
                collisions=fmean(es,lambda e:score(e)['names_stats']['collisions']),
                controls=fmean(es,lambda e:len(score(e)['names'])),
                current=concentration(lambda e:e['current_id']),
                target_token=concentration(lambda e:score(e)['target_id']),
                distance=dict(C.Counter(e['distance'] for e in es)),
                variants={v:dict(events=len(xs), families=len({e['family'] for e in xs}),gap=fmean(xs,gap))
                          for v in ('base','corrupted','reorder') if (xs:=[e for e in es if e['variant']==v])},
                distance_gap={d:fmean([e for e in es if e['distance']==d],gap) for d in ('self','previous','long')})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    assert args.source.resolve()!=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    ps={p['id']:p for p in lines(args.source/'prompts.jsonl.gz')}
    original={e['event_id']:e for e in lines(args.source/'events.jsonl.gz')}
    den=C.Counter()
    for p in ps.values():
        for fact in p['test_facts']:
            for scope in ['all_facts']+(['query_fact'] if fact['query'] else []):
                den[(scope,'all_test',p['family'])]+=len(p['token_ids'])-max(fact['s'],fact['ol'])
                den[(scope,'final',p['family'])]+=1
    candidates=json.loads((args.source/'candidates.json').read_text())['candidates']
    heads={f"L{c['layer']}H{c['head']}" for c in candidates}|{'L3H11','L9H22'}
    summary={}
    with (args.source/'head_summary.csv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream):
            if r['distance']=='all' and r['variant']=='pooled' and r['position'] in ('all_test','final'):
                key=(f"L{r['layer']}H{r['head']}",r['scope'],r['position'],r['anchor'])
                summary[key]=r
    audit=C.Counter(); diagnostics=[]; examples={}; token_text={}; final_head_counts={}
    seen=set(); selected_events={}
    for path in sorted((args.source/'heads').glob('*.jsonl.gz')):
        es=list(lines(path)); head=path.name.split('.')[0]
        for e in es:
            assert e['event_id'] not in seen
            seen.add(e['event_id'])
            old=original[e['event_id']]
            assert all(e[k]==v for k,v in old.items())
            p=ps[e['id']]; fact=next(f for f in p['test_facts'] if f['fi']==e['fact'])
            assert e['j']>=max(fact['s'],fact['ol'])
            assert e['dominance']>2.2
            assert (e['position']=='final')==(e['j']==len(p['token_ids'])-1)
            audit['events']+=1; audit['final']+=e['position']=='final'
            audit['query_final']+=e['position']=='final' and e['query_fact']
            token_text[e['current_id']]=p['tokens'][e['j']]
            start,end=p['offsets'][e['j']]
            # Word-span overlap, not token-ID equality: leading spaces can change IDs.
            def overlaps(name):
                return any(m.start()<end and m.end()>start for m in re.finditer(r'\b'+re.escape(name)+r'\b',p['prompt']))
            e['current_role']='target_name' if overlaps(e['target']) else ('source_name' if overlaps(e['source']) else 'other')
            audit['scored']+=e['scores'] is not None
            if e['scores'] is None:
                continue
            for anchor,s in e['scores'].items():
                token_text[s['target_id']]=p['tokens'][fact['of' if anchor=='first' else 'ol']]
                assert math.isfinite(s['target']) and 0<=s['target']<=1
                for kind in ('names','words'):
                    items=s[kind]; st=s[kind+'_stats']
                    assert st['n']==len(items)
                    if items:
                        ctrl=mean(c['score'] for c in items)
                        assert abs(ctrl-st['mean'])<1e-12
                        assert abs(s['target']-ctrl-st['gap'])<1e-12
                        higher=sum(c['score']>s['target']+1e-7 for c in items)
                        ties=sum(abs(c['score']-s['target'])<=1e-7 for c in items)
                        assert st['rank_min']==1+higher and st['rank_max']==1+higher+ties
                    for c in items:
                        assert math.isfinite(c['score']) and 0<=c['score']<=1
                        assert c['position']<=e['j']
                        assert c['token_id']==p['token_ids'][c['position']]
                        if c['collision']:
                            assert c['score']==s['target']
        final_head_counts[head]=dict(all=sum(e['position']=='final' for e in es),
                                    query=sum(e['position']=='final' and e['query_fact'] for e in es))
        # Independent recomputation for every active head and both anchors.
        for scope in ('all_facts','query_fact'):
            for pos in ('all_test','final'):
                subset=[e for e in es if (scope=='all_facts' or e['query_fact']) and (pos=='all_test' or e['position']=='final')]
                scored=[e for e in subset if e['scores']]
                for anchor in ('first','last'):
                    row=summary[(head,scope,pos,anchor)]
                    assert int(row['passes'])==len(subset)
                    pass_families=C.Counter(e['family'] for e in subset)
                    ds={fam:n for (sc,po,fam),n in den.items() if sc==scope and po==pos}
                    assert int(row['opportunities'])==sum(ds.values())
                    assert abs(float(row['frequency'])-mean(pass_families[f]/n for f,n in ds.items()))<1e-12
                    for metric in ('target','names_gap','words_gap'):
                        if metric=='target':
                            keep=scored; value=lambda e:e['scores'][anchor]['target']
                        else:
                            kind=metric.split('_')[0]
                            keep=[e for e in scored if e['scores'][anchor][kind]]
                            value=lambda e:e['scores'][anchor][kind+'_stats']['gap']
                        assert int(row[metric+'_events'])==len(keep)
                        assert int(row[metric+'_families'])==len({e['family'] for e in keep})
                        calc=fmean(keep,value)
                        assert (calc is None and not row[metric+'_mean']) or (calc is not None and abs(calc-float(row[metric+'_mean']))<1e-12)
                        audit['summary_checks']+=1
                    if head in heads:
                        d=diagnostic(subset,anchor)
                        if d:
                            matched=[e for e in subset if e['scores'] and e['scores'][anchor]['names']]
                            def by_role(xs):
                                return dict(events=len(xs),families=len({e['family'] for e in xs}),
                                            gap=fmean(xs,lambda e:e['scores'][anchor]['names_stats']['gap']))
                            d['current_roles']={role:by_role([e for e in matched if e['current_role']==role]) for role in ('target_name','source_name','other')}
                            d['without_current_target']=by_role([e for e in matched if e['current_role']!='target_name'])
                            d.update(head=head,scope=scope,position=pos,anchor=anchor,
                                     passes=len(subset),frequency=float(row['frequency']),
                                     all_target_events=int(row['target_events']),all_target_families=int(row['target_families']))
                            diagnostics.append(d)
        if head in heads:
            selected_events[head]=es
            examples[head]=[]
            for anchor in ('first','last'):
                for e in sorted([e for e in es if e['scores'] and e['scores'][anchor]['names']],
                                key=lambda e:e['scores'][anchor]['names_stats']['gap'],reverse=True)[:3]:
                    examples[head].append(dict(anchor=anchor,id=e['id'],fact=ps[e['id']]['facts'][e['fact']]['line'],
                        source=e['source'],target=e['target'],current=token_text[e['current_id']],current_id=e['current_id'],
                        target_token=token_text[e['scores'][anchor]['target_id']],j=e['j'],distance=e['distance'],
                        scores=e['scores'][anchor]))
    assert seen==set(original)
    for (head,scope,pos,anchor),row in summary.items():
        if head not in final_head_counts:
            assert int(row['passes'])==0 and row['target_mean']=='' and row['names_gap_mean']==''
    audit['active_heads']=len(final_head_counts)
    paired=C.defaultdict(list)
    with (args.source/'paired_variants.csv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream):
            head=f"L{r['layer']}H{r['head']}"
            if head in heads:
                for scope in ['all_facts']+(['query_fact'] if r['query_fact']=='True' else []):
                    paired[(head,scope,r['position'],r['anchor'],r['comparison'])].append(r)
    pairs=[]
    for key,rs in paired.items():
        both=[r for r in rs if r['both_pass']=='True']
        valid=[r for r in both if r['base_names_gap'] and r['twin_names_gap']]
        fams=C.defaultdict(list)
        for r in valid:
            fams[r['family']].append((float(r['base_names_gap']),float(r['twin_names_gap'])))
        fm=[(mean(x[0] for x in xs),mean(x[1] for x in xs)) for xs in fams.values()]
        pairs.append(dict(zip(('head','scope','position','anchor','comparison'),key),
            either_pairs=len(rs),both_pairs=len(both),base_pass_pairs=sum(int(r['base_passes'])>0 for r in rs),
            twin_pass_pairs=sum(int(r['twin_passes'])>0 for r in rs),both_families=len({r['family'] for r in both}),
            comparable_pairs=len(valid), comparable_families=len(fm),
            both_positive_families=sum(x>1e-7 and y>1e-7 for x,y in fm),
            base_gap=mean(x for x,y in fm),twin_gap=mean(y for x,y in fm),
            target_changed_pairs=sum(r['base_target_name']!=r['twin_target_name'] for r in valid)))
    fixed_final=[]
    by_variant={(p['family'],p['order'],p['variant']):p for p in ps.values()}
    for head,es in selected_events.items():
        indexed={(e['id'],e['source']):e for e in es if e['position']=='final' and e['query_fact'] and e['scores']}
        for anchor in ('first','last'):
            pairs_delta=[]; same_sets=0
            for e in indexed.values():
                if e['variant']!='base':
                    continue
                p=ps[e['id']]; twin=by_variant[(p['family'],p['order'],'corrupted')]
                b=indexed.get((twin['id'],e['source']))
                if b is None or b['target']==e['target']:
                    continue
                old_s=e['scores'][anchor]; new_s=b['scores'][anchor]
                old_controls={c['text']:c['score'] for c in old_s['names']}
                new_controls={c['text']:c['score'] for c in new_s['names']}
                if b['target'] not in old_controls or e['target'] not in new_controls:
                    continue
                same_sets+=set(p['token_ids'])==set(twin['token_ids']) and e['current_id']==b['current_id']
                pairs_delta.append((old_s['target']-old_controls[b['target']],new_s['target']-new_controls[e['target']]))
            if pairs_delta:
                fixed_final.append(dict(head=head,anchor=anchor,pairs=len(pairs_delta),same_current_and_context_set=same_sets,
                    both_positive=sum(x>1e-7 and y>1e-7 for x,y in pairs_delta),
                    max_antisymmetry_error=max(abs(x+y) for x,y in pairs_delta)))
    # Freeze detailed outputs before writing human interpretation.
    result=dict(audit=dict(audit),candidate_count=len(candidates),
                names_gap_candidates=sum(any(r['metric']=='names_gap' for r in c['reasons']) for c in candidates),
                final_head_counts=final_head_counts,token_text=token_text,diagnostics=diagnostics,paired=pairs,examples=examples,
                final_counterfactual=fixed_final)
    (args.out/'post_selection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(audit),indent=2))
    print('Saved',args.out/'post_selection_audit.json')


if __name__=='__main__':
    main()
