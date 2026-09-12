"""Summarize head events without treating prompt twins as independent families.
Usage: python analyze_stage1_audit.py RUN_DIRECTORY
"""
import argparse
import collections
import gzip
import json
from pathlib import Path


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def summarize(run, heads=((9, 22), (3, 11))):
    selected = set(heads)
    groups = collections.defaultdict(lambda: dict(evals=0, passes=0, scored=0,
        first_sum=0., last_sum=0., control_n=0, target_minus_control_sum=0., collisions=0))
    prompts = {r['id']: r for r in lines(run / 'audit_prompts.jsonl.gz')}
    def key(r, h, block):
        return (h[0], h[1], r['family'], r['id'], r['fact_block'], block)
    for r in lines(run / 'ri_opportunities.jsonl.gz'):
        for h in selected:
            if r['layer'] == h[0]:
                for block, n in r['positions_by_block'].items():
                    groups[key(r, h, block)]['evals'] += n
    examples = collections.defaultdict(list)
    signatures = collections.defaultdict(set)
    for r in lines(run / 'ri_events.jsonl.gz'):
        h = (r['layer'], r['head'])
        if h not in selected:
            continue
        a = groups[key(r, h, r['position_block'])]
        a['passes'] += 1
        if not r['scored']:
            continue
        a['scored'] += 1
        a['first_sum'] += r['strength_first']; a['last_sum'] += r['strength_last']
        controls = [c['score'] for c in r['entity_controls'].values()
                    if c['score'] is not None and not c['token_collision']]
        a['collisions'] += sum(c['token_collision'] for c in r['entity_controls'].values())
        if controls:
            a['control_n'] += 1
            a['target_minus_control_sum'] += r['strength_first'] - sum(controls)/len(controls)
        p = prompts[r['id']]
        # Causal decoder state before j is determined by this token prefix.
        # Count exact repeated measurement conditions separately from raw events.
        sig = (tuple(p['token_ids'][:r['j']+1]), r['fact'], r['target_first_id'], r['target_last_id'])
        signatures[h].add(sig)
        item = dict(r, fact_text=p['facts'][r['fact']]['line'],
                    current_token=p['tokens'][r['j']],
                    target=p['facts'][r['fact']]['tail'])
        examples[h].append(item)
    output = []
    for k, a in sorted(groups.items()):
        output.append(dict(layer=k[0], head=k[1], family=k[2], id=k[3],
            fact_block=k[4], position_block=k[5], **a,
            frequency=a['passes']/a['evals'] if a['evals'] else None,
            strength_first=a['first_sum']/a['scored'] if a['scored'] else None,
            strength_last=a['last_sum']/a['scored'] if a['scored'] else None))
    (run / 'candidate_breakdown.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    summary = {}
    for h in sorted(selected):
        subset = [r for r in output if (r['layer'],r['head']) == h]
        fam = collections.defaultdict(lambda: [0.,0])
        for r in subset:
            fam[r['family']][0] += r['first_sum']; fam[r['family']][1] += r['scored']
        scored_fam = [s/n for s,n in fam.values() if n]
        summary[f'L{h[0]}H{h[1]}'] = dict(
            families_evaluated=len(fam), families_with_scored_events=len(scored_fam),
            raw_scored_events=sum(r['scored'] for r in subset),
            unique_prefix_fact_conditions=len(signatures[h]),
            mean_of_family_conditional_means=sum(scored_fam)/len(scored_fam) if scored_fam else None,
            top_events=sorted(examples[h], key=lambda r:r['strength_first'], reverse=True)[:20])
    (run / 'candidate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path)
    args = ap.parse_args(); summarize(args.run)
