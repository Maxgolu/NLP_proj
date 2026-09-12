"""Family-blocked matched-target randomization, no model or GPU required.

Primary population: scored QK events for TEST facts whose tail is one of the
two answer candidates. Both candidate mentions must be fully visible, distinct
in first token, and equal in token length at their fact occurrences.

This is a NEW conditional test-target statistic, not the original pooled RI.
Its randomization interpretation assumes true/control assignment is exchangeable
within each family under the null, conditional on fixed QK selection and scores.
It is not a causal test. Shared ICL demonstrations are not independent samples.
"""
import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
from analyze_stage1_audit import lines
from stage1_audit import file_hash


def holm_adjust(p):
    """Holm step-down adjustment over ALL heads, including unsupported p=1."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p); result[order] = adjusted
    return result


def span_tokens(prompt, span):
    return [i for i, (a, b) in enumerate(prompt['offsets']) if a < span[1] and b > span[0]]


def collect(run, n_layers, n_heads):
    prompts = {p['id']: p for p in lines(run / 'audit_prompts.jsonl.gz')}
    if not prompts or any('candidates' not in p for p in prompts.values()):
        raise ValueError('Use the updated scan: audit_prompts must include candidates.')
    totals = collections.defaultdict(lambda: np.zeros(3, dtype=float))
    coverage = collections.defaultdict(collections.Counter)
    pair_by_family = {}
    for e in lines(run / 'ri_events.jsonl.gz'):
        h = e['layer'] * n_heads + e['head']
        if not (0 <= e['layer'] < n_layers and 0 <= e['head'] < n_heads):
            raise ValueError('Head dimensions do not match this run')
        c = coverage[h]; c['all_qk_events'] += 1
        if e['fact_block'] != 'test' or e['position_block'] != 'test':
            c['outside_test_scope'] += 1; continue
        if not e['scored']:
            c['unscored'] += 1; continue
        p = prompts[e['id']]
        pair = tuple(sorted(p['candidates']))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError('Expected two distinct answer names')
        if pair_by_family.setdefault(e['family'], pair) != pair:
            raise ValueError('Candidate names changed within family')
        fact = p['facts'][e['fact']]
        target = fact['tail']
        if target not in pair:
            c['noncandidate_fact'] += 1; continue
        other = next(n for n in pair if n != target)
        candidates = [f for f in p['facts'] if f['block'] == 'test' and f['tail'] == other]
        if len(candidates) != 1:
            c['ambiguous_control_mention'] += 1; continue
        ti = span_tokens(p, fact['tail_span']); ci = span_tokens(p, candidates[0]['tail_span'])
        if not ti or not ci or max(ti[-1],ci[-1]) > e['j']:
            c['not_fully_visible'] += 1; continue
        if len(ti) != len(ci):
            c['token_length_mismatch'] += 1; continue
        ctrl = e['entity_controls'].get(other)
        if ctrl is None or ctrl['score'] is None:
            c['missing_control_score'] += 1; continue
        if ctrl['token_collision'] or p['token_ids'][ti[0]] == p['token_ids'][ci[0]]:
            c['first_token_collision'] += 1; continue
        if ctrl['token_id'] != p['token_ids'][ci[0]] or e['target_first_id'] != p['token_ids'][ti[0]]:
            raise ValueError('Token anchor mismatch in saved event')
        vals = np.array([e['strength_first'],ctrl['score']], dtype=float)
        if not np.isfinite(vals).all() or np.any(vals < 0) or np.any(vals > 1):
            raise ValueError('Invalid saved score')
        totals[(h,e['family'])] += [vals[0],vals[1],1]
        c['matched_events'] += 1
    return totals, coverage


def calibrate(totals, total_heads=1024, draws=100000, seed=20260913,
              min_families=10, min_events=50, alpha=.05):
    """Equal weight per active family; a shared family coin for all events/heads.

    Matrix multiplication is batched over 16 heads. No B x H array is retained.
    Missing families do not count as zero-score evidence for a head.
    """
    if draws < 1 or min_families < 1 or min_events < 1:
        raise ValueError('Draws and minimum support must be positive')
    families = sorted({f for h,f in totals})
    fi = {f:i for i,f in enumerate(families)}
    true = np.zeros((len(families),total_heads),dtype=float)
    other = np.zeros_like(true); active = np.zeros_like(true,dtype=bool)
    counts = np.zeros(total_heads,dtype=int)
    table = []
    for (h,f), (a,b,n) in sorted(totals.items()):
        if not n: continue
        true[fi[f],h] = a/n; other[fi[f],h] = b/n; active[fi[f],h] = True
        counts[h] += int(n)
        table.append(dict(head_index=h,family=f,n_events=int(n),true_mean=a/n,control_mean=b/n))
    nf = active.sum(axis=0)
    eligible = (nf >= min_families) & (counts >= min_events)
    denom = np.maximum(nf,1)
    observed = true.sum(axis=0)/denom
    control = other.sum(axis=0)/denom
    center = (observed+control)/2
    weights = (true-other)/(2*denom)
    # One coin per family per draw, reused for every head and all six twins.
    rng = np.random.default_rng(seed)
    signs = rng.integers(0,2,size=(draws,len(families)),dtype=np.int8).astype(float)
    signs *= 2; signs -= 1
    raw = np.ones(total_heads)
    summaries = {}
    indices = np.flatnonzero(eligible)
    for start in range(0,len(indices),16):
        hs = indices[start:start+16]
        null = signs @ weights[:,hs] + center[hs]
        tol = 100*np.finfo(float).eps*np.maximum(1.,np.abs(observed[hs]))
        raw[hs] = (1+np.sum(null >= observed[hs]-tol,axis=0))/(draws+1)
        quantiles = np.percentile(null,[95,99,99.9],axis=0)
        for j,h in enumerate(hs):
            summaries[int(h)] = dict(null_mean=float(null[:,j].mean()),
                null_std=float(null[:,j].std()), null_p95=float(quantiles[0,j]),
                null_p99=float(quantiles[1,j]),null_p99_9=float(quantiles[2,j]),
                p_randomization=float(raw[h]))
    adjusted = holm_adjust(raw)
    rows = []
    for h in range(total_heads):
        row = dict(head_index=h,matched_events=int(counts[h]),families=int(nf[h]),
            status='tested' if eligible[h] else 'insufficient_support',
            true_mean=float(observed[h]) if nf[h] else None,
            control_mean=float(control[h]) if nf[h] else None,
            effect=float(observed[h]-control[h]) if nf[h] else None,
            p_randomization=None,p_holm=None,selected=False,
            null_mean=None,null_std=None,null_p95=None,null_p99=None,null_p99_9=None)
        if eligible[h]:
            row.update(summaries[h],p_holm=float(adjusted[h]),
                       selected=bool(adjusted[h] <= alpha and observed[h] > control[h]))
        rows.append(row)
    return rows,table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('--draws',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=20260913)
    ap.add_argument('--min-families',type=int,default=10)
    ap.add_argument('--min-events',type=int,default=50)
    ap.add_argument('--alpha',type=float,default=.05)
    ap.add_argument('--output-name',default='null_calibration')
    args = ap.parse_args()
    if Path(args.output_name).name != args.output_name:
        ap.error('output-name must be a directory name, not a path')
    out = args.run/args.output_name; out.mkdir(exist_ok=False)
    start=time.monotonic()
    with (args.run/'head_stats.csv').open() as f: heads=list(csv.DictReader(f))
    nl=1+max(int(r['layer']) for r in heads); nh=1+max(int(r['head']) for r in heads)
    print('Reading saved events; no model loading.',flush=True)
    totals,coverage=collect(args.run,nl,nh)
    print(f'Randomizing {len(totals)} head/family groups, {args.draws} draws.',flush=True)
    rows,table=calibrate(totals,nl*nh,args.draws,args.seed,args.min_families,args.min_events,args.alpha)
    for r in rows:
        r['layer'],r['head']=divmod(r['head_index'],nh)
    with (out/'head_null_calibration.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'family_scores.json').write_text(json.dumps(table,indent=1))
    (out/'coverage.json').write_text(json.dumps(dict(coverage),indent=1))
    metadata=dict(draws=args.draws,seed=args.seed,min_families=args.min_families,
        min_events=args.min_events,alpha=args.alpha,total_heads=nl*nh,
        statistic='equal-family mean of first-target score, matched test events only',
        scope='test fact -> test position; tail among answer candidates; both full mentions visible; equal token lengths; distinct first tokens',
        randomization='one fair true/control swap per family, shared across all events, variants, orders and heads',
        correction='Holm across all heads; unsupported heads assigned p=1 internally',
        assumptions='conditional exchangeability of true/control assignment within family under the null; fixed QK events; not a causal test',
        limitations='subset statistic, not pooled RI; name/token preferences and working-set selection can limit exchangeability; demos descriptive only',
        hashes={name:file_hash(args.run/name) for name in ('ri_events.jsonl.gz','audit_prompts.jsonl.gz','head_stats.csv')},
        script_sha256=file_hash(__file__),elapsed_s=round(time.monotonic()-start,2))
    (out/'method.json').write_text(json.dumps(metadata,indent=1))
    print(f'Done: {sum(r["status"]=="tested" for r in rows)} tested; '
          f'{sum(r["selected"] for r in rows)} Holm-selected. Outputs: {out}',flush=True)


if __name__ == '__main__': main()
