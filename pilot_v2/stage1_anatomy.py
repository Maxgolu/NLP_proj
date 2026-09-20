"""CPU-only anatomy of saved QK passes; independent of stage 2."""
import argparse
import collections
import csv
import json
from pathlib import Path
from analyze_stage1_audit import lines
from stage1_audit import file_hash


def distance_bucket(j, source):
    d = j-source
    if d < 0: raise ValueError('Current position before source')
    return 'self' if d == 0 else 'previous' if d == 1 else 'distance_ge_2'


def position_bucket(p, j):
    if j == len(p['token_ids'])-1: return 'answer_prediction'
    start = p['prompt'].rfind('\n\n')+2 if '\n\n' in p['prompt'] else 0
    return 'test_other' if p['offsets'][j][1] > start else 'demo'


def analyze(run, out):
    out.mkdir(parents=True, exist_ok=False)
    prompts = {p['id']:p for p in lines(run/'audit_prompts.jsonl.gz')}
    denominators = collections.Counter()
    anchors = {}
    for p in prompts.values():
        for fi, f in enumerate(p['facts']):
            def anchor(span):
                ids=[i for i,(a,b) in enumerate(p['offsets']) if a<span[1] and b>span[0]]
                if not ids: raise ValueError('Missing fact anchor')
                return ids[-1]
            s=anchor(f['head_span']); o=anchor(f['tail_span'])
            anchors[(p['id'],fi)]=s
            for j in range(max(s,o),len(p['token_ids'])):
                denominators[(f['block'],position_bucket(p,j),distance_bucket(j,s))]+=1
    stats=collections.defaultdict(lambda:dict(passes=0,scored=0,first_sum=0.,last_sum=0.,families=set()))
    for e in lines(run/'ri_events.jsonl.gz'):
        p=prompts[e['id']]
        scope=(e['fact_block'],position_bucket(p,e['j']),distance_bucket(e['j'],anchors[(e['id'],e['fact'])]))
        a=stats[(e['layer'],e['head'],*scope)]
        a['passes']+=1; a['families'].add(e['family'])
        if e['scored']:
            a['scored']+=1; a['first_sum']+=e['strength_first']; a['last_sum']+=e['strength_last']
    with (run/'head_stats.csv').open() as f: original=list(csv.DictReader(f))
    heads=sorted({(int(r['layer']),int(r['head'])) for r in original})
    rows=[]
    for h in heads:
        for scope,n in sorted(denominators.items()):
            a=stats[(*h,*scope)]
            rows.append(dict(layer=h[0],head=h[1],fact_block=scope[0],position=scope[1],
                distance=scope[2],evals=n,qk_passes=a['passes'],scored=a['scored'],
                families=len(a['families']),frequency=a['passes']/n,
                strength_first=a['first_sum']/a['scored'] if a['scored'] else None,
                strength_last=a['last_sum']/a['scored'] if a['scored'] else None))
    # Reconcile partition counts with the original pooled scan, including zeros.
    for h in heads:
        old=[r for r in original if (int(r['layer']),int(r['head']))==h]
        new=[r for r in rows if (r['layer'],r['head'])==h]
        for a,b in [('evals','evals'),('qk_passes','qk_passes'),('n_a','scored')]:
            if sum(int(r[a]) for r in old)!=sum(r[b] for r in new):
                raise ValueError(f'Partition reconciliation failed: {h} {a}')
    with (out/'head_anatomy.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=collections.Counter()
    for r in rows:summary[(r['fact_block'],r['position'],r['distance'])]+=r['qk_passes']
    text=['# Stage 1 saved-event anatomy','',
        'Only QK-passing events are characterized. The answer position is the last',
        'token of the bare prompt, before any shared answer prefix used by stage 2.',
        'Distance is current position minus the annotated source anchor.', '',
        '| Fact | Current position | Distance | QK passes |','|---|---|---|---:|']
    text += [f'| {k[0]} | {k[1]} | {k[2]} | {n} |' for k,n in sorted(summary.items())]
    (out/'summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(dict(run=str(run),
        files={n:file_hash(run/n) for n in ['audit_prompts.jsonl.gz','ri_events.jsonl.gz','head_stats.csv']},
        script_sha256=file_hash(__file__),partition_reconciliation='passed'),indent=2))
    print(f'Anatomy complete: {out}; partition counts match the original scan.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();analyze(a.run,a.out)
