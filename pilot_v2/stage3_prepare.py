"""Prepare portable Stage-3 inputs and correct two saved-analysis diagnostics.

Reads original runs; writes a new directory. No model load or old-result edits.
"""
import argparse
import collections
import csv
import tarfile
from pathlib import Path
import numpy as np
from stage3_common import *

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--out',type=Path,default=ROOT/'results/stage3_inputs_v1')
    a=ap.parse_args(); R=a.results; out=a.out
    if out.exists(): raise ValueError('Output exists. Use a new --out; original preparation stays immutable.')
    inv=json_read(R/'stage3_plan/inventory_policy.json'); heads=inv['selected_flat_ids']
    if len(heads)!=105: raise ValueError('Unexpected inventory')
    v1=json_read(R/'stage2_v1/manifest.json'); v2=json_read(R/'stage2_v2_extension/manifest.json')
    for run in ['stage2_v1','stage2_v2_extension']:
        if not json_read(R/run/'summary.json')['complete']: raise ValueError('Incomplete source run')
    if v1['model']!=v2['model']: raise ValueError('Model identities differ')
    pairs=v1['pairs']; families=sorted({p['family'] for p in pairs})
    if len(pairs)!=178 or len(families)!=89 or any(f%2 for f in families): raise ValueError('Discovery split mismatch')
    for p in pairs:
        if p['clean']['split']!='discovery' or p['corr']['split']!='discovery': raise ValueError('Validation leak')
    D=json_read(R/'stage2_v1/shortlist.json')['heads']; hp=v2['scopeP_heads']; hf=v2['scopeF_heads']
    rows=[]; attr=[]; exact=[]; scopeP=[]; scopeF=[]; logits=[]; baselines=[]
    # Read the saved baseline directly from its archive; no inference is repeated.
    with tarfile.open(R/'singlehop_results.tar.gz') as archive:
        baseline_bytes=archive.extractfile('olmo2_singlehop_4shot/results.jsonl').read()
    if hashlib.sha256(baseline_bytes).hexdigest()!=v1['input_hashes'][v1['baseline']]:
        raise ValueError('Behavioral baseline archive does not match Stage 2')
    baseline={r['id']:r['gold_minus_best_other'] for r in map(json.loads,baseline_bytes.decode().splitlines())}
    gate_ids={r['id'] for p in pairs if p['family'] in v1['gate_families'] for r in [p['clean'],p['corr']]}
    baseline={k:baseline[k] for k in sorted(gate_ids)}
    for p in pairs:
        name=f"family_{p['family']:03d}_order{p['order']}.npz"
        with np.load(R/'stage2_v1/screen'/name) as z:
            if str(z['pair_id'])!=p['id']: raise ValueError('Pair identity mismatch')
            attr.append(z['attribution'].copy())
            exact.append(z['exact_delta'].copy() if p['exact_all'] else np.full(1024,np.nan))
            baselines.append([float(z['m_clean']),float(z['m_corr_fixed_sign'])])
        P={};F={}; L={}
        with np.load(R/'stage2_v1/extension'/name) as z:
            if list(z['head_indices'])!=D: raise ValueError('v1 head IDs differ')
            P.update(zip(D,z['exact_delta']))
        with np.load(R/'stage2_v2_extension/pairs'/name) as z:
            if str(z['pair_id'])!=p['id'] or list(z['scopeP_heads'])!=hp or list(z['scopeF_heads'])!=hf: raise ValueError('v2 pair/head mismatch')
            P.update(zip(hp,z['scopeP_delta']));F.update(zip(hf,z['scopeF_delta']))
            L.update(zip(hf,z['scopeF_readout'][:,:3]))
            clean_logits=z['clean_logits'][:2].copy()
        scopeP.append([P[h] for h in heads]);scopeF.append([F.get(h,np.nan) for h in heads])
        logits.append([L.get(h,[np.nan]*3) for h in heads])
        rows.append(dict(id=p['id'],family=p['family'],order=p['order'],exact_all=p['exact_all'],
                         clean=p['clean'],corr=p['corr'],shared_prefix=v1['shared_prefixes'][p['id']],clean_logits=clean_logits.tolist()))
    A=np.array(attr); E=np.array(exact); P=np.array(scopeP); F=np.array(scopeF)
    common=np.array([p['exact_all'] for p in pairs]); fam=np.array([p['family'] for p in pairs])
    if not np.isfinite(A).all() or not np.isfinite(E[common]).all() or not np.isfinite(P).all(): raise ValueError('Nonfinite required measurements')
    prompts=list(read_lines(R/'ri_test_v2/prompts.jsonl.gz')); byid={p['id']:p for p in prompts}
    if len(byid)!=534 or {p['family'] for p in prompts}!=set(families): raise ValueError('RI prompt coverage differs')
    for p in pairs:
        for r in [p['clean'],p['corr']]:
            if byid[r['id']]['prompt']!=r['prompt']: raise ValueError('RI and causal inputs differ')
    sources={str(p.relative_to(R)):digest(p) for p in [R/'stage2_v1/manifest.json',R/'stage2_v2_extension/manifest.json',R/'stage3_plan/inventory_policy.json',R/'ri_test_v2/candidates.json',R/'ri_test_v2/prompts.jsonl.gz']}
    with (R/'ri_test_v2/head_summary.csv').open(encoding='utf-8',newline='') as f:
        support={int(r['layer'])*32+int(r['head']):int(r['passes']) for r in csv.DictReader(f)
                 if r['scope']=='all_facts' and r['position']=='all_test' and r['distance']=='all'
                 and r['variant']=='pooled' and r['anchor']=='first'}
    sources['ri_test_v2/head_summary.csv']=digest(R/'ri_test_v2/head_summary.csv')
    events=[]; concentration=collections.defaultdict(list)
    for h in heads:
        path=R/'ri_test_v2/heads'/f'{head_name(h)}.jsonl.gz'
        if not path.exists():
            if support.get(h)!=0: raise FileNotFoundError(path)
            continue  # Explicitly verified zero-pass head, not missing data.
        sources[str(path.relative_to(R))]=digest(path)
        for e in read_lines(path):
            events.append(e)
            if not e['scores']: continue
            prompt=byid[e['id']]; start,end=prompt['offsets'][e['j']]
            def overlap(name):
                return any(m.start()<end and m.end()>start for m in re.finditer(r'\b'+re.escape(name)+r'\b',prompt['prompt']))
            for anchor,s in e['scores'].items():
                if not s['names']: continue
                role='target' if overlap(e['target']) else ('control' if any(overlap(c['text']) for c in s['names']) else 'neither')
                concentration[(h,anchor,role,e['family'])].append([s['target'],s['names_stats']['mean'],s['names_stats']['gap']])
    out.mkdir(parents=True)
    write_lines(out/'pairs.jsonl.gz',rows); write_lines(out/'prompts.jsonl.gz',prompts)
    write_lines(out/'events.jsonl.gz',events)
    json_write(out/'behavioral_reference.json',baseline)
    npz_write(out/'references.npz',heads=heads,scopeP_delta=P,scopeF_delta=F,scopeF_readout=logits,
              baselines=baselines,attribution=A,exact=E,families=fam,common=common)
    with (out/'matched_approximation_error.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['head','exact_importance_40','estimated_importance_40','absolute_error_40'])
        for h in range(1024):
            x,y=-E[common,h].mean(),-A[common,h].mean();w.writerow([head_name(h),x,y,abs(x-y)])
    controls=json_read(R/'stage2_v1/shortlist.json')['random_controls']
    cands=json_read(R/'ri_test_v2/candidates.json')['candidates']; ri=[c['layer']*32+c['head'] for c in cands]
    corrected=bootstrap_median_difference(family_means(-P[:,[heads.index(h) for h in ri]],fam),
                                        family_means(-P[:,[heads.index(h) for h in controls]],fam))
    json_write(out/'corrected_median_comparison.json',corrected)
    conc=[]
    for (h,anchor,role),_ in collections.Counter(k[:3] for k in concentration).items():
        chunks=[v for k,v in concentration.items() if k[:3]==(h,anchor,role)]
        means=np.array([np.mean(x,axis=0) for x in chunks]).mean(0)
        conc.append(dict(head=head_name(h),anchor=anchor,current_role=role,events=sum(map(len,chunks)),
                         families=len(chunks),target=float(means[0]),control=float(means[1]),gap=float(means[2])))
    json_write(out/'current_name_diagnostics.json',conc)
    json_write(out/'inventory.json',inv)
    files={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    json_write(out/'plan.json',dict(policy=POLICY,heads=heads,model=v1['model'],source_hashes=sources,
                files=files,common_families=sorted(set(fam[common].tolist())),
                reverse_heads=[int(s[1:].split('H')[0])*32+int(s.split('H')[1]) for s in inv['reverse_patch_heads']],
                gate_families=v1['gate_families'],pairs=178,prompts=534,events=len(events),
                source_baseline=v1['baseline'],source_run_name='stage2_v1'))
    print(f'Prepared {out}: {len(heads)} heads, {len(events)} saved RI events; corrected diagnostics written separately.')


if __name__=='__main__': main()
