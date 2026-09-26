"""Final freeze manifest: the only key that opens the sealed held-out families.

The manifest lists every discovery stage (schedule, run manifests, analysis outputs) with
hashes, the chosen background and its status, candidate/structure/control/mean identities,
code and data versions, the validation specification, primary signs and thresholds, and a
signature over all of it. Validation code refuses anything that does not validate here.
"""
import hashlib
import json
from pathlib import Path
from stage3_common import digest,json_read,json_write
from s45_plan import POLICY,STRUCTURES,C33,C50,CANDIDATES,REFERENCES,ATTACHMENTS,CELLS

REQUIRED_STAGES=['means','stage_a','stage_b','stage_c','stage_d','full_discovery']

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def signature(body):return hashlib.sha256(canonical(body).encode()).hexdigest()

def build(inputs,out,stages,B,B_status,selection,signs,bank_manifest,code):
    """stages: label -> dict(schedule=path|None, runs=[paths], analysis=[paths])."""
    plan=json_read(inputs/'plan.json');listing={}
    for label,st in stages.items():
        listing[label]=dict(schedule=digest(st['schedule']) if st.get('schedule') else None,
            runs={str(Path(r).name):digest(Path(r)/'manifest.json') for r in st['runs']},
            done=all(json_read(Path(r)/'done.json')['complete'] for r in st['runs']),
            analysis={str(Path(x).name):digest(x) for x in st.get('analysis',[])})
    missing=[s for s in REQUIRED_STAGES if s not in listing or not listing[s]['done']]
    if missing:raise ValueError('Discovery stages incomplete: '+','.join(missing))
    body=dict(policy=POLICY,plan_hash=digest(inputs/'plan.json'),pair_hash=plan['pair_hash'],model=plan['model'],code=code,
        mean_bank=dict(manifest_hash=digest(bank_manifest),mode_discovery='lofo',mode_validation='full'),
        stages=listing,B=dict(members=B,status=B_status,size=len(B)),structures=STRUCTURES,C33=C33,C50=C50,candidates=CANDIDATES,references=REFERENCES,
        attachments=[dict(structures=ts,candidates=hs,attachment=att) for ts,hs,att in ATTACHMENTS],control_rosters=plan['control_rosters'],
        original59=plan['original59'],selection=selection,primary_signs=signs,thresholds=dict(tau=POLICY['tau'],coherent_fraction=POLICY['coherent_fraction'],
        heterogeneous_fraction=POLICY['heterogeneous_fraction'],fidelity_F=POLICY['fidelity_F'],fidelity_L=POLICY['fidelity_L'],accuracy_drop=POLICY['accuracy_drop']),
        validation_spec=dict(populations='87 sealed families, both orders, 174 pairs',cells=CELLS,
            items=['four-cell means: full, empty, B, B-R(B), extra state per selected candidate (<=7)',
                   'selected Gamma pairs: both B_h states, both directions, original axis (<=3)',
                   'selected attachments and matched controls in frozen diagnostic backgrounds, both directions (<=6)',
                   'B, B-R(B), selected toggles under paired-donor replacement, original axis (<=5)'],
            rule='no replacement of failed candidates, no new links, frozen signs and thresholds'),
        held_out=plan['heldout'],frozen_before_heldout=True)
    manifest=dict(body,signature=signature(body))
    json_write(out,manifest);return manifest

def validate_freeze(inputs,freeze,code=None,require_files=True):
    body={k:v for k,v in freeze.items() if k!='signature'}
    if signature(body)!=freeze['signature']:raise ValueError('Freeze signature mismatch')
    plan=json_read(inputs/'plan.json');ref=plan.get('discovery_plan_hash') or digest(inputs/'plan.json')
    if freeze['plan_hash']!=ref or freeze['policy']!=POLICY:raise ValueError('Freeze does not match the frozen discovery plan')
    if freeze['structures']!=STRUCTURES or freeze['C33']!=C33 or freeze['C50']!=C50 or freeze['candidates']!=CANDIDATES:raise ValueError('Roster drift')
    if freeze['control_rosters']!=plan['control_rosters'] or freeze['original59']!=plan['original59']:raise ValueError('Control/RI identity drift')
    if not freeze['frozen_before_heldout'] or not freeze['held_out']['sealed']:raise ValueError('Held-out not sealed at freeze time')
    missing=[s for s in REQUIRED_STAGES if s not in freeze['stages'] or not freeze['stages'][s]['done']]
    if missing:raise ValueError('Freeze lists incomplete discovery stages: '+','.join(missing))
    if freeze['B']['status'] not in ['pass','partial'] or sorted(freeze['B']['members'])!=sorted(freeze['B']['members']):raise ValueError('B status')
    sel=freeze['selection']
    if len(sel['selected'])>POLICY['max_selected'] or len({s['candidate'] for s in sel['selected']})>POLICY['max_selected']:raise ValueError('Selection exceeds caps')
    if code is not None and any(freeze['code'].get(k)!=v for k,v in code.items() if k in freeze['code']):raise ValueError('Code changed since freeze')
    return True
