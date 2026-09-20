"""Post-run verification for stage2_v2_extension. CPU only. Usage: python3 verify_stage2_extension_v2.py RUN_DIR [SOURCE_RUN_DIR]"""
import json, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); src = Path(sys.argv[2]) if len(sys.argv) > 2 else None
m = json.loads((run / 'manifest.json').read_text()); s = json.loads((run / 'summary.json').read_text())
assert s['complete'], 'summary not complete'
hp, hf = m['scopeP_heads'], m['scopeF_heads']; pairs = m['pairs']
problems = []
n_new = 0; secs = []
for p in pairs:
    f = run / 'pairs' / f'family_{p["family"]:03d}_order{p["order"]}.npz'
    if not f.exists(): problems.append('missing ' + f.name); continue
    with np.load(f, allow_pickle=False) as z:
        if str(z['pair_id']) != p['id']: problems.append('id mismatch ' + f.name)
        if z['scopeP_delta'].shape != (len(hp),) or z['scopeF_delta'].shape != (len(hf),): problems.append('shape ' + f.name)
        if not (np.isfinite(z['scopeP_delta']).all() and np.isfinite(z['scopeF_delta']).all()): problems.append('nonfinite ' + f.name)
        R = z['scopeF_readout']
        if np.max(np.abs(R[:, 0] - (R[:, 1] - R[:, 2]))) > 1e-4: problems.append('readout inconsistency ' + f.name)
        if bool(z['in_v1_exact_subset']) != bool(p['exact_all']): problems.append('subset flag ' + f.name)
        if not p['exact_all']:
            n_new += 1; secs.append(float(z['scopeP_seconds']) + float(z['scopeF_seconds']))
            RP = z['scopeP_readout']
            if np.max(np.abs(RP[:, 0] - (RP[:, 1] - RP[:, 2]))) > 1e-4: problems.append('scopeP readout inconsistency ' + f.name)
        elif src is not None:
            with np.load(src / 'screen' / f.name) as old:
                if not np.array_equal(old['exact_delta'][hp], z['scopeP_delta']): problems.append('v1 copy mismatch ' + f.name)
        if int(z['first_diff']) >= int(z['n']): problems.append('first_diff ' + f.name)
gates = sorted(p for p in run.glob('gate_*.json') if p.stem[5:].isdigit())
for g in gates:
    G = json.loads(g.read_text())
    if not G.get('passed'): problems.append(g.name + ' not passed')
    for r in G.get('v2', []):
        for k, tol in [('scopeF_self_patch_max', 1e-3), ('prefix_patch_max', 1e-3), ('readout_consistency', 1e-4)]:
            if r[k] >= tol: problems.append(f'{g.name} {r["id"]} {k}={r[k]}')
print(json.dumps(dict(pairs=len(pairs), new_scopeP_pairs=n_new, expected_new_scopeP_pairs=sum(not p['exact_all'] for p in pairs),
                      scopeP_heads=len(hp), scopeF_heads=len(hf), gates=len(gates),
                      mean_seconds_per_new_pair=float(np.mean(secs)) if secs else None,
                      v1_replication_max=[r.get('v1_replication_max') for g in gates for r in json.loads(g.read_text()).get('v2', []) if 'v1_replication_max' in r],
                      problems=problems), indent=1))
sys.exit(1 if problems else 0)
