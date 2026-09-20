"""Stage-2 extension v2: targeted exact head patching for the updated Stage-1 candidates.

Reads the completed, immutable stage2_v1 run (manifest, pairs, gate families, per-pair
screen files) and a fixed head list (extension_v2_heads.json, produced by
stage2_coverage_audit_v2.py). It computes, with the same model, data, metric and
patch mechanics as stage2_v1:

  Scope P  exact all-prompt-position patching for `scopeP_heads_138_pairs`
           on the pairs that were NOT in the v1 all-head exact subset
           (v1 already has these heads on the 40 common pairs).
  Scope F  exact final-prompt-position patching for `scopeF_heads_178_pairs`
           on all 178 pairs.

For every patched forward it stores the contrast M and, separately, the clean-answer
logit, the corrupted-answer logit and the source-name (child) logit.

Nothing under the source run is modified. stage2_engine.py / stage2_run.py are imported,
not edited, so the v1 code hashes stay valid.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
from stage1_audit import file_hash
from stage2_common import atomic_json, groups
from stage2_engine import Engine

ROOT = Path(__file__).resolve().parent
CODE_FILES = ['stage2_extension_v2.py', 'stage2_engine.py', 'stage2_common.py', 'stage1_scan.py', 'stage1_audit.py']
H = 32


class EngineV2(Engine):
    """Adds position-restricted patching and separate logit readout."""

    def positional_patch(self, li, hi, donor, positions):
        import contextlib
        positions = list(positions)
        @contextlib.contextmanager
        def ctx():
            def hook(mod, inp):
                z = inp[0]; n = donor.shape[1]
                if n > z.shape[1] or max(positions) >= n:
                    raise ValueError('Patch positions must lie inside the original prompt')
                result = z.clone(); sl = slice(hi * self.DH, (hi + 1) * self.DH)
                idx = self.t.tensor(positions, device=z.device)
                result[:, idx, sl] = donor[:, idx, sl].to(z.device, z.dtype)
                return (result,)
            h = self.layers[li].self_attn.o_proj.register_forward_pre_hook(hook)
            try: yield
            finally: h.remove()
        return ctx()

    def readout(self, ids, g, d, s):
        with self.t.no_grad():
            logits = self.model(ids, use_cache=False).logits[0, -1].float()
            return float(logits[g] - logits[d]), float(logits[g]), float(logits[d]), float(logits[s])

    def exact_positions(self, spec, donor, heads, positions, s):
        """Return array [len(heads), 4] = (M, logit_g, logit_d, logit_s) per patched forward."""
        out = np.zeros((len(heads), 4)); self.sync(); start = time.monotonic()
        for i, (li, hi) in enumerate(heads):
            with self.positional_patch(li, hi, donor[li], positions):
                out[i] = self.readout(spec['clean'], spec['g'], spec['d'], s)
            if len(heads) >= 64 and (i + 1) % 64 == 0: print(f'  heads {i+1}/{len(heads)}', flush=True)
        if not np.isfinite(out).all(): raise ValueError('Non-finite patched readout')
        self.sync()
        return out, time.monotonic() - start


def first_diff(spec):
    a = spec['clean'][0, :spec['n']].tolist(); b = spec['corr'][0, :spec['n']].tolist()
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


def source_token(tok, row):
    ids = tok.encode(' ' + row['question_entity'], add_special_tokens=False)
    return ids[0]


def pair_file(out, p):
    return out / 'pairs' / f'family_{p["family"]:03d}_order{p["order"]}.npz'


def save_npz(path, **data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **data)
    temp.replace(path)


def plan(pairs, heads_p, heads_f, replicas):
    """Whole families per replica, balanced by expected patched forwards."""
    fams = sorted({p['family'] for p in pairs})
    cost = {f: sum((0 if p['exact_all'] else len(heads_p)) + len(heads_f) for p in pairs if p['family'] == f) for f in fams}
    loads = [0] * replicas; assign = {}
    for f in sorted(fams, key=lambda f: (-cost[f], f)):
        i = min(range(replicas), key=lambda i: (loads[i], i)); assign[f] = i; loads[i] += cost[f]
    return [dict(p, worker=assign[p['family']]) for p in pairs], loads


# --------------------------------------------------------------------------------------
def worker(out, rank):
    from stage1_scan import load_model, runner_score
    m = json.loads((out / 'manifest.json').read_text()); pairs = m['pairs']
    src = Path(m['source_run'])
    heads_p = [divmod(h, H) for h in m['scopeP_heads']]; heads_f = [divmod(h, H) for h in m['scopeF_heads']]
    def state(status, **kw): atomic_json(out / f'state_{rank}.json', dict(status=status, **kw))
    if rank:
        state('waiting for previous replica load')
        while not (out / f'loaded_{rank-1}.json').exists(): time.sleep(1)
    state('loading model'); print(f'Replica {rank}: loading pinned model.', flush=True)
    start = time.monotonic(); torch, tok, model, lock = load_model()
    if lock != m['model']: raise ValueError('Model lock differs from manifest')
    e = EngineV2(torch, tok, model)
    if (e.L, e.H) != (32, 32): raise ValueError('Expected 32 x 32 heads')
    atomic_json(out / f'loaded_{rank}.json', dict(seconds=time.monotonic() - start,
        visible=os.environ.get('CUDA_VISIBLE_DEVICES'), device_map={k: str(v) for k, v in model.hf_device_map.items()}))
    for p in pairs:
        spec = e.encode(p)
        if spec['shared_prefix'] != m['shared_prefixes'][p['id']]: raise ValueError('Token specification mismatch: ' + p['id'])

    # ---- gate: v1 gate + v2 checks ------------------------------------------------------
    state('running gate')
    refs = {json.loads(l)['id']: json.loads(l)['gold_minus_best_other'] for l in open(m['baseline'], encoding='utf-8')}
    gate_pairs = [p for p in pairs if p['family'] in m['gate_families']]
    gate = e.gate(gate_pairs, refs, runner_score)
    v2 = []
    probe = [(3, 11), (9, 22), (16, 4), (16, 21), (16, 1), (27, 6)]
    for p in gate_pairs:
        spec, clean, corr, m0, mc = e.prepare(p); n = spec['n']; k = first_diff(spec); s = source_token(tok, p['clean'])
        rep = dict(id=p['id'], n=n, first_diff=k, source_token=s)
        # (a) Scope-F self patch: donor == recipient at the final position -> exactly the clean metric.
        selfF, _ = e.exact_positions(spec, clean, probe, [n - 1], s)
        rep['scopeF_self_patch_max'] = float(np.max(np.abs(selfF[:, 0] - m0)))
        # (b) Patching corrupted activations at positions BEFORE the first differing token must be neutral
        #     (causal attention on an identical prefix). fp16 tolerance 1e-3.
        pre, _ = e.exact_positions(spec, corr, probe, list(range(k)), s)
        rep['prefix_patch_max'] = float(np.max(np.abs(pre[:, 0] - m0)))
        # (c) Scope-P replication of v1 for pairs in the v1 exact subset.
        if p['exact_all']:
            with np.load(src / 'screen' / f'family_{p["family"]:03d}_order{p["order"]}.npz') as z:
                old = z['exact_delta']
            full, _ = e.exact_positions(spec, corr, probe, list(range(n)), s)
            rep['v1_replication_max'] = float(np.max(np.abs((full[:, 0] - m0) - old[[li * H + hi for li, hi in probe]])))
        # (d) readout consistency: M equals logit_g - logit_d.
        rep['readout_consistency'] = float(np.max(np.abs(pre[:, 0] - (pre[:, 1] - pre[:, 2]))))
        # (e) prefix positions are identical activations (direct check of the algebraic claim).
        rep['prefix_activation_max_abs_diff'] = float(max((clean[li][:, :k] - corr[li][:, :k]).abs().max().item() for li in range(e.L)))
        v2.append(rep)
        if rep['scopeF_self_patch_max'] >= 1e-3: raise ValueError('Scope-F self patch failed: ' + p['id'])
        if rep['prefix_patch_max'] >= 1e-3: raise ValueError('Prefix patch not neutral: ' + p['id'])
        if rep['readout_consistency'] >= 1e-4: raise ValueError('Readout inconsistency: ' + p['id'])
        if 'v1_replication_max' in rep and rep['v1_replication_max'] >= m['replication_tolerance']:
            raise ValueError('v1 replication failed: ' + p['id'])
        print(f'Gate v2 pair passed: {p["id"]} {rep}', flush=True)
        del clean, corr
    gate['v2'] = v2
    atomic_json(out / f'gate_{rank}.json', gate); state('gate passed; waiting for all replicas')
    while not (out / 'gate_passed.json').exists(): time.sleep(1)

    # ---- main work ----------------------------------------------------------------------
    assigned = [p for p in pairs if p['worker'] == rank]
    for i, p in enumerate(assigned):
        path = pair_file(out, p)
        if path.exists():
            with np.load(path) as z:
                if not (np.array_equal(z['scopeP_heads'], m['scopeP_heads']) and np.array_equal(z['scopeF_heads'], m['scopeF_heads'])):
                    raise ValueError('Cached pair file has a different head list')
            continue
        state('patching', pair=p['id'], index=i + 1, total=len(assigned))
        print(f'Replica {rank}: {i+1}/{len(assigned)} {p["id"]}', flush=True)
        spec, clean, corr, m0, mc = e.prepare(p); n = spec['n']; s = source_token(tok, p['clean'])
        base = e.readout(spec['clean'], spec['g'], spec['d'], s)
        if abs(base[0] - m0) >= 1e-3: raise ValueError('Baseline readout mismatch: ' + p['id'])
        data = dict(pair_id=p['id'], family=p['family'], order=p['order'], n=n, first_diff=first_diff(spec),
                    m_clean=m0, m_corr_fixed_sign=mc, clean_logits=np.array(base[1:]), source_token=s,
                    scopeP_heads=np.array(m['scopeP_heads']), scopeF_heads=np.array(m['scopeF_heads']),
                    in_v1_exact_subset=bool(p['exact_all']))
        start = time.monotonic()
        if p['exact_all']:
            with np.load(src / 'screen' / f'family_{p["family"]:03d}_order{p["order"]}.npz') as z:
                data['scopeP_delta'] = z['exact_delta'][m['scopeP_heads']]
            data['scopeP_readout'] = np.full((len(heads_p), 4), np.nan)  # v1 stored only the contrast
        else:
            P, secP = e.exact_positions(spec, corr, heads_p, list(range(n)), s)
            data['scopeP_delta'] = P[:, 0] - m0; data['scopeP_readout'] = P; data['scopeP_seconds'] = secP
        F, secF = e.exact_positions(spec, corr, heads_f, [n - 1], s)
        data['scopeF_delta'] = F[:, 0] - m0; data['scopeF_readout'] = F; data['scopeF_seconds'] = secF
        data['seconds'] = time.monotonic() - start
        save_npz(path, **data)
        del clean, corr
    atomic_json(out / f'done_{rank}.json', dict(complete=True)); state('complete')


# --------------------------------------------------------------------------------------
def controller(a):
    import torch
    from stage1_scan import RUNS
    src = RUNS / a.source
    sm = json.loads((src / 'manifest.json').read_text()); ss = json.loads((src / 'summary.json').read_text())
    if not ss.get('complete'): raise ValueError('Source run is not complete')
    if json.loads((ROOT / 'model_lock_olmo2.json').read_text()) != sm['model']: raise ValueError('Model lock differs from source run')
    for path, h in sm['input_hashes'].items():
        if file_hash(path) != h: raise ValueError('Source input changed: ' + path)
    heads = json.loads(Path(a.heads).read_text())
    hp, hf = sorted(heads['scopeP_heads_138_pairs']), sorted(heads['scopeF_heads_178_pairs'])
    if not hp or not hf or max(hp + hf) >= 1024 or min(hp + hf) < 0: raise ValueError('Bad head list')
    v1_ext = set(json.loads((src / 'shortlist.json').read_text())['heads'])
    if set(hp) & v1_ext: raise ValueError('Scope-P list overlaps the v1 extension; those heads are already measured')
    if torch.cuda.device_count() != a.gpus: raise ValueError('GPU count differs from Slurm allocation')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(a.gpus)))).split(',')
    gpu_groups = groups(devices, a.gpus_per_replica)
    pairs, loads = plan(sm['pairs'], hp, hf, len(gpu_groups))
    manifest = dict(version=2, source_run=str(src), source_manifest_hashes=dict(code=sm['code_hashes'], inputs=sm['input_hashes']),
        baseline=sm['baseline'], model=sm['model'], code_hashes={n: file_hash(ROOT / n) for n in CODE_FILES},
        heads_file_hash=file_hash(a.heads), scopeP_heads=hp, scopeF_heads=hf, pairs=pairs,
        shared_prefixes=sm['shared_prefixes'], gate_families=sm['gate_families'], gpu_group_sizes=list(map(len, gpu_groups)),
        expected_forwards_per_replica=loads, replication_tolerance=a.replication_tolerance,
        metric='first divergent token; fixed clean-gold sign (as stage2_v1)',
        scopeP='corrupted head output at all original prompt positions; answer prefix recomputed (as stage2_v1)',
        scopeF='corrupted head output at the last prompt position only; answer prefix recomputed',
        readout='(M, logit_clean_answer, logit_corrupted_answer, logit_source_name) at the first divergent token')
    out = RUNS / a.name
    if out.exists():
        if not a.resume or json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError('Run exists. Resume requires identical code, inputs and settings.')
        for name in ['gate_passed.json', 'summary.json', *[f'{k}_{i}.json' for i in range(len(gpu_groups)) for k in ['loaded', 'gate', 'state', 'done']]]:
            (out / name).unlink(missing_ok=True)
    else:
        out.mkdir(parents=True); atomic_json(out / 'manifest.json', manifest)
    children = []; logs = []
    def wait_for(kind):
        last = 0.
        while True:
            if any(c.poll() not in (None, 0) for c in children): raise RuntimeError('Worker failed; see worker logs.')
            if all((out / f'{kind}_{i}.json').exists() for i in range(len(children))): return
            if any(c.poll() == 0 and not (out / f'{kind}_{i}.json').exists() for i, c in enumerate(children)):
                raise RuntimeError('Worker exited before completing its phase')
            if time.monotonic() - last > 30:
                states = [json.loads((out / f'state_{i}.json').read_text()) if (out / f'state_{i}.json').exists() else {'status': 'starting'} for i in range(len(children))]
                print(kind, states, flush=True); last = time.monotonic()
            time.sleep(1)
    def stop(*_): raise InterruptedError('Job interrupted; use --resume with identical settings.')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for i, g in enumerate(gpu_groups):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g))
            env['OMP_NUM_THREADS'] = str(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')) // len(gpu_groups)))
            env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS']
            f = (out / f'worker_{i}.log').open('a', encoding='utf-8'); logs.append(f)
            children.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(i), '--out', str(out)], env=env, stdout=f, stderr=subprocess.STDOUT))
        wait_for('gate')
        for i in range(len(children)):
            if not json.loads((out / f'gate_{i}.json').read_text()).get('passed'): raise RuntimeError('Gate failed')
        atomic_json(out / 'gate_passed.json', dict(passed=True, replicas=len(children)))
        print('ALL REPLICAS PASSED THE GATE.', flush=True)
        if a.gate_only: return
        wait_for('done')
        summarize(out, manifest)
    finally:
        for c in children:
            if c.poll() is None: c.terminate()
        for c in children:
            try: c.wait(timeout=10)
            except subprocess.TimeoutExpired: c.kill(); c.wait()
        for f in logs: f.close()
        for sig, h in previous.items(): signal.signal(sig, h)


def summarize(out, m):
    import csv
    pairs = m['pairs']; hp, hf = m['scopeP_heads'], m['scopeF_heads']
    P = np.zeros((len(pairs), len(hp))); F = np.zeros((len(pairs), len(hf))); fam = []
    Fr = np.zeros((len(pairs), len(hf), 4))
    for i, p in enumerate(pairs):
        with np.load(pair_file(out, p), allow_pickle=False) as z:
            if str(z['pair_id']) != p['id']: raise ValueError('Pair identity mismatch')
            P[i] = z['scopeP_delta']; F[i] = z['scopeF_delta']; Fr[i] = z['scopeF_readout']
        fam.append(p['family'])
    if not (np.isfinite(P).all() and np.isfinite(F).all()): raise ValueError('Non-finite results')
    fam = np.array(fam); fams = sorted(set(fam))
    def fmean(X):  # average orders within family, then over families
        return np.stack([X[fam == f].mean(axis=0) for f in fams]).mean(axis=0)
    with (out / 'head_effects_v2.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'head', 'flat_id', 'scopeP_importance_178', 'scopeP_pairs', 'scopeF_importance_178', 'scopeF_pairs',
                    'scopeF_mean_dlogit_clean_answer', 'scopeF_mean_dlogit_corrupted_answer', 'scopeF_mean_dlogit_source'])
        base = np.stack([np.load(pair_file(out, p))['clean_logits'] for p in pairs])  # [pairs,3]
        for h in sorted(set(hp) | set(hf)):
            rowP = (-fmean(P[:, hp.index(h)]), len(pairs)) if h in hp else ('', 0)
            if h in hf:
                j = hf.index(h); d = Fr[:, j, 1:] - base
                rowF = (-fmean(F[:, j]), len(pairs), *fmean(d))
            else: rowF = ('', 0, '', '', '')
            w.writerow([h // H, h % H, h, *rowP, *rowF])
    atomic_json(out / 'summary.json', dict(complete=True, pairs=len(pairs), scopeP_heads=len(hp), scopeF_heads=len(hf),
        scopeP_new_forwards=int(sum(len(hp) for p in pairs if not p['exact_all'])), scopeF_forwards=len(pairs) * len(hf),
        importance='minus mean delta; family-level means over 89 families (orders averaged first)',
        note='Scope-P values for the 40 v1 exact-subset pairs are copied from stage2_v1/screen; Scope-F values are all new.'))
    print('EXTENSION V2 COMPLETE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int); ap.add_argument('--out', type=Path)
    ap.add_argument('--name', default='stage2_v2_extension'); ap.add_argument('--source', default='stage2_v1')
    ap.add_argument('--heads', default=str(ROOT / 'extension_v2_heads.json'))
    ap.add_argument('--gpus', type=int, default=6); ap.add_argument('--gpus-per-replica', type=int, default=2)
    ap.add_argument('--replication-tolerance', type=float, default=0.05)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--gate-only', action='store_true')
    a = ap.parse_args()
    if a.worker is not None: worker(a.out, a.worker); return
    import re
    if not re.fullmatch('[A-Za-z0-9_-]+', a.name) or a.name == a.source: ap.error('Invalid run name')
    controller(a)


if __name__ == '__main__': main()
