"""Read-only loading diagnostics and failure snapshots; never alter RI scores."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def environment(model_dir):
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'safetensors', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(host=platform.node(), python=platform.python_version(), packages=versions,
                model_dir=str(model_dir.resolve()),
                slurm={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURMD_NODENAME',
                     'CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')},
                mount=command(['findmnt','-T',str(model_dir)]),
                gpus=command(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader']))


def io_state():
    result = {}
    for name in ('io','status'):
        path = Path('/proc/self') / name
        if path.exists():
            result[name] = path.read_text()
    return result


def probe_weights(model_dir, report, save, bytes_per_shard=64*2**20):
    """Bounded sequential sample, NOT a full-file benchmark or cold-cache test."""
    paths = sorted(model_dir.glob('*.safetensors'))
    if not paths:
        raise ValueError('No local safetensors shards found')
    report['read_probe'] = dict(bytes_per_shard=bytes_per_shard, shards=[],
        caveat='Reads only each shard prefix; may warm cache. Following load is not a cold-load measurement.')
    save(report)
    for path in paths:
        start, total = time.monotonic(), 0
        with path.open('rb') as f:
            while total < bytes_per_shard:
                chunk = f.read(min(8*2**20,bytes_per_shard-total))
                if not chunk:
                    break
                total += len(chunk)
        seconds = time.monotonic()-start
        row = dict(file=path.name, file_bytes=path.stat().st_size, bytes_read=total,
                   seconds=seconds, mib_per_second=total/2**20/max(seconds,1e-9))
        report['read_probe']['shards'].append(row)
        save(report)
        print(f'Read probe {path.name}: {total/2**20:.1f} MiB in {seconds:.2f}s',flush=True)


def array_stats(a):
    a = np.asarray(a)
    finite = a[np.isfinite(a)]
    return dict(shape=list(a.shape), dtype=str(a.dtype), nonfinite=int(a.size-finite.size),
                zeros=int(np.count_nonzero(a == 0)),
                minimum=float(finite.min()) if finite.size else None,
                maximum=float(finite.max()) if finite.size else None)


def capture_failure(torch, model, out, e, p, tids, visible, probabilities, reason, write_json):
    """Record actual cached probabilities, then replay the same batch for intermediates.

    Replay is diagnostic only; results never replace the measurement or bypass the gate.
    """
    report = dict(reason=reason, event=e, current_token=p['prompt'][slice(*p['offsets'][e['j']])],
                  probabilities=array_stats(probabilities))
    q = np.clip(probabilities-probabilities.mean(),0,None)
    denominator = float(q.sum())
    report['q'] = array_stats(q)
    report['denominator'] = denominator if np.isfinite(denominator) else None
    # Persist identity and observed values BEFORE any replay, even if replay fails.
    write_json(out/'gate_failure.json',report)
    arrays = dict(visible_ids=np.asarray(visible), cached_probabilities=probabilities,
                  q=q, historical_first=np.asarray(e.get('strength_first',np.nan)),
                  historical_last=np.asarray(e.get('strength_last',np.nan)))
    np.savez_compressed(out/'gate_failure_arrays.npz',**arrays)
    try:
        li,hi = e['layer'],e['head']
        idx=tids.index(e['current_id']); start=(idx//8)*8
        batch=tids[start:start+8]; row=idx-start
        dh=model.config.hidden_size//model.config.num_attention_heads
        layer=model.model.layers[li].self_attn
        E,U=model.get_input_embeddings().weight,model.lm_head.weight
        with torch.inference_mode():
            X=E[torch.tensor(batch,device=E.device)].to(layer.v_proj.weight.device,torch.float16)
            V=X @ layer.v_proj.weight.T[:,hi*dh:(hi+1)*dh]
            Z=V @ layer.o_proj.weight.T[hi*dh:(hi+1)*dh,:]
            logits=Z.to(U.device) @ U.T
            probs=logits.float().softmax(-1)
            for name,tensor in (('embedding',X),('V',V),('Z',Z),('logits',logits),('softmax',probs)):
                arrays[name]=tensor[row].float().cpu().numpy()
                report[name]=array_stats(arrays[name])
        report['replay_batch_ids']=batch
        arrays['replayed_visible']=arrays['softmax'][visible]
        report['replay_matches_cached']=bool(np.array_equal(arrays['replayed_visible'],probabilities))
    except Exception as exc:
        report['replay_error']=repr(exc)
    np.savez_compressed(out/'gate_failure_arrays.npz',**arrays)
    write_json(out/'gate_failure.json',report)
