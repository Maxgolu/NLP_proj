#!/usr/bin/env python3
"""Stage 1 of Phase A: mandatory preflight block + the observational RI scan
over ALL 1024 heads.

PREFLIGHT (aborts the job on P1-P3 failure, ~3 minutes):
  P1  Replicate the behavioral runner's candidate scoring exactly (each
      candidate teacher-forced on its own continuation, summed log-probs)
      on 6 prompts; compare to results.jsonl within tolerance.
  P2  Inert hooks preserve logits bit-for-bit; the attention-capturing
      forward's metric drift is measured and reported.
  P3  Self-patching (a head replaced by its own activation) changes the
      metric by exactly ~0.
  P4  Twin-patching 8 early-layer heads: recorded as an empirical zero-floor
      (report only).

SCAN (Ren et al. 2024, Findings of ACL, Eqs. 1-3, implemented from the paper):
  For every annotated triplet (t_s = child mention, t_o = mother/container
  mention) and every current position j >= max(s, o):
    QK stage:  s = argmax_k A^h_{j,k}  and  A^h_{j,s} / max_{k != s} A^h_{j,k} > tau (=2.2).
    OV stage:  p^{h,j} = softmax(x_j W_OV^h W_U)  with x_j the RAW EMBEDDING of
               the token at j (the paper operates on input embeddings);
               q_t = max(0, p_t - mean over unique context tokens k <= j);
               a^{h,j} = q_{t_o} / sum_k q_{t_k}.
  Multi-token adaptation (documented deviation; Ren used single-letter
  entities): t_s anchor = LAST token of the source span; t_o = FIRST token id
  of the target occurrence (primary) and LAST token id (sensitivity).
  Nulls: t_o replaced by (a) the token at position 10 (Ren's fake-tail) and
  (b) a random unique context token per evaluation.
  Also collected: the QK dominance-ratio distribution (for tau*_ours = its
  95th percentile) and per-variant breakdowns (base / corrupted / reorder).

Output: storage/runs/<name>/ with preflight.json, head_stats.csv,
null_samples.npz, dominance_percentiles.json, config.json.
Usage: python3 stage1_scan.py --name stage1_v1 [--shots 4] [--tau 2.2]
       [--limit-prompts 0] [--gate-prompts 6]
"""
import argparse, collections, json, os, random, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('PILOT_STORAGE', str(ROOT / 'storage'))).resolve()
RUNS = Path(os.environ.get('PILOT_RUNS', str(STATE / 'runs'))).resolve()
os.environ.setdefault('HF_HOME', str(STATE / 'hf'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    lock = json.loads((ROOT / 'model_lock_olmo2.json').read_text())
    model_dir = STATE / 'model' / lock['revision']
    memory = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        memory[i] = max(0, free - 2 * 2**30)
    memory['cpu'] = 0
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16,
        use_safetensors=True, attn_implementation='sdpa',
        low_cpu_mem_usage=True, device_map='auto', max_memory=memory)
    if any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
        raise RuntimeError('CPU offload -- not enough GPU memory.')
    model.eval(); model.config.use_cache = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch, tok, model, lock


def runner_score(torch, tok, model, device, prompt, name):
    """EXACTLY the behavioral runner's candidate score: summed log-probs of
    ' name.' teacher-forced on its own continuation."""
    base = tok.encode(prompt, add_special_tokens=False)
    full = tok.encode(prompt + ' ' + name + '.', add_special_tokens=False)
    tail = full[len(base):]
    ids = torch.tensor([base + tail[:-1]], device=device)
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits
    sel = logits[0, len(base)-1: len(base)+len(tail)-1].float().log_softmax(-1)
    tgt = torch.tensor(tail, device=sel.device).unsqueeze(-1)
    return sel.gather(-1, tgt).sum().item()


def first_split_metric(torch, tok, model, device, row, patch_ctx=None):
    """Primary intervention metric: logit difference between the two
    candidates' FIRST continuation tokens, at the answer position, single
    forward on the bare prompt. Sign: + = prefers this row's gold."""
    prompt = row['prompt']
    gold = row['gold']; distr = [c for c in row['candidates'] if c != gold][0]
    base = tok.encode(prompt, add_special_tokens=False)
    g0 = tok.encode(prompt + ' ' + gold, add_special_tokens=False)[len(base)]
    d0 = tok.encode(prompt + ' ' + distr, add_special_tokens=False)[len(base)]
    ids = torch.tensor([base], device=device)
    with torch.inference_mode():
        if patch_ctx is None:
            logits = model(ids, use_cache=False).logits
        else:
            with patch_ctx:
                logits = model(ids, use_cache=False).logits
    lg = logits[0, -1].float()
    return (lg[g0] - lg[d0]).item(), g0, d0


class HeadPatcher:
    """Context manager: replace head (li, hi) slice of o_proj input with a
    cached tensor during the forward."""
    def __init__(self, layers, dh, li, hi, cached):
        self.layer, self.dh, self.li, self.hi, self.cached = layers[li], dh, li, hi, cached
        self.hook = None
    def __enter__(self):
        def swap(mod, inp):
            z = inp[0].clone()
            z[:, :, self.hi*self.dh:(self.hi+1)*self.dh] = \
                self.cached[:, :z.shape[1], self.hi*self.dh:(self.hi+1)*self.dh].to(z.device, z.dtype)
            return (z,)
        self.hook = self.layer.self_attn.o_proj.register_forward_pre_hook(swap)
        return self
    def __exit__(self, *a):
        self.hook.remove()


def cache_head_inputs(torch, model, layers, ids):
    saved = {}
    hooks = []
    def keep(li):
        def f(mod, inp): saved[li] = inp[0].detach().clone()
        return f
    for li, lay in enumerate(layers):
        hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(keep(li)))
    with torch.inference_mode():
        model(ids, use_cache=False)
    for h in hooks: h.remove()
    return saved


def token_anchor(offsets, span, which='last'):
    idx = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0]]
    if not idx: return None
    return idx[-1] if which == 'last' else idx[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='stage1_v1')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--tau', type=float, default=2.2)
    ap.add_argument('--limit-prompts', type=int, default=0)
    ap.add_argument('--gate-prompts', type=int, default=6)
    ap.add_argument('--data', default=None)
    args = ap.parse_args()
    data = args.data or str(ROOT / 'data' / f'singlehop_v1_{args.shots}shot.jsonl')
    out = RUNS / args.name; out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260911)

    rows = [json.loads(l) for l in open(data, encoding='utf-8')]
    by_id = {r['id']: r for r in rows}

    # -------- prompt selection: working-set discovery families --------
    res = RUNS / f'olmo2_singlehop_{args.shots}shot' / 'results.jsonl'
    fams = None
    if res.exists():
        ok = collections.defaultdict(list)
        for line in res.open():
            r = json.loads(line)
            if r['variant'] in ('base', 'corrupted'):
                ok[r['family']].append(r['correct'])
        fams = {f for f, v in ok.items() if all(v)}
        print(f'working set from baseline: {len(fams)} families')
    scan_rows = [r for r in rows if r['split'] == 'discovery'
                 and (fams is None or r['family'] in fams)]
    if args.limit_prompts: scan_rows = scan_rows[:args.limit_prompts]
    print(f'scan set: {len(scan_rows)} prompts '
          f'({len({r["family"] for r in scan_rows})} families x variants x orders)', flush=True)

    torch, tok, model, lock = load_model()
    device = model.get_input_embeddings().weight.device
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    DH = cfg.hidden_size // H
    layers = model.model.layers
    W_E = model.get_input_embeddings().weight          # [V, d]
    W_U = model.lm_head.weight                          # [V, d]
    preflight = {}

    # =================== PREFLIGHT ===================
    print('=== preflight ===', flush=True)
    # P1: exact runner replication
    drifts = []
    if res.exists():
        ref = {json.loads(l)['id']: json.loads(l)['gold_minus_best_other'] for l in res.open()}
        for row in scan_rows[:args.gate_prompts]:
            gold = row['gold']; distr = [c for c in row['candidates'] if c != gold][0]
            m = runner_score(torch, tok, model, device, row['prompt'], gold) \
                - runner_score(torch, tok, model, device, row['prompt'], distr)
            if row['id'] in ref: drifts.append(abs(m - ref[row['id']]))
    preflight['P1_runner_replication'] = dict(max_drift=max(drifts) if drifts else None, n=len(drifts))
    p1_ok = bool(drifts) and max(drifts) < 0.05
    print(f'P1 runner replication: max drift {max(drifts) if drifts else None} -> {"OK" if p1_ok else "FAIL"}', flush=True)

    # P2: inert hooks preserve logits; attention-capture drift measured
    row0 = scan_rows[0]
    ids0 = torch.tensor([tok.encode(row0['prompt'], add_special_tokens=False)], device=device)
    with torch.inference_mode():
        base_logits = model(ids0, use_cache=False).logits[0, -1].float().cpu()
    inert = [lay.self_attn.o_proj.register_forward_pre_hook(lambda m, i: None) for lay in layers]
    with torch.inference_mode():
        hooked_logits = model(ids0, use_cache=False).logits[0, -1].float().cpu()
    for h in inert: h.remove()
    inert_diff = (base_logits - hooked_logits).abs().max().item()
    try:
        with torch.inference_mode():
            attn_out = model(ids0, use_cache=False, output_attentions=True)
        attn_logits = attn_out.logits[0, -1].float().cpu()
        attn_diff = (base_logits - attn_logits).abs().max().item()
        attn_available = attn_out.attentions is not None and attn_out.attentions[0] is not None
        del attn_out
    except Exception as e:
        attn_diff, attn_available = None, False
    preflight['P2_hooks'] = dict(inert_hook_max_logit_diff=inert_diff,
                                 attention_forward_max_logit_diff=attn_diff,
                                 attention_available=attn_available)
    p2_ok = inert_diff == 0.0 and attn_available
    print(f'P2 inert hooks diff {inert_diff}, attention-forward diff {attn_diff}, '
          f'attn available {attn_available} -> {"OK" if p2_ok else "FAIL"}', flush=True)

    # P3: self-patch is a no-op; P4: early-head twin patches = empirical floor
    m0, _, _ = first_split_metric(torch, tok, model, device, row0)
    zself = cache_head_inputs(torch, model, layers, ids0)
    m_self, _, _ = first_split_metric(torch, tok, model, device, row0,
                                      patch_ctx=HeadPatcher(layers, DH, 16, 0, zself[16]))
    preflight['P3_self_patch'] = dict(m_clean=m0, m_self=m_self, delta=abs(m_self - m0))
    p3_ok = abs(m_self - m0) < 1e-3
    print(f'P3 self-patch delta {abs(m_self - m0):.6f} -> {"OK" if p3_ok else "FAIL"}', flush=True)

    twin = by_id.get(row0['paired_base_id']) if row0.get('paired_base_id') else \
        by_id.get(row0['id'].replace('/base/', '/corrupted/').replace('/0/', '/1/'))
    floor = {}
    if twin is not None:
        ids_t = torch.tensor([tok.encode(twin['prompt'], add_special_tokens=False)], device=device)
        if ids_t.shape == ids0.shape:
            ztwin = cache_head_inputs(torch, model, layers, ids_t)
            for li, hi in [(0, 3), (0, 17), (1, 8), (2, 25), (3, 1), (3, 30), (2, 5), (1, 22)]:
                m_p, _, _ = first_split_metric(torch, tok, model, device, row0,
                                               patch_ctx=HeadPatcher(layers, DH, li, hi, ztwin[li]))
                floor[f'L{li}H{hi}'] = round(m_p - m0, 5)
            del ztwin
    preflight['P4_early_head_floor'] = floor
    print('P4 early-head twin-patch floor:', floor, flush=True)

    (out / 'preflight.json').write_text(json.dumps(preflight, indent=1))
    if not (p1_ok and p2_ok and p3_ok):
        print('PREFLIGHT FAILED -- aborting before the scan.', flush=True)
        sys.exit(2)
    del zself
    print('preflight PASSED; starting scan.', flush=True)

    # =================== SCAN ===================
    # accumulators: key (layer, head, variant)
    acc = collections.defaultdict(lambda: dict(evals=0, passes=0,
                                               a_first=0.0, a_last=0.0,
                                               null10=0.0, nullr=0.0, n_a=0))
    null_pool = collections.defaultdict(list)     # (layer, head) -> reservoir of null a values
    dom_sample = []                                # dominance ratios at true sources
    t_start = time.monotonic()

    for pi, row in enumerate(scan_rows):
        enc = tok(row['prompt'], return_offsets_mapping=True, add_special_tokens=False)
        ids_l, offs = enc['input_ids'], enc['offset_mapping']
        n = len(ids_l)
        ids = torch.tensor([ids_l], device=device)
        with torch.inference_mode():
            o = model(ids, use_cache=False, output_attentions=True)
        attn = [a[0].float().cpu() for a in o.attentions]     # L x [H, n, n]
        del o
        variant = row['variant']

        triples = []
        for f in row['facts']:
            s = token_anchor(offs, f['head_span'], 'last')
            o_first = token_anchor(offs, f['tail_span'], 'first')
            o_last = token_anchor(offs, f['tail_span'], 'last')
            if None in (s, o_first, o_last): continue
            triples.append((s, o_first, o_last))
        uniq_ids = {}
        for k, t in enumerate(ids_l):
            uniq_ids.setdefault(t, k)             # first position of each unique token

        # per (layer, head): set of j token-ids needing an OV row
        need = collections.defaultdict(set)       # (li, hi) -> {token_id_j}
        passes = []                                # (li, hi, j, o_first_id, o_last_id, jmax)
        for li in range(L):
            A = attn[li]                          # [H, n, n]
            top2 = A.topk(2, dim=-1)
            ratio = top2.values[..., 0] / (top2.values[..., 1] + 1e-9)   # [H, n]
            arg = top2.indices[..., 0]                                    # [H, n]
            for (s, o_f, o_l) in triples:
                jmin = max(s, o_l)
                hit = (arg[:, jmin:] == s) & (ratio[:, jmin:] > args.tau)
                # dominance sample for tau*_ours: ratios where argmax == s
                sel = (arg[:, jmin:] == s)
                if li % 8 == 0 and len(dom_sample) < 200000:
                    dom_sample.extend(ratio[:, jmin:][sel].tolist()[:200])
                n_eval = H * (n - jmin)
                hh, jj = hit.nonzero(as_tuple=True)
                for hi in range(H):
                    acc[(li, hi, variant)]['evals'] += (n - jmin)
                for hi, j_off in zip(hh.tolist(), jj.tolist()):
                    j = jmin + j_off
                    acc[(li, hi, variant)]['passes'] += 1
                    need[(li, hi)].add(ids_l[j])
                    passes.append((li, hi, j, ids_l[o_f], ids_l[o_l]))

        # OV stage: p^{h,j} = softmax(emb[token_j] W_OV^h W_U), cached per (li,hi,tokid)
        ctx_ids = torch.tensor(sorted(uniq_ids), device=W_U.device)
        pcache = {}
        for (li, hi), tokids in need.items():
            blk = layers[li].self_attn
            dev = blk.v_proj.weight.device
            tl = sorted(tokids)
            X = W_E[torch.tensor(tl, device=W_E.device)].to(dev, torch.float16)   # [m, d]
            V = X @ blk.v_proj.weight.T[:, hi*DH:(hi+1)*DH]
            Z = V @ blk.o_proj.weight.T[hi*DH:(hi+1)*DH, :]
            P = torch.softmax((Z.to(W_U.device) @ W_U.T).float(), dim=-1)          # [m, V]
            Pc = P[:, ctx_ids]                                                     # restrict to context tokens
            for r_i, t_id in enumerate(tl):
                pcache[(li, hi, t_id)] = Pc[r_i].cpu()
        ctx_list = sorted(uniq_ids)                      # unique token ids, ascending
        ctx_index = {t: i for i, t in enumerate(ctx_list)}
        ctx_pos_arr = np.array([uniq_ids[t] for t in ctx_list])   # first occurrence of each
        tok10 = ids_l[10] if n > 10 else ids_l[-1]

        for (li, hi, j, of_id, ol_id) in passes:
            p = pcache[(li, hi, ids_l[j])].numpy()       # p over ALL unique ctx tokens
            vis = ctx_pos_arr <= j                       # which unique tokens occur at/before j
            pc = p[vis]
            q = np.clip(pc - pc.mean(), 0, None)
            denom = q.sum()
            if denom <= 0: continue
            # map a token id -> its index within the visible subset
            vis_cum = np.cumsum(vis) - 1                 # position within pc for visible entries

            def share(tid):
                idx = ctx_index.get(tid)
                if idx is None or not vis[idx]: return None
                return float(q[vis_cum[idx]] / denom)

            a_f = share(of_id); a_l = share(ol_id)
            if a_f is None or a_l is None: continue
            n10 = share(tok10)
            vis_tokens = [t for t, v in zip(ctx_list, vis) if v]
            rnd = share(vis_tokens[rng.randrange(len(vis_tokens))])
            a = acc[(li, hi, variant)]
            a['a_first'] += a_f; a['a_last'] += a_l; a['n_a'] += 1
            if n10 is not None: a['null10'] += n10
            if rnd is not None:
                a['nullr'] += rnd
                pool = null_pool[(li, hi)]
                if len(pool) < 200: pool.append(rnd)
                elif rng.random() < 0.05: pool[rng.randrange(200)] = rnd
        if (pi + 1) % 10 == 0:
            el = time.monotonic() - t_start
            print(f'{pi+1}/{len(scan_rows)} prompts; {el/(pi+1):.2f} s/prompt', flush=True)

    # =================== OUTPUT ===================
    with (out / 'head_stats.csv').open('w') as f:
        f.write('layer,head,variant,evals,qk_passes,freq,n_a,strength_first,strength_last,null_tok10,null_random\n')
        for (li, hi, variant), a in sorted(acc.items()):
            freq = a['passes'] / a['evals'] if a['evals'] else 0
            sf = a['a_first'] / a['n_a'] if a['n_a'] else 0
            sl = a['a_last'] / a['n_a'] if a['n_a'] else 0
            n10 = a['null10'] / a['n_a'] if a['n_a'] else 0
            nr = a['nullr'] / a['n_a'] if a['n_a'] else 0
            f.write(f'{li},{hi},{variant},{a["evals"]},{a["passes"]},{freq:.6g},{a["n_a"]},{sf:.6g},{sl:.6g},{n10:.6g},{nr:.6g}\n')
    np.savez_compressed(out / 'null_samples.npz',
                        **{f'L{li}H{hi}': np.array(v) for (li, hi), v in null_pool.items() if v})
    dom = np.array(dom_sample)
    (out / 'dominance_percentiles.json').write_text(json.dumps(
        {f'p{p}': float(np.percentile(dom, p)) for p in (50, 75, 90, 95, 99)} | {'n': len(dom)}, indent=1))
    (out / 'config.json').write_text(json.dumps(dict(
        model=lock, data=data, tau=args.tau, shots=args.shots,
        n_prompts=len(scan_rows), metric='ren2024_eq1-3_embedding_level',
        anchors='source=last-token, target=first-token primary / last-token sensitivity',
        elapsed_s=round(time.monotonic() - t_start, 1)), indent=1))
    print(f'done: {len(scan_rows)} prompts in {(time.monotonic()-t_start)/60:.1f} min; '
          f'outputs in {out}', flush=True)


if __name__ == '__main__':
    main()
