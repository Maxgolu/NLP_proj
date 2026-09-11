#!/usr/bin/env python3
"""Stage 0 of Phase A: hook infrastructure + the four budget measurements.

Runs as ONE GPU job (2 GPUs) from pilot_v2/. Produces storage/runs/<name>/
stage0_report.json plus a printed summary with extrapolated stage-1/2 budgets.

What it does, in order (each part is fenced; a failure is recorded and the
script continues):
  gate  Correctness gate: hooked/eager model reproduces the behavioral runner's
        candidate scores (sdpa, no hooks) on N prompts within tolerance.
  m1    Forward with attention + hidden-state caching: sec/prompt, peak GiB,
        cache size.
  m2    Full Relation-Index pass per prompt from the cache (QK dominance filter
        at tau=2.2, then OV vocabulary projection only for survivors at fact
        anchors): sec/prompt, survivor counts. NOTE: cost-faithful
        implementation; the exact Ren-et-al. normalization details are
        finalized in stage 1 against the paper -- FLOP profile is identical.
  m3    Exact head patching: replace one head's output slice (o_proj input)
        with its value from the corrupted twin, all positions; sec/patched
        forward. Also the LESSON: a 32-head sweep over one layer on one twin
        pair, printing the per-head logit-diff effect.
  m4    Backward pass feasibility: params frozen, grads captured on o_proj
        inputs, backward of the logit-diff metric; peak GiB or OOM. If it
        fits, computes REAL attribution-patching estimates for all 1024 heads
        on the demo pair and prints the top 10.

Usage (inside the job):
  python3 stage0_costing.py --name stage0_v1 [--pairs 10] [--layer 16]
"""
import argparse, json, os, sys, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('PILOT_STORAGE', str(ROOT / 'storage'))).resolve()
RUNS = Path(os.environ.get('PILOT_RUNS', str(STATE / 'runs'))).resolve()
os.environ.setdefault('HF_HOME', str(STATE / 'hf'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

TAU = 2.2
REPORT = {}

def fence(key):
    def deco(fn):
        def wrap(*a, **k):
            t0 = time.monotonic()
            try:
                out = fn(*a, **k)
                REPORT[key] = dict(ok=True, wall_s=round(time.monotonic()-t0, 2), **(out or {}))
                return out
            except Exception as e:
                import traceback; traceback.print_exc()
                REPORT[key] = dict(ok=False, error=f'{type(e).__name__}: {e}')
                return None
        return wrap
    return deco


def load_model(attn_impl):
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
        use_safetensors=True, attn_implementation=attn_impl,
        low_cpu_mem_usage=True, device_map='auto', max_memory=memory)
    if any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
        raise RuntimeError('CPU offload -- not enough GPU memory.')
    model.eval()
    model.config.use_cache = False
    cfg = model.config
    assert cfg.num_attention_heads == 32 and cfg.num_hidden_layers == 32
    if getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads) != cfg.num_attention_heads:
        print('WARNING: GQA model -- OV stage needs kv-head mapping (handled in stage 1).')
    return torch, tok, model


def pick_pairs(rows, n_pairs):
    """n_pairs (base, corrupted) twin pairs from DISCOVERY families, preferring
    families the baseline run answered correctly in both variants+orders."""
    by_id = {r['id']: r for r in rows}
    correct_fams = None
    res = RUNS / 'olmo2_singlehop_4shot' / 'results.jsonl'
    if res.exists():
        ok = {}
        for line in res.open():
            r = json.loads(line)
            ok.setdefault(r['family'], []).append(r['correct'] if r['variant'] in ('base','corrupted') else True)
        correct_fams = {f for f, v in ok.items() if all(v)}
    pairs = []
    for r in rows:
        if r['variant'] != 'corrupted' or r['option_order'] != 0: continue
        if r['split'] != 'discovery': continue
        if correct_fams is not None and r['family'] not in correct_fams: continue
        base = by_id.get(r['paired_base_id'])
        if base: pairs.append((base, r))
        if len(pairs) == n_pairs: break
    assert pairs, 'no twin pairs found'
    return pairs


def encode_with_tail(torch, tok, row, device):
    """prompt + ' gold.' ids, plus per-position (gold, distractor) target ids
    for the multi-token logit-diff metric (candidates are token-length matched)."""
    prompt = row['prompt']
    gold = row['gold']; distr = [c for c in row['candidates'] if c != gold][0]
    base = tok.encode(prompt, add_special_tokens=False)
    g = tok.encode(prompt + ' ' + gold + '.', add_special_tokens=False)[len(base):]
    d = tok.encode(prompt + ' ' + distr + '.', add_special_tokens=False)[len(base):]
    assert len(g) == len(d), 'tail length mismatch -- audit should have caught this'
    ids = torch.tensor([base + g], device=device)
    return ids, len(base), g, d


def logit_diff(logits, base_len, g, d):
    total = 0.0
    for i in range(len(g)):
        pos = base_len - 1 + i
        total += (logits[0, pos, g[i]] - logits[0, pos, d[i]]).item()
    return total


def head_slices(model):
    cfg = model.config
    dh = cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers, cfg.num_attention_heads, dh


def token_anchor(offsets, span):
    """last token overlapping the char span (last-token anchor policy)."""
    idx = [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0]]
    return idx[-1] if idx else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='stage0_v1')
    ap.add_argument('--pairs', type=int, default=10)
    ap.add_argument('--layer', type=int, default=16, help='layer for the 32-head lesson sweep')
    ap.add_argument('--data', default=str(ROOT / 'data' / 'singlehop_v1_4shot.jsonl'))
    args = ap.parse_args()
    out = RUNS / args.name; out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.data, encoding='utf-8')]
    # Load ONCE with sdpa (the runner's proven-correct config); eager on this
    # OLMo-2 + transformers + 2-GPU setup returned degenerate (uniform) logits.
    torch, tok, model = load_model('sdpa')
    # Force the MATH sdpa kernel globally, exactly like the behavioral runner
    # (guarantees finite, correct logits; supports backward for m4). This makes
    # every model() call below use MATH without per-call wrapping.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = model.get_input_embeddings().weight.device
    L, H, DH = head_slices(model)
    layers = model.model.layers
    pairs = pick_pairs(rows, args.pairs)
    print(f'{len(pairs)} twin pairs; families {[b["family"] for b, _ in pairs]}', flush=True)
    prompts = [b for b, _ in pairs] + [c for _, c in pairs]

    # optional: does TransformerLens even import here? (no load attempt -- that is its own experiment)
    try:
        import transformer_lens  # noqa
        REPORT['tlens_importable'] = True
    except Exception as e:
        REPORT['tlens_importable'] = f'no: {type(e).__name__}'

    # ---------------- correctness gate ----------------
    @fence('gate')
    def gate():
        diffs = []
        with torch.inference_mode():
            for row in prompts[:6]:
                ids, bl, g, d = encode_with_tail(torch, tok, row, device)
                lg = model(ids, use_cache=False).logits.float()
                diffs.append(logit_diff(lg, bl, g, d))
                del lg
        res = RUNS / 'olmo2_singlehop_4shot' / 'results.jsonl'
        max_drift = None
        if res.exists():
            ref = {json.loads(l)['id']: json.loads(l)['gold_minus_best_other'] for l in res.open()}
            drift = [abs(m - ref[row['id']]) for m, row in zip(diffs, prompts[:6]) if row['id'] in ref]
            max_drift = max(drift) if drift else None
        passed = all(x > 0 for x in diffs) and (max_drift is None or max_drift < 0.1)
        print('gate: metric values', [round(x, 2) for x in diffs], '| max drift vs runner:', max_drift,
              '| PASSED' if passed else '| WARNING (not matching runner)', flush=True)
        if not passed:
            # diagnostic: what does the model actually predict at the first answer position?
            with torch.inference_mode():
                ids, bl, g, d = encode_with_tail(torch, tok, prompts[0], device)
                top = model(ids, use_cache=False).logits[0, bl-1].float().topk(5).indices.tolist()
            print('  first-answer top-5 tokens:', [tok.decode([t]) for t in top], flush=True)
        return dict(passed=passed, metric_values=[round(x, 3) for x in diffs], max_drift_vs_runner=max_drift)
    gate()

    # ---------------- m1: forward with caching ----------------
    caches = {}
    @fence('m1_forward_with_cache')
    def m1():
        for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
        times, cache_bytes = [], 0
        attn_ok = True
        with torch.inference_mode():
            for row in prompts:
                ids, bl, g, d = encode_with_tail(torch, tok, row, device)
                t0 = time.monotonic()
                try:
                    o = model(ids, use_cache=False, output_attentions=True, output_hidden_states=True)
                    attns = o.attentions
                    if attns is None or attns[0] is None:
                        raise RuntimeError('attentions not returned')
                except Exception as e:
                    # sdpa may refuse output_attentions; capture hidden states only.
                    attn_ok = False
                    o = model(ids, use_cache=False, output_hidden_states=True)
                    attns = None
                    if row is prompts[0]:
                        print('  m1: attention weights unavailable via output_attentions '
                              f'({e}); RI timing (m2) will need manual attention. hidden states OK.', flush=True)
                torch.cuda.synchronize()
                times.append(time.monotonic() - t0)
                attn = [a[0].to('cpu', torch.float16) for a in attns] if attns is not None else None
                hid = [h[0].to('cpu', torch.float16) for h in o.hidden_states[:-1]]
                cache_bytes = (sum(a.numel() for a in attn) * 2 if attn else 0) + sum(h.numel() for h in hid) * 2
                caches[row['id']] = dict(attn=attn, hid=hid, n=ids.shape[1], row=row)
                del o
        peak = [round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(torch.cuda.device_count())]
        return dict(sec_per_prompt=round(sum(times)/len(times), 3), attention_captured=attn_ok,
                    peak_gib=peak, attn_cache_mib_per_prompt=round(cache_bytes/2**20, 1),
                    n_tokens=[caches[r['id']]['n'] for r in prompts[:3]])
    m1()

    # ---------------- m2: Relation-Index pass from cache ----------------
    @fence('m2_ri_pass')
    def m2():
        # precompute unembedding on the model device of the last layer
        W_U = model.lm_head.weight            # [V, d]
        if any(caches[r['id']]['attn'] is None for r in prompts):
            return dict(skipped='attention weights unavailable (see m1); RI timing deferred to stage 1 with manual attention capture')
        times, qk_survivors, ov_evals = [], [], 0
        nonlocal_ov = 0
        for row in prompts:
            c = caches[row['id']]
            enc = tok(row['prompt'], return_offsets_mapping=True, add_special_tokens=False)
            offs = enc['offset_mapping']
            anchors = []
            for f in row['facts']:
                s = token_anchor(offs, f['head_span']); o_ = token_anchor(offs, f['tail_span'])
                if s is not None and o_ is not None: anchors.append((s, o_))
            t0 = time.monotonic()
            survivors = []   # (layer, head, j, s_pos)
            for li, A in enumerate(c['attn']):          # A: [H, n, n] cpu fp16
                Af = A.float()
                top2 = Af.topk(2, dim=-1)               # values [H, n, 2], idx [H, n, 2]
                for (s, o_) in anchors:
                    jmin = max(s, o_)
                    idx = top2.indices[:, jmin:, 0]     # argmax target per (head, j)
                    dom = top2.values[:, jmin:, 0] / (top2.values[:, jmin:, 1] + 1e-9)
                    hit = (idx == s) & (dom > TAU)
                    hh, jj = hit.nonzero(as_tuple=True)
                    for h_i, j_i in zip(hh.tolist(), jj.tolist()):
                        survivors.append((li, h_i, jmin + j_i, s, o_))
            # OV stage only for survivors, vectorized per (layer, source-position)
            from collections import defaultdict
            group = defaultdict(set)
            for li, h_i, j, s, o_ in survivors: group[(li, s)].add(h_i)
            for (li, s), heads in group.items():
                blk = layers[li].self_attn
                x = c['hid'][li][s].to(blk.v_proj.weight.device, torch.float16)  # [d], on the layer's GPU
                for h_i in heads:
                    v = x @ blk.v_proj.weight.T[:, h_i*DH:(h_i+1)*DH]      # [dh]
                    z = v @ blk.o_proj.weight.T[h_i*DH:(h_i+1)*DH, :]      # [d]
                    logits_v = (z.to(W_U.device) @ W_U.T).float()          # [V]
                    _ = torch.softmax(logits_v, -1)                        # tail-share arithmetic is negligible
                    nonlocal_ov += 1
            times.append(time.monotonic() - t0)
            qk_survivors.append(len(survivors))
        return dict(sec_per_prompt=round(sum(times)/len(times), 3),
                    mean_qk_survivor_pairs=round(sum(qk_survivors)/len(qk_survivors), 1),
                    ov_projections_total=nonlocal_ov,
                    anchors_per_prompt=len(anchors))
    m2()
    caches.clear(); gc.collect(); torch.cuda.empty_cache()

    # ---------------- m3: exact head patching + the 32-head lesson ----------------
    @fence('m3_exact_patching')
    def m3():
        base, corr = pairs[0]
        ids_b, bl, g, d = encode_with_tail(torch, tok, base, device)
        ids_c, blc, gc_, dc = encode_with_tail(torch, tok, corr, device)
        assert ids_b.shape == ids_c.shape, 'twins must be token-aligned'
        # cache corrupted o_proj inputs (z) per layer
        z_corr = {}
        hooks = []
        def save(li):
            def f(mod, inp): z_corr[li] = inp[0].detach().clone()
            return f
        for li, lay in enumerate(layers):
            hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(save(li)))
        with torch.inference_mode():
            m_corr = logit_diff(model(ids_c, use_cache=False).logits.float(), blc, gc_, dc)
        for h in hooks: h.remove()
        with torch.inference_mode():
            m_clean = logit_diff(model(ids_b, use_cache=False).logits.float(), bl, g, d)
        print(f'lesson pair: M_clean={m_clean:.2f}  M_corr(own gold)={m_corr:.2f}', flush=True)

        def patched_metric(li, h_i):
            def swap(mod, inp):
                z = inp[0].clone()
                z[:, :, h_i*DH:(h_i+1)*DH] = z_corr[li][:, :, h_i*DH:(h_i+1)*DH].to(z.device, z.dtype)
                return (z,)
            hk = layers[li].self_attn.o_proj.register_forward_pre_hook(swap)
            try:
                with torch.inference_mode():
                    return logit_diff(model(ids_b, use_cache=False).logits.float(), bl, g, d)
            finally:
                hk.remove()

        t0 = time.monotonic()
        sweep = {h_i: round(patched_metric(args.layer, h_i) - m_clean, 3) for h_i in range(H)}
        per_fwd = (time.monotonic() - t0) / H
        top = sorted(sweep.items(), key=lambda kv: kv[1])[:5]
        print(f'LESSON — layer {args.layer}, per-head patch effect on M (negative = pushes toward the corrupted answer):', flush=True)
        print('  strongest 5:', top, flush=True)
        return dict(sec_per_patched_forward=round(per_fwd, 3), m_clean=round(m_clean, 3),
                    m_corr=round(m_corr, 3), lesson_layer=args.layer, lesson_sweep=sweep)
    m3()

    # ---------------- m4: backward feasibility + real attribution demo ----------------
    @fence('m4_backward_attribution')
    def m4():
        for p in model.parameters(): p.requires_grad_(False)
        # With all params frozen no autograd graph would be built at all; force
        # the graph to exist from the embeddings onward (embedding output is a
        # leaf when params are frozen, so the flag may be set on it).
        def force_grad(mod, inp, out):
            out.requires_grad_(True)
            return out
        embed_hook = model.get_input_embeddings().register_forward_hook(force_grad)
        for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
        base, corr = pairs[0]
        ids_b, bl, g, d = encode_with_tail(torch, tok, base, device)
        ids_c, blc, gc_, dc = encode_with_tail(torch, tok, corr, device)
        saved, grads, hooks = {}, {}, []
        def save(li):
            def f(mod, inp):
                z = inp[0]
                z.retain_grad() if z.requires_grad else None
                saved[li] = z
                if z.requires_grad:
                    z.register_hook(lambda gr, li=li: grads.__setitem__(li, gr.detach()))
                return None
            return f
        for li, lay in enumerate(layers):
            hooks.append(lay.self_attn.o_proj.register_forward_pre_hook(save(li)))
        # corrupted pass (no grad) for the deltas
        with torch.no_grad():
            model(ids_c, use_cache=False)
        z_corr = {li: saved[li].detach().clone() for li in saved}
        saved.clear()
        # clean pass WITH graph, backward on the metric
        t0 = time.monotonic()
        logits = model(ids_b, use_cache=False).logits
        m_val = 0
        for i in range(len(g)):
            pos = bl - 1 + i
            m_val = m_val + logits[0, pos, g[i]].float() - logits[0, pos, d[i]].float()
        m_val.backward()
        torch.cuda.synchronize()
        bw_time = time.monotonic() - t0
        for h in hooks: h.remove()
        embed_hook.remove()
        peak = [round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(torch.cuda.device_count())]
        # attribution for all L*H heads on this pair
        attr = {}
        for li in grads:
            delta = (z_corr[li] - saved[li].detach()).float()
            prod = (delta * grads[li].float()).sum(dim=1)[0]     # sum over positions -> [d_model]
            for h_i in range(H):
                attr[f'L{li}H{h_i}'] = round(prod[h_i*DH:(h_i+1)*DH].sum().item(), 4)
        top = sorted(attr.items(), key=lambda kv: kv[1])[:10]
        print('attribution demo — 10 most negative heads (predicted to push toward corrupted answer):', flush=True)
        for k, v in top: print('  ', k, v, flush=True)
        return dict(fits=True, backward_plus_forward_s=round(bw_time, 2), peak_gib=peak,
                    n_heads_estimated=len(attr), top10=dict(top))
    m4()

    # ---------------- extrapolation + report ----------------
    ext = {}
    try:
        n_scan_prompts = 89 * 3 * 2          # discovery families x variants x orders
        ri_s = REPORT['m2_ri_pass'].get('sec_per_prompt', REPORT['m1_forward_with_cache']['sec_per_prompt'])
        ext['stage1_scan_hours'] = round(n_scan_prompts * (REPORT['m1_forward_with_cache']['sec_per_prompt'] + ri_s) / 3600, 2)
        n_pairs = 89 * 2
        per_pair = 4 * REPORT['m1_forward_with_cache']['sec_per_prompt']
        ext['stage2_attribution_screen_hours'] = round(n_pairs * per_pair / 3600, 2)
        ext['stage2_exact_verify_hours_per_100heads_40pairs'] = round(100 * 40 * REPORT['m3_exact_patching']['sec_per_patched_forward'] / 3600, 2)
    except Exception as e:
        ext['error'] = str(e)
    REPORT['extrapolation'] = ext
    (out / 'stage0_report.json').write_text(json.dumps(REPORT, indent=1))
    print('\n==== STAGE 0 SUMMARY ====')
    for k, v in REPORT.items():
        s = json.dumps(v) if not isinstance(v, dict) else json.dumps({kk: vv for kk, vv in v.items() if kk != 'lesson_sweep' and kk != 'top10'})
        print(f'{k}: {s[:300]}')
    print('report:', out / 'stage0_report.json')


if __name__ == '__main__':
    main()
