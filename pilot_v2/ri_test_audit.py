#!/usr/bin/env python3
"""RI-only discovery extension: prepare on CPU, complete OV on GPU, analyze on CPU.

No model forward pass and no use of Stage-2 effects. Original QK events/tau and
visible-context normalization are retained. See README_RI_TEST_AUDIT.md.
"""
import argparse
import collections
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
VERSION = 1
POLICY = dict(seed=20260916, word_controls=3, min_events=20, min_families=10,
              top_k=10, tau=2.2, anchors=['first', 'last'],
              selection='Union of top10 positive family-mean target/name-gap/word-gap '
              'for each anchor, scope and all_test/final position; >=20 scored '
              'events and >=10 active families for the corresponding metric. '
              'Descriptive discovery shortlist, not a significance claim.',
              scopes=['all_facts', 'query_fact'],
              normalization='Original unique visible FULL-context token IDs, including demos',
              words='Up to 3 unique ASCII alphabetic non-name word types in visible test; '
              'sample once per prompt/position, shared across heads/facts; first/last token anchors',
              names='All other fully visible fact-tail names in test, equal weight per name; '
              'no token-length exclusion; collisions retained as ties',
              families='Conditional event means within family; equal active-family weight. '
              'Frequency also reported with zero-pass families included.',
              post_selection='Token concentration and leave-family-out diagnostics deferred')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_lines(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


@contextlib.contextmanager
def run_lock(out):
    path = out / '.running.lock'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Run is locked: {path}. Check previous job before removing stale lock.')
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(dict(pid=os.getpid(), job=os.environ.get('SLURM_JOB_ID'))))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def token_span(p, span):
    ids = [i for i, (a, b) in enumerate(p['offsets']) if a < span[1] and b > span[0]]
    if not ids:
        raise ValueError(f'Missing token span {p["id"]}: {span}')
    return ids


def enrich_prompt(p):
    p = dict(p)
    test_start = p['prompt'].rfind('\n\n') + 2
    match = re.search(r'Question: Who is the mother of ([A-Za-z]+)\?\s*Answer:\s*$',
                      p['prompt'][test_start:])
    if not match:
        raise ValueError('Unsupported test template: ' + p['id'])
    p['query_source'] = match.group(1)
    p['test_start'] = test_start
    facts = []
    for fi, fact in enumerate(p['facts']):
        if fact['block'] != 'test':
            continue
        s = token_span(p, fact['head_span'])[-1]
        o = token_span(p, fact['tail_span'])
        facts.append(dict(fi=fi, source=fact['head'], target=fact['tail'], s=s,
                          of=o[0], ol=o[-1], query=fact['head'] == p['query_source']))
    relevant = [f for f in facts if f['query']]
    if len(facts) != 4 or len(relevant) != 1 or relevant[0]['target'] != p['gold']:
        raise ValueError('Query/fact/gold mismatch: ' + p['id'])
    p['test_facts'] = facts
    p['order'] = int(p['id'].rsplit('order', 1)[1])
    # General words deliberately exclude ALL entity names, not just the target.
    names = {f[k].lower() for f in p['facts'] for k in ('head', 'tail')}
    words = []
    for m in re.finditer(r'\b[A-Za-z]+\b', p['prompt'][test_start:]):
        if m.group().lower() in names:
            continue
        span = [test_start + m.start(), test_start + m.end()]
        ts = token_span(p, span)
        words.append(dict(text=m.group(), first=ts[0], last=ts[-1]))
    p['test_words'] = words
    return p


def controls(p, fact, j):
    names = [dict(text=f['target'], first=f['of'], last=f['ol'])
             for f in p['test_facts'] if f['target'] != fact['target'] and f['ol'] <= j]
    unique = {}
    for w in p['test_words']:
        if w['last'] <= j:
            unique.setdefault(w['text'], w)
    pool = [unique[k] for k in sorted(unique)]
    seed = int(hashlib.sha256(f'{POLICY["seed"]}|{p["id"]}|{j}'.encode()).hexdigest(), 16)
    words = random.Random(seed).sample(pool, min(POLICY['word_controls'], len(pool)))
    return names, words


def position(p, j):
    return 'final' if j == len(p['token_ids']) - 1 else 'earlier'


def distance(fact, j):
    d = j - fact['s']
    if d < 0:
        raise ValueError('Source not visible')
    return 'self' if d == 0 else 'previous' if d == 1 else 'long'


def code_hashes():
    return {n: digest(ROOT / n) for n in ('ri_test_audit.py', 'ri_test_diagnostics.py', 'stage1_scan.py',
                                          'stage1_audit.py', 'model_lock_olmo2.json')}


def check_manifest(out):
    m = read_json(out / 'manifest.json')
    if m['version'] != VERSION or m['policy'] != POLICY or m['code'] != code_hashes():
        raise ValueError('Code/policy changed. Use a new output directory, not resume.')
    for n, sha in m['prepared'].items():
        if digest(out / n) != sha:
            raise ValueError('Prepared input changed: ' + n)
    return m


def prepare(source, out):
    required = ['config.json', 'provenance.json', 'head_stats.csv',
                'audit_prompts.jsonl.gz', 'ri_events.jsonl.gz']
    inputs = {n: digest(source / n) for n in required}
    config = read_json(source / 'config.json')
    if config['model'] != read_json(ROOT / 'model_lock_olmo2.json') or config['tau'] != 2.2:
        raise ValueError('Expected pinned model and original tau=2.2')
    if (out / 'manifest.json').exists():
        m = check_manifest(out)
        if m['inputs'] != inputs:
            raise ValueError('Source changed; cannot resume')
        print('Prepared input already verified.', flush=True)
        return
    raw = list(lines(source / 'audit_prompts.jsonl.gz'))
    ps = {p['id']: enrich_prompt(p) for p in raw}
    families = sorted({p['family'] for p in raw})
    if len(ps) != 534 or len(raw) != 534 or len(families) != 89 or any(f % 2 for f in families):
        raise ValueError('Expected 534 unique prompts / 89 even-ID discovery families')
    for fam in families:
        group = [p for p in ps.values() if p['family'] == fam]
        if {(p['variant'], p['order']) for p in group} != {
                (v, o) for v in ('base', 'corrupted', 'reorder') for o in (0, 1)}:
            raise ValueError('Incomplete family: ' + str(fam))
    events, totals, scored, seen = [], collections.Counter(), collections.Counter(), set()
    for e in lines(source / 'ri_events.jsonl.gz'):
        key = (e['layer'], e['head'], e['variant'])
        totals[key] += 1
        scored[key] += int(e['scored'])
        if e['fact_block'] != 'test' or e['position_block'] != 'test':
            continue
        p = ps[e['id']]
        if e['family'] != p['family'] or e['variant'] != p['variant']:
            raise ValueError('Event/prompt identity mismatch')
        f = next(f for f in p['test_facts'] if f['fi'] == e['fact'])
        uid = (e['layer'], e['head'], e['id'], e['fact'], e['j'])
        if uid in seen or e['j'] < max(f['s'], f['ol']) or e['dominance'] <= 2.2:
            raise ValueError('Invalid or duplicate QK event')
        seen.add(uid)
        if e['target_first_id'] != p['token_ids'][f['of']] or e['target_last_id'] != p['token_ids'][f['ol']]:
            raise ValueError('Target anchor mismatch')
        events.append(dict(e, event_id=len(events), query_fact=f['query'],
                           source=f['source'], target=f['target'],
                           order=p['order'], position=position(p, e['j']),
                           distance=distance(f, e['j'])))
    with (source / 'head_stats.csv').open() as fp:
        stats = list(csv.DictReader(fp))
    for r in stats:
        k = (int(r['layer']), int(r['head']), r['variant'])
        if totals[k] != int(r['qk_passes']) or scored[k] != int(r['n_a']):
            raise ValueError('Event counts disagree with head_stats')
    heads = sorted({(int(r['layer']), int(r['head'])) for r in stats})
    if heads != [(l, h) for l in range(32) for h in range(32)]:
        raise ValueError('Expected all 1024 heads')
    write_lines(out / 'prompts.jsonl.gz', ps.values())
    write_lines(out / 'events.jsonl.gz', events)
    manifest = dict(version=VERSION, policy=POLICY, inputs=inputs, code=code_hashes(),
                    source_config=config, source_provenance=read_json(source / 'provenance.json'),
                    families=families, prompts=len(ps), events=len(events), heads=heads,
                    prepared={n: digest(out / n) for n in ('prompts.jsonl.gz', 'events.jsonl.gz')})
    write_json(out / 'manifest.json', manifest)
    print(f'Prepared {len(events)} test QK events, {len(ps)} prompts; policy frozen.', flush=True)


def normalized_scores(probabilities):
    q = np.clip(probabilities - probabilities.mean(), 0, None)
    denom = q.sum()
    return None if denom <= 0 else q / denom


def ov_probabilities(torch, model, li, hi, token_ids, context_ids, batch=8):
    """Same fp16 products + fp32 vocabulary softmax as the historical scan."""
    layer = model.model.layers[li].self_attn
    dh = model.config.hidden_size // model.config.num_attention_heads
    E, U = model.get_input_embeddings().weight, model.lm_head.weight
    result = []
    with torch.inference_mode():
        ci = torch.tensor(context_ids, device=U.device)
        for start in range(0, len(token_ids), batch):
            ts = token_ids[start:start + batch]
            X = E[torch.tensor(ts, device=E.device)].to(layer.v_proj.weight.device, torch.float16)
            V = X @ layer.v_proj.weight.T[:, hi*dh:(hi+1)*dh]
            Z = V @ layer.o_proj.weight.T[hi*dh:(hi+1)*dh, :]
            P = torch.softmax((Z.to(U.device) @ U.T).float(), dim=-1)
            result.append(P[:, ci].cpu().numpy())
    return np.concatenate(result)


def compare_scores(target, items):
    if not items:
        return dict(n=0, mean=None, gap=None, rank_min=None, rank_max=None,
                    wins=None, ties=None, top=None, strict_top=None, collisions=0)
    vals = [c['score'] for c in items]
    # Numerical ties include exact token collisions. No artificial win from a collision.
    ties = sum(abs(x-target) <= 1e-7 for x in vals)
    higher = sum(x > target + 1e-7 for x in vals)
    return dict(n=len(vals), mean=float(np.mean(vals)), gap=float(target-np.mean(vals)),
                rank_min=1+higher, rank_max=1+higher+ties,
                wins=sum(target > x+1e-7 for x in vals)/len(vals), ties=ties/len(vals),
                top=int(higher == 0), strict_top=int(higher == 0 and ties == 0),
                collisions=sum(c['collision'] for c in items))


def complete_ov(out):
    m = check_manifest(out)
    ps = {p['id']: p for p in lines(out / 'prompts.jsonl.gz')}
    groups = collections.defaultdict(list)
    for e in lines(out / 'events.jsonl.gz'):
        groups[(e['layer'], e['head'])].append(e)
    shards = out / 'heads'
    shards.mkdir(exist_ok=True)
    manifest_sha = digest(out / 'manifest.json')
    pending = []
    for key, es in sorted(groups.items()):
        stem = f'L{key[0]}H{key[1]}'
        meta = shards / (stem + '.json')
        data = shards / (stem + '.jsonl.gz')
        if meta.exists():
            info = read_json(meta)
            if info['manifest'] != manifest_sha or info['sha256'] != digest(data) or info['events'] != len(es):
                raise ValueError('Invalid completed shard: ' + stem)
        else:
            pending.append((key, es))
    if not pending:
        print('All OV shards already complete.', flush=True)
        return
    from stage1_scan import load_model, STATE
    from ri_test_diagnostics import environment, io_state, probe_weights, capture_failure
    model_dir = STATE / 'model' / m['source_config']['model']['revision']
    loading = environment(model_dir)
    save_loading = lambda report: write_json(out/'loading_diagnostics.json', report)
    print('Measuring bounded reads from checkpoint shards before loading.', flush=True)
    probe_weights(model_dir, loading, save_loading)
    loading['before_load'] = io_state()
    loading['status'] = 'loading'
    save_loading(loading)
    print(f'Loading pinned local model for {len(pending)} remaining heads (no forward passes).', flush=True)
    load_start = time.monotonic()
    torch, tok, model, lock = load_model()
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_id)
    loading.update(load_seconds=time.monotonic()-load_start, after_load=io_state(), status='loaded')
    save_loading(loading)
    print(f'Model loading: {loading["load_seconds"]:.2f}s (after bounded read probe).', flush=True)
    if lock != m['source_config']['model'] or model.config.num_attention_heads != 32 or len(model.model.layers) != 32:
        raise ValueError('Loaded model identity/shape mismatch')
    if getattr(model.config, 'num_key_value_heads', 32) != 32:
        raise ValueError('This implementation requires full multi-head attention')
    for p in ps.values():
        enc = tok(p['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        if enc['input_ids'] != p['token_ids'] or [list(x) for x in enc['offset_mapping']] != p['offsets']:
            raise ValueError('Tokenizer identity mismatch: ' + p['id'])
    write_json(out / 'model_gate.json', dict(tokenizer_prompts=len(ps), model=lock,
               cuda=torch.version.cuda, torch=torch.__version__, devices=str(model.hf_device_map),
               atol=0.0002, rtol=0.002,
               note='Per-head saved-score replication must also pass; no forward passes.'))
    start = time.monotonic()
    for done, ((li, hi), es) in enumerate(pending, 1):
        pids = {e['id'] for e in es}
        ctx = sorted({t for pid in pids for t in ps[pid]['token_ids']})
        tids = sorted({e['current_id'] for e in es})
        probs = ov_probabilities(torch, model, li, hi, tids, ctx)
        ctx_idx, tid_idx = {t:i for i,t in enumerate(ctx)}, {t:i for i,t in enumerate(tids)}
        enriched, errors = [], []
        checks = 0
        for e in es:
            p = ps[e['id']]
            f = next(f for f in p['test_facts'] if f['fi'] == e['fact'])
            visible = sorted(set(p['token_ids'][:e['j']+1]))
            event_probs = probs[tid_idx[e['current_id']], [ctx_idx[t] for t in visible]]
            def fail(reason):
                capture_failure(torch, model, out, e, p, tids, visible, event_probs, reason, write_json)
                raise ValueError(reason + '; see gate_failure.json and gate_failure_arrays.npz')
            if not np.isfinite(event_probs).all():
                fail('Nonfinite OV probabilities')
            a = normalized_scores(event_probs)
            if a is None:
                if e['scored']:
                    fail('Previously scored event now has zero denominator')
                enriched.append(dict(e, scores=None))
                continue
            if not e['scored']:
                fail('Previously unscored event now scored; inspect numeric drift')
            lookup = dict(zip(visible, a))
            pairs = [(e['strength_first'], lookup[e['target_first_id']]),
                     (e['strength_last'], lookup[e['target_last_id']])]
            pairs += [(c['score'], lookup[c['token_id']]) for c in e['entity_controls'].values()
                      if c['score'] is not None]
            if e.get('null_tok10') is not None:
                pairs.append((e['null_tok10'], lookup[p['token_ids'][10]]))
            for old, new in pairs:
                checks += 1
                errors.append(abs(float(old)-float(new)))
                if not np.isclose(old, new, atol=0.0002, rtol=0.002):
                    fail(f'OV replication failed: original={old}, recomputed={float(new)}, error={errors[-1]}')
            names, words = controls(p, f, e['j'])
            scores = {}
            for anchor, poskey in (('first', 'of'), ('last', 'ol')):
                target_id = p['token_ids'][f[poskey]]
                target = float(lookup[target_id])
                s = dict(target=target, target_id=target_id)
                for kind, cs in (('names', names), ('words', words)):
                    items = [dict(text=c['text'], position=c[anchor],
                                  token_id=p['token_ids'][c[anchor]],
                                  score=float(lookup[p['token_ids'][c[anchor]]]),
                                  collision=p['token_ids'][c[anchor]] == target_id) for c in cs]
                    s[kind] = items
                    s[kind + '_stats'] = compare_scores(target, items)
                scores[anchor] = s
            enriched.append(dict(e, scores=scores))
        stem = f'L{li}H{hi}'
        path = shards / (stem + '.jsonl.gz')
        write_lines(path, enriched)
        write_json(shards / (stem + '.json'), dict(manifest=manifest_sha, events=len(es),
                   sha256=digest(path), checks=checks, max_error=max(errors, default=0.0)))
        print(f'OV {done}/{len(pending)} {stem}: {len(es)} events; '
              f'max error {max(errors, default=0):.3g}; elapsed {time.monotonic()-start:.0f}s', flush=True)
    (out / 'gate_failure.json').unlink(missing_ok=True)


def memberships(query, pos, dist, variant):
    for scope in (['all_facts', 'query_fact'] if query else ['all_facts']):
        for location in ('all_test', pos):
            for separation in ('all', dist):
                for version in ('pooled', variant):
                    yield scope, location, separation, version


def denominators(ps):
    den = collections.Counter()
    for p in ps.values():
        for f in p['test_facts']:
            for j in range(max(f['s'], f['ol']), len(p['token_ids'])):
                for group in memberships(f['query'], position(p, j), distance(f, j), p['variant']):
                    den[(group, p['family'])] += 1
    return den


def sample_summary(values):
    vals = list(values)
    return None if not vals else float(np.mean(vals))


def analyze(out):
    m = check_manifest(out)
    ps = {p['id']: p for p in lines(out / 'prompts.jsonl.gz')}
    original = list(lines(out / 'events.jsonl.gz'))
    expected = collections.defaultdict(set)
    for e in original:
        expected[(e['layer'], e['head'])].add(e['event_id'])
    den = denominators(ps)
    all_groups = sorted({g for g, fam in den})
    metrics = ('target', 'names_target', 'words_target', 'names_control', 'words_control',
               'names_gap', 'words_gap', 'names_wins', 'words_wins',
               'names_ties', 'words_ties', 'names_top', 'words_top',
               'names_strict_top', 'words_strict_top', 'names_collisions', 'words_collisions',
               'names_rank_min', 'names_rank_max', 'words_rank_min', 'words_rank_max')
    # Summary is computed one head at a time to bound host memory.
    summary, family_rows, paired_rows = [], [], []
    fingerprint = digest(out / 'manifest.json')
    total_read = 0
    shard_hashes = {}
    by_variant = {(p['family'], p['order'], p['variant']): p for p in ps.values()}
    bases = [p for p in ps.values() if p['variant'] == 'base']
    for li, hi in m['heads']:
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        pass_counts = collections.Counter()
        per_prompt = collections.defaultdict(lambda: collections.defaultdict(list))
        prompt_passes = collections.Counter()
        stem = f'L{li}H{hi}'
        if (li, hi) in expected:
            path = out / 'heads' / (stem + '.jsonl.gz')
            meta = read_json(out / 'heads' / (stem + '.json'))
            if meta['manifest'] != fingerprint or digest(path) != meta['sha256']:
                raise ValueError('Unverified OV shard: ' + stem)
            shard_hashes[stem] = meta['sha256']
            es = list(lines(path))
            if len(es) != len(expected[(li, hi)]) or {e['event_id'] for e in es} != expected[(li, hi)]:
                raise ValueError('Incomplete/duplicate event shard: ' + stem)
        else:
            es = []
        total_read += len(es)
        for e in es:
            prompt_passes[(e['id'], e['source'], 'all_test')] += 1
            prompt_passes[(e['id'], e['source'], e['position'])] += 1
            for group in memberships(e['query_fact'], e['position'], e['distance'], e['variant']):
                pass_counts[(group, e['family'])] += 1
                if e['scores'] is None:
                    continue
                for anchor, s in e['scores'].items():
                    a = acc[(group, e['family'], anchor)]
                    a['target'].append(s['target'])
                    for kind in ('names', 'words'):
                        st = s[kind + '_stats']
                        if st['n']:
                            a[kind+'_target'].append(s['target'])
                            a[kind+'_control'].append(st['mean'])
                            for metric in ('gap', 'wins', 'ties', 'top', 'strict_top', 'collisions', 'rank_min', 'rank_max'):
                                a[kind+'_'+metric].append(st[metric])
            # Stable source-name correspondence across reorder/corruption; no fact-index matching.
            if e['scores'] is not None:
                for anchor, s in e['scores'].items():
                    for loc in ('all_test', e['position']):
                        key = (e['family'], e['source'], e['order'], e['variant'], loc, anchor)
                        a = per_prompt[key]
                        a['target'].append(s['target'])
                        for kind in ('names', 'words'):
                            if s[kind+'_stats']['n']:
                                a[kind+'_gap'].append(s[kind+'_stats']['gap'])
        for group in all_groups:
            fams = [f for f in m['families'] if den[(group, f)]]
            for anchor in POLICY['anchors']:
                n_pass = sum(pass_counts[(group, f)] for f in fams)
                row = dict(layer=li, head=hi, scope=group[0], position=group[1], distance=group[2],
                           variant=group[3], anchor=anchor, opportunities=sum(den[(group, f)] for f in fams),
                           passes=n_pass, families_evaluated=len(fams),
                           pass_families=sum(pass_counts[(group, f)] > 0 for f in fams),
                           frequency_event_weighted=n_pass/sum(den[(group, f)] for f in fams),
                           frequency=sample_summary(pass_counts[(group, f)]/den[(group, f)] for f in fams))
                for metric in metrics:
                    arrays = [acc.get((group, f, anchor), {}).get(metric, []) for f in fams]
                    row[metric+'_events'] = sum(map(len, arrays))
                    row[metric+'_families'] = sum(bool(a) for a in arrays)
                    row[metric+'_mean'] = sample_summary(np.mean(a) for a in arrays if a)
                summary.append(row)
                # Family output restricted to unstratified distance; event files retain all strata.
                if group[2] == 'all':
                    for f in fams:
                        a = acc.get((group, f, anchor), {})
                        if not pass_counts[(group, f)]:
                            continue
                        fr = {k:row[k] for k in ('layer','head','scope','position','variant','anchor')}
                        fr.update(family=f, passes=pass_counts[(group, f)], opportunities=den[(group, f)])
                        for metric in ('target','names_gap','words_gap'):
                            fr[metric+'_mean'] = sample_summary(a.get(metric, []))
                            fr[metric+'_events'] = len(a.get(metric, []))
                        family_rows.append(fr)
        # All source facts/order pairs included, even when neither variant has a pass.
        for p in bases:
            for fact in p['test_facts']:
                for comparison in ('corrupted', 'reorder'):
                    twin = by_variant[(p['family'], p['order'], comparison)]
                    tf = next(f for f in twin['test_facts'] if f['source'] == fact['source'])
                    for loc in ('all_test', 'final'):
                        for anchor in POLICY['anchors']:
                            keys = [(p['family'], fact['source'], p['order'], v, loc, anchor)
                                    for v in ('base', comparison)]
                            a, b = [per_prompt.get(k, {}) for k in keys]
                            # QK counts must include unscored passes, not just scored RI events.
                            counts = [prompt_passes[(q['id'], fact['source'], loc)] for q in (p, twin)]
                            pr = dict(layer=li, head=hi, family=p['family'], source=fact['source'],
                                      query_fact=fact['query'], order=p['order'], comparison=comparison,
                                      position=loc, anchor=anchor, base_target_name=fact['target'],
                                      twin_target_name=tf['target'], base_passes=counts[0], twin_passes=counts[1],
                                      both_pass=bool(counts[0] and counts[1]))
                            for prefix, q, fq, count in zip(('base','twin'), (p,twin), (fact,tf), counts):
                                opportunities = 1 if loc == 'final' else len(q['token_ids'])-max(fq['s'],fq['ol'])
                                pr[prefix+'_opportunities'] = opportunities
                                pr[prefix+'_frequency'] = count/opportunities
                            for metric in ('target','names_gap','words_gap'):
                                x, y = sample_summary(a.get(metric, [])), sample_summary(b.get(metric, []))
                                pr['base_'+metric] = x
                                pr['twin_'+metric] = y
                                pr['change_'+metric] = None if x is None or y is None else y-x
                            # Sparse output: neither-pass pairs are inferable from the frozen prompt universe.
                            if counts[0] or counts[1]:
                                paired_rows.append(pr)
        if hi == 31:
            print(f'Analyzed layer {li}/31', flush=True)
    if total_read != m['events']:
        raise ValueError('Global event count mismatch')
    write_csv(out / 'head_summary.csv', summary)
    write_csv(out / 'family_metrics.csv', family_rows)
    write_csv(out / 'paired_variants.csv', paired_rows)
    reasons = collections.defaultdict(list)
    for scope in POLICY['scopes']:
        for loc in ('all_test', 'final'):
            for anchor in POLICY['anchors']:
                for metric in ('target','names_gap','words_gap'):
                    eligible = [r for r in summary if r['scope'] == scope and r['position'] == loc
                                and r['distance'] == 'all' and r['variant'] == 'pooled'
                                and r['anchor'] == anchor and r[metric+'_events'] >= POLICY['min_events']
                                and r[metric+'_families'] >= POLICY['min_families']
                                and r[metric+'_mean'] is not None and r[metric+'_mean'] > 0]
                    eligible.sort(key=lambda r: (-r[metric+'_mean'], r['layer'], r['head']))
                    for rank, r in enumerate(eligible[:POLICY['top_k']], 1):
                        reasons[(r['layer'],r['head'])].append(dict(scope=scope, position=loc,
                            anchor=anchor, metric=metric, rank=rank, score=r[metric+'_mean'],
                            events=r[metric+'_events'], families=r[metric+'_families']))
    shortlist = [dict(layer=l, head=h, reasons=rs) for (l,h),rs in sorted(reasons.items())]
    write_json(out / 'candidates.json', dict(policy=POLICY, candidates=shortlist,
               historical_reference_heads=['L3H11','L9H22'],
               note='Descriptive union; no p-values, no semantic/causal claim. Post-selection audit deferred.'))
    outputs = {n:digest(out/n) for n in ('head_summary.csv','family_metrics.csv',
                                         'paired_variants.csv','candidates.json')}
    write_json(out / 'summary.json', dict(complete=True, manifest=fingerprint, events=total_read,
               heads=len(m['heads']), candidates=len(shortlist), outputs=outputs, shards=shard_hashes,
               gate='Every completed OV shard replicated saved scores within atol=2e-4, rtol=2e-3',
               limitations=['Discovery only; no significance claims',
                            'Raw current-token OV, not contextual output',
                            'Original QK selection and full-context normalization',
                            'Paired all_test means need not use identical positions; final is position-aligned',
                            'Family metrics are conditional on passes; shared demo bank remains',
                            'General controls are common non-name words in a fixed template']))
    print(f'COMPLETE: {len(shortlist)} discovery candidates. Outputs: {out}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare','ov','analyze','run'])
    ap.add_argument('--source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if args.source and args.source.resolve() == out:
        ap.error('Source and output must differ')
    out.mkdir(parents=True, exist_ok=True)
    with run_lock(out):
        if args.phase in ('prepare','run'):
            if args.source is None:
                ap.error('--source is required')
            prepare(args.source.resolve(), out)
        if args.phase in ('ov','run'):
            complete_ov(out)
        if args.phase in ('analyze','run'):
            analyze(out)


if __name__ == '__main__':
    main()
