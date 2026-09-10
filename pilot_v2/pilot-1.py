"""Multi-model distinct-name behavioral pilot (Pythia-6.9B / OLMo-2-7B).

Data and reporting need only Python's stdlib. Every dataset is bound to the
tokenizer(s) it was length-matched for; the runner refuses a model/dataset
combination that was not audited together (see pilot_v3/audit_tokenizer.py).
"""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('PILOT_STORAGE', str(ROOT / 'storage'))).resolve()
RUNS = Path(os.environ.get('PILOT_RUNS', str(STATE / 'runs'))).resolve()
os.environ.setdefault('HF_HOME', str(STATE / 'hf'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODELS = {
    'pythia-6.9b': dict(model_id='EleutherAI/pythia-6.9b', lock='model_lock.json'),
    'olmo2-7b': dict(model_id='allenai/OLMo-2-1124-7B', lock='model_lock_olmo2.json'),
}
DEFAULT_MODEL = 'pythia-6.9b'
LABELS = (' Yes', ' No')

PROMPT_VERSIONS = ['v2', 'v3', 'completion', 'completion_v2',
                   'completion_v3', 'completion_v3_controls',
                   'completion_v3_olmo2', 'completion_v3_olmo2_controls',
                   'singlehop_v1', 'singlehop_loc_v1']


def dataset_path(shots, version):
    fixed = {
        'completion_v3_controls': 'controls_v3_4shot.jsonl',
        'completion_v3_olmo2_controls': 'controls_v3_olmo2_4shot.jsonl',
    }
    if version in fixed:
        if shots != 4:
            raise ValueError('Control files are built at 4 shots only.')
        return ROOT / 'data' / fixed[version]
    per_shot = {
        'completion': 'completion_{s}shot.jsonl',
        'completion_v2': 'completion_v2_{s}shot.jsonl',
        'completion_v3': 'completion_v3_{s}shot.jsonl',
        'completion_v3_olmo2': 'completion_v3_olmo2_{s}shot.jsonl',
        'singlehop_v1': 'singlehop_v1_{s}shot.jsonl',
        'singlehop_loc_v1': 'singlehop_loc_v1_{s}shot.jsonl',
    }
    if version in per_shot:
        return ROOT / 'data' / per_shot[version].format(s=shots)
    if shots == 0:
        raise ValueError('Zero-shot is supported by the completion datasets only.')
    suffix = '_prompt_v3' if version == 'v3' else ''
    return ROOT / 'data' / f'pilot_{shots}shot{suffix}.jsonl'


def check_data_model_match(rows, model_key):
    """A dataset may only be scored by a model whose tokenizer it was
    length-matched for. Legacy rows (no matched_tokenizers field) were matched
    for Pythia only."""
    model_id = MODELS[model_key]['model_id']
    tags = {tuple(r.get('matched_tokenizers', ['EleutherAI/pythia-6.9b'])) for r in rows}
    bad = [t for t in tags if model_id not in t]
    if bad:
        raise ValueError(
            f'Dataset was not tokenizer-matched for {model_id} (matched for: {sorted(bad)}). '
            f'Use the dataset variant built for this model '
            f'(e.g. --prompt-version completion_v3_olmo2 for olmo2-7b).')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    spec = MODELS[args.model]
    lock_path = ROOT / spec['lock']
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock['model_id'] != spec['model_id']:
            raise ValueError(f'{spec["lock"]} does not match {spec["model_id"]}.')
    else:
        info = HfApi().model_info(spec['model_id'], revision='main')
        lock = {'model_id': spec['model_id'], 'revision': info.sha, 'requested_revision': 'main'}
        dump(lock_path, lock)
        print(f'Pinned {spec["model_id"]} at revision {info.sha}. '
              f'Commit {spec["lock"]} to git for reproducibility.')
    target = STATE / 'model' / lock['revision']
    snapshot_download(spec['model_id'], revision=lock['revision'], local_dir=str(target),
                      allow_patterns=['*.safetensors', '*.safetensors.index.json',
                                      'config.json', 'tokenizer.json', 'tokenizer_config.json',
                                      'special_tokens_map.json', 'generation_config.json'])
    print(f'Model ready: {target}\nKeep {spec["lock"]} when moving between providers.')


def load_runtime(model_key):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Submit a GPU job first.')
    spec = MODELS[model_key]
    lock = json.loads((ROOT / spec['lock']).read_text())
    if lock['model_id'] != spec['model_id']:
        raise ValueError('Wrong model lock.')
    model_dir = STATE / 'model' / lock['revision']
    torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32 = False
    memory = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        memory[i] = max(0, free - 2 * 2**30)
    memory['cpu'] = 0
    print(f'Loading {spec["model_id"]}; GPU budgets:', memory, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16,
        use_safetensors=True, attn_implementation='sdpa', low_cpu_mem_usage=True,
        device_map='auto', max_memory=memory)
    mapping = model.hf_device_map
    if any(str(v) in ('cpu', 'disk') for v in mapping.values()):
        raise RuntimeError('GPU memory is insufficient. Use a larger GPU or two GPUs; CPU offload is disabled.')
    model.eval()
    model.config.use_cache = False
    print('Model loaded. Device map:', mapping, flush=True)
    return torch, tokenizer, model, lock


def label_tokens(tokenizer, prompt, label):
    base = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + label, add_special_tokens=False)
    if full[:len(base)] != base:
        raise ValueError('Answer boundary changes prompt tokenization.')
    return base, full[len(base):]


def evaluate_one(torch, tokenizer, model, row):
    if row.get('task') == 'name_completion':
        from completion_evaluation import evaluate_one as evaluate_completion
        return evaluate_completion(torch, tokenizer, model, row)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    prompt = row['prompt']
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    base = enc['input_ids']
    if len(base) + 8 > model.config.max_position_embeddings:
        raise ValueError('Context limit exceeded; truncation is disabled.')
    labels = {}
    for label in LABELS:
        prefix_ids, tail = label_tokens(tokenizer, prompt, label)
        if prefix_ids != base or len(tail) != 1:
            raise ValueError('This runner requires one-token answer labels.')
        labels[label.strip()] = tail[0]
    input_device = model.get_input_embeddings().weight.device
    ids = torch.tensor([base], device=input_device)
    # Match the successful numerical probe: FP16 weights, math-only SDPA.
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        # One shared forward pass: scoring and first generated token use identical logits.
        first = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        if not torch.isfinite(first).all():
            raise ValueError('Non-finite logits.')
        logp = first.log_softmax(-1)
        scores = {label: logp[token].item() for label,token in labels.items()}
        first_logits = {label: first[token].item() for label,token in labels.items()}
        tail = []
        next_logits = first
        for step in range(8):
            token = next_logits.argmax().item()
            tail.append(token)
            if token == tokenizer.eos_token_id or step == 7:
                break
            ids = torch.cat([ids, torch.tensor([[token]], device=input_device)], dim=1)
            next_logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
            if not torch.isfinite(next_logits).all():
                raise ValueError(f'Non-finite generation logits for {row["id"]}, step {step + 1}.')
    generated = tokenizer.decode(tail, skip_special_tokens=True)
    match = re.match(r'^\s*(Yes|No)\b', generated)
    free_label = match.group(1) if match else None
    margin = scores['Yes'] - scores['No']
    predicted = 'Yes' if margin > 0 else 'No' if margin < 0 else 'Tie'
    anchors = [dict(span, token_indices=[j for j,(a,b) in enumerate(enc['offset_mapping'])
                                        if a < span['end'] and b > span['start']])
               for span in row['entity_spans']]
    return dict(row, candidate_logprobs=scores, first_answer_logits=first_logits,
                answer_token_ids=labels, generated_token_ids=tail,
                yes_minus_no=margin, predicted=predicted, correct=predicted==row['gold'],
                generated=generated, free_label=free_label, format_ok=free_label is not None,
                free_correct=free_label==row['gold'], token_ids=base,
                token_offsets=enc['offset_mapping'], token_anchors=anchors)


def read_results(path, repair=False):
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, valid_bytes = [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
            valid_bytes += len(line)
        except (ValueError, UnicodeDecodeError):
            if repair and i == len(lines)-1 and not line.endswith(b'\n'):
                with path.open('r+b') as handle:
                    handle.truncate(valid_bytes)
            else:
                raise
    # A complete JSON record without newline also needs a separator before append.
    if repair and rows and path.read_bytes() and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as handle:
            handle.write(b'\n')
    return rows


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.96
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [center-radius, center+radius]


def summarize(out):
    rows = read_results(out / 'results.jsonl')
    if rows and rows[0].get('task') == 'name_completion':
        from completion_evaluation import summarize as summarize_completion
        return summarize_completion(out, rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['check'], row['variant'], row['gold'])].append(row)
    stats = []
    for (check, variant, gold), values in sorted(groups.items()):
        n = len(values)
        k = sum(v['correct'] for v in values)
        low, high = wilson(k, n)
        stats.append(dict(check=check, variant=variant, gold=gold, n=n,
                          candidate_accuracy=k/n, tie_rate=sum(v['predicted']=='Tie' for v in values)/n, ci95_low=low, ci95_high=high,
                          format_rate=sum(v['format_ok'] for v in values)/n,
                          free_accuracy=sum(v['free_correct'] for v in values)/n,
                          mean_yes_minus_no=sum(v['yes_minus_no'] for v in values)/n))
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        if stats:
            writer = csv.DictWriter(f, fieldnames=list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)
    families = defaultdict(dict)
    for r in rows:
        families[r['family']][(r['check'], r['variant'], r['polarity'])] = r
    paired = defaultdict(list)
    for values in families.values():
        clean = values.get(('twohop', 'base', 'positive'))
        for variant in ('first', 'second'):
            broken = values.get(('broken', variant, 'negative'))
            if clean and broken:
                paired['clean_and_broken_correct_' + variant].append(clean['correct'] and broken['correct'])
                paired['margin_decreases_' + variant].append(clean['yes_minus_no'] > broken['yes_minus_no'])
        for variant in ('reorder', 'distractors', 'rename'):
            for polarity in ('positive', 'negative'):
                original = values.get(('twohop', 'base', polarity))
                changed = values.get(('robustness', variant, polarity))
                if original and changed:
                    key = variant + '_' + polarity
                    paired['both_correct_' + key].append(original['correct'] and changed['correct'])
                    paired['prediction_agrees_' + key].append(original['predicted'] == changed['predicted'])
    dump(out / 'paired_metrics.json', {k: {'n_families': len(v), 'rate': sum(v)/len(v),
                                          'ci95': wilson(sum(v), len(v))} for k,v in paired.items()})
    print(f'{len(rows)} completed prompts. Read {out / "summary.csv"}')


def run(args):
    path = dataset_path(args.shots, args.prompt_version)
    rows = read_results(path)
    if not rows:
        raise ValueError('Dataset missing or empty; run data first.')
    check_data_model_match(rows, args.model)
    if args.limit_families:
        rows = [r for r in rows if r['family'] < args.limit_families]
    rows = [r for r in rows if r['family'] % args.shard_count == args.shard_index]
    if not rows:
        raise ValueError('Empty worker partition; reduce the number of workers.')
    torch, tokenizer, model, lock = load_runtime(args.model)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'transformers', 'huggingface-hub', 'safetensors', 'accelerate']}
    settings = dict(model=lock, model_key=args.model,
                    data_sha256=digest(path), code_sha256=digest(Path(__file__)),
                    shots=args.shots, limit_families=args.limit_families,
                    shard_index=args.shard_index, shard_count=args.shard_count, dtype='float16',
                    attention='sdpa_math', versions=versions, generation_max_new_tokens=8)
    if args.prompt_version.startswith(('completion', 'singlehop')):
        settings['completion_code_sha256'] = digest(ROOT / 'completion_evaluation.py')
        settings['candidate_scoring'] = 'sum_logprobs_name_and_period'
    manifest = out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            raise ValueError('Run settings differ: choose a new --name to avoid mixing experiments.')
    else:
        dump(manifest, settings)
    gpu = torch.cuda.get_device_properties(0)
    session_id = str(time.time_ns())
    session = dict(python=sys.version, platform=platform.platform(), gpu=gpu.name,
                   gpu_gib=gpu.total_memory/2**30, cuda=torch.version.cuda,
                   all_gpus=[dict(name=torch.cuda.get_device_name(i), total_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(torch.cuda.device_count())],
                   device_map={k:str(v) for k,v in model.hf_device_map.items()},
                   disk_free_gib=shutil.disk_usage(STATE).free/2**30,
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'), versions=versions)
    dump(out / f'session_{session_id}.json', session)
    old = read_results(out / 'results.jsonl', repair=True)
    done = {r['id'] for r in old}
    if len(done) != len(old):
        raise ValueError('Duplicate result IDs; do not run concurrent writers with the same name.')
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    start = time.monotonic()
    count = 0
    with (out / 'results.jsonl').open('a', encoding='utf-8') as f:
        for row in rows:
            if row['id'] in done:
                continue
            result = evaluate_one(torch, tokenizer, model, row)
            result['session_id'] = session_id
            f.write(json.dumps(result) + '\n')
            f.flush()
            os.fsync(f.fileno())
            count += 1
            if count % 12 == 0:
                elapsed = time.monotonic()-start
                print(f'{len(done)+count}/{len(rows)} prompts; {elapsed/count:.2f} s/prompt', flush=True)
    session.update(elapsed_seconds=time.monotonic()-start, completed_this_session=count,
                   peak_allocated_gpu_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())])
    dump(out / f'session_{session_id}.json', session)
    summarize(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    dl = sub.add_parser('download')
    dl.add_argument('--model', choices=sorted(MODELS), default=DEFAULT_MODEL)
    eval_parser = sub.add_parser('run')
    eval_parser.add_argument('--name', required=True)
    eval_parser.add_argument('--model', choices=sorted(MODELS), default=DEFAULT_MODEL)
    eval_parser.add_argument('--shots', type=int, choices=[0,4,12], default=12)
    eval_parser.add_argument('--prompt-version', choices=PROMPT_VERSIONS, default='v2')
    eval_parser.add_argument('--limit-families', type=int, default=0)
    eval_parser.add_argument('--shard-index', type=int, default=0)
    eval_parser.add_argument('--shard-count', type=int, default=1)
    report = sub.add_parser('report')
    report.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.cmd == 'run' and not (args.shard_count > 0 and 0 <= args.shard_index < args.shard_count):
        parser.error('Invalid shard index/count.')
    if getattr(args, 'name', None) and not re.fullmatch(r'[A-Za-z0-9_-]+', args.name):
        parser.error('--name must contain only letters, numbers, underscores or hyphens.')
    if getattr(args, 'limit_families', 0) < 0 or getattr(args, 'families', 1) <= 0:
        parser.error('Family counts must be positive (0 means no run limit).')
    if args.cmd == 'download': download(args)
    elif args.cmd == 'run':
        # Kernel lock releases automatically on job termination (Linux providers).
        import fcntl
        # Keep locks on compute storage: Colab Drive FUSE may not support flock.
        lock_dir = STATE / 'locks'
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / (args.name + '.lock')).open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('This run already has an active writer. Use another --name.')
            run(args)
    elif args.cmd == 'report': summarize(RUNS / args.name)


if __name__ == '__main__':
    main()
