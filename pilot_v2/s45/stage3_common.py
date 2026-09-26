"""Portable Stage-3 protocol, annotation and file helpers (CPU only)."""
import contextlib
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

VERSION = 1
SEED = 20260921
POLICY = dict(version=VERSION, seed=SEED, heads=105, families=89,
              position_families=20, dominance=2.2, min_events=20, min_families=10,
              synthetic_trials=32, repeat_length=32, retrieval_records=16,
              weight_nonname_tokens=128, output_rank_population='visible name anchors plus current token',
              patch_scope='original prompt only; answer prefix recomputed',
              intervention='clean recipient, corrupt donor; fixed clean-answer sign',
              attention_value_order=['clean_clean', 'corrupt_attention', 'corrupt_values', 'corrupt_both'],
              probe_tolerance=0.05, inert_tolerance=0.001, reconstruction_tolerance=0.02,
              ri_atol=0.0002, ri_rtol=0.002, posterior_selection='discovery, descriptive only')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def json_read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def read_lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_lines(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def npz_write(path, **data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **data)
    temp.replace(path)


def slug(s): return re.sub(r'[^A-Za-z0-9_.-]', '_', s)
def head_name(h): return f'L{h//32}H{h%32}'


@contextlib.contextmanager
def run_lock(out):
    path = Path(out)/'.running.lock'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(pid=os.getpid(), job=os.environ.get('SLURM_JOB_ID')), f)
    try: yield
    finally: path.unlink(missing_ok=True)


def span_tokens(offsets, span):
    ids = [i for i,(a,b) in enumerate(offsets) if a < span[1] and b > span[0]]
    if not ids: raise ValueError(f'Token span missing: {span}')
    return ids


def annotate(tok, row):
    enc = tok(row['prompt'], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc['input_ids'], enc['offset_mapping']
    start = row['prompt'].rfind('\n\n')+2
    test = [i for i,(a,b) in enumerate(offsets) if b > start]
    facts = []
    for fi,f in enumerate(row['facts']):
        if f['block'] != 'test': continue
        facts.append(dict(index=fi, source=f['head'], target=f['tail'],
                          source_positions=span_tokens(offsets,f['head_span']),
                          target_positions=span_tokens(offsets,f['tail_span']),
                          query=f['head']==row.get('question_entity',row.get('query_source'))))
    if len(facts)!=4 or sum(f['query'] for f in facts)!=1: raise ValueError('Fact annotation mismatch')
    return dict(ids=ids, offsets=offsets, test_positions=test, facts=facts)


def changed_query(row):
    """Change only the queried child; use the other answer-side fact already present."""
    query=row.get('question_entity',row.get('query_source'))
    alternatives=[f for f in row['facts'] if f['block']=='test' and f['tail'] in row['candidates'] and f['head']!=query]
    if len(alternatives)!=1: raise ValueError('Ambiguous alternative query')
    fact=alternatives[0]; old='Question: Who is the mother of '+query+'?'
    i=row['prompt'].rfind(old)
    if i<0: raise ValueError('Missing final question')
    new=dict(row, id=row['id']+'/query_change', variant='query_change',
             question_entity=fact['head'], gold=fact['tail'],order=row.get('order',row.get('option_order')))
    new['prompt']=row['prompt'][:i]+row['prompt'][i:].replace(old,'Question: Who is the mother of '+fact['head']+'?',1)
    return new


def family_means(values, families):
    values=np.asarray(values); families=np.asarray(families)
    return np.stack([values[families==f].mean(axis=0) for f in sorted(set(families))])


def bootstrap_median_difference(a, b, draws=20000, seed=SEED):
    """Paired family resampling; never borrow a sample size from another analysis."""
    if a.shape[0]!=b.shape[0] or a.ndim!=2 or b.ndim!=2: raise ValueError('Family arrays differ')
    rng=np.random.default_rng(seed); n=a.shape[0]; samples=[]
    for _ in range(draws):
        ix=rng.integers(0,n,n)
        samples.append(float(np.median(a[ix].mean(0))-np.median(b[ix].mean(0))))
    return dict(difference=float(np.median(a.mean(0))-np.median(b.mean(0))),
                interval=np.quantile(samples,[.025,.975]).tolist(), families=n, draws=draws,
                seed=seed, interpretation='descriptive; head selection held fixed')


def normalized_ri(probabilities):
    x=np.maximum(np.asarray(probabilities,float)-np.mean(probabilities),0)
    return x/x.sum() if x.sum()>0 else None
