#!/usr/bin/env python3
"""Phase-A single-hop dataset (SIH causal-audit study).

Purpose: causally test whether heads passing Ren et al.'s SIH definition carry
the model's single-hop semantic-relation ability. Anchor relation: mother-of
(kinship, families 000-199). Generalization world: inside (location, families
200-249), built identically but kept in separate files.

What this adds over v3.1's direct condition (the five gaps):
 1. A CORRUPTED TWIN for every direct question: the two answer-side names
    (qg <-> dg) are swapped inside the fact lines. The former distractor
    becomes the correct answer; mention counts, token counts, and every other
    token are unchanged, so clean and corrupted prompts are token-aligned
    position-by-position under both tokenizers -- exactly what activation
    patching needs. paired_base_id carries the FULL id incl. order suffix.
 2. FACT-LEVEL ANNOTATIONS: every fact line in the prompt (demos included)
    is recorded as a triple with character spans for head (child/inner) and
    tail (mother/container) -- the anchors the Relation Index needs.
 3. A REORDER control for the direct question (is 1-hop layout-robust?).
 4. SINGLE-HOP-ONLY demonstration banks (mother-only / location-only), order
    fully crossed, so demos inject no two-hop mechanism into the context.
 5. A discovery/validation SPLIT (family parity) fixed at build time.

Names: candidate pairs (qg, dg) drawn from the same DUAL-tokenizer length
bucket (Pythia + OLMo-2), so one dataset serves both models.
Random-triple RI floor is an analysis-time control and needs no data here.
"""
import collections, json, random, re
from pathlib import Path

SEED = 20260910
HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_singlehop'; OUT.mkdir(exist_ok=True)
rng = random.Random(SEED)

from tokenizers import Tokenizer
PYTHIA = Tokenizer.from_file(str(HERE.parent / 'pilot_v2' / 'tokenizer_source' / 'tokenizer.json'))
OLMO = Tokenizer.from_file(str(HERE / 'olmo2_tokenizer' / 'tokenizer.json'))
def sig(name):
    """Token counts in every context a candidate name occupies in a prompt:
    line start (no prefix), mid-line (' '+name), and before the period
    (' '+name+'.'). Pairs sharing the FULL signature keep clean and corrupted
    prompts token-aligned position-by-position under both tokenizers."""
    out = []
    for tok in (PYTHIA, OLMO):
        for ctx in (name, ' ' + name, ' ' + name + '.'):
            out.append(len(tok.encode(ctx, add_special_tokens=False).ids))
    return tuple(out)

# ---------- name grammars (same as v3) ----------
def kinship_names():
    onsets = list('BDFGHJKLMNPRSTVWZ')
    v1 = ['a','e','i','o','u']
    mids = ['ln','rn','ld','rl','st','nd','lm','rv','lt','sm','rd','nt']
    ends = ['a','ia','na','la','ra']
    combos = [(o,a,m,e) for o in onsets for a in v1 for m in mids for e in ends]
    rng.shuffle(combos)
    for o,a,m,e in combos: yield o + a + m + e
def location_names():
    onsets = list('bdfgjklmnprstvz')
    v = ['a','e','i','o','u']
    mids = ['rv','st','ld','nt','rp','lg','sk','nd','rm','lt']
    ends = ['el','in','ot','ar','ul']
    combos = [(o,a,m,e) for o in onsets for a in v for m in mids for e in ends]
    rng.shuffle(combos)
    for o,a,m,e in combos: yield o + a + m + e

def make_pools(builder):
    by_sig, flat = collections.defaultdict(list), []
    for name in builder():
        by_sig[sig(name)].append(name); flat.append(name)
    return by_sig, flat

POOLS = {'kinship': make_pools(kinship_names), 'location': make_pools(location_names)}
USED = set()

def draw_pair(world):
    """Two names with identical dual-tokenizer signature (the candidate pair)."""
    by_sig, _ = POOLS[world]
    for s in sorted(by_sig, key=lambda s: -len(by_sig[s])):
        avail = [n for n in by_sig[s] if n not in USED]
        if len(avail) >= 2:
            a, b = avail[0], avail[1]
            USED.add(a); USED.add(b)
            return a, b
    raise RuntimeError('candidate pool exhausted')

def draw(world, n):
    _, flat = POOLS[world]
    out = []
    for name in flat:
        if name in USED: continue
        USED.add(name); out.append(name)
        if len(out) == n: return out
    raise RuntimeError('pool exhausted')

# ---------- world grammar ----------
W = {
 'kinship': dict(
    fact=lambda child, parent: f'{parent} is the mother of {child}.',
    q=lambda e: f'Question: Who is the mother of {e}?',
    fact_rx=re.compile(r'^([A-Za-z]+) is the mother of ([A-Za-z]+)\.$'),  # (tail, head)
    relation='mother_of', answer_line=True),
 'location': dict(
    fact=lambda inner, outer: f'The {inner} is inside the {outer}.',
    q=lambda e: f'Question: Where is the {e}?',
    fact_rx=re.compile(r'^The ([a-z]+) is inside the ([a-z]+)\.$'),       # (head, tail)
    relation='inside', answer_line=True),
}

def new_family(world):
    qg, dg = draw_pair(world)                     # candidate pair: same dual signature
    qc, qm, dc, dm = draw(world, 4)
    return dict(qc=qc, qm=qm, qg=qg, dc=dc, dm=dm, dg=dg)

def facts_of(world, fam, order, swap=False):
    g, d = (fam['dg'], fam['qg']) if swap else (fam['qg'], fam['dg'])
    F = W[world]['fact']
    q = [F(fam['qc'], fam['qm']), F(fam['qm'], g)]
    dd = [F(fam['dc'], fam['dm']), F(fam['dm'], d)]
    return q + dd if order == 0 else dd + q

def block(world, fam, facts, answer=None):
    lines = facts + [W[world]['q'](fam['qm'])]
    lines.append('Answer:' + (f' {answer}.' if answer else ''))
    return '\n'.join(lines)

# ---------- demonstrations: SINGLE-HOP ONLY, order fully crossed ----------
DEMOS = {}
for world in W:
    demos = []
    for i, order in enumerate([0, 1] * 8):        # 16 demos, 8 per order cell
        fam = new_family(world)
        demos.append(dict(id=f'{world}/sh-demo/{i}', world=world, order=order, hops=1,
                          text=block(world, fam, facts_of(world, fam, order), answer=fam['qg']),
                          names=[fam[k] for k in ('qc','qm','qg','dc','dm','dg')]))
    DEMOS[world] = demos

def demo_ids_for(world, qseed, shots):
    r = random.Random(qseed)
    cells = collections.defaultdict(list)
    for d in DEMOS[world]: cells[d['order']].append(d)
    take = []
    for o in (0, 1): take += r.sample(cells[o], shots // 2)
    r.shuffle(take)
    return take

# ---------- annotations ----------
def annotate(prompt, world):
    """Every fact line in the prompt as a triple with char spans; plus spans of all entities."""
    rx = W[world]['fact_rx']
    facts, offset = [], 0
    n_blocks = prompt.count('\n\n') + 1
    for bi, blk in enumerate(prompt.split('\n\n')):
        for line in blk.split('\n'):
            m = rx.match(line)
            if m:
                if world == 'kinship': tail, head = m.group(1), m.group(2)
                else: head, tail = m.group(1), m.group(2)
                hm = re.search(r'(?<![A-Za-z])' + re.escape(head) + r'(?![A-Za-z])', line)
                tm = re.search(r'(?<![A-Za-z])' + re.escape(tail) + r'(?![A-Za-z])', line)
                facts.append(dict(
                    block='test' if bi == n_blocks - 1 else 'demo',
                    relation=W[world]['relation'], head=head, tail=tail,
                    head_span=[offset + hm.start(), offset + hm.end()],
                    tail_span=[offset + tm.start(), offset + tm.end()], line=line))
            offset += len(line) + 1               # +1 for the newline
        offset += 1                               # the blank line between blocks
    return facts

def spans(prompt, names):
    out = []
    for n in set(names):
        for m in re.finditer(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', prompt):
            out.append(dict(entity=n, start=m.start(), end=m.end()))
    return sorted(out, key=lambda s: s['start'])

# ---------- build ----------
FILES = {'kinship': ('singlehop_v1_{s}shot.jsonl', range(0, 200)),
         'location': ('singlehop_loc_v1_{s}shot.jsonl', range(200, 250))}
MODEL_TAGS = ['EleutherAI/pythia-6.9b', 'allenai/OLMo-2-1124-7B']
problems = []
audit = collections.defaultdict(int)

for world, (pattern, fam_range) in FILES.items():
    rows_by_shots = {0: [], 4: [], 12: []}
    demo_names = {n for d in DEMOS[world] for n in d['names']}
    for fidx in fam_range:
        fam = new_family(world)
        variants = [('base', 0, False, None), ('corrupted', 0, True, None),
                    ('reorder', 0, False, 'shuffle')]
        for qi, (variant, _, swap, mode) in enumerate(variants):
            for order in (0, 1):
                facts = facts_of(world, fam, order, swap=swap)
                if mode == 'shuffle':
                    random.Random(7000 * fidx + order).shuffle(facts)
                gold = fam['dg'] if swap else fam['qg']
                distractor = fam['qg'] if swap else fam['dg']
                test_block = block(world, fam, facts)
                qid = f'{fidx:03d}/direct/{variant}/{qi}'
                # audit: balance
                ftext = '\n'.join(facts)
                c = lambda n: len(re.findall(r'(?<![A-Za-z])' + re.escape(n) + r'(?![A-Za-z])', ftext))
                if c(gold) != c(distractor): problems.append(('imbalance', qid, order))
                if sig(gold) != sig(distractor): problems.append(('sig', qid))
                if gold in demo_names or distractor in demo_names: problems.append(('demo_leak', qid))
                for shots in (0, 4, 12):
                    if shots:
                        # One demo set per (family, shots), shared by ALL variants and
                        # both orders: clean/corrupted/reorder rows are patching twins
                        # and must differ ONLY in the test block.
                        demos = demo_ids_for(world, f'{world}/{fidx}/{shots}', shots)
                        prompt = '\n\n'.join([d['text'] for d in demos] + [test_block])
                        dids = [d['id'] for d in demos]
                    else:
                        prompt, dids = test_block, []
                    cands = [gold, distractor]
                    random.Random(qid + str(order)).shuffle(cands)
                    rows_by_shots[shots].append(dict(
                        id=f'{qid}/order{order}', question_id=qid, family=fidx, world=world,
                        task='name_completion', check='direct', variant=variant, hops=1,
                        relation=W[world]['relation'], question_entity=fam['qm'],
                        gold=gold, candidates=cands, bridge=None,
                        corruption='answer_swap' if swap else None,
                        option_order=order, option_pair_id=f'{qid}/order{1 - order}',
                        paired_base_id=f'{fidx:03d}/direct/base/0/order{order}' if variant != 'base' else None,
                        split='discovery' if fidx % 2 == 0 else 'validation',
                        dataset_version='singlehop_v1', seed=SEED, shots=shots,
                        matched_tokenizers=MODEL_TAGS, length_matched=True,
                        demo_ids=dids, prompt=prompt,
                        entity_spans=spans(prompt, [gold, distractor, fam['qm']]),
                        facts=annotate(prompt, world)))
                    audit[(world, variant, shots)] += 1
    for shots, rows in rows_by_shots.items():
        p = OUT / pattern.format(s=shots)
        with p.open('w', encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r) + '\n')
        print(p.name, len(rows))

json.dump({'demos': {w: [d['text'] for d in DEMOS[w]] for w in DEMOS}},
          (OUT / 'demonstrations_singlehop.json').open('w'), indent=1)
json.dump({'counts': {str(k): v for k, v in audit.items()}, 'problems': problems},
          (OUT / 'build_audit_singlehop.json').open('w'), indent=1)
print('problems:', len(problems))
for p in problems[:10]: print(p)
