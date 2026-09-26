"""CPU-only helpers for the stage-1 observational audit."""
import hashlib
from pathlib import Path
import numpy as np


def first_divergence(tok, row):
    """Condition on the shared answer prefix; never on a candidate-only token."""
    prompt = tok.encode(row['prompt'], add_special_tokens=False)
    other = next(c for c in row['candidates'] if c != row['gold'])
    tails = []
    for name in (row['gold'], other):
        full = tok.encode(row['prompt'] + ' ' + name + '.', add_special_tokens=False)
        if full[:len(prompt)] != prompt:
            raise ValueError('Tokenizer changed prompt boundary: ' + row['id'])
        tails.append(full[len(prompt):])
    a, b = tails
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if k is None:
        raise ValueError('Candidates have no divergent token: ' + row['id'])
    return dict(input_ids=prompt + a[:k], prompt_length=len(prompt),
                shared_prefix=a[:k], gold_token=a[k], other_token=b[k])


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class DominanceStore:
    """All qualifying float32 ratios, on disk; exact empirical quantiles.

    Files are sorted/partitioned in place by quantile computation. They contain
    values, not event identities. Global weighting is one vote per eligible
    fact/head/position event, including repeated demo prefixes.
    """
    def __init__(self, out, layers):
        self.paths = [Path(out) / 'dominance_all.f32'] + [
            Path(out) / f'dominance_L{i}.f32' for i in range(layers)]
        self.files = [p.open('wb') for p in self.paths]

    def add(self, layer, values):
        a = np.asarray(values, dtype='<f4').reshape(-1)
        if not np.isfinite(a).all():
            raise ValueError('Non-finite dominance ratio')
        a.tofile(self.files[0]); a.tofile(self.files[layer + 1])

    def finish(self):
        for f in self.files:
            f.close()
        def stats(p):
            n = p.stat().st_size // 4
            if not n:
                return {'n': 0, **{f'p{k}': None for k in (50, 75, 90, 95, 99)}}
            a = np.memmap(p, dtype='<f4', mode='r+', shape=(n,))
            below = int(np.count_nonzero(a <= 2.2))
            q = np.percentile(a, [50, 75, 90, 95, 99], overwrite_input=True)
            a.flush(); del a
            return dict(n=n, fraction_le_2_2=below/n,
                        **{f'p{k}': float(v) for k, v in zip((50,75,90,95,99), q)})
        result = stats(self.paths[0])
        result['by_layer'] = {str(i): stats(p) for i, p in enumerate(self.paths[1:])}
        result['population'] = 'all heads; all annotated fact/position events with argmax == source; no tau filter'
        result['weighting'] = 'event weighted, including repeated demonstration prefixes'
        return result
