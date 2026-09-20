"""Stage-2 analysis after the extension-v2 run: new test-only RI vs exact causal importance,
Scope P vs Scope F, logit decomposition, and the audit verdicts under the new definition.

CPU only. Inputs (relative to the project root unless --results is given):
  results/stage2_v1/{screen,extension}/*.npz, head_effects.csv, shortlist.json, manifest.json
  results/stage2_v2_extension/pairs/*.npz, manifest.json
  results/ri_test_v2/head_summary.csv, candidates.json
  results/stage2_v2_coverage/coverage_v2.csv
Outputs: results/stage2_v2_analysis/  (tables as CSV/JSON, figures as PNG, summary.md)

Conventions (as in analyze_stage2_results.py): importance I_h = -delta; family-level means (orders
averaged first); percentile bootstrap over whole families, 20,000 draws, seed 20260915. All intervals
are descriptive; the head groups were selected on the same discovery families.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

H = 32; SEED = 20260915; DRAWS = 20000
ROOT = Path(__file__).resolve().parent


def name(f): return f'L{f // H}H{f % H}'


def rankdata(a):
    a = np.asarray(a, float); o = np.argsort(a, kind='stable'); r = np.empty(len(a)); i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[o[j]] == a[o[i]]: j += 1
        r[o[i:j]] = (i + j - 1) / 2; i = j
    return r


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return None
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0: return None
    return float(np.corrcoef(ra, rb)[0, 1])


def fam_mean(X, fam):
    """X: [pairs, ...]; average orders within family, then over families."""
    fams = sorted(set(fam)); return np.stack([X[fam == f].mean(axis=0) for f in fams]).mean(axis=0), fams


def fam_matrix(X, fam):
    fams = sorted(set(fam)); return np.stack([X[fam == f].mean(axis=0) for f in fams])


def boot_ci(F, rng, draws=DRAWS):
    """F: [families, k] family-level values. Returns (lo, hi) per column."""
    n = F.shape[0]; idx = rng.integers(0, n, size=(draws, n))
    means = F[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    R = Path(a.results); out = Path(a.out or R / 'stage2_v2_analysis'); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- load v1
    m1 = json.loads((R / 'stage2_v1' / 'manifest.json').read_text()); pairs = m1['pairs']
    fam = np.array([p['family'] for p in pairs]); order = np.array([p['order'] for p in pairs])
    exact_all = np.array([p['exact_all'] for p in pairs])
    D = json.loads((R / 'stage2_v1' / 'shortlist.json').read_text())['heads']
    attr = np.zeros((len(pairs), 1024)); ext = np.zeros((len(pairs), len(D))); ex40 = {}
    m_clean = np.zeros(len(pairs)); m_corr = np.zeros(len(pairs))
    for i, p in enumerate(pairs):
        fn = f'family_{p["family"]:03d}_order{p["order"]}.npz'
        with np.load(R / 'stage2_v1' / 'screen' / fn) as z:
            attr[i] = z['attribution']; m_clean[i] = z['m_clean']; m_corr[i] = z['m_corr_fixed_sign']
            if p['exact_all']: ex40[i] = z['exact_delta']
        with np.load(R / 'stage2_v1' / 'extension' / fn) as z:
            assert list(z['head_indices']) == D; ext[i] = z['exact_delta']
    idx40 = sorted(ex40); X40 = np.stack([ex40[i] for i in idx40]); fam40 = fam[idx40]
    I40, _ = fam_mean(-X40, fam40)                       # importance on the 40 common pairs, all heads
    I40_by_fam = fam_matrix(-X40, fam40)

    # ---------------------------------------------------------------- load v2
    m2 = json.loads((R / 'stage2_v2_extension' / 'manifest.json').read_text())
    hp, hf = m2['scopeP_heads'], m2['scopeF_heads']
    P = np.zeros((len(pairs), len(hp))); F = np.zeros((len(pairs), len(hf))); FR = np.zeros((len(pairs), len(hf), 4))
    base = np.zeros((len(pairs), 3)); first_diff = np.zeros(len(pairs), int); nlen = np.zeros(len(pairs), int)
    for i, p in enumerate(pairs):
        fn = f'family_{p["family"]:03d}_order{p["order"]}.npz'
        with np.load(R / 'stage2_v2_extension' / 'pairs' / fn) as z:
            assert str(z['pair_id']) == p['id']
            P[i] = z['scopeP_delta']; F[i] = z['scopeF_delta']; FR[i] = z['scopeF_readout']; base[i] = z['clean_logits']
            first_diff[i] = z['first_diff']; nlen[i] = z['n']
    # consistency: v2 Scope-P on the 40 common pairs equals v1 exact
    for i in idx40:
        assert np.array_equal(P[i], ex40[i][hp])

    # full-coverage Scope-P importance on 178 pairs for D (from v1) and hp (from v2)
    IP = {}; IP_fam = {}
    E_fam = fam_matrix(-ext, fam); P_fam = fam_matrix(-P, fam)
    for j, h in enumerate(D): IP[h] = E_fam[:, j].mean(); IP_fam[h] = E_fam[:, j]
    for j, h in enumerate(hp): IP[h] = P_fam[:, j].mean(); IP_fam[h] = P_fam[:, j]
    F_fam = fam_matrix(-F, fam); IF = {h: F_fam[:, j].mean() for j, h in enumerate(hf)}; IF_fam = {h: F_fam[:, j] for j, h in enumerate(hf)}
    # logit decomposition under Scope F: change of each logit relative to clean (family-level)
    dlog = FR[:, :, 1:] - base[:, None, :]              # [pairs, hf, 3] = d clean-answer, d corrupted-answer, d source
    dlog_fam = fam_matrix(dlog, fam)                     # [fams, hf, 3]
    gap = fam_mean(m_clean - m_corr, fam)[0]

    # ---------------------------------------------------------------- groups & RI
    cov = {int(r['flat_id']): r for r in csv.DictReader(open(R / 'stage2_v2_coverage' / 'coverage_v2.csv'))}
    A = {f for f, r in cov.items() if r['in_A'] == 'True'}; B = {f for f, r in cov.items() if r['in_B'] == 'True'}
    C = {f for f, r in cov.items() if r['in_C'] == 'True'}; Dset = set(D)
    controls = set(json.loads((R / 'stage2_v1' / 'shortlist.json').read_text())['random_controls'])
    hist = {int(r['layer']) * H + int(r['head']): (float(r['ri_first']), float(r['ri_last'])) for r in csv.DictReader(open(R / 'stage2_v1' / 'head_effects.csv'))}
    cand = {c['layer'] * H + c['head']: c['reasons'] for c in json.loads((R / 'ri_test_v2' / 'candidates.json').read_text())['candidates']}
    ri = {}  # (scope, position, anchor) -> {flat: row}
    for r in csv.DictReader(open(R / 'ri_test_v2' / 'head_summary.csv')):
        if r['distance'] != 'all' or r['variant'] != 'pooled': continue
        ri.setdefault((r['scope'], r['position'], r['anchor']), {})[int(r['layer']) * H + int(r['head'])] = r
    def stat(key, metric, f, min_ev=20, min_fam=10):
        r = ri[key].get(f)
        if r is None or r[f'{metric}_mean'] == '': return np.nan
        if int(r[f'{metric}_events']) < min_ev or int(r[f'{metric}_families']) < min_fam: return np.nan
        return float(r[f'{metric}_mean'])
    MAIN = ('all_facts', 'all_test', 'first')

    # ---------------------------------------------------------------- 1. head table (148 full-coverage heads)
    full = sorted(IP); rows = []
    ci_cols = {}
    for h in full:
        lo, hi = boot_ci(IP_fam[h][:, None], rng)
        ci_cols[h] = (float(lo[0]), float(hi[0]))
    for h in full:
        r = dict(head=name(h), flat_id=h, layer=h // H, in_A=h in A, in_B=h in B, in_C=h in C, in_D=h in Dset, random_control=h in controls,
                 hist_ri_first=hist[h][0], ng_first=stat(MAIN, 'names_gap', h), ng_last=stat(('all_facts', 'all_test', 'last'), 'names_gap', h),
                 target_first=stat(MAIN, 'target', h), attribution_178=-attr[:, h].mean(),
                 I_P_178=IP[h], I_P_lo=ci_cols[h][0], I_P_hi=ci_cols[h][1], I_P_pos_fam=float((IP_fam[h] > 0).mean()),
                 I_P_gap_fraction=IP[h] / gap, I_40=I40[h])
        if h in IF:
            lo, hi = boot_ci(IF_fam[h][:, None], rng); j = hf.index(h); dl = dlog_fam[:, j, :].mean(axis=0)
            r.update(I_F_178=IF[h], I_F_lo=float(lo[0]), I_F_hi=float(hi[0]), I_F_pos_fam=float((IF_fam[h] > 0).mean()),
                     F_over_P=IF[h] / IP[h] if abs(IP[h]) > 1e-9 else np.nan,
                     F_dlogit_clean=dl[0], F_dlogit_corrupted=dl[1], F_dlogit_source=dl[2])
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(out / 'head_table_v2.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + [k for k in keys if k not in rows[0]]); w.writeheader(); w.writerows(rows)
    byhead = {r['flat_id']: r for r in rows}

    # ---------------------------------------------------------------- 2. RI vs causal
    corr = []
    allh = np.arange(1024)
    corr.append(dict(x='historical RI first (zero-filled)', y='I_40 all heads', n=1024, rho=spearman([hist[h][0] for h in allh], I40)))
    for key in sorted(ri):
        for metric in ('target', 'names_gap'):
            x = np.array([stat(key, metric, h) for h in allh])
            corr.append(dict(x=f'test-only {metric} {key[0]}/{key[1]}/{key[2]} (supported)', y='I_40 all heads', n=int(np.isfinite(x).sum()), rho=spearman(x, I40)))
            # layer-controlled: ranks within layer
            xs, ys = [], []
            for L in range(32):
                sel = [h for h in allh if h // H == L and np.isfinite(x[h])]
                if len(sel) >= 3:
                    xs += list(rankdata(x[sel]) / (len(sel) - 1)); ys += list(rankdata(I40[sel]) / (len(sel) - 1))
            corr.append(dict(x=f'test-only {metric} {key[0]}/{key[1]}/{key[2]} (supported, within-layer ranks)', y='I_40 all heads', n=len(xs), rho=spearman(xs, ys)))
    # on the 148 full-coverage heads (selected sample) with I_P_178
    x = np.array([stat(MAIN, 'names_gap', h) for h in full]); y = np.array([IP[h] for h in full])
    corr.append(dict(x='test-only names_gap all_facts/all_test/first (supported)', y='I_P_178 full-coverage heads', n=int(np.isfinite(x).sum()), rho=spearman(x, y)))
    x = np.array([hist[h][0] for h in full])
    corr.append(dict(x='historical RI first', y='I_P_178 full-coverage heads', n=len(full), rho=spearman(x, y)))
    # family bootstrap of the main new correlation (I_40 recomputed per resample)
    xng = np.array([stat(MAIN, 'names_gap', h) for h in allh]); okh = np.isfinite(xng)
    nf = I40_by_fam.shape[0]; bs = []
    for _ in range(2000):
        idx = rng.integers(0, nf, nf); Ib = I40_by_fam[idx].mean(axis=0); bs.append(spearman(xng[okh], Ib[okh]))
    ng_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    with open(out / 'ri_vs_causal_v2.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['x', 'y', 'n', 'rho']); w.writeheader(); w.writerows(corr)
    # drivers of the negative correlation: drop layers 16-18 / drop group C
    sup = [h for h in allh if np.isfinite(xng[h])]
    drivers = dict(
        all_supported=dict(n=len(sup), rho=spearman(xng[sup], I40[sup])),
        without_layers_16_18=dict(n=len([h for h in sup if not 16 <= h // H <= 18]), rho=spearman([xng[h] for h in sup if not 16 <= h // H <= 18], [I40[h] for h in sup if not 16 <= h // H <= 18])),
        without_top25_exact=dict(n=len([h for h in sup if h not in C]), rho=spearman([xng[h] for h in sup if h not in C], [I40[h] for h in sup if h not in C])),
        positive_gap_only=dict(n=int((xng[sup] > 0).sum()), rho=spearman([xng[h] for h in sup if xng[h] > 0], [I40[h] for h in sup if xng[h] > 0])),
        layer_vs_I40=dict(n=1024, rho=spearman(allh // H, I40)),
        layer_vs_namegap=dict(n=len(sup), rho=spearman([h // H for h in sup], xng[sup])),
        mean_layer_supported=float(np.mean([h // H for h in sup])),
        mean_layer_top25_exact=float(np.mean([h // H for h in C])))
    # top-k Jaccard overlap curves
    ks = [10, 25, 50, 100]; jac = []
    rank_I = list(np.argsort(-I40)); rank_hist = list(np.argsort(-np.array([hist[h][0] for h in allh])))
    rank_ng = sorted(sup, key=lambda h: -xng[h])
    for k in ks:
        top = set(rank_I[:k])
        jac.append(dict(k=k, hist=len(top & set(rank_hist[:k])) / len(top | set(rank_hist[:k])),
                        names_gap=len(top & set(rank_ng[:k])) / len(top | set(rank_ng[:k])) if len(rank_ng) >= k else None,
                        attribution=len(top & set(np.argsort(-np.abs(attr[idx40].mean(axis=0)))[:k])) / len(top | set(np.argsort(-np.abs(attr[idx40].mean(axis=0)))[:k]))))

    # ---------------------------------------------------------------- 3. verdicts under the new definition
    # sound: A heads vs same-layer non-A heads, I_40 (all heads have it)
    diffs = []; layer_rows = []
    for L in sorted({h // H for h in A}):
        a_h = [h for h in A if h // H == L]; o_h = [h for h in allh if h // H == L and h not in A]
        d = I40[a_h].mean() - I40[o_h].mean(); diffs.append(d)
        layer_rows.append(dict(layer=L, n_A=len(a_h), mean_A=I40[a_h].mean(), mean_other=I40[o_h].mean(), diff=d, median_A=np.median(I40[a_h]), median_other=np.median(I40[o_h])))
    Aarr = sorted(A); notA = [h for h in allh if h not in A]
    # family bootstrap of pooled layer-matched difference
    def layer_matched_diff(I):
        return float(np.mean([I[[h for h in A if h // H == L]].mean() - I[[h for h in allh if h // H == L and h not in A]].mean() for L in sorted({h // H for h in A})]))
    bsd = []
    for _ in range(2000):
        idx = rng.integers(0, nf, nf); bsd.append(layer_matched_diff(I40_by_fam[idx].mean(axis=0)))
    sound = dict(pooled_layer_matched_diff=layer_matched_diff(I40), ci=[float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))],
                 layers_with_positive_diff=int(sum(d > 0 for d in diffs)), layers=len(diffs),
                 median_A=float(np.median(I40[Aarr])), median_notA=float(np.median(I40[notA])),
                 mean_A=float(I40[Aarr].mean()), mean_notA=float(I40[notA].mean()), per_layer=layer_rows)
    # incomplete: top decile of I_40 outside A with non-positive (or unsupported) name gap at both anchors
    dec = rank_I[:102]; ng_last = np.array([stat(('all_facts', 'all_test', 'last'), 'names_gap', h) for h in allh])
    outside = [h for h in dec if h not in A]
    nonpos = [h for h in outside if not (xng[h] > 0) and not (ng_last[h] > 0)]
    incomplete = dict(top_decile=102, outside_A=len(outside), outside_A_nonpositive_gap_both_anchors=len(nonpos),
                      examples=[dict(head=name(h), I_40=float(I40[h]), ng_first=None if not np.isfinite(xng[h]) else float(xng[h]), ng_last=None if not np.isfinite(ng_last[h]) else float(ng_last[h])) for h in nonpos[:10]],
                      top10_in_A=[name(h) for h in rank_I[:10] if h in A])
    # misleading: median I_P_178 of A vs random controls (both have 178-pair exact)
    medA = float(np.median([IP[h] for h in A])); medR = float(np.median([IP[h] for h in controls]))
    bsm = []
    Afam = np.stack([IP_fam[h] for h in sorted(A)], axis=1); Rfam = np.stack([IP_fam[h] for h in sorted(controls)], axis=1)
    for _ in range(2000):
        idx = rng.integers(0, nf, nf); bsm.append(float(np.median(Afam[idx].mean(axis=0)) - np.median(Rfam[idx].mean(axis=0))))
    misleading = dict(median_A=medA, median_random=medR, diff=medA - medR, diff_ci=[float(np.percentile(bsm, 2.5)), float(np.percentile(bsm, 97.5))],
                      mean_A=float(np.mean([IP[h] for h in A])), mean_random=float(np.mean([IP[h] for h in controls])),
                      A_positive_share=float(np.mean([IP[h] > 0 for h in A])), random_positive_share=float(np.mean([IP[h] > 0 for h in controls])),
                      A_abs_gt_0_1=int(sum(abs(IP[h]) > 0.1 for h in A)), A_heads=len(A))

    # ---------------------------------------------------------------- 4. Scope F
    AB = sorted(A | B); assert set(AB) == set(hf)
    scopeF = dict(heads=len(AB), rho_P_vs_F=spearman([IP[h] for h in AB], [IF[h] for h in AB]),
                  top_F=[dict(head=name(h), I_F=IF[h], I_P=IP[h], ratio=IF[h] / IP[h] if abs(IP[h]) > 1e-9 else None,
                              d_clean=byhead[h]['F_dlogit_clean'], d_corrupted=byhead[h]['F_dlogit_corrupted'], d_source=byhead[h]['F_dlogit_source'],
                              in_A=h in A, in_B=h in B) for h in sorted(AB, key=lambda h: -abs(IF[h]))[:12]],
                  F_share_of_P=dict((name(h), IF[h] / IP[h]) for h in AB if abs(IP[h]) > 0.3),
                  active_positions_mean=float((nlen - first_diff).mean()), prompt_len_mean=float(nlen.mean()))

    # ---------------------------------------------------------------- 5. priority heads
    prio = ['L11H4', 'L17H5', 'L9H16', 'L12H2', 'L25H18', 'L23H10', 'L15H18', 'L6H2', 'L16H21', 'L16H1', 'L16H4', 'L16H24', 'L3H11', 'L9H22', 'L17H1', 'L27H6']
    prow = [byhead[h] for h in full if name(h) in prio]

    # ---------------------------------------------------------------- summary json
    summ = dict(pairs=len(pairs), full_coverage_heads=len(full), groups=dict(A=len(A), B=len(B), C=len(C), D=len(D), random_controls=len(controls)),
                gap_mean=float(gap), active_positions_mean=scopeF['active_positions_mean'], prompt_len_mean=scopeF['prompt_len_mean'],
                correlations=corr, names_gap_vs_I40_family_bootstrap_ci=ng_ci, negative_correlation_drivers=drivers, jaccard=jac,
                verdict_sound=sound, verdict_incomplete=incomplete, verdict_misleading=misleading, scopeF=scopeF,
                priority_heads=[{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in prow])
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    json.dump(clean(summ), open(out / 'summary.json', 'w'), indent=1)

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing; tables written, figures skipped'); return
    col = dict(A='#c0392b', B='#2471a3', AB='#7d3c98', other='#9e9e9e', C='#1e8449')
    def annotate(ax_, items):
        offs = [(4, 3), (4, -9), (-30, 5), (4, 9), (-34, -9)]
        for i, (lab, x_, y_) in enumerate(sorted(items, key=lambda t: (-t[2], t[1]))):
            ax_.annotate(lab, (x_, y_), fontsize=6.5, xytext=offs[i % len(offs)], textcoords='offset points')
    def group_color(h):
        if h in A and h in B: return col['AB']
        if h in A: return col['A']
        if h in B: return col['B']
        return col['other']
    # Fig 1: two panels RI vs I_40
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xh = np.array([hist[h][0] for h in allh])
    for h in allh: ax[0].scatter(xh[h], I40[h], s=10, color=group_color(h), alpha=.75, linewidths=0)
    ax[0].set_xlabel('historical pooled RI (first target)'); ax[0].set_ylabel('exact importance $I_h$, 40 common pairs (logits)')
    ax[0].set_yscale('symlog', linthresh=0.05); ax[0].set_title(f'historical RI: Spearman = {corr[0]["rho"]:.3f} (n=1024)')
    annotate(ax[0], [(name(h), xh[h], I40[h]) for h in allh if I40[h] > 1.5 or xh[h] > 0.2])
    rho_ng = next(c['rho'] for c in corr if c['x'].startswith('test-only names_gap all_facts/all_test/first (supported)') and c['y'] == 'I_40 all heads')
    for h in sup: ax[1].scatter(xng[h], I40[h], s=14, color=group_color(h), alpha=.8, linewidths=0)
    ax[1].axvline(0, color='k', lw=.5); ax[1].set_yscale('symlog', linthresh=0.05)
    ax[1].set_xlabel('test-only name gap (all facts, all test positions, first anchor)'); ax[1].set_title(f'new RI (supported heads): Spearman = {rho_ng:.3f} (n={len(sup)})')
    annotate(ax[1], [(name(h), xng[h], I40[h]) for h in sup if I40[h] > 0.5 or xng[h] > 0.012 or I40[h] < -0.5])
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], marker='o', ls='', color=col[k], label=l) for k, l in [('A', 'group A (new RI)'), ('B', 'group B (final QK)'), ('AB', 'A and B'), ('other', 'other')]], fontsize=7, loc='lower right')
    for a_ in ax: a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'stage2_v2_fig1_ri_vs_causal_v2.png', dpi=170); plt.close(fig)
    # Fig 2: Scope P vs Scope F
    fig, ax = plt.subplots(figsize=(5.6, 5))
    xs = np.array([IP[h] for h in AB]); ys = np.array([IF[h] for h in AB])
    lim = max(abs(xs).max(), abs(ys).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color='k', lw=.6, ls='--'); ax.axhline(0, color='k', lw=.4); ax.axvline(0, color='k', lw=.4)
    for h in AB: ax.scatter(IP[h], IF[h], s=22, color=group_color(h), linewidths=0)
    annotate(ax, [(name(h), IP[h], IF[h]) for h in AB if abs(IP[h]) > 0.09 or abs(IF[h]) > 0.09])
    ax.set_xscale('symlog', linthresh=0.05); ax.set_yscale('symlog', linthresh=0.05)
    ax.set_xlabel('Scope P importance (all prompt positions), 178 pairs'); ax.set_ylabel('Scope F importance (final position only), 178 pairs')
    ax.set_title(f'A ∪ B heads (n={len(AB)}); Spearman = {scopeF["rho_P_vs_F"]:.3f}'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'stage2_v2_fig2_scopeP_vs_scopeF.png', dpi=170); plt.close(fig)
    # Fig 3: logit decomposition for top Scope-F heads
    top = sorted(AB, key=lambda h: -abs(IF[h]))[:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = .27; xs = np.arange(len(top))
    ax.bar(xs - w, [byhead[h]['F_dlogit_clean'] for h in top], w, label='Δ logit clean answer', color='#1e8449')
    ax.bar(xs, [byhead[h]['F_dlogit_corrupted'] for h in top], w, label='Δ logit corrupted answer', color='#c0392b')
    ax.bar(xs + w, [byhead[h]['F_dlogit_source'] for h in top], w, label='Δ logit source name', color='#7f8c8d')
    ax.axhline(0, color='k', lw=.5); ax.set_xticks(xs); ax.set_xticklabels([name(h) for h in top], rotation=45, ha='right')
    ax.set_ylabel('mean change under Scope-F patch (logits)'); ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
    ax.set_title('Scope F: which logit moves when the head is patched at the final position')
    fig.tight_layout(); fig.savefig(out / 'stage2_v2_fig3_scopeF_logit_decomposition.png', dpi=170); plt.close(fig)
    # Fig 4: group A heads, I_P with CIs, sorted; colored by name-gap sign
    fig, ax = plt.subplots(figsize=(10, 4.2))
    As = sorted(A, key=lambda h: -IP[h])
    for i, h in enumerate(As):
        c = '#c0392b' if (xng[h] > 0) else ('#2471a3' if np.isfinite(xng[h]) else '#9e9e9e')
        ax.errorbar(i, IP[h], yerr=[[IP[h] - byhead[h]['I_P_lo']], [byhead[h]['I_P_hi'] - IP[h]]], fmt='o', ms=3.5, color=c, elinewidth=.8, capsize=0)
    ax.axhline(0, color='k', lw=.5); ax.set_yscale('symlog', linthresh=0.02)
    ax.set_xticks(range(len(As))); ax.set_xticklabels([name(h) for h in As], rotation=90, fontsize=6.5)
    ax.set_ylabel('Scope P importance, 178 pairs (logits)'); ax.grid(alpha=.3, axis='y')
    ax.set_title('Group A (59 test-only RI candidates): Scope-P importance, 178 pairs, 95% family-bootstrap intervals', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'stage2_v2_fig4_groupA_importance.png', dpi=170); plt.close(fig)
    # Fig 5: Scope-F share by layer
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sel = [h for h in AB if abs(IP[h]) > 0.04]
    for h in sel:
        ax.scatter(h // H, IF[h] / IP[h], s=60 * min(1, abs(IP[h])) + 12, color=group_color(h), linewidths=0, alpha=.85)
        ax.annotate(name(h), (h // H, IF[h] / IP[h]), fontsize=6.5, xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='k', lw=.5); ax.axhline(1, color='k', lw=.5, ls='--'); ax.set_ylim(-0.25, 1.35)
    ax.set_xlabel('layer'); ax.set_ylabel('Scope F / Scope P importance'); ax.grid(alpha=.3)
    ax.set_title(f'Final-position share of the Scope-P effect (A ∪ B heads with |I_P| > 0.04, n={len(sel)}; marker size ∝ |I_P|)', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'stage2_v2_fig5_scopeF_share_by_layer.png', dpi=170); plt.close(fig)
    print(json.dumps(clean(dict(correlations=[c for c in corr if 'within-layer' not in c['x']][:6], ng_ci=ng_ci, drivers=drivers, jaccard=jac,
                                 sound={k: v for k, v in sound.items() if k != 'per_layer'}, incomplete=incomplete, misleading=misleading,
                                 scopeF={k: v for k, v in scopeF.items() if k != 'F_share_of_P'})), indent=1))


if __name__ == '__main__': main()
