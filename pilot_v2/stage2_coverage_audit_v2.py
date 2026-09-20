"""Stage-2 coverage audit for the extension-v2 head list. CPU only; reads saved results.

Groups (see methodology Section 5, "Exact extension to all 178 pairs"):
  A  59-head Stage-1 test-only RI discovery union      (results/ri_test_v2/candidates.json)
  B  heads with >=1 final-position test QK event         (results/stage1_v4_review/anatomy_all_heads/head_anatomy.csv)
  C  top-25 |exact| on the common 40-pair subset         (results/stage2_v1/head_effects.csv)
  D  the 92 heads of the stage2_v1 exact extension       (results/stage2_v1/shortlist.json)

Outputs (results/stage2_v2_coverage/):
  coverage_v2.csv          one row per head in A|B|C|D with group flags and measured coverage
  extension_v2_heads.json  heads needing Scope-P exact on the 138 remaining pairs; Scope-F list
  ri_new_vs_exact.csv      Spearman of the new RI statistics against exact importance on 40 pairs
  summary.json

Flattened head id = layer*32 + head (zero-based), as in stage2_run.py.
Sign convention: head_effects.csv stores Delta_h (patched minus clean); importance I_h = -Delta_h.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

H = 32
ROOT = Path(__file__).resolve().parents[1]


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3: return None, int(len(a))
    def rank(x):
        o = np.argsort(x, kind='stable'); r = np.empty(len(x)); i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and x[o[j]] == x[o[i]]: j += 1
            r[o[i:j]] = (i + j - 1) / 2; i = j
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0: return None, int(len(a))
    return float(np.corrcoef(ra, rb)[0, 1]), int(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(ROOT / 'results'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--top-exact', type=int, default=25)
    ap.add_argument('--seconds-per-forward', type=float, default=0.152)
    args = ap.parse_args()
    R = Path(args.results); out = Path(args.out or R / 'stage2_v2_coverage'); out.mkdir(parents=True, exist_ok=True)

    # ---- stage2_v1 head table -------------------------------------------------
    eff = {}
    for r in csv.DictReader(open(R / 'stage2_v1' / 'head_effects.csv')):
        fid = int(r['layer']) * H + int(r['head'])
        eff[fid] = dict(layer=int(r['layer']), head=int(r['head']),
                        ri_first_hist=float(r['ri_first']), ri_last_hist=float(r['ri_last']),
                        delta_first_order=float(r['first_order_mean']),
                        delta_exact40=float(r['exact_subset_mean']),
                        abs_exact40=float(r['exact_subset_mean_abs']),
                        delta_exact178=float(r['exact_all_pairs_mean']) if r['exact_all_pairs_mean'] else None)
    assert len(eff) == 1024, len(eff)
    D = set(json.load(open(R / 'stage2_v1' / 'shortlist.json'))['heads'])
    have178 = {f for f, e in eff.items() if e['delta_exact178'] is not None}
    assert have178 == D, 'exact-178 coverage in head_effects.csv does not match shortlist.json'
    assert len(D) == 92

    # ---- group A ----------------------------------------------------------------
    cand = json.load(open(R / 'ri_test_v2' / 'candidates.json'))['candidates']
    A = {c['layer'] * H + c['head']: c['reasons'] for c in cand}
    assert len(A) == 59, len(A)

    # ---- group B ----------------------------------------------------------------
    B = {}
    for r in csv.DictReader(open(R / 'stage1_v4_review' / 'anatomy_all_heads' / 'head_anatomy.csv')):
        if r['fact_block'] == 'test' and r['position'] == 'answer_prediction' and int(r['qk_passes']) > 0:
            B[int(r['layer']) * H + int(r['head'])] = dict(events=int(r['qk_passes']), families=int(r['families']))
    assert sum(v['events'] for v in B.values()) == 744, sum(v['events'] for v in B.values())

    # ---- group C ----------------------------------------------------------------
    C = set(sorted(eff, key=lambda f: -abs(eff[f]['delta_exact40']))[:args.top_exact])

    # ---- coverage table ---------------------------------------------------------
    union = set(A) | set(B) | C | D
    rows = []
    for f in sorted(union):
        e = eff[f]
        rows.append(dict(head=f"L{e['layer']}H{e['head']}", flat_id=f, layer=e['layer'], head_index=e['head'],
                         in_A=f in A, in_B=f in B, in_C=f in C, in_D=f in D,
                         A_reasons=len(A.get(f, [])),
                         A_names_gap_rank=any(x['metric'] == 'names_gap' for x in A.get(f, [])),
                         B_final_events=B.get(f, {}).get('events'), B_final_families=B.get(f, {}).get('families'),
                         first_order_pairs=178, exact_pairs=178 if f in D else 40, exact_families=89 if f in D else 20,
                         importance_exact40=-e['delta_exact40'], importance_first_order=-e['delta_first_order'],
                         importance_exact178=(-e['delta_exact178']) if e['delta_exact178'] is not None else None,
                         residual_40=abs(e['delta_exact40'] - e['delta_first_order']),
                         needs_scopeP_138=(f in (set(A) | set(B) | C)) and f not in D,
                         needs_scopeF_178=f in (set(A) | set(B))))
    with open(out / 'coverage_v2.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    missingP = sorted(f for f in (set(A) | set(B) | C) if f not in D)
    scopeF = sorted(set(A) | set(B))
    fwdP = len(missingP) * 138; fwdF = len(scopeF) * 178
    json.dump(dict(scopeP_heads_138_pairs=missingP, scopeF_heads_178_pairs=scopeF,
                   groups=dict(A=sorted(A), B=sorted(B), C=sorted(C), D=sorted(D)),
                   rule='(A|B|C)\\D exact Scope P on the 138 pairs outside the common subset; A|B exact Scope F on all 178 pairs',
                   flat_id='layer*32+head'), open(out / 'extension_v2_heads.json', 'w'), indent=1)

    # ---- new RI statistics vs exact importance (common 40 pairs, all heads) ----
    imp40 = np.array([-eff[f]['delta_exact40'] for f in range(1024)])
    ri_rows = []
    hist_rho, _ = spearman([eff[f]['ri_first_hist'] for f in range(1024)], imp40)
    ri_rows.append(dict(statistic='historical pooled RI first', scope='pooled', position='all', anchor='first',
                        support_rule='none (zero-filled)', heads=1024, spearman=hist_rho))
    summ = {}
    for r in csv.DictReader(open(R / 'ri_test_v2' / 'head_summary.csv')):
        if r['distance'] != 'all' or r['variant'] != 'pooled': continue
        key = (r['scope'], r['position'], r['anchor'])
        summ.setdefault(key, {})[int(r['layer']) * H + int(r['head'])] = r
    for key, tab in sorted(summ.items()):
        for metric in ('target', 'names_gap'):
            vals = np.full(1024, np.nan); zero = np.zeros(1024); n_sup = 0
            for f, r in tab.items():
                m = r[f'{metric}_mean']; ev = int(r[f'{metric}_events']); fa = int(r[f'{metric}_families'])
                if m != '' and ev >= 20 and fa >= 10:
                    vals[f] = float(m); zero[f] = float(m); n_sup += 1
            rho_s, n = spearman(vals, imp40)
            rho_z, _ = spearman(zero, imp40)
            ri_rows.append(dict(statistic=f'test-only {metric}', scope=key[0], position=key[1], anchor=key[2],
                                support_rule='>=20 events & >=10 families', heads=n, spearman=rho_s,
                                spearman_zero_filled_1024=rho_z))
    with open(out / 'ri_new_vs_exact.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['statistic', 'scope', 'position', 'anchor', 'support_rule', 'heads', 'spearman', 'spearman_zero_filled_1024'])
        w.writeheader(); w.writerows(ri_rows)

    # ---- summary ------------------------------------------------------------------
    s = dict(
        groups=dict(A=len(A), B=len(B), C=len(C), D=len(D), union=len(union)),
        overlaps=dict(A_and_D=len(set(A) & D), B_and_D=len(set(B) & D), C_and_D=len(C & D),
                      A_and_B=len(set(A) & set(B)), A_and_C=len(set(A) & C), B_and_C=len(set(B) & C)),
        already_full_exact=dict(A=len(set(A) & D), B=len(set(B) & D), C=len(C & D)),
        missing_scopeP=dict(heads=len(missingP), pairs_each=138, forwards=fwdP,
                            hours_one_replica=round(fwdP * args.seconds_per_forward / 3600, 2)),
        scopeF=dict(heads=len(scopeF), pairs_each=178, forwards=fwdF,
                    hours_one_replica=round(fwdF * args.seconds_per_forward / 3600, 2)),
        total_forwards=fwdP + fwdF,
        total_hours_one_replica=round((fwdP + fwdF) * args.seconds_per_forward / 3600, 2),
        historical_ri_first_vs_exact40_spearman=hist_rho,
        top_exact40=[dict(head=f"L{eff[f]['layer']}H{eff[f]['head']}", flat_id=f, importance_exact40=-eff[f]['delta_exact40'],
                          in_A=f in A, in_B=f in B, in_D=f in D) for f in sorted(C, key=lambda f: -abs(eff[f]['delta_exact40']))],
        note='Coverage read from saved stage2_v1 outputs; no model forwards. Spearman values are exploratory descriptives on discovery families.')
    json.dump(s, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != 'top_exact40'}, indent=1))
    print('\nTop exact-40 heads:')
    for t in s['top_exact40']: print(f"  {t['head']:8s} I={t['importance_exact40']:+.4f}  A={t['in_A']} B={t['in_B']} D={t['in_D']}")
    print('\nNew RI vs exact-40 Spearman:')
    for r in ri_rows: print(f"  {r['statistic']:28s} {r['scope']:11s} {r['position']:9s} {r['anchor']:6s} n={r['heads']:4d} rho={r['spearman'] if r['spearman'] is None else round(r['spearman'],3)}  zero-filled={r.get('spearman_zero_filled_1024') if r.get('spearman_zero_filled_1024') is None else round(r['spearman_zero_filled_1024'],3)}")


if __name__ == '__main__':
    main()
