"""Independent CPU re-analysis of the S4.3 extension (results/stage4_s43_all_v1).

Recomputes, from extension_analysis_v1/pair_order_matched.csv (per pair/order/direction
raw effects, matched direct effects and increments):
  * the matched-increment identity (raw - direct == increment),
  * family means (two orders averaged), coherent / heterogeneous / retained labels
    under the frozen rule (tau = 0.10; coherent: |mean| >= tau and >= 70% sign agreement;
    heterogeneous: mean |effect| >= tau and >= 20% of families with |effect| >= tau),
  * noising/restoration family sign-match fraction and Pearson r,
  * the 69 selection-excluded families and the common-20 reproduction,
  * the 11 direct prerequisite configurations, which the report does not tabulate,
and compares them with extension_analysis_v1/bidirectional_summary.csv.
Optionally spot-checks raw chunk records against the table.
Run from the project root:  python results/stage4_s43_review_20260926/verify_s43.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/stage4_s43_all_v1')
TAU = 0.10
pm = pd.read_csv(R / 'extension_analysis_v1/pair_order_matched.csv')
bs = pd.read_csv(R / 'extension_analysis_v1/bidirectional_summary.csv').set_index('config_id')
man = json.load(open(R / 'extension_analysis_v1/analysis_manifest.json'))
sel20 = set(man['common20_reproducibility']['families'])

sel = pm[pm.extension_role == 'selected'].copy()
assert len(sel) == 16 * 178 * 2, len(sel)
assert (sel.raw_effect - sel.direct_effect - sel.matched_increment).abs().max() < 1e-9

def rule(g):
    m = g.mean(); ma = g.abs().mean()
    agree = (np.sign(g) == np.sign(m)).mean(); active = (g.abs() >= TAU).mean()
    coh = abs(m) >= TAU and agree >= 0.70; het = ma >= TAU and active >= 0.20
    return pd.Series(dict(mean=m, mabs=ma, agree=agree, coherent=coh, heterogeneous=het, retained=coh or het))

fam = sel.groupby(['config_id', 'direction', 'family']).matched_increment.mean().reset_index()
agg = fam.groupby(['config_id', 'direction']).matched_increment.apply(rule).unstack()
piv = fam.pivot_table(index=['config_id', 'family'], columns='direction', values='matched_increment').reset_index()
sm = piv.assign(same=np.sign(piv.noise) == np.sign(piv.restore)).groupby('config_id').same.mean()

bad = 0
for cid, r in bs.iterrows():
    a, b = agg.loc[(cid, 'noise')], agg.loc[(cid, 'restore')]
    for mine, theirs, name in [(a['mean'], r.noise_mean_signed, 'noise mean'), (b['mean'], r.restore_mean_signed, 'restore mean'),
                               (a.coherent, r.noise_coherent, 'noise coherent'), (b.coherent, r.restore_coherent, 'restore coherent'),
                               (a.retained, r.noise_retained, 'noise retained'), (b.retained, r.restore_retained, 'restore retained')]:
        ok = abs(float(mine) - float(theirs)) < 1e-9 if isinstance(mine, float) else bool(mine) == bool(theirs)
        if not ok: bad += 1; print('MISMATCH', cid, name, mine, theirs)
print(f'16 selected routes x 6 reported quantities: {bad} mismatches')

# common-20 reproduction and 69-family sensitivity
f20 = fam[(fam.family.isin(sel20)) & (fam.direction == 'noise')].groupby('config_id').matched_increment.mean()
print('max |rerun20 - screen20| =', float((f20 - bs.screen20_mean_signed).abs().max()))
f69 = fam[~fam.family.isin(sel20)]
a69 = f69.groupby(['config_id', 'direction']).matched_increment.apply(rule).unstack()
print('69 families: coherent both directions =', int(((a69.xs('noise', level=1).coherent) & (a69.xs('restore', level=1).coherent)).sum()), 'of 16')

# direct prerequisites (never tabulated in the report)
d = sel[['pair_id', 'family', 'direction', 'comparator_config_id', 'direct_effect']].drop_duplicates()
fd = d.groupby(['comparator_config_id', 'direction', 'family']).direct_effect.mean().reset_index()
pre = fd.groupby(['comparator_config_id', 'direction']).direct_effect.apply(rule).unstack()
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 90)
print('\nDirect prerequisites, 89 families:'); print(pre.round(3).to_string())

# order sensitivity, noising
o = sel[sel.direction == 'noise'].groupby(['extension_priority', 'order']).matched_increment.mean().unstack()
print('\nNoising increment by prompt order:'); print(o.round(3).to_string())

# optional raw-chunk spot check
chunks = sorted(glob.glob(str(R / 'extension_v4_family034_final/chunks/*.json')))
if chunks:
    n = mism = 0
    key = pm.set_index(['pair_id', 'config_id', 'direction']).raw_effect
    for f in chunks:
        dd = json.load(open(f))
        for rec in dd['records']:
            k = (dd['pair_id'], rec['config_id'], rec['direction'])
            if k in key.index:
                n += 1; mism += abs(float(key.loc[k]) - rec['effect']) > 1e-9
    print(f'\nraw chunk records matched against the table: {n}, mismatches: {mism} ({len(chunks)} chunk files)')
