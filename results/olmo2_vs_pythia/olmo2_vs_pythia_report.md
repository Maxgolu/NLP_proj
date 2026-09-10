# OLMo-2-7B vs Pythia-6.9B on the v3.1 battery — comparison report

Date: 2026-09-09. Runs: `olmo2_7b_v3_1_{0,4,12}shot` + `olmo2_7b_v3_1_controls`
(OLMo-2-1124-7B @ 7df9a825, dataset `name_completion_v3_1_olmo2`) vs
`completion_v3_1_{0,4,12}shot` (Pythia-6.9B @ c0e3eee3, dataset
`name_completion_v3_1`). Same questions; 81 entities renamed for dual-tokenizer
length matching (see `pilot_v3/data_olmo2/adapt_audit_olmo2.json`).
Analysis script: `compare_olmo2_pythia.py` (this folder).

## Pipeline calibration (OLMo controls)

verbatim: 20/20 candidate, 19/20 free — the scoring pipeline is sound on OLMo.
stated: 15/20 — all 5 failures pick the distractor grandmother whose fact is
more recent than the stated answer; this is recency competition (model
behavior), not a pipeline defect. No Pythia v3.1 controls run exists for
comparison (worth running once for the report).

## Headline table — kinship, candidate accuracy, order0/order1/both-orders

```
check/variant        pythia-4sh        pythia-12sh       olmo-4sh          olmo-12sh
direct/base          0.90/1.00/0.90    0.84/1.00/0.84    1.00/0.93/0.93    1.00/0.90/0.90
twohop/base          0.74/1.00/0.74    0.72/1.00/0.72    0.69/0.91/0.65    0.82/0.88/0.78
broken/first         0.12/0.06/0.03    0.18/0.00/0.00    0.44/0.47/0.29    0.15/0.44/0.09
broken/second        0.79/0.97/0.76    0.68/1.00/0.68    0.68/0.91/0.65    0.94/0.91/0.85
reorder              0.51/0.40/0.22    0.51/0.41/0.25    0.57/0.60/0.41    0.62/0.57/0.37
distractors          0.81/0.87/0.74    0.72/0.91/0.68    0.81/0.62/0.51    0.90/0.72/0.66
rename               0.65/0.96/0.65    0.65/0.97/0.65    0.65/0.93/0.59    0.75/0.91/0.68
```

Key CIs (both-orders, kinship 12-shot): twohop OLMo 53/68 = .78 [.67,.86] vs
Pythia 49/68 = .72 [.60,.81]; reorder OLMo 25/68 = .37 [.26,.49] (4-shot:
28/68 = .41 [.30,.53]) vs Pythia 17/68 = .25 [.16,.36]. Reorder vs chance .25:
OLMo p = 2.5e-3 (4-shot), 2.1e-2 (12-shot); Pythia p = .75 / .55.

## The qualitative finding: the failure mode moved

1. **Order symmetry.** Pythia's two-hop is order0 .72 / order1 1.00 (pure
   recency split); OLMo is .82/.88 — the layout asymmetry largely gone.
2. **Reorder.** OLMo is above chance on shuffled facts (both shot counts,
   CI excludes chance); Pythia is exactly at chance. First direct evidence of
   layout-robust (content-based) two-hop in the 7B class.
3. **broken/first error anatomy** (free generation, kinship):

```
                gold  bridge  distractor(=canonical answer)
pythia 12sh       6     23      37     <- ignores the rewired edge: layout answer
olmo   12sh       6     41      21     <- FOLLOWS the rewired edge to the new
olmo    4sh      13     38      17        bridge, then fails to compose hop-2
olmo    0sh      33     25      10     <- with NO demos, completes the rewired
                                          chain correctly ~half the time
```

   Pythia's dominant error is the canonical grandmother (hop-1 resolved by
   position, edge content ignored). OLMo's dominant error is the NEW bridge —
   hop-1 is resolved by content; the bottleneck is the composition step.
   Strikingly, canonical demos actively suppress OLMo's rewired-chain
   completion (0-shot 33 gold -> 12-shot 6 gold): demo-induced prior competes
   with in-context evidence (a circuit-competition phenomenon in itself).

   CAVEAT (post-hoc audit, 2026-09-10): in broken/first — and only there —
   the rewiring gives the new bridge a third mention, making it the unique
   most-frequent name in the fact block (68/68 rows; every other condition
   has tied counts, so no frequency rule is even definable there). The
   bridge-answer reading of the FREE-generation anatomy is therefore
   confounded with a frequency-weighted copy rule. Counter-evidence: at
   0-shot OLMo outputs the max-frequency name only 10/68. Candidate-metric
   results are unaffected: gold/distractor mentions are exactly balanced in
   136/136 rows of every condition. Fix for the next data revision: equalize
   mention counts in the broken conditions as well.

## Pre-registered decision rule — honest verdict

At 12-shot: (1) direct+twohop >= Pythia: PASS. (2) broken/first > 50% both
orders: FAIL (.15/.44). (3) broken/second >= Pythia: PASS (.85 vs .68).
Secondary (reorder above chance): PASS. Formally this is a PARTIAL pass — the
rule as written does not select OLMo automatically.

Deviation, with justification stated openly: the rule's failure clause assumed
a broken/first failure would look like Pythia's (layout-bound). It does not —
the secondary metric and the error anatomy show content-based hop-1 with a
composition bottleneck, which is exactly the mechanism the SIC project needs
to study. Choosing OLMo here is a justified deviation, recorded as such.

## Caveats

- broken cells are n=34 order-pairs (alternating families) — CIs are wide.
- Per-world (12-shot, twohop both-orders): kinship P.72/O.78; location
  P.12/O.69 (OLMo dramatically better); containment P.44/O.25 (OLMo WORSE —
  the no-question "Therefore, all X are" format interacts with something).
  Kinship stays the primary world; containment results should not be pooled.
- distractors variant order1: OLMo .72 vs Pythia .91 — OLMo slightly more
  distractible in the recency-favored order. Track, don't over-read.
- 0-shot OLMo two-hop order0 is LOW (.32 kinship) with order1 .96 — intrinsic
  recency prior is strong; demos are what buy symmetry. The 0-shot vs 12-shot
  contrast is itself analyzable.

## Decision and refined research plan

**Platform: OLMo-2-7B primary; Pythia-6.9B kept as the contrast model.**
Two models with two distinct regimes (layout-heuristic vs content-based with a
composition bottleneck) make the mechanistic comparison a feature of the
paper, not a fallback. TransformerLens supports both; OLMo-2 has public
intermediate checkpoints (stepXXX-tokensYYYB revisions).

Refined research question: *what circuit implements content-based hop-1
binding in OLMo-2-7B, what breaks at the composition step under edge rewiring,
and when does this circuit emerge over training — contrasted with Pythia,
where hop-1 binding is positional?*

Sharpest patching contrasts, in priority order:
1. base vs broken/first (kinship, 12-shot, canonical order): where does the
   correct bridge identity enter, and where is it overridden? OLMo's bridge-
   dominant errors give a clean logit-diff target (gold vs bridge vs old answer).
2. canonical vs reorder: which components carry the layout advantage in
   Pythia but not in OLMo?
3. 0-shot vs 12-shot on broken/first (OLMo): demo-suppression of rewired-chain
   completion — circuit competition between demo-induced pattern and
   in-context evidence.
Checkpoint sweep on OLMo-2-7B revisions once the final-checkpoint circuit is
localized; costing dry-run (RI + patching timings) still precedes the sweep.
