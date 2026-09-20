# Stage 1 v4: independent analysis

Run: `stage1_v4_calibrated`, job 886892. OLMo-2-1124-7B, revision
7df9a82518afdecae4e8c026b27adccc8c1f0032, 4 shots, tau=2.2.

## Scope and execution

The run scored 534 prompts from 89 behaviorally eligible discovery families,
crossing three variants and two orders. It scanned 1,024 heads. It did not
train the model or perform a new broad causal head screen.

The archive and nested JSONL streams parsed successfully; all three result
hashes recorded by calibration match. Recorded scan/helper hashes match the
local code. Preflight P1 and P3 passed; inert hooks had zero drift. Attention
capture still changes answer-position vocabulary logits by at most 0.0390625
on the tested prompt, as in the previous run. This is not an all-prompt
equivalence test. The first-divergent-token specifications cover shared prefixes
in families 168 and 174 (12 variant/order rows), but the small causal preflight
does not specifically validate those families on GPU.

Model shard loading: 38m44s. Scan: 749.2s (12.5m). Calibration: 7.2s.
The long startup was checkpoint loading, not the randomization analysis.

## Replication versus interpretation

Every parsed row of `head_stats.csv` is identical to the earlier stage1_v2
archive. The original pooled score was replicated; the added event records
change its interpretation, not its numerical value.

| Head | Pooled strength | Demo->demo events / mean | Test->test events / mean | Demo->test events |
|---|---:|---:|---:|---:|
| L3H11 | 0.243067 | 144 / 0.250173 | 5 / 0.038429 | 0 |
| L9H22 | 0.373918 | 72 / 0.445312 | 14 / 0.006744 | 0 |

All events in this table have defined OV scores. The arrows denote annotated
fact block -> current attention-query position block, not circuit edges.
Demo events are 96.6% and 83.7% of the respective heads' events and account
for over 99% of their respective summed first-target scores.

L3H11 has events in 17/89 families, with test events confined to family 138.
L9H22 has events in 15/89 families, with test events in 52, 84, 168 and 170.
The saved analyzer counts only 10 and 13 distinct prefix/fact conditions,
respectively. Repeated demonstrations across twins and families are not
independent observations. These counts are not numbers of distinct sentences.

## Local attention and token preferences

Reconstructing the source anchor from the saved character offsets shows:

- L9H22: ALL 86 QK passes have `j == source` (self-attention).
- L3H11: ALL 149 passes have `j == source + 1` (previous-token attention).
- L3H11's current token is always token 627, representing a period plus
  newline. L9H22's 72 demonstration events all use current token `la`.

Concrete example: in family 14, the demonstration starts with
`Mulnla is the mother of Garnla.` L9H22 attends from the final `la` of
`Garnla` to that same position. Its embedding-level OV projection gives the
first target token `Mul` a normalized share of about 0.528. That is not an
observation of answering the test question or retrieving a distant mention.

For L3H11, demonstration events for `Tesmla is the mother of Rusma.` score
about 0.768 for `Tes`, while those for `Toltia is the mother of Fernla.`
score zero for `T`. In both cases the OV input is the SAME period/newline
embedding. The static projected vocabulary vector is therefore the same for
this head; target choice and visible-context normalization change the reported
share. This supplies a concrete token-preference explanation for the pooled
high scores. It does not prove either head is causally irrelevant or that it
can never participate in a semantic computation.

## Matched-target randomization

The NEW calibration statistic is intentionally narrower than pooled RI: test
facts with targets among the answer candidates, both candidate mentions fully
visible, equal token lengths, distinct first tokens, equal weight per active
family. It retains QK selection and compares saved true/other candidate scores.
It uses 100,000 family-consistent swaps, at least 50 matched events and 10
families, and Holm adjustment across 1,024 heads.

Event accounting: 278,134 total QK passes; 231,500 outside test->test scope;
17,870 noncandidate facts; 8,221 not fully visible; 642 first-token collisions;
19,901 matched events. Eighty heads meet the support criteria; 944 do not.

Neither earlier candidate enters the test:

- L3H11: 144 events in demos; the remaining five test events concern a
  noncandidate fact. Zero matched events.
- L9H22: 72 in demos; eight noncandidate test events, three not fully visible,
  three first-token collisions. Zero matched events.

Thus the null result is NOT a statistical rejection of those two heads.

| Head | Matched events | Families | True mean | Control mean | Raw p |
|---|---:|---:|---:|---:|---:|
| L16H4 | 87 | 40 | 0.014713 | 0.009302 | 0.005810 |
| L9H18 | 187 | 54 | 0.013037 | 0.010034 | 0.022540 |
| L13H13 | 148 | 57 | 0.013663 | 0.010534 | 0.025270 |
| L11H4 | 79 | 35 | 0.014335 | 0.009312 | 0.029340 |

All Holm-adjusted p-values for tested heads are 1.0. Even correcting only
across 80 heads would not select any: the smallest raw p exceeds 0.05/80.
The inference remains conditional on the exchangeability assumptions and the
behaviorally selected discovery set. Absence of selection is not evidence of
absence of semantic capability or of causal head effects.

## Full dominance population and offline sensitivity

All-layer collection contains 1,455,940 eligible dominance ratios. Tau=2.2
is at cumulative fraction 80.90%; the full-population p95 is 4.481804,
versus the earlier restricted-sample p95 of 3.65812. This is still event-weighted
and includes repeated demo contexts.

Because p95 is ABOVE 2.2, every event needed for this sensitivity is already
in the archive. Filtering saved events at 4.481804 requires no model forward:
72,797 passes remain, versus 278,134 at 2.2. L3H11 retains 30 passes with
mean first-target score zero. L9H22 retains 49, mean approximately 0.517388,
just below the old 50-pass support cutoff. This illustrates score and support
sensitivity, not a monotonic notion of head quality.

No fresh p95 model run or matched-null recalibration at p95 was performed in
this review. The offline filter uses the stored ratios and scores, preserving
the original random-control draws.

## Recommended next decisions

1. Keep this run as a successful replication plus an audit of why pooled RI
   selected the two heads. Do not discard it or tune thresholds until a head
   passes. Separate original RI from newly scoped statistics in reporting.
2. Use saved events to report self, previous-token and nonlocal attention
   separately for all heads, and answer-position events separately from other
   test positions. Changing these filters changes the estimand and must be
   labelled as a new analysis rather than an exact paper replication.
3. Next empirical priority: a small exact causal preview on multiple discovery
   families with fixed metric/patch scope, including the earlier RI candidates,
   descriptive matched-score candidates such as L16H4, and controls. Include
   shared-prefix families in metric sanity checks. This tests a different
   question from token-level observational RI.
4. Only after preview validation, proceed to broad attribution/exact patching
   and held-out validation. Test contextual OV as a separately labelled
   diagnostic, not a silent replacement for the published embedding score.

No stage-2 implementation or model run was added during this analysis.
