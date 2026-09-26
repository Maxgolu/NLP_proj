# Final S4.4/S4.5 proposal after S4.3

Status: proposed on 26 September 2026; not implemented, run, or retrospectively preregistered. The final executable registry must be frozen before new measurements. This document does not modify the historical protocols or results.

## 1. Question and achievable claim

Does a bounded attention-head mechanism preserve contextual mother-of retrieval when both fact assignments and the queried child change? Which originally RI-selected heads participate in this mechanism, in which tested backgrounds, and through which already measured communications?

Semantic ability here means following the queried relation in these counterfactual single-hop prompts, not general semantic understanding, a literal relation representation, or a new relation-domain generalization. A retained head mechanism remains conditional on live MLPs, embeddings, normalization, demonstrations, and shared-answer-prefix computation. A faithful node mask is not an isolated or edge-faithful circuit.

## 2. Evidence motivating the experiment

Sources: `Stage4_Overleaf/main.tex`, `Stage4_Overleaf/s42_results.tex`, `Stage4_3_Methodology_and_Results.pdf`, `results/stage4_s43_review_20260926/S43_review.md`, and the primary S4.3 extension tables. The authoritative prior prospective design is `results/stage4_s42_analysis_20260924/next_stage_plan.json`.

- L17H1 is a strong fact-position source, with mother-sensitive controlled-readout evidence. S4.1 measures writer-union V routes into L18H18 (+3.400) and L18H19 (+2.074). S4.2 receiver blocking removes about 89% of its restoration effect, conditional on that intervention background.
- S4.3 measures L17H1 -> {L18H18,L18H19} -> L27H6 Q (+1.635 noising, +1.511 restoration), with smaller continuations into L21H6, L21H18, and L22H5. Joint and single L18 releases support more than one contributing branch; their effects are not additive shares.
- L15H25 child-last -> L16H1/L16H21 V is supported by S4.1. S4.2 establishes dependence on the pair; S4.3 supports pair-mediated continuation to L18 Q (child-last to L18H18: +0.172/+0.179). This is evidence of a composed dependency, not individual necessity of both intermediates or proof that Q encodes query identity.
- L8H15 is an RI-selected upstream contributor missed by the original strong-source roster. Its direct query-sentence V routes to L18H18 and L18H19 are +0.465/+0.442 and +0.460/+0.446. MLP9 release attenuates these positive effects; the negative increments are not negative raw routes.
- Opposing RI heads have stable conditional contributions in S4.2. L26H23 has a direct bypass (-0.172/-0.173) and additional block-mediated effects. Head-specific attribution of the block effect remains unresolved.
- RI23 is collectively relevant but heterogeneous and baseline-sensitive. L26H31 did not pass its dedicated backup test. L20H7 is not a stable directed negative route.

Two accounting corrections to the review prose: the extension has nine selected chain configurations sharing seven distinct structural-zero comparators. All chain direct effects are exactly zero. The block27-only mean increment (-0.089) is about 49% of the full-release increment (-0.183), not most of it, and this quotient is not an additive mediation fraction. L27H6 remains a plausible constituent to investigate later, not an identified mediator.

## 3. Fixed candidates and backgrounds

Inherit the exact C0 (33 heads), R17, C50, RI31, RI23, and W/D/P/A/N/R6/T/U rosters from `next_stage_plan.json`. S4.3 adds evidence for existing candidates, not new heads outside these sets.

The named D={L16H1,L16H21} and P={L18H18,L18H19} groups are especially important for interpreting the composed routes. W, D, P, A, N, R6, T, and U are reduction blocks, not assertions that each block is a separate circuit.

For any retained C, R(C)=C intersect the original RI selection; within the named candidate pool this equals C intersect RI31. Non-RI(C)=C minus R(C). Update these sets when C changes; never keep the initial 14/19 partition by mistake. The original 59-head selection and the two historical extras remain separate in the membership ledger.

All retained heads are live at ALL test-block positions. Replace excluded head-output slices with the existing role-mean policy. Role keys may use syntax, fact slot, and within-name token bucket, never name identity, target label, or query relevance. This matches the implemented `s42_plan.mean_roles` policy. Retain no oracle query-fact-only masks in the mechanism experiment.

Extend the mean bank to all 1,024 heads without changing its frozen original-query, 89-family, two-order, clean/corrupted reference distribution. Use leave-one-family-out means on discovery; use the full frozen discovery bank on held-out. New query conditions do not alter the bank. MLPs and the other stated background components remain live.

## 4. Retention and bounded reduction

On the original 20 common families, both orders and both fact conditions:

1. Evaluate full, empty-head mask, and C0. Evaluate the sole C50 fallback only if C0 fails. Preserve compatible saved endpoints.
2. For each family, g(C) is the clean-minus-corrupted margin gap averaged over orders. Require F=mean(g(C))/mean(g(full)) in [0.8,1.2], L=mean(abs(g(C)-g(full)))/mean(abs(g(full))) <=0.2, and no more than five percentage points of candidate-accuracy loss separately for clean and corrupted conditions and their order strata. Report order-specific F/L and first-token full-vocabulary behavior as well.
3. If the empty-head mask passes the same guards, the retention criterion does not isolate a head explanation. Continue only the already bounded participation tests; do not call the result a self-contained circuit or start an MLP search.
4. Starting from a passing candidate, perform group-first greedy reduction with the historical tie-break, at most two passes, and at most 32 distinct trial masks rather than 96. Recompute removal effects after each accepted deletion. Any corrective discovery addback/retest consumes this same 32-mask allowance. Do not optimize accuracy by dropping opposing heads while ignoring fidelity.
5. Test full/C completeness on the core panel for W,D,P,A,N,R6,R(C),Non-RI(C). For the six functional blocks, remove the frozen block roster from full and its intersection with C from C: members absent from C are already clamped, not silently removed from the full-model comparator. This can expose missing alternative responses even when intact C is faithful. For RI/non-RI comparisons, use the exact retained-member roster on BOTH sides. Compare family gaps (normalized absolute discrepancy <=0.2) and condition/order accuracies (<=5 points). All these removals use all-test masks, not old colon-only group endpoints.
6. A failed completeness check permits only addback of previously deleted blocks or the one R17 block, within the cap. If repair is not available, keep the bounded set and report incomplete explanation. Do not expand the pool.
7. On all 89 discovery families evaluate the chosen mask and the primary original-axis configurations below. If the reduced mask fails retention, permit one return to its starting passing parent (C0 or C50), without further full-discovery search. If that parent also fails, report failure. Freeze the resulting mask before held-out evaluation.

The final mask is bounded and approximately reduced, not uniquely or individually minimal. Failure to retain behavior is an admissible scientific outcome.

## 5. One six-configuration fact-by-query panel

Use the four cells x00, x10, x01, x11: original/swapped mothers crossed with original/alternative queried child. Derive answers from the facts; with fixed candidate names a,b the correct pattern is a,b,b,a. Use both orders, preserve errors, retokenize new questions and recompute their prefixes and role positions.

The six configurations are:

1. Full model.
2. Final retained C.
3. C minus R(C).
4. C minus Non-RI(C).
5. C minus D intersect C.
6. C minus P intersect C.

Items 5 and 6 replace the old full-minus-O3 and full-minus-RI23 crossed-panel slots. Those old full-background groups already have extensive S4.2 measurements. The new slots ask whether the specific S4.3-supported stages matter inside the retained relational mechanism. RI23 still motivates C50 and its candidates remain in the reduction/completeness scope; no new 23-head sweep is added.

Measure the panel on 20 discovery families, then run the same frozen panel on all 87 held-out families. Do not add the previous conditional 89-family crossed-panel expansion: the independent final panel supplies the test of reproducibility. A discovery crossed-panel failure is reported, not used to start another mask search.

Report every condition's accuracy and fixed-sign margin, each adjacent fact-axis and query-axis contrast, both orders, and B=mean(M00-M10-M01+M11)/4. Report B ratios only with a stable full-model denominator. A positive B alone is insufficient: the four condition outcomes must support the expected binding pattern. Apply the five-point condition/order accuracy fidelity guard to C versus full.

Interpret removal effects cautiously: a decrease in relational behavior supports contribution, while preserved behavior does not prove absence because of redundancy. Selective impairment on an axis describes functional dependence under the intervention; it does not uniquely decode the information represented by a head.

## 6. Targeted individual RI participation

The only new individual-head panel targets the nine already evidence-supported RI heads:

L8H15, L16H1, L16H21, L17H24, L18H19, L19H16, L22H5, L23H15, L26H23.

For each member retained in C, compare C against C-minus-h with the same role-mean baseline, all-test mask, original clean/corrupted inputs and both orders. Measure on the core discovery panel and once on the held-out panel. This is a new-background membership question, not a repeat of intact singleton screening. Do not adaptively replace missing/failed members with other candidates.

Use the signed contrast M(C)-M(C-minus-h) on clean inputs and its sign-aligned counterpart M(C-minus-h)-M(C) on corrupted inputs. Retain both directions, mean absolute family effects, and the inherited descriptive coherent/heterogeneous rule (tau=0.1). An opposing member need not improve task accuracy; contribution can be negative. Record fidelity change as well as performance change.

A head can have a supported full-model route and prior conditional contribution yet be dispensable in this particular reduced mask. Label that status explicitly. Group dependence alone does not establish every member. Do not require all nine to pass or erase members absent from C.

## 7. Final sensitivity and held-out suite

On the 89 discovery families, original fact axis only, repeat four final configurations with paired opposite-fact donors: C, C-minus-R(C), C-minus-Non-RI(C), C-minus-N intersect C. Apply donors symmetrically and compare to the corresponding means; the C-minus-N mean comes from the matched completeness scope. Donors on a changed-query axis are not needed in this bounded sensitivity panel.

If meanings conflict across baselines, report baseline-sensitive membership or fidelity. Do not substitute the more favorable intervention as primary. No new mean bank is built from held-out examples.

Freeze one held-out suite on all 87 families:

- The six four-cell configurations above, both orders.
- Empty mask and full-minus-C on the original two cells.
- Matched full/C removals for D,P,N,R(C),Non-RI(C), original cells, with the same functional-block versus retained-RI roster policy as discovery (reuse C-side D/P/RI/non-RI panel cells). W,A,R6 completeness remains discovery-only.
- The up-to-nine retained individual RI tests, original cells.
- The four paired-donor sensitivity configurations, original cells, and their required mean references.
- Four frozen representative route measurements, original pairs, noising and restoration: (i) L17H1 writer-union -> {L18H18,L18H19} -> L27H6 Q; (ii) L15H25 child-last -> {L16H1,L16H21} -> L18H18 Q; (iii) L8H15 query-sentence -> L18H19 V; (iv) L26H23 colon -> logits direct bypass. The chain no-release terms are structural zeros and need a gate, not redundant per-pair scientific runs.

These four validate representative communications, not the entire historical graph. Other routes remain discovery-supported. No new head search, S4.3 block decomposition, or all-16-route extension is included.

After held-out evaluation, mark each claim reproduced, inconclusive, or not reproduced. Do not change C, thresholds, routes, means or head lists. In particular, low or failing C fidelity does not invalidate every measured route, and a passing C does not certify every drawn edge.

## 8. Workload envelope

Counts below are scored endpoint evaluations before exact reuse, not GPU hours or all forward passes. One original-axis mask on 20 families costs 20*2 orders*2 conditions=80 endpoints; on 89 families, 356; on 87 families, 348. A six-mask four-cell held-out panel costs 6*87*2*4=4,176.

Core discovery upper bounds:

| Panel | Endpoints |
|---|---:|
| Four pilot masks, including conditional C50 | 320 |
| 32 reduction/addback trials | 2,560 |
| Eight matched full/C completeness groups | 1,280 |
| Nine targeted retained RI removals | 720 |
| Six four-cell configurations | 960 |
| Total core ceiling | 5,840 |

The previous reduction-plus-crossed-panel allowance alone was 7,680+960=8,640 endpoints. The smaller reduction therefore funds the targeted membership panel without expanding the core workload.

Full discovery: at most ten mean configurations on the original axis (the six panel masks, empty, full-minus-C, C-minus-N, and one parent/fallback evaluation): 3,560. Four donor configurations: 1,424. The ten-mask count is an allowance, not an instruction to rerun cached conditions.

Held-out upper bound: six four-cell masks 4,176; empty/complement 696; additional matched completeness cells 2,088; nine individual removals 3,132; four donor configurations 1,392; four bidirectional routes 1,392. Total 12,876. C-minus-N means are included in the additional completeness cells.

Overall ceiling: 5,840+3,560+1,424+12,876=23,700 scored endpoints before reuse. The four held-out route tests additionally require up to 1,392 hybrid construction forwards. Mean-bank construction, intact capture passes and small implementation gates are separate mandatory overhead and must be counted in the executable schedule. All-head mean-bank construction was already required by the old S4.5 design.

The old plan did not state a single total endpoint cap for final completeness and validation. Therefore 23,700 is a new explicit ceiling, not a recovered historical total. Relative to that plan, no population, candidate pool, four-cell slot count, or reduction allowance is enlarged; representative validation is narrowed, the adaptive 89-family crossed-panel extension is removed, and the added targeted singleton work is funded by reduction savings. Cache every fully identical intervention exactly once.

## 9. Analysis, implementation gates, and exclusions

Average orders within family for primary estimates; also report each order and mean absolute within-family order changes. Use paired 20,000-draw family bootstraps, with frozen seed, for descriptive intervals. Do not label exploratory thresholds significance tests, or combine selected discovery families with held-out for confirmatory estimates. Preserve both candidate logits, gold signs, shared prefixes, full-vocabulary top token, baseline identity, and the full intervention key.

Small gates: all-live mask reproduces full model; self donors are identity; selected C/full mask endpoints agree with compatible historical controls; changing only the question cannot change earlier fact activations; both swapped answer tokens and shared-prefix cases use correct metric positions; role means never depend on answer identity; excluded heads are actually replaced and live MLPs remain live. Preserve the documented S4.3 family-034 compatibility reference rather than silently relaxing a tolerance.

No new model training, observational RI, synthetic probes, representation readout, attention redirection, spectral analysis, generic all-head screen, dedicated L26H31 backup study, exhaustive coalitions, or whole-block decomposition. Do not add L26H23->L27H6 merely because it is plausible: the existing bypass plus conditional tests already allow bounded opposing participation to be assessed. Naming the unresolved block hypothesis is sufficient for this project's stopping boundary.

## 10. Decision table

| Outcome | Permitted conclusion |
|---|---|
| C passes fidelity and four-cell held-out behavior; RI removals and tested routes reproduce | Selected RI heads participate in a mixed RI/non-RI mechanism implementing this contextual relational task; report their measured roles and signs. |
| C passes original axis but fails crossed panel | A fact-swap-faithful head subset, not an established mechanism preserving joint fact/query behavior. |
| C passes behavior, individual RI tests fail but group RI removal matters | Collective RI dependence; no unsupported individual necessity claim. |
| C fails but selected routes/conditional tests reproduce | Supported partial communication mechanism and selective RI participation; no complete retained mechanism. |
| Empty head mask passes | The chosen scope/metric does not isolate a self-contained head explanation. |
| Mean/donor or order results differ | Baseline- or order-conditional conclusion, not universal necessity or invariance. |

Completion is the frozen evaluation and honest decision table, not successful recovery of a preferred circuit.
