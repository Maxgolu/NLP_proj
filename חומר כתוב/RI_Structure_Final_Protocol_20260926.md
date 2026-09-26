# RI participation in measured structures: implementation specification

Status: prospective proposal, 26 September 2026. Not implemented or executed. Supersedes the experimental emphasis of `Final_S44_S45_Proposal_20260926.md`; it does not alter historical experimental results. Freeze the executable registry and code before new measurements. This revision prioritizes additional RI candidates and structure-specific participation, not greedy circuit minimization.

## 1. Fixed populations and estimands

- Initial discovery: the existing 20 common families, two orders (40 aligned fact-swap pairs).
- Extension: all 89 discovery families, two orders (178 pairs).
- Final validation: all 87 sealed families, two orders (174 pairs), one frozen suite.
- Four input cells: original/swapped mother assignments x original/alternative queried child, x00/x10/x01/x11. Their gold answers are a/b/b/a under a fixed candidate a-minus-b metric. Preserve model errors; do not filter on new-cell success.
- Metric: candidate logits at the first divergent answer token, teacher-forcing the shared prefix. Changed-query inputs require new tokenization, offsets and position/prefix checks. The same names a,b define the fixed sign in all four cells.
- Primary aggregation: average the two orders within family, then families. Also report each order separately, absolute family effects, and mean absolute within-family order differences.
- Family bootstrap: 20,000 paired resamples, seed 20260926. Discovery intervals are descriptive and not selection-adjusted. Freeze validation signs and selected comparisons before opening validation.

## 2. Five structures, with exact primary route anchors

All head indices are zero-based. The retained-node tests keep the listed heads at ALL test positions. The route tests retain their own narrower historical source/channel definitions. These are different interventions and must have different IDs.

| ID | Retained head set | Primary route anchor for the participation matrix | Historical I/J (logits) |
|---|---|---|---|
| T1 | L17H1,L18H18,L18H19,L27H6 | L17H1 query_writer_union -> live {L18H18,L18H19} -> L27H6 Q at colon | +1.635/+1.511 |
| T2 | L15H25,L16H1,L16H21,L18H18,L18H19 | L15H25 query_child_last -> live {L16H1,L16H21} -> L18H18 Q at colon | +0.172/+0.179 |
| T3 | L17H1,L18H18,L27H6 | L17H1 query_writer_union -> live {L18H18} -> L27H6 Q at colon | +0.876/+0.809 |
| T4 | L8H15,L18H18 | L8H15 query_sentence -> L18H18 V; no MLP release | +0.465/+0.442 |
| T5 | L20H1,L27H6 | L20H1 colon -> L27H6 Q; direct route | -0.236/-0.223 |

T3 and T5 have no member of the original 59-head RI selection. T3 is deliberately nested in T1; this is a useful comparison, not independent replication. T4 and T5 are measured two-head routes, not three-stage composed chains. T2's retained-node set contains both L18 receivers because both have supporting S4.3 evidence, but its PRIMARY matrix anchor is only the child-last-to-L18H18 branch. Do not describe a change of that anchor as a measured change of both branches or of their unmeasured joint release.

Additional L17H1/pair continuations to L21H6, L22H5 and L21H18 remain historical context, not extra screening targets. Do not concatenate separate chains into an unmeasured longer route.

For T1/T2/T3, the no-intermediate comparator is structurally zero. Verify it in gates and report the route effect itself. Do not reinterpret a structural null subtraction as incremental mediation evidence.

## 3. Exact broad sets

C33 (inherit the original C0 without new selection):

```
L6H24 L8H15 L13H10 L13H18 L14H23 L14H26 L15H3 L15H25
L16H1 L16H21 L16H31 L17H1 L17H3 L17H17 L17H24
L18H18 L18H19 L18H24 L19H16 L19H22 L20H1 L20H7
L21H6 L21H18 L21H23 L22H5 L23H15 L24H19 L25H17
L26H23 L27H6 L30H13 L30H18
```

C50 = C33 plus the following R17, and nothing else:

```
L1H27 L8H8 L9H6 L9H17 L11H4 L11H7 L12H2 L16H4
L16H16 L16H24 L17H5 L21H25 L23H10 L25H18 L26H31 L27H23 L30H3
```

All T sets are subsets of C33. C50 is a conditional fallback, not a second adaptive search. No greedy reduction or role trimming is part of this revision.

Select the broad background B after the core retention test: C33 if it passes; otherwise test C50 and use it if it passes. If neither passes, B=C50 as a PREDECLARED PARTIAL background, with no sufficient-mechanism claim. Later failure does not trigger a new search or switching after inspecting validation.

## 4. Candidates and reference heads

Freeze six additional RI candidates:

| Head | Stage-1 reason |
|---|---|
| L1H27 | Query-fact first-anchor target RI rank 1; broad support |
| L11H4 | Query-fact first-anchor name gap rank 2; survives concentration checks |
| L17H5 | Query-fact first-anchor name gap rank 1; nonlocal attention with template dependence |
| L23H10 | All-facts first-anchor name gap rank 1; current-target alternative remains relevant |
| L25H18 | All-facts last-anchor name gap rank 2; limited query-fact support is retained as a qualification |
| L9H16 | Previously prioritized last-anchor candidate; broad support but order sensitivity; not in RI31 |

The five candidates other than L9H16 failed the G2 retention rule in the tested weakened background. This is a new structure/background question, not an independent replication of successful singleton effects. The shortlist spans pre-existing RI diagnostics; it is not a new global ranking or a claim of statistical selection validity.

Reference heads: L18H19 and L8H15. Measure their functional toggle in B on the core four-cell panel to calibrate against known participants. Do not force a positive outcome under the new background, and do not add them to the six-candidate route-modulation search.

RI membership always comes from the ORIGINAL candidate JSON. The historical pooled extras are not silently included in RI31. Define R(B)=B intersect the original 59-head set. All selected broad-set RI heads are already in RI31.

## 5. Shared masking and baseline policy

Keep retained heads live at every original test-block position; replace other head-output slices with role means. Keep MLPs, embeddings, demonstrations, shared normalization and answer-prefix computation live. These are head-subset mechanisms conditional on that background, not isolated circuits.

Use the existing role policy: syntax/slot/token bucket only, never name identity, query relevance or gold answer. Build missing all-1024-head mean statistics from the frozen original-query discovery reference distribution, both fact variants and orders. Discovery uses leave-one-family-out; held-out uses the fixed full discovery bank. Do not add changed-query or held-out data to the bank.

For candidate h define B_h_minus=B without h, and B_h_plus=B union h. Exactly one of these is B. Thus there are at most seven distinct B states for all six candidates, not twelve independent base contexts. If B=C50, L9H16 addition makes one 51-head TEST configuration; this is not a new 51-head fallback or an expansion of B itself. Record its membership explicitly.

Capture intact clean and corrupted donor activations inside the SAME B_h state for route tests. Do not mix full-model source donors with a retained-model recipient without declaring a different intervention.

## 6. Stage A: assess the structures' task behavior

Core four-cell panel, both orders:

- Full model, empty head mask, T1,T2,T3,T4,T5,C33.
- C50 only if C33 fails the guards below.

For the original fact axis, let g_f(D)=mean_over_orders[M_D(x00)-M_D(x10)]. Require F=mean(g(D))/mean(g(full)) in [0.8,1.2] and L=mean(abs(g(D)-g(full)))/mean(abs(g(full))) <=0.2. Require candidate accuracy no more than five percentage points below full in each of four cells and each order stratum. Report all margins, full-vocabulary first-token outcomes and order-specific gap ratios.

Define b_f(D)=mean_over_orders[(M00-M10-M01+M11)/4], and B(D)=mean_f b_f(D). Report both fact-axis and both query-axis contrasts, not just B. Use B/full ratios only with a stable denominator.

A T mask that fails is not an independently faithful mechanism. Its measured route remains eligible for Stage C in the broad background. In particular, T5 is an opposing route; it is not expected to solve the task alone. If the empty mask passes, do not make an isolated head-mechanism claim even if B also passes.

Stage A is a descriptive characterization and selection of B, not permission to tune T rosters.

## 7. Stage B: functional participation of each candidate

In the chosen B, run B_h_minus and B_h_plus on all four cells, both orders, for each of six h. Reuse the common B endpoint. Also run B-minus-L18H19, B-minus-L8H15, and B-minus-R(B) on the same panel.

For each h store the four fixed-sign margin differences, the four gold-oriented differences, change in condition accuracies, change in F/L and d_b(f)=b_f(B_h_plus)-b_f(B_h_minus). Positive d_b means greater correct binding contrast; negative d_b means an opposing contribution. A large d_b without the actual four-cell outcomes is not enough to claim improved relational behavior.

Use the inherited descriptive rule separately for d_b and for the original-cell gold-oriented margin differences: coherent if abs(mean)>=0.1 and sign agreement>=70%; heterogeneous if mean(abs)>=0.1 and >=20% of families exceed 0.1 in magnitude. Report both; there is no new significance test. Preserve zero/null results.

This stage assesses functional participation in B. It DOES NOT yet associate h with a specific T route. The design intentionally avoids requiring every tiny T mask to solve the task, or running an unbounded T+h rescue search.

## 8. Stage C: the complete six-by-five structure-association matrix

For every candidate h and primary route anchor t, measure the route's fact-swap noising effect in B_h_plus and B_h_minus on the 40 common pairs:

```
I_t(B_state) = M_intact(B_state, clean)
               - M_route_patched(B_state, clean <- corrupted)
Gamma_h,t = I_t(B_h_plus) - I_t(B_h_minus)
```

This is a difference of matched intervention effects, not a comparison of raw patched margins. Captures, frozen branch increments, source donor, receiver donor and fresh recipient must all respect the declared B_state. Preserve S4.1/S4.3 source normalization and selective-live-head semantics; a new B cannot reuse old full-model route endpoints.

The same five baseline routes in B are shared across candidates. At most seven B states x five anchors x40 pairs=1,400 scored core route endpoints are needed. Cached source/receiver captures are shared wherever the complete identity matches.

Gamma asks whether h changes the efficacy of this measured communication in this background. It is an interaction, NOT proof that h lies on the serial route or that the whole head group has changed equally. In particular, downstream amplification can change Gamma. A route must itself pass the inherited effect rule in at least one B_state for the pair to be eligible; otherwise mark it unavailable/weak in this background, not an established association.

Apply the same coherent/heterogeneous rule to family Gamma. Screen all 30 pairs; do not silently omit late candidates, negative effects, or failed T masks.

Freeze up to THREE informative (h,t) pairs for extension. Deterministic ordering: coherent-Gamma pairs first, then heterogeneous; within class decreasing mean(abs(Gamma)), then layer/head then T index. First pass selects at most one pair per candidate to cover different candidates; a second pass fills remaining slots with at most two pairs per candidate. Fewer than three is valid. Do not fill a slot with a below-rule result. Label whether Stage B also detected a functional effect; Gamma-only pairs are conditional interaction candidates, not confirmed semantic members.

If fewer than three route-pair slots are occupied, fill unused slots with candidates that passed the Stage-B d_b rule but have no selected pair, ordered coherent first, then mean(abs(d_b)), then head ID. Label these records `structure=B, functional_only`; they receive behavioral extension/validation but no invented Gamma or attachment. Across both record types there are at most three selected slots and at most three distinct candidates. Zero selected slots is a valid stop: validate only the fixed B/collective-RI configurations. Unselected discoveries remain explicitly discovery-only.

## 9. Stage D: confirm selected pairs and test a named connection

For each frozen pair, complete Gamma in BOTH noising and restoration directions on the SAME 40 common pairs, reusing the noising screen. Restoration uses M_route_restored(B_state, corrupt <- clean)-M_intact(B_state, corrupt). Compare effects within direction before comparing directions. Report same/opposite signs, family sign agreement, means, absolute means and each order; do not replace a selected pair if restoration disappoints. New link/attachment measurements are NOT additionally expanded to all 89 discovery families: the frozen 87-family validation supplies the independent replication. The 89-family expansion below is restricted to broad-background behavioral endpoints.

Also test one PREDECLARED direct attachment and one matched control per selected pair. Attachments are deliberately limited; a null does not exclude every alternative channel.

| Target structure | Candidate | Named attachment |
|---|---|---|
| T1,T3,T4 | L1H27,L9H16,L11H4,L17H5 | h, all four fact sentences -> L18H18 V, output row at colon |
| T2 | L1H27,L9H16,L11H4 | h, all four fact sentences -> L16H1 V, output row at colon |
| T2 | L17H5 | h, colon -> L18H18 Q, output row at colon |
| T1,T2,T3,T4 | L23H10,L25H18 | L18H18, colon -> h Q, output row at colon |
| T5 | L1H27,L9H16,L11H4,L17H5 | h, all four fact sentences -> L20H1 V, output row at colon |
| T5 | L23H10,L25H18 | h, colon -> L27H6 Q, output row at colon |

All-fact masks include each entire annotated fact sentence, excluding question/colon; they are not oracle query-only masks. V injection is at those fact positions with the receiving head's recomputed output used at the colon. Colon-Q tests never infer a direct fact-position-to-Q route. All sources precede receivers in layer order.

Run attachments in B_h_plus, both directions, the 40 common pairs. For each possible direct attachment freeze a same-layer alternative receiver before new results using Random(20260926), sorted eligible head IDs, excluding C50, all T members, all original RI candidates and the tested h. Process unique attachment keys in lexical order so random selection is reproducible. Keep source, channel, source positions and direction identical. Enable that control receiver in BOTH compared contexts if it is normally mean-clamped; equivalently, the scored attachment/control comparison runs in B_h_plus plus the control receiver, and the target attachment is recomputed in that SAME background. Never compare a live target receiver to a disabled control. Generate exact control rosters for all possible attachments in the pre-run manifest, not after effect inspection.

The matched attachment/control background is a local diagnostic background, distinct from the primary B_h context used for Gamma; record it explicitly. These are at most six attachment/control endpoint configurations, not a new all-head screen.

A retained named route plus reproduced Gamma supports a connection to a member/branch of the structure. It does not establish exclusive mediation through the entire chain. Shared L18H18 or L27H6 endpoints must be described as shared, not as independent structure-specific discoveries.

For final classification apply the inherited coherent/heterogeneous rule separately to Gamma and the attachment in each direction. A coherent bidirectional claim requires both directions coherent with the same mean sign and at least 70% family sign agreement between directions; otherwise label the outcome heterogeneous, asymmetric or unreplicated as applicable. Report the paired difference of absolute family attachment effects versus the matched control; if the target does not exceed the control, do not claim receiver selectivity. Its bootstrap interval is descriptive, not a new post-selection significance test. Negative Gamma can strengthen a negative route or attenuate a positive one: always print raw route effects in both h states beside Gamma.

## 10. Full-discovery behavior and baseline sensitivity

On the original fact axis, all 89 families, evaluate at most seven mean configurations: full, empty, B, B-minus-R(B), and the additional B_h state for each of at most three selected candidates. A selected candidate's other state is already B. Reuse every exact core cell.

Repeat B, B-minus-R(B), and the up-to-three candidate-toggle configurations with paired opposite-fact donor replacement, symmetrically on clean/corrupted inputs. These five configurations test baseline sensitivity; they are not required to pass identical numerical fidelity thresholds. Report discordance rather than choosing the favorable baseline. Do not rerun a general weakened-background survey.

If B fails full-discovery fidelity, keep its frozen membership and classify it partial. Do not restart C selection, add heads or redefine thresholds. Route claims remain separately assessable.

## 11. Freeze and final held-out suite

Freeze all hashes, means, B, candidate/structure rosters, selected pairs and controls, primary signs, thresholds, input construction and gates BEFORE accessing the 87 held-out families. No replacement of failed candidates or selection of new links is allowed.

On held-out, both orders:

1. Four-cell means: full, empty, B, B-minus-R(B), and the extra state for each selected candidate (<=7 configurations).
2. Selected Gamma pairs: both B_h states, both fact-swap directions, original axis (<=3 pairs).
3. Selected named attachments and their matched controls in their frozen diagnostic backgrounds, both directions (<=6 configurations).
4. B, B-minus-R(B), and selected candidate toggles under paired-donor replacement, original axis (<=5 configurations).

T1..T5 standalone behavior and unselected candidates remain discovery-only. Do not imply that they were independently validated. The representative selected route anchors are validated within the frozen B backgrounds, not as a new all-routes full-model validation.

Final labels: (a) reproduced functional and communication participation, with sign/background; (b) functional participant, attachment unresolved; (c) route/interaction modulation only, overall functional contribution unresolved; (d) no effect detected in this bounded scope; (e) order/baseline sensitive or not reproduced. Any semantic claim is restricted to this single-hop contextual task and the stated live background.

## 12. Gates, implementation hooks and data products

Reuse the frozen S4.1/S4.3 route executors, not just their scalar results. Add a retained-head background layer with explicit precedence: mean masks define B; source/receiver interventions override only their declared live coordinates; cached branch increments are captured after applying the declared background. Reconstruct joint attention/value output with OLMo shared normalization handled exactly as in the existing gated engine.

Required gates on a small fixed discovery subset, including a shared-prefix case: all-live reproduces full; self-donor is identity in EACH new background type; no-intermediate cross-position Q is zero; changing the question cannot alter earlier facts; mean role keys exclude identities/relevance; disabled receiver controls are not silently interpreted as nulls; full-background route regression reproduces historical anchors within the documented hardware policy. Keep the S4.3 family-034 compatibility accommodation explicit and do not relax tolerances automatically.

One cache key includes model/tokenizer hashes, input/cell/order, metric prefix, live mask, mean-bank/LOFO ID, source mask, receiver/channel/output row, mediator/live-head list, direction, donor-background identity and control status. Store both candidate logits, baseline margins, top vocabulary token, intervention norms, family/order, head/structure IDs, and reusable captures. Do not subtract across different keys except explicitly defined paired contrasts.

Deliver: frozen_plan.json; structure_registry.json; candidate_registry.json; mean_bank_manifest.json; core_behavior.csv; core_route_effects.csv; candidate_structure_matrix.csv; frozen_selection.json; extension_behavior.csv; attachment_controls.csv; core_bidirectional_gamma.csv; validation_*.csv; and a final RI membership ledger. Summaries must distinguish node-mask behavior, historical routes, new Gamma interactions and newly tested attachments.

## 13. Hard workload envelope

Scored endpoint caps BEFORE reuse:

| Work | Maximum |
|---|---:|
| Core full/empty/five T/C33/C50 four-cell pilot | 1,440 |
| Core six candidate toggles, two references, collective RI removal | 1,440 |
| Core route matrix: seven B states x five anchors x40 | 1,400 |
| Full-discovery original-axis mean configurations: 7x356 | 2,492 |
| Three selected Gamma pairs, core: 3x2 states x2 directions x40 | 480 |
| Three named attachments and three controls, core: 6x2x40 | 480 |
| Five full-discovery donor configurations: 5x356 | 1,780 |
| Held-out four-cell means: 7x696 | 4,872 |
| Held-out Gamma: 3x2x2x174 | 2,088 |
| Held-out attachments/controls: 6x2x174 | 2,088 |
| Held-out donor configurations: 5x348 | 1,740 |
| Total scored endpoints | 20,300 |

Most shared core/extension cells reduce this total. Route-hybrid construction, diagnostic-background captures, all-head mean collection and gates are ADDITIONAL forwards, not hidden in the endpoint count. The route endpoint total is at most 6,536; constructing their hybrids can require up to another 6,536 forwards with the existing engines, before cache reuse. A wall-time or total-forward comparison must include these costs. This proposal removes all 32/96-mask greedy reduction work and the previous broad completeness/individual-known-member panel. The old S4.4/S4.5 design had no single complete forward/GPU-hour cap, so matching endpoint ceilings alone does not prove equal runtime; the executable scheduler must inventory captures and hybrid passes before a job is submitted. No new run is authorized by this document itself.

Never exceed three selected links, six additional RI candidates, five route anchors, or the frozen populations. Do not fill unused capacity by adding new hypotheses. The final experiment can legitimately conclude that the new high-RI candidates remain unconnected under the tested scope.
