# Stage 3 measurement and report review

Reviewed 22 September 2026. Sources: the current `new_stage_3_experiment.pdf`, its revised LaTeX plan and attended-name paragraph, `Stage3_Results_and_Analysis_local.pdf` and its LaTeX/figure sources, the measurement/analysis code in `pilot_v2`, `results/stage3_inputs_v1`, and `results/stage3_v1`.

**Verdict:** the saved run and principal numerical summaries are internally consistent and reproducible. I found no material implementation defect in the main intervention calculations that warrants repeating Stage 3. Several mechanistic conclusions in the report go beyond those measurements, and signed averages conceal important heterogeneity. A planned diagnostic relevant to the RI copying hypothesis was computed but omitted from the report. A narrowly scoped optional readout is justified; a broad extension is not required.

## 1. What was actually checked

- Read the measurement, capture/intervention, preparation and aggregation code; checked the definitions against the current plan.
- Re-ran the complete CPU analyzer from saved checkpoints, redirecting all new outputs to `reproduced/`. Its input, checksum, gate, coverage, shape, finite-value and attention-probability checks passed. This covers 105 heads, 178 pairs/89 families, 40 common position-scan pairs, 160 core prompts and 34,526 contextual-RI anchor comparisons.
- Compared 12 exported CSV tables. Eleven agree at absolute/relative tolerance `1e-12`. The attended-event CSV has tiny numerical export differences: maximum `2.813e-7` for dominance and `4.986e-8` for contrasts. Event identities, missingness and contrast signs agree; head-level classifications and statistics reproduce. All arrays in `family_effects.npz` agree exactly.
- Independently recomputed position additivity errors, attention-change magnitudes, position-role support, attended-name site restrictions, the F decomposition and AV interaction summaries from saved numeric data.
- Verified the original analysis files and summary were unchanged by the regeneration. All computations in this review are CPU checks of saved measurements; **no model/GPU experiment or optional §1.5 test was run**.

Reproduction details: `reproduced/comparison.json`; the strict event-table mismatch is explained separately in `serialization_check.json`. Review scripts and supplementary CSVs are in this directory. Regenerating the same code is a reproducibility check, not independent proof of its scientific interpretation; the code/definition review and extra calculations address that separately.

## 2. Measurement logic that is sound

**Causal readout and signs.** The outcome remains the clean-answer minus corrupted-answer logit contrast at the first divergent answer token. Shared prefixes are retained. Scope F patches the last original prompt token, even when evaluation follows an appended shared prefix. The code preserves saved F measurements where available and uses their corresponding saved baselines. The promotion/suppression identity holds; the largest family-level residual is `4.77e-7` logits.

**Head patch and attention/value factorial.** Patching the appropriate input slice of `o_proj` replaces one head's pre-output-projection result while allowing the full model, including shared normalization, to run normally. Attention reconstruction uses the actual Q/K after normalization and RoPE, the actual mask and actual V. Clean/clean and corrupt/corrupt factorial endpoints are checked against the ordinary outputs. With `M00`, `M10`, `M01`, `M11` denoting clean/clean, corrupt-attention/clean-values, clean-attention/corrupt-values and corrupt/corrupt:

`routing = M00-M10`, `values = M00-M01`,
`interaction = M10+M01-M00-M11`.

Their sum is `M00-M11`. The small discrepancy from separately saved ordinary F measurements is expected reconstruction/runtime drift within the gate; its maximum at family level is `0.01245` logits, below the declared `0.05` readout tolerance.

**Contextual RI and copying diagnostics.** Contextual RI replaces only the value-projection input, checks raw replication and retains matched event/control support. The weight copying contrast correctly compares each token's self projection with other tokens in the declared population. The attended-name rule is implemented as an event-level descriptive rule, with explicit insufficient support and sensitivity restrictions. The reproduced result is six positive movers, one negative mover, 79 neither and 19 insufficient. This operational label is not itself a causal mechanism.

**Aggregation.** Causal measurements average the two orders within each of the 89 families; the core anatomy analyses use 20 families, and synthetic probes use trials. The attended-name classification explicitly counts eligible events rather than giving each family equal weight. Thus the report's blanket opening claim that all results are family-level descriptions on 89 families should be narrowed, even though the later sections often state the right population.

## 3. Material corrections to the report

### A. Small signed averages do not establish weak interactions or invariance

The arithmetic in the reported averages is correct. The inference from those averages is the problem.

**Position additivity (§4, PDF p.3).** For each pair, define `r = sum(single-position importance) - exact Scope-P importance`, using the same 40 pairs. For L17H1:

| Statistic | Logits |
|---|---:|
| Signed mean residual | -0.13457 |
| Mean absolute pair residual | 0.94405 |
| Maximum absolute pair residual | 2.85337 |
| Mean absolute residual after averaging orders within family | 0.88890 |
| Maximum absolute family residual | 2.33452 |

The report's `0.135` is the magnitude of a signed mean, not the largest discrepancy on an example. Positive and negative errors cancel. Therefore remove the claim that this establishes negligible within-head position interactions or licenses an additive decomposition. Keep the location finding: L17H1 is causally important at specific query-fact positions, and L15H25 at the child position. Call these **single-position intervention profiles**, not a proven partition of the total effect.

**AV interactions (§7.4, PDF pp.8–9).** The same issue occurs in the statement that interactions are within about `±0.27`. Those are means over families:

| Head | Signed mean interaction | Mean absolute family interaction | Maximum absolute family interaction |
|---|---:|---:|---:|
| L27H6 | +0.205 | 1.254 | 4.579 |
| L18H18 | -0.271 | 0.811 | 3.271 |
| L18H19 | -0.159 | 0.542 | 3.254 |

Value replacement carries the larger average effect for the leading answer-position heads. That conclusion survives. Uniformly weak routing/value interaction does not. Preserve the signed averages and add absolute magnitudes or family distributions before assuming separability in Stage 4.

**Reordering (§5, PDF p.5).** The reported maximum `0.11` is also a signed mean over examples, not a bound on individual attention changes. For L21H18, query-mother attention change has signed mean `-0.024`, mean absolute pair change `0.154` and maximum `0.488`. Changed-query evidence still strongly supports query-sensitive routing; it does not establish positional invariance. Report signed changes alongside absolute changes and retain the narrower query-tracking conclusion. Also, the query preference statistic is a difference of two attention contrasts, so values above one are legitimate; it is not a single probability change.

Evidence: `additivity_per_pair.csv`, `additivity_by_family.csv`, `av_magnitudes_by_head.csv`, `serialization_check.json`, `attention_change_magnitudes.csv`.

### B. The report has not established direct versus downstream-mediated causal effects

In §6 (PDF pp.5–6), the contextual output is the clean head vector projected through the unembedding **before shared normalization**. F is a clean-to-donor intervention in the full model. In addition:

- Output contrast: query mother's first token versus mean distractor mothers; causal contrast: clean versus corrupted answer at their first divergent token.
- Output population: 20 core families; main F population: 89 families.
- Output: one clean vector; causal effect: a change between two runs, including subsequent normalization and computation.

Thus `+2.99` projected units for L27H6 versus about `+3.06` F logits is not an attribution equality. Likewise, L18H19's `+0.16` projection versus `+3.08` causal effect does not prove later heads/MLPs mediate its effect. A shared normalization response is among the alternatives, and a few shared-prefix cases also involve a later evaluation position.

Replace “entirely indirect,” “these heads act through later components,” and the corresponding definitive explanation of L23H15 with **hypotheses for path interventions**. Keep the useful observation that the selected vocabulary projection does not explain the measured effect. A matched clean-minus-donor projection under an explicitly defined normalization reference would be a better diagnostic; isolating a downstream communication path requires Stage 4 intervention.

The sentence that self-attenders “read no fact position” is directly contradicted by the attention data. For example, L17H17 puts about `0.219` total mass on the eight fact-name spans; L21H23 puts `0.147`. Self mass is only `0.186` and `0.204`, respectively. Even L25H17's self mass `0.570` is not exclusive self attention. Do not rule out their fact-reading paths. See `self_attender_fact_mass.csv`.

### C. Keep mover labels and mechanism claims separate

The six-plus-one classifications are reproduced. The report correctly rejects automatically calling L23H15 a negative name mover, but failing that criterion does not prove downstream mediation. The L30H18 observations support the hypothesis of distractor-name promotion; they do not establish that this is the causal explanation without a targeted intervention.

For L26H31, the reported roughly 6% answer selection pools **all earlier and final sites**. Earlier sites include positions before the relevant answer is available and positions with no reason to retrieve the final answer. This is not an answer-position retrieval accuracy. Its answer fraction is `12.9%` among all final-site name-attention events, and `25%` among the 16 base final-site events; the latter has only nine families. These remain descriptive. The conclusion that it has strong attended-name projections but small individual F importance (`+0.027`) is sound. “Only moving plus selecting causes an effect” is not established: redundancy, context, and the specific intervention are still alternatives.

One consequential support detail omitted from the narrative: L18H19 has a supported overall mover label, but its final-site subset has only 17 events/nine families and is **insufficient** under the frozen rule. This does not negate its strong causal F effect or overall mover status; it limits claims about that label specifically at the causal answer position. See `attended_site_comparison_valid.csv` and the original sensitivity columns.

Similarly, no fingerprint on two synthetic tasks means absence on those probes, not proof that a head is specific to kinship.

### D. A planned diagnostic is absent from the report

The current plan §1.4 explicitly requests partitioning raw RI events by whether the **current token** belongs to the target name, a control name, or neither. The computation already exists in `results/stage3_inputs_v1/current_name_diagnostics.json`; it is different from the attended-name mover test.

First-anchor examples, using the saved family-weighted raw target-minus-control gap:

| Head | Current token belongs to control: gap (events/families) | Neither: gap (events/families) |
|---|---|---|
| L18H19 | -0.022820 (111/48) | +0.000337 (51/32) |
| L21H6 | -0.024536 (421/85) | -0.004385 (116/70) |
| L19H16 | +0.005161 (92/59) | -0.001735 (144/61) |

This is relevant evidence that current-name/control overlap accompanies the negative RI name gap for some positive-copying heads, with a contrasting pattern for the anti-copying head. The partitions differ in event composition, so this is an association, not an isolated causal explanation of RI failure. Include support; several other heads have insufficient or almost no partitioned events. No new GPU work is needed to add this missing analysis.

### E. One concrete count correction

In §3 and the conclusion, **20**, not 21, of the 25 strong heads have `F/P` in `[0.96, 1.07]`. Besides the four listed exceptions, L17H3 has `P=-0.5460`, `F=-0.4022`, ratio `0.7367`. It is acknowledged later in the position section but omitted from the opening count. Treat it as a mixed-position candidate in Stage 4.

## 4. Recommendation on optional §1.5

**Run a small controlled representation readout if the aim is to clarify what the early causal sites contain; do not run all optional extensions or postpone circuit work until they are done.** The trigger in the plan is met: a site is causally localized, but the diagnostic projection has not identified its representation.

Priority candidates:

1. **L17H1:** the query-fact `is` and period positions. Ask whether a controlled readout recovers the mother's identity, the relation, or both, and how the answer changes when this head's output is replaced by the matched corrupted donor.
2. **L15H25:** the last token of the queried child in the fact. Ask whether the representation exposes that child's mother binding, beyond just the child's lexical identity.

Use the existing 20 common discovery families and both orders. Fix capture sites and readout prompts in advance. Compare intact versus head-intervened residuals at the same site, using the same readout prompt; include no injection, irrelevant injection and entity-swap controls. Avoid readout prompts that reveal the desired mother. Record probability contrasts alongside decoded text. A residual readout contains contributions from multiple components; even a successful readout is supporting evidence, not proof of a head-to-head channel.

The saved attention/projection exports do not contain the full residual vectors required for this injection, so this proposal needs a **new, small capture/readout run**; it cannot be obtained just by relabeling the current CSVs. No such run was performed in this review. Broad spectral copying analysis and function-vector mediation do not yet answer a comparably focused outstanding question. If the immediate question is who receives these writers' information, proceed directly to Stage 4 path interventions instead.

Suggested order: correct the report's aggregation-based and mechanistic claims, add the already-computed current-token diagnostic, then perform the small readout if content clarification is useful while preparing Stage 4. The existing Stage-3 causal measurements remain usable.
