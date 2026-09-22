# Stage 3 §1.5: controlled readout analysis

## Main finding

L17H1 has a reproducible intervention-dependent effect on the readout's preference for the query mother's first token, especially at the query-fact `is` position. Replacing this head's local output with its corrupted donor produces most of the average contrast shift produced by replacing the entire source prompt with its corrupted twin. This supports a contribution to an identity-sensitive representation at that site. It does not identify a downstream channel, establish a relational computation, or explain a quantified fraction of the original answer effect.

The L15H25 result at the queried child's last token is inconclusive about content: even the full entity swap produces little mother-specific readout response. Its previously measured causal importance remains valid. The optional experiment therefore sharpened the L17H1 hypothesis but did not resolve L15H25's representation.

## Verification and measurement

The run contains exactly the 1,920 unique planned records: 40 pairs × four writer/site combinations × six conditions × two readouts. All combinations, candidate-name lists, positions and family/order labels were checked against the frozen plan. The plan and input hashes match the manifest/source hashes; all 160 stored site importances agree with the original Stage-3 position table. The gate passed with zero identity-injection/self-patch drift and exact reproduction of the three saved causal probe values.

All four analysis CSVs were regenerated in a separate directory and match the supplied files byte-for-byte. The local `stage3_readout.py` hash differs from the run manifest only because local hash-checked plan discovery was added to the analyzer. The source version recovered from the upload bundle matches the manifest; measurement code is unchanged. Original results were not modified. No new model or GPU experiment was performed.

The RUNBOOK's opening “has not run” statement describes its preparation-time status; it is superseded by the completed records and gate. Its interpretation suggestions are hypotheses, not additional results.

For each source residual, the readout measures

`C = log p(first token of query mother) - log p(first token of other candidate mother)`.

The reported shift is `C(condition) - C(intact_clean)`. A negative shift means movement toward the swapped-in mother. Orders are averaged within each family, then the 20 family means are averaged. These are descriptive results on the selected discovery families. Probabilities concern the first token, not the full name. The representation is taken at the output of the writer's entire decoder block, including its subsequent MLP and shared normalization; it is not an isolated head vector.

## Results

| Writer/site | Readout | Clean C | Full source-swap shift | Local head-replacement shift | Families with negative head shift | Ratio of mean shifts |
|---|---|---:|---:|---:|---:|---:|
| L17H1 / `is` | identity | +0.51465 | -0.52070 | -0.45840 | 20/20 | 0.880 |
| L17H1 / period | identity | +0.28789 | -0.04609 | -0.04121 | 20/20 | 0.894 |
| L17H1 / `is` | mother_of | +0.25459 | -0.00903 | -0.00981 | 17/20 | 1.086 |
| L17H1 / period | mother_of | +0.26479 | -0.02686 | -0.02319 | 20/20 | 0.864 |
| L15H25 / child-last | identity | +0.18213 | +0.02021 | +0.00605 | 8/20 | not informative |
| L15H25 / child-last | mother_of | +0.24312 | +0.00181 | -0.00020 | 9/20 | not informative |
| L15H25 / `is` | identity | +0.29395 | -0.15117 | +0.00303 | 7/20 | approximately zero |
| L15H25 / `is` | mother_of | +0.25063 | -0.00171 | +0.00020 | 5/20 | not informative |

Ratios divide two signed mean shifts. They are not proportions of represented information, mediated answer effects, or per-family explained variance. In particular, a ratio near one with a tiny denominator does not establish strong decoding.

### L17H1 at `is`: the strongest positive result

The clean identity contrast is +0.515; it falls to +0.056 after local head replacement and to -0.006 for the corrupted source. A residual from the same role in the other answer-side fact gives +0.002, close to the corrupted source. Thus the effect is aligned with both relevant controls, rather than merely differing from the no-injection prompt.

The head-induced shift is negative in every family, with median -0.459 and range [-0.701, -0.234]. Unlike the earlier signed-average concerns, the direction here is consistent across families. Its magnitude is about 88% of the mean full-swap shift in this readout. The supported interpretation is that L17H1's local output contributes to a mother-token preference exposed by the identity readout at this site.

This site has mean original-task single-position importance **+1.769 logits**. The approximately **+5.7** figure is the head's all-position importance over 89 families, not its importance at this site. The readout shift of -0.458 is a different experiment and cannot be subtracted from 1.769 or 5.7 to estimate an unexplained remainder.

### L17H1 at the period: consistent but weaker under identity readout

The identity head shift is -0.0412 in all 20 families, with median -0.0410 and range [-0.0723, -0.0117]. The full source-swap shift is -0.0461 and the other-fact shift -0.0497. This is the same directional pattern as `is`, but around an order of magnitude smaller.

The period retains substantial original-task single-position importance, **+1.377 logits**. Its small identity-readout response therefore does not imply a correspondingly small role in the original computation. It demonstrates the limits of treating readout magnitude as a causal-importance proxy.

The `mother_of` readout at the period also moves consistently: head shift -0.0232 and source swap -0.0269, each negative in all families. This is a small identity-sensitive response to that prompt; the prompt's wording alone does not turn it into evidence of decoding a relational binding.

### L15H25: the principal content question remains unresolved

At the child's last token, the original-task single-position importance is **+0.429 logits**, positive in every family. The replacement also changes the captured residual: its mean norm difference is 1.106, so this is not a numerically inert source intervention.

Nevertheless, the `mother_of` head shift is approximately -0.00020, with median zero; the whole source swap is only +0.00181 and is negative in just four families. The identity readout similarly fails to show the expected robust mother-swap response at this site. Without a sensitive source-swap control, a small head contrast cannot distinguish absent information from a readout that does not expose it.

The child-last `mother_of` candidate accuracy is 55% for the clean source, replaced head, corrupted source **and no injection**. This percentage is therefore not evidence of recovered mother binding. Nor is there strong evidence that identity readout cleanly recovers the child itself: the child's first token is the highest-scoring candidate in only 15% of intact identity probes.

At L15H25's secondary `is` site, source swapping does affect identity readout (-0.151; negative in 18/20 families), while replacing L15H25 has essentially no systematic effect (+0.003). This shows that readout sensitivity to source identity at block 15 does not automatically entail dependence on L15H25 there. It agrees with the site's near-zero original-task importance (+0.00394). It does not assign L15H25 an “address” function at its different, causal child-last site.

## Two limits that matter for reporting

### The decoder has not freely recovered the names

For every intact-clean probe, the most probable next token is ` hello` for identity and ` Mary` for mother_of. The successful observation is a change in a candidate contrast, not successful unrestricted name generation.

At L17H1/`is` under identity, the mean absolute probability of the query mother's first token is 0.000373 (0.0373%), versus 0.0000429 without injection, 0.000282 from the other-fact residual, 0.000290 after head replacement and 0.000285 for the corrupted source. This supplies a positive probability-relevance check and a source-specific effect, but also makes the low absolute decoding probability explicit. These means average probabilities of each example's own target token; they are not exponentiated mean log probabilities.

For mother_of, candidate-top fractions remain at the no-injection value of 55% across all clean and head-replaced writer/site groups. Small continuous contrast changes can still be real, but the current results do not demonstrate successful relation verbalization. `irrelevant` is a residual from the competing answer fact, which actually contains alternative relevant information; it is an other-fact control, not an information-free vector.

### `head_span` versus `head_site` is structurally redundant here

All 320 paired readout outputs are exactly identical between these conditions, including candidate log probabilities and saved top-five outputs. The code patches a head at the input of its output projection, after that layer's attention has already mixed tokens. Between this point and the captured block output, the operations are per-position projection, normalization, residual addition and MLP. Consequently, extra replacements at earlier positions cannot affect the captured position in that same block. Both conditions replace the same vector at the captured position.

This equality is expected from the intervention/capture design. It is **not** evidence that earlier writes do not matter or that information does not accumulate across positions. Testing their downstream propagation would require capturing after a later token-mixing layer or a Stage-4 path intervention. Report one local head-replacement effect here; do not count the span condition as independent corroboration.

## Consequence for Stage 4

Prioritize the hypothesis that **L17H1 writes mother-identity-sensitive information at `is`, and more weakly exposes it at the period, which later components may consume**. Test consumers of these actual positions, including potential intermediate components; a reader attending mother-name positions has not thereby been shown to read `is` or the period. Name-sensitive content and routing roles can coexist, so values versus keys remain an experimental question.

Keep **L15H25 at child-last** as a causal communication candidate with unresolved content. The current null readout does not justify dropping it, calling it an address head, or concluding that the site lacks a binding. If content clarification remains essential, first calibrate a revised readout on intact versus full-swap sources and then examine head dependence. Otherwise, proceed to path/group interventions without a broad new readout sweep.

The optional experiment can be reported as complete with one clear positive contrast result and one unresolved content question. It does not require every site to yield a successful decoder.

## Supporting files

- `verification.json`: coverage, hashes, source preservation and byte-level regeneration.
- `code_changes_since_run.diff`: the local analyzer's plan-location change.
- `family_details.csv` and `descriptive_checks.csv`: paired family shifts and their distributions.
- `site_importance_by_family.csv`: same-site original-task effects.
- `absolute_probabilities.csv` and `injection_relevance.csv`: probability-scale controls.
- `review_readout.py`: reproduction and supplementary analysis, writing only to this review directory.
