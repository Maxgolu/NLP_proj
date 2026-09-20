# Stage 1: Test-only RI extension — results and analysis

Analysis date: 18 September 2026. Model: OLMo-2-1124-7B. GPU job: 905839.

## 1. Conclusion

The run completed successfully, and the local artifacts pass both the supplied verifier and an independent event-level audit. The extension identifies useful **RI discovery candidates beyond L3H11 and L9H22**, but does not establish semantic heads.

For further investigation of relational RI, I would prioritize **L11H4 (first token), L17H5 (first token), and L9H16 (last token)**, for different reasons and with different limitations. L12H2 and L25H18 are secondary candidates. This is an interpretive prioritization after diagnostics, not a new preregistered selection rule or a claim of statistical significance.

L23H10 has the largest eligible first-token name contrast, but part of its apparent advantage occurs while processing the target name itself. L15H18 provides an especially clear example of this issue. L6H2 is a useful high-RI self-attention comparator. At the answer boundary, L16H21 is interesting primarily for its QK selection of the relevant fact, not as proof that the measured OV score dynamically retrieves the correct mother.

## 2. Completion and numerical integrity

**Question and rationale.** Did we obtain the complete intended calculation, rather than a partial or numerically incompatible result? Slurm completion alone cannot answer this.

- The supplied accounting output reports COMPLETED, 00:47:06, exit code 0:0.
- The local verifier confirms 46,634 events, summaries for all 1,024 heads, and 59 descriptive candidates. There are 340 heads with at least one eligible test QK event; 684 have none under this gate.
- All 46,634 collected events have scores. Shard metadata records 237,685 saved-score replication checks, with maximum absolute discrepancy 0.000195891, within the existing absolute/relative tolerances of 0.0002 and 0.002.
- The independent audit checks event identity and uniqueness, complete coverage, QK dominance, token positions, finite scores, control means, contrasts, ranks and collision ties. It reproduces 8,160 aggregate target/name-gap/word-gap values and their supports across all active heads, both anchors and both fact/position scopes. It separately checks frequency denominators and the empty-head summaries.
- These are consistency checks on the saved calculation, not an independent model forward-pass replication. No new forward passes or held-out experiments were performed.

Loading still took approximately 38 minutes 38 seconds. Completion does not mean that the earlier loading-performance issue has disappeared. None of these integrity checks establishes the semantic interpretation of RI.

## 3. What the extension measures

**Question and rationale.** Does a head that satisfies the original QK gate give the true target a higher RI score than plausible alternative targets in the test? Separate word controls ask the weaker question of whether names score above ordinary template words.

The dataset contains 534 discovery prompts from 89 families, with six variants per family. Only events whose fact and current position are in the test block are included. Demonstration events are excluded, but demonstrations remain in the model input and in the original full-visible-context normalization.

An event is a head, a fact and an eligible current position. The gate requires dominant attention to the annotated source's final token, with ratio greater than 2.2. Failure to pass is **not** a zero RI observation. Actual scored zeros remain zeros. Conditional RI means and QK frequency are separate quantities.

For each passing event we report:

1. The target RI at the first or last token of its annotated name occurrence.
2. The target score minus the mean score of other fully visible fact-tail names in the test. There is no equal-token-length restriction. Shared anchor IDs remain ties, not successes.
3. The analogous contrast against up to three sampled, fully visible non-name word types from the test. These are mostly fixed-template words, not a comprehensive linguistic null.
4. The target's rank, including ties, and the fraction of events where it strictly exceeds every available name control.

Each metric is averaged within family and then across families with eligible observations for that metric. Six variants are not six independent families. QK frequency includes zero-pass families, and its denominator is eligible fact–position opportunities, not generally prompts. At the final position for the query fact, there is exactly one opportunity per prompt, so the denominator there is 534.

The frozen selection is the union of the top ten positive scores for each combination of three metrics, two anchors, two fact scopes and two position scopes. Each ranking requires at least 20 scored events and ten active families. This produces **59 heads; 36 enter at least one name-gap top-ten list**. The other 23 enter through target RI or word contrast. These are not 59 significant discoveries.

All-test includes the final position; the two position summaries overlap. Likewise, the query-fact subset is contained in all-facts. Neither is an independent replication.

## 4. Relational candidates within the test

**Question and rationale.** Which positive contrasts survive a closer look at support, lexical concentration and query relevance? Comparing with other names is more informative than raw RI alone, because it reduces the possibility that a head simply favors names in general.

The following table uses all test facts and all eligible test positions. Counts and all means in this table use the **same name-control-eligible subset**, not the potentially larger raw-target subset. Events count comparisons performed, not wins. Families count families contributing at least one such comparison.

| Head | Anchor | Events | Families | Target RI | Name control | Difference |
|---|---|---:|---:|---:|---:|---:|
| L11H4 | First | 94 | 39 | 0.01466 | 0.00814 | +0.00652 |
| L17H5 | First | 65 | 34 | 0.02288 | 0.01682 | +0.00606 |
| L9H16 | Last | 343 | 87 | 0.02087 | 0.01347 | +0.00740 |
| L12H2 | First | 52 | 32 | 0.02009 | 0.01293 | +0.00716 |
| L25H18 | Last | 46 | 17 | 0.02869 | 0.01562 | +0.01307 |
| L23H10 | First | 60 | 24 | 0.02827 | 0.00677 | +0.02150 |

### L11H4: a relatively well-supported first-token candidate

Its contrast is positive in 22 of 39 active families, and positive separately in base, corrupted and reorder summaries. Removing any single family leaves a positive mean, with worst-case +0.00498. Removing the target token contributing the most to the family-weighted target RI leaves +0.00576; removing the dominant current token leaves +0.00524. Its target anchors span 19 token IDs.

The effect is not merely attention to itself: 78 of 94 comparisons have source distance greater than one, with contrast +0.00728. None occurs inside a mention of the target name. Restricting to the query-relevant fact leaves 53 comparisons across 29 families and +0.00708. The word contrast is also positive.

Limits: its overall QK frequency is only about 0.146% of eligible all-fact test opportunities. The target is strictly above all name controls in only 27.1% of the family-averaged comparisons. The last-token contrast is slightly negative. This is evidence for a selective, anchor-specific RI pattern, not reliable answer retrieval.

### L17H5: query-relevant first-token signal, with template dependence

All 65 comparisons involve nonlocal source attention. There are no current-target-name events. The query-fact subset retains +0.00818 across 32 comparisons and 20 families. The full contrast remains positive after deleting any one family, the dominant target token, or the dominant current token. The target is strictly highest in 51.5% of the family-averaged comparisons.

However, the current token ` mother` supplies about 66.3% of the family-weighted target RI. This is consistent with sensitivity to a fixed syntactic cue and is not evidence by itself for relation-general semantics. The last-token contrast is negative. The family-resampling distribution is broad and includes zero; strong certainty would be unwarranted even before allowing for selection.

### L9H16: broader support for a last-token effect, mostly after reordering

The name contrast is positive in 51 of 87 active families and survives removal of any one family, the dominant target token, or the dominant current token. All 343 comparisons are nonlocal, none processes the target name itself, and 311 occur within a mention of the source name. The query-fact subset retains +0.00694 across 244 comparisons and 76 families.

The central caveat is variant dependence: 299 comparisons are in reorder, versus 21 in base and 23 in corrupted. Their respective contrasts are +0.00721, -0.00013 and +0.01280. Thus broad family coverage does not establish order-independent relational behavior. The first-token contrast is negative. Keep this candidate, but label the observed pattern as last-token and reorder-dominated.

### Secondary candidates

L12H2 has a positive first-token contrast across all three variant summaries and survives the single-family and dominant-token deletions. Excluding current-target-name events leaves 44 comparisons across 30 families and +0.00360. However, the query-fact subset has only seven comparisons across six families, below the discovery support threshold. It is a fact-processing candidate, not currently a supported query-specific candidate.

L25H18 has a strong last-token contrast and retains +0.00840 after excluding current-target-name events. Its support is smaller, and its matched query-fact subset spans only nine families. It is worth retaining as a secondary candidate rather than treating its larger mean as stronger evidence than L11H4's.

## 5. Why some impressive scores are not convincing semantic evidence

**Question and rationale.** Could a score arise from copying, suffix compatibility or a stereotyped position? We examined source distance, the current token's role, and removal of dominant token contributions. These diagnostics test alternative explanations without changing the frozen candidate list.

- **L23H10:** the largest eligible first-token name gap is real as a descriptive calculation. But 31 of 60 comparisons occur while the current token is inside the target name. Removing those leaves +0.00718 across 29 events and 17 families: the effect is reduced, not eliminated. Only four query-fact comparisons remain in the original full subset. Its 13 jointly passing base/corrupted pairs all concern facts whose target did **not** change, so their stability is not evidence of following a reassigned mother. Keep it as a high-RI mechanism-comparison candidate, not the strongest answer-semantic candidate.
- **L15H18:** both anchors have positive name contrasts (+0.00541 and +0.00789), and the pattern is not dominated by a single token. Nevertheless, 104 of 110 comparisons process the target name itself, and only one comparison concerns the query fact. Diversification across tokens does not remove this role-based confound. Six non-target-name events are too few to establish the alternative interpretation.
- **L6H2:** the last-token contrast is +0.00668 across 927 comparisons and all 89 families, and survives the deletion checks. Yet every comparison is self-attention at the source token. This is a strong RI observation, but its geometry differs substantially from nonlocal retrieval. It is a useful comparator for asking what the metric recognizes, not something to dismiss merely for being self-attention.
- **L9H6:** its first-token contrast is +0.01155, but every name comparison uses the same sentence-ending token and attends to the immediately preceding source. The score cannot establish context-dependent target selection by that raw current-token OV vector.

The historical heads remain poorly supported on test-only data. L3H11 has five test events in one family; only three admit a name comparison. L9H22 has 14 test events in four families; only nine admit a name comparison, with contrast approximately +0.000018. Neither meets the new support requirements. This does not prove they are unimportant for other model functions.

## 6. The answer boundary: fact selection versus target promotion

**Question and rationale.** Does the pattern occur at the position where the model is about to answer, and does it concern the fact needed for that answer? This prevents earlier fact processing from being conflated with answer selection.

There are 744 test QK events at the final prompt position, of which 718 concern the query fact. These are head–prompt events and must not be read as 744 distinct prompts. The table below concerns only the query fact at that position.

| Head | Events | Families | QK frequency | First-token name gap | Last-token name gap |
|---|---:|---:|---:|---:|---:|
| L16H21 | 222 | 77 | 41.6% | -0.00042 | +0.00335 |
| L16H1 | 223 | 76 | 41.8% | -0.00250 | -0.00342 |
| L16H24 | 68 | 35 | 12.7% | -0.00283 | +0.00359 |
| L16H4 | 63 | 32 | 11.8% | +0.00042 | -0.00715 |
| L16H31 | 23 | 14 | 4.3% | -0.00411 | +0.00383 |

L16H21 is interesting for fact selection: 222 of its 230 final test events select the query fact. Its positive last-token mean is not driven by a single family, but approximately 55.9% of its target RI comes from the suffix `la`. Removing that target token changes the name gap to -0.00641. For L16H24, `na` contributes approximately 71.9%, and removing it changes the gap to -0.01317. The small first-token advantage of L16H4 becomes negative under one-family deletion. L16H31's last-token advantage also fails that deletion check.

### A structural limitation, directly checked in the results

The measured OV distribution uses the **raw embedding of the current token** through the head's value/output matrices and unembedding. It is not the actual context-dependent head output. At the final prompt position, the current token is always `:`.

For a fixed head, swapping the mothers while retaining the same set of visible token IDs therefore leaves this measured score for each candidate token unchanged. Reassigning the correct answer changes which fixed score we call the target; it does not cause the raw OV vector to learn the new assignment. Normalization can vary between unrelated prompts, but is identical in these matched swaps.

For L16H21, 49 base/corrupted query pairs pass QK in both versions. In every pair, the current token and visible token-ID set are identical. Comparing the old correct name specifically with the new correct name gives exactly opposite contrasts in the two versions, at both anchors. There are zero strict correct-target wins in both versions of a pair. The same algebraic pattern appears for the other supported final-position heads.

This is **not a newly introduced numerical bug, nor evidence that the model cannot follow the swap**. It limits what this particular raw-OV measurement can demonstrate. QK can still change with context, and actual contextual head outputs may behave differently. Consequently, final-position QK selectivity and raw-OV semantic specificity must not be equated.

## 7. Paired variants and robustness

**Question and rationale.** Does a pattern follow the same source when facts are reordered or its mother is changed? Pairing by family, source and order avoids mistaking unrelated positive pooled means for consistency.

The evidence is incomplete. For example, L11H4 has 21 jointly passing base/corrupted source–order pairs across 18 families; all 21 change the target. Only five of those families have positive family-mean name gaps in both versions. Under reorder, only five pairs pass in both versions. L17H5 has 17 jointly passing corruption pairs but no jointly passing reorder pairs. This means paired order robustness is **not established**, not that a missing QK observation has negative RI.

L9H16's reorder concentration similarly prevents its broad family coverage from being interpreted as broad paired stability. Moreover, all-test pair means can involve different current positions and different visible controls. Only the final-position pairing aligns the answer position by construction.

Leave-one-family-out results diagnose single-family dominance, not significance. Removing a dominant token recomputes means over the remaining active families and can change the evaluated population. The accompanying audit also includes 5,000 family-bootstrap resamples as descriptive sensitivity distributions. They are not selection-adjusted confidence intervals or held-out validation. In particular, L17H5's range includes zero. No multiplicity-corrected discovery claim is made for any head.

## 8. Interpretation and handoff

The extension succeeds at its intended discovery task: it exposes candidates and failure modes hidden by the original pooled demonstration-heavy selection.

The useful separation is:

- **Relational-RI investigation:** L11H4, L17H5 and L9H16, with their anchor and position/variant restrictions preserved. L12H2 and L25H18 are secondary.
- **High-RI alternative-mechanism comparators:** L23H10 and L15H18 for current-target-name overlap; L6H2 for self-attention; the historical L3H11 and L9H22 as references.
- **Answer-fact-selection investigation:** L16H21, with L16H1 as a useful contrast showing strong QK selection without positive name-specific RI. These are not interchangeable with the first group.

This is a stronger and more informative candidate set than the historical pair alone. It is **not yet a validated list of semantic heads**. The remaining uncertainties concern held-out generalization, lexical/positional alternatives and actual contextual or causal contributions. Those belong to validation and later methods, not to a stronger interpretation of the present discovery means. No additional GPU run or intervention was started during this analysis.

## Reproducibility

Source artifacts: `results/ri_test_v2/`, especially the frozen manifest and policy, per-head event shards, head summaries, paired variants and replication metadata. Source results were not modified.

Integrity verification: `pilot_v2/verify_ri_test_results.py`.

Independent calculations and post-selection diagnostics: `pilot_v2/analyze_ri_test_results.py`, with machine-readable output in `results/ri_test_v2_analysis/post_selection_audit.json`. The audit includes diagnostics for all 59 selected heads and both historical heads, not just the examples discussed here. All head coordinates are zero-based, as in the source files.
