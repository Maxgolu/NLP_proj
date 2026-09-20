# Stage 1: independent review

Scope: read-only inspection of `results/stage1_results.tar.gz`, current `pilot_v2/stage1_scan.py`, the prior stage-0 code, single-hop data and baseline archive, and the supplied conversation. No stage-2 code or model runs were produced. Numbers below were recomputed from the archived CSV; the archive does not contain hashes of the executed script or dataset, so exact execution provenance cannot be independently established.

## Reproduced observations

- Config: OLMo-2-1124-7B revision `7df9a82518afdecae4e8c026b27adccc8c1f0032`, 4 shots, tau 2.2, 534 prompts, 358.8 seconds.
- 3,072 head/variant rows: 1,024 heads times three variants. 637 heads have no QK passes. 244 have at least 50 scored passes.
- Weighted aggregation uses `n_a` for strength and sums QK passes / evaluations for frequency.
- The 99.9th percentile of eligible heads' random-null means is approximately 0.182739. Exactly two first-target-token strengths exceed this threshold:

| Head | Scored passes | Conditional strength | Own random-null mean | QK pass frequency |
|---|---:|---:|---:|---:|
| L9H22 | 86 | 0.373918 | 0.086091 | 0.000055817 |
| L3H11 | 149 | 0.243067 | 0.088241 | 0.000096706 |

These frequencies are lower than the approximate values in the supplied report. They refer to eligible triple-position evaluations, not to the fraction of prompts solved or attended correctly. Conditional strength is not task accuracy.

Last-target-token ranking is different: L29H14 has strength 0.587577 with 56 passes; L9H22 0.311916; L28H19 0.305236; L3H20 0.265764. This is a substantial token-policy sensitivity.

## Improvements since stage 0

- Runner replication now scores each complete candidate under its own continuation. Archived maximum drift is exactly zero on six prompts.
- Inert hooks preserve the checked output exactly; self-patching one head on one prompt changes the metric by zero.
- Attention-capture forward maximum answer-position vocabulary-logit drift is 0.0390625 on the checked prompt. The code checks availability, not a numerical tolerance on this drift.
- Interventions use the bare prompt rather than adding the answer continuation. They still patch all prompt positions.
- The intervention metric is now the difference between the first continuation tokens. This is distinct from whole-name candidate likelihood.
- P1-P3 failures abort before scanning; P4 remains descriptive.
- OV projection uses raw embeddings at the current position j, unlike the earlier costing proxy. This follows the explicit embedding-level x_j formula in the paper text inspected. It does not measure the head's actual contextual output.

The six default preflight rows come from the first selected family, not six independent families or the 20 prompts specified in the methodology.

## Conclusions that are not yet supported

### 1. The null threshold is not a calibrated false-positive guarantee

Taking a percentile across 244 heterogeneous heads' null means does not establish a 0.001 false-positive probability for each observed strength. Heads differ in pass count, positions and score distribution. One random draw per event is not a distribution of repeated head-level null means. Consequently, 244 * 0.001 = 0.244 is not a justified expected false-positive count here.

For each event, the normalized shares sum to one over K visible unique context tokens. A uniformly sampled null target therefore has conditional expectation 1/K. Different position distributions can explain different null means. Random punctuation and common subwords are also not necessarily appropriate controls for an entity's initial token.

Next analysis should distinguish the published fake-target control from a new, explicitly documented entity-matched null, use repeated null assignments and family-level uncertainty, and address multiple comparisons. The archived null pool is insufficient for exact reconstruction: at most 200 event values per head are saved with a non-uniform replacement scheme, without family IDs.

### 2. Stable variant averages do not establish stable test-question behavior

All demonstration and test triples are pooled. The four demonstrations contribute 16 of the 20 facts, and causal ordering gives earlier facts many more later positions. Demonstrations are identical across a family's base/corrupted/reorder variants. Prefix computations before the changed test block are identical by causality.

Thus stable scores can reflect shared demonstration activity. The CSV does not distinguish demonstration/test blocks, families, source/target names, or current positions. We cannot determine where the 86 or 149 passes occurred or how many independent families contributed.

Save per-family and per-block summaries plus the scored events of selected heads; separate true-target from distractor promotion and inspect the token identities. Do not interpret a high first-subword score as recognition of the complete entity.

### 3. Stage-0 attribution is not a valid causal counterpart to the current RI scan

The supplied top-ten list came from one pair with the old multi-token metric under the gold continuation, including patches to continuation positions. Stage 1 changed both the metric and the input scope. Zero overlap is a descriptive comparison of incompatible diagnostics, not evidence that RI misses causally important heads.

Recompute a causal preview under a fixed valid metric and patch scope before classifying heads. Retain old candidates as historical exploratory candidates only.

### 4. P4 does not prove structural zero or test the selected heads

P4 tests eight heads from layers 0-3 on one pair. It tests neither L3H11 nor L9H22. Small observed effects, at most 0.03784, do not imply that all early heads are inactive. The intervention replaces all prompt positions, so early information can in principle affect later computation. A structural-zero claim would require a specific causal-path argument for the exact intervention.

### 5. The dominance percentile is sampled narrowly

The saved sample has p95 = 3.65812. The code samples only layers 0, 8, 16 and 24, takes the first 200 eligible ratios in a slice, and stops near 200,000 values. It is not a representative all-head/all-position distribution. The archive also does not preserve the sample needed to independently recover the claimed percentile rank of 2.2.

The paper's appendix examines all heads and eligible current positions and notes that fewer than 5% of ratios exceed 2.2. Keep 2.2 for the reference scan; a representative, explicitly stratified sample is needed before labeling 3.66 the model's transferred threshold. Even then, differences combine model, data and tokenization effects.

### 6. The first-token intervention metric has collisions

In the 4-shot single-hop data, 30 of 1,200 rows have identical first continuation tokens for the candidate names. Two eligible discovery families are affected:

- Family 168: Vuldna / Velnla, both start with token 650.
- Family 174: Gindra / Gerdna, both start with token 480.

Their first-token difference is identically zero, regardless of model behavior or intervention. This affects two of the planned 89 discovery families (four of 178 order-specific base/corrupted pairs). It does not invalidate their whole-name behavioral scores or the observational RI scan.

Use a documented first-diverging-token contrast under the shared candidate prefix, or full-sequence candidate scores. Specify patch positions separately; do not silently regenerate only inconvenient names or count identically zero metrics as negative causal findings.

## Interpretation and next priorities

The scan establishes a small, sparse tail of high conditional embedding-level target scores under one token policy. It does not yet establish relation-specific heads, calibrated significance, or a failure of the SIH definition.

Before a full causal study:

1. Fix the intervention metric specification and collision handling; repeat the preview on several discovery families and the same patch scope.
2. Recover block/family/event provenance for RI, inspect the top first- and last-token heads, and compute test-only summaries without losing the separate demonstration analysis.
3. Calibrate nulls and dominance sampling, retaining tau 2.2 as the reference and documenting deviations from the original methodology (target-anchor policy, minimum pass count, null aggregation).
4. Test the actual selected heads with exact patching plus matched controls. Only then interpret an RI-versus-effect comparison. Single-head inactivity is not proof of irrelevance without redundancy tests.
5. Treat contextual OV/residual variants as additional diagnostics. OLMo's normalization and actual value inputs must be accounted for; raw embeddings are not automatically faithful in early layers, and residual substitution is not an exact reproduction of RI.

The existing file cannot support family confidence intervals or demo/test separation retrospectively. A targeted observational rerun with better logging is justified. No such run was initiated in this review.
