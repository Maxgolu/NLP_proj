# S4.5 discovery run — analysis (27 September 2026)

Run: `s45_discovery_v2` (Slurm 937878, s-005, 6× RTX 2080 Ti; means reused from `s45_discovery_v1`, job 933563).
Protocol: `חומר כתוב/RI_Structure_Final_Protocol_20260926.md`. All numbers below come from
`results/stage4_s45_discovery_v2/analysis/**` and `runs/gate_r0/gate.json`. Discovery only: 20 common
families (40 pairs) for Stages A–C, 89 families (178 pairs) for the behavioural extension. The 87
held-out families remain sealed; nothing here is validated.

## 0. Gates

All six gate pairs (families 0, 16, 168; 168 has a shared answer prefix) passed with exact zeros where
zeros are required: saved baselines reproduced (error 0.000, no compatibility exception needed on this
GPU), inert instrumentation, SDPA reconstruction, prefix invariance under a question change (0.000),
all-live background ≡ full model (0.000), self-donor identity in every background type (0.000),
no-intermediate cross-position Q structurally zero (0.000), mean-clamped control receiver exactly null.
The empty mask replaces head-output slices with total norm ≈ 105–110 per prompt and changes the margin
by 0.4–10 logits. Full-background anchors on the 20 core families reproduce the historical means:
T1 +1.664 (hist. +1.635), T2 +0.179 (+0.172), T3 +0.859 (+0.876), T4 +0.477 (+0.465), T5 −0.210
(−0.236); all within 0.05 (T5: 0.026). fp16 arithmetic on this hardware was deterministic
(repeat drift 0.000 in the diagnostic run 935537).

## 1. Stage A — no retained set is a faithful mechanism; B = C50, status PARTIAL

Fact-axis gap g = mean over orders of M(x00) − M(x10), core families (full model 11.20, CI [10.36, 12.06]):

| mask | g | F | L | b (interaction) | q0 = M00−M01 | candidate acc x00 (o0/o1) | acc x01 (o0/o1) |
|---|---:|---:|---:|---:|---:|---|---|
| full | 11.20 | 1.00 | 0.00 | 5.56 | 10.9 | 1.00/1.00 | 0.90/1.00 |
| empty (all 1,024 heads mean-replaced) | 0.00 | 0.00 | 1.00 | 0.00 | 0.0 | 0.45/0.45 | 0.55/0.55 |
| T1 | −0.21 | −0.02 | 1.02 | −0.03 | 0.0 | 0.85/0.25 | 0.15/0.75 |
| T2, T4, T5 | ≈0 | ≈0 | ≈1 | ≈0 | ≈0 | 0.45/0.45 | 0.55/0.55 |
| T3 | −0.19 | −0.02 | 1.02 | −0.02 | 0.0 | 0.75/0.25 | 0.20/0.75 |
| C33 | 0.58 | 0.05 | 0.95 | 0.44 | 0.9 | 0.90/0.20 | 0.10/0.80 |
| C50 | 1.28 | 0.11 | 0.89 | 0.75 | 1.5 | 0.90/0.20 | 0.10/0.80 |

Every mask fails F, L and the accuracy guard in every cell × order. Per the protocol B = C50 as a
predeclared partial background with no sufficient-mechanism claim. The empty mask is a true zero
(g = 0, constant output, top token is never a candidate name), so the mean replacement is a valid
baseline and the C33/C50 numbers are real but small: 5–11 % of the fact gap, 8–14 % of the four-cell
interaction b.

The order-resolved numbers show what the retained heads do instead of the task. Under C50
(20 families):

| | M00 | M10 | M01 | M11 |
|---|---:|---:|---:|---:|
| order 0 (queried chain listed first) | +6.4 (18/20 > 0) | −4.1 (16/20 < 0) | +5.1 (only 2/20 < 0) | −2.4 (4/20 > 0) |
| order 1 (queried chain listed second) | −3.0 (4/20 > 0) | +5.0 (2/20 < 0) | −4.7 (16/20 < 0) | +6.3 (18/20 > 0) |

The sign of the margin is fixed by the position of the mothers, not by the question: with the
strong heads live and everything else at its role mean, the model outputs the mother stated in the
first chain (a primacy copy). The query change moves the margin by ~1.3–1.7 logits (full model:
10.5–11.3), i.e. the question-conditioned selection is almost entirely outside C50. The full model
itself has a milder primacy asymmetry (g 13.9 vs 8.5 by order; M01 −3.2 vs −7.0), so the retained
set exaggerates a bias that the rest of the network normally corrects.

Caveat that applies to every "mean" number below: replacing 974 heads by slot/role means keeps
positional structure (means are per slot and bucket) while erasing name identity, so the primacy
behaviour is a property of the retained heads *in that background*, not a claim about the full model.
The paired-donor baseline (§5) gives a different, order-symmetric picture.

## 2. Stage B — functional participation in B (four-cell panel, 20 families)

d_b = b_f(B_h+) − b_f(B_h−); coherent rule |mean| ≥ 0.1 with ≥ 70 % sign agreement.

| head | in B | d_b | 20-family CI | sign agreement | order 0 / order 1 | class |
|---|---|---:|---|---:|---|---|
| **L9H16** (not in RI31) | no | **+0.601** | [0.52, 0.69] | 20/20 | +0.64 / +0.57 | coherent |
| **L1H27** | yes | **+0.260** | [0.22, 0.31] | 20/20 | +0.26 / +0.26 | coherent |
| L11H4 | yes | +0.004 | [0.00, 0.01] | 17/20 | | below rule |
| L17H5 | yes | −0.009 | [−0.02, 0.00] | 15/20 | | below rule |
| L23H10 | yes | +0.004 | [0.00, 0.01] | 13/20 | | below rule |
| L25H18 | yes | +0.000 | [0.00, 0.00] | 13/20 | | below rule |
| L18H19 (reference) | yes | +0.247 | [0.20, 0.30] | 20/20 | | coherent |
| L8H15 (reference) | yes | +0.133 | [0.10, 0.16] | 20/20 | | coherent |
| R(B) = RI31 collectively | — | +0.759 | [0.64, 0.88] | 20/20 | +0.71 / +0.81 | coherent |

Gold-oriented single-cell differences agree (L9H16: +0.55 to +0.67 in all four cells, 20/20; L1H27:
+0.22 to +0.29, 17–19/20). No candidate changes any condition accuracy (the primacy pattern is
unchanged); ΔF is +0.11 for L9H16 and +0.05 for L1H27. Removing the 31 RI heads from B removes the
whole fact-axis signal under means (g +1.28 → −0.13; b 0.75 → −0.01): what C50 carries on the fact
axis is carried by its RI members, and B−R(B) = the 19 non-RI strong heads is inert on this axis.

Scale: L9H16's +0.60 is 11 % of the full-model interaction (5.56) and 80 % of the interaction that B
itself achieves (0.75); it is the largest single-head contribution measured in this background,
larger than the reference readers.

## 3. Stage C — routes survive at half strength in B; no candidate modulates them

Route noising effects (family mean, 20 families) in B versus the full background:

| anchor | full | in B | fraction | B order 0 / order 1 | class in B |
|---|---:|---:|---:|---|---|
| T1 L17H1 → {L18H18,L18H19} → L27H6 Q | +1.66 | +0.62 | 37 % | +1.06 / +0.18 | coherent (20/20) |
| T2 L15H25 child-last → {L16H1,L16H21} → L18H18 Q | +0.18 | +0.04 | 23 % | +0.05 / +0.03 | below rule |
| T3 L17H1 → {L18H18} → L27H6 Q | +0.86 | +0.39 | 46 % | +0.66 / +0.13 | coherent (20/20) |
| T4 L8H15 query_sentence → L18H18 V | +0.48 | +0.26 | 53 % | +0.38 / +0.13 | coherent (19/20) |
| T5 L20H1 colon → L27H6 Q | −0.21 | −0.02 | 10 % | −0.03 / −0.01 | below rule |

The three L17H1/L8H15 → L18H18/L27H6 communications are real inside the 50-head background but
retain only 37–53 % of their efficacy, and mostly in order 0; T2 and T5 fall below the rule. The
efficacy of every measured route therefore depends on the mean-replaced remainder (amplification or
context provided by heads outside C50).

Gamma matrix (6 candidates × 5 anchors, 30 pairs; 18 eligible where the route is retained in at
least one state): **no Γ reaches the rule.** Largest: L9H16×T1 +0.030 (CI [0.016, 0.046], 17/20),
L9H16×T5 −0.013, L25H18×T1 +0.012, L17H5×T1 −0.010; all other |Γ| < 0.01. Relative to the routes
themselves these are ≤ 5 %. No route pair was selected; two functional-only records (L9H16, L1H27)
were frozen, so Stage D (restoration Γ, attachments, controls) was not needed and did not run.

## 4. Full discovery (89 families, original axis, means)

full g 11.17 [10.62, 11.72]; empty 0.02; B 1.34 [1.13, 1.57] (F 0.12; order 0 +9.9 / order 1 −7.3;
top-token accuracy 0.87/0.02); B−R(B) −0.11 [−0.27, 0.05]; B+L9H16 2.55 [2.27, 2.86] (F 0.23);
B−L1H27 0.79 [0.59, 1.00]. The core-family picture generalises to all 89 families with the same
primacy pattern (candidate accuracy 0.90/0.16–0.18 by order for B, B±h alike).

## 5. Baseline sensitivity — the candidates' contributions disappear under paired-donor replacement

With the non-retained heads carrying the *swapped-fact* twin's outputs instead of means (89 families):

| configuration | donor g | donor F | (means g) |
|---|---:|---:|---:|
| B | +2.27 [1.72, 2.89] | 0.20 | 1.34 |
| B − R(B) | −2.34 [−3.04, −1.56] | −0.21 | −0.11 |
| B + L9H16 | +2.28 | 0.20 | 2.55 |
| B − L1H27 | +2.27 | 0.20 | 0.79 |

Three things change. (i) Against a swapped-world background the retained heads recover 20 % of the gap
symmetrically across orders (+3.1 / +1.4; candidate accuracy 0.76/0.67) — no primacy flip; the flip in
§1 is a mean-background phenomenon. (ii) The RI31 members of B are what holds the correct-fact signal
against the swapped background: without them the 19 non-RI heads follow the donor (−2.3). (iii)
**L9H16 and L1H27 contribute nothing under the donor baseline** (ΔF = 0.000 and −0.001) although they
contribute +1.2 and +0.55 logits under means. Their functional participation is therefore
background-conditional: they matter when the rest of the early network is silent, not when it carries
competing (swapped) name information. Under the protocol's final labels this is the signature of
category (e), "order/baseline sensitive", regardless of what the held-out replication shows for the
mean configurations.

## 6. What this says about RI and the semantic structures

1. The heads that RI ranks highest by name gap at the first anchor (L11H4, L17H5, L23H10, L25H18) do
   nothing in this scope: |d_b| < 0.015 (and < 0.04 in every single cell), |Γ| < 0.012 on every
   measured route, in a background where every strong causal head is live. After S4.2's weakened
   background this is a second, independent background in which they are inert. Within the tested
   scope they are not connected to the measured structures, functionally or by communication.
2. The two candidates with functional effects (L9H16, L1H27) are not route members: Γ ≤ 0.03. Their
   effect is additive/parallel and vanishes under the donor baseline, i.e. it is a redundant carrier
   unmasked by mean replacement, not a structural role. L9H16 is not even in RI31.
3. RI heads as a collective inside B carry the fact-axis signal (B−R inert under means, negative under
   donor) — but the RI heads that matter here are the ones that were already strong in the causal map
   (L18H19, L8H15, L16H1/L16H21, L23H15, …), i.e. RI is confirmed where it overlaps the causal map and
   adds nothing where it does not.
4. Independently of RI: the 50 heads that contain every strong causal head and every RI candidate
   are not a mechanism for the task (F 0.11–0.20). They implement fact copying (routes at half strength)
   but not question-conditioned selection (query contrast 1.5 of 10.9); in the mean background they
   default to a primacy copy. The selection step is distributed over the mean-replaced remainder.

## 7. Recommendation for the held-out suite

Run it. The frozen suite is behavioural only (7 mean configurations × 4 cells, 5 donor configurations,
174 pairs; ≈ 6.6k forwards, ~2 h incl. loading), no Γ or attachments. It validates: B's partial
status and the primacy pattern, the collective RI31 contribution, and L9H16/L1H27's d_b under means,
with the donor null as the baseline-sensitivity check. The null Stage-C matrix and the inert
candidates remain discovery-only by construction and must be described as such.

Known limitations to state: single task, single model, mean/donor replacement of 974 heads at test
positions is an out-of-distribution background; 20-family core for Γ; descriptive rule and bootstrap
intervals, no selection-adjusted inference.
