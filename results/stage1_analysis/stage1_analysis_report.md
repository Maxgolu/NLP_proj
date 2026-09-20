# Stage 1 — RI scan over all 1,024 heads: analysis

Run: `stage1_v2` (OLMo-2-1124-7B @ 7df9a825; 534 prompts = 89 working-set
discovery families × {base, corrupted, reorder} × 2 orders; 4-shot;
Ren et al. Eqs. 1–3 at τ=2.2, embedding-level OV; source anchor = last token,
target = first token primary / last token sensitivity; scan wall time 6 min).

## Preflight (all passed)
P1 runner replication: max drift 0.0. P2 inert hooks: bitwise identical;
attention-capture forward drift 0.039 logits. P3 self-patch: exactly 0.
P4 early-head twin-patch floor: |ΔM| ≤ 0.038 on M≈5.45 — confirms the
structural-zero prediction for early layers.

## Selection under the pre-registered rule
Null = each head's own random-context-token mean (same evaluations); threshold
= 99.9th percentile of null means over the 244 heads with ≥50 scored passes
(expected false positives ≈ 0.24). 637/1024 heads never pass the QK dominance
filter at τ=2.2.

**RI-passing heads: 2** (5 at the 99th percentile):

| head  | freq   | strength(first-tok) | own null | ratio | n   | base/corr/reorder |
|-------|--------|--------------------:|---------:|------:|-----|-------------------|
| L9H22 | 1.0e-4 | 0.374 | 0.086 | 4.3× | 86  | .370/.397/.357 |
| L3H11 | 1.4e-4 | 0.243 | 0.088 | 2.8× | 149 | .242/.242/.246 |

Both are cross-variant stable (signature of real behavior). Both sit in EARLY
layers. Strength distribution over active heads: p50=.018, p99=.154, max=.374
— a thin separated tail.

Last-token target sensitivity reorders the top: L29H14 (.588), L9H22 (.312),
L28H19 (.305), L3H20 (.266)… — target-token policy matters; both variants are
carried forward.

## The headline cross: RI vs the stage-0 causal preview
The stage-0 attribution top-10 (one twin pair; layers 15–28: L21H18 −4.05,
L18H19, L28H11, L24H17, L17H1, L18H18, L21H27, L15H2, L21H25, L18H10) has
**zero overlap** with the RI-passing set. Every attribution-top head scores at
or BELOW its own RI null (e.g., L21H18: strength .0096 vs null .0208).
Conversely, the RI-passing heads live in early layers where twin-patching is
structurally near-zero (preflight P4).

If this holds in stage 2 (exact patching, 178 pairs), the audit verdict trends
toward *misleading + incomplete*. Cautions: the causal side is currently ONE
pair and a first-order estimate; stage 2 decides.

Candidate mechanism for the pattern (itself an audit claim): the definition's
OV stage projects the RAW EMBEDDING through the head (per the paper's
formulation). In early layers the residual ≈ embedding, so the score is
faithful there; late-layer heads operate on transformed representations, so an
embedding-level projection systematically understates them — a built-in
early-layer bias of the definition. Testable: recompute the OV stage on the
layer-input residual instead of the embedding and see whether late-layer
causal heads recover.

## τ transfer finding (pre-registered)
Ren's recipe (95th percentile of dominance ratios) yields τ*_ours = **3.66**;
the published constant 2.2 sits at ≈ the 87th percentile here. The constant
does not transfer across models. Sensitivity rerun at τ=3.66 costs 6 minutes
and is scheduled with stage 2.

## Stage-2 shortlist
Exact-patching candidates: {L9H22, L3H11} (RI-passing) ∪ stage-0 attribution
top-10 ∪ last-token top (L29H14, L28H19, L3H20) ∪ 25 random controls; plus the
full attribution screen over all 178 pairs and exact patching of ALL heads on
a 40-pair subset (≈1.7 h at the measured 0.152 s/patched forward).
