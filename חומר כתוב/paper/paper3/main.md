# Introduction {#sec:intro}

Attention and weight patterns suggest functions for attention heads, but
do not establish their causal contribution. We test whether the relation
index (RI) used to identify semantic induction heads [@ren2024semantic]
predicts head importance in OLMo-2-7B.

On controlled single-hop relation prompts, we score all 1,024 heads and
measure their effects with activation, path, and group interventions.
Public training checkpoints provide a developmental comparison. RI
misses major causal contributors and selects many weak heads, although
some selected heads participate in writer--reader paths or contribute
collectively. A retained 50-head set fails to preserve the task, and
early routing changes do not identify mature high-RI heads. We examine
these discrepancies in the sections that follow.

# Background and Related Work {#sec:background}

Two lines of work assign functions to attention heads. The first
*describes* a head by its attention pattern and weights. The second
*intervenes* on the head and measures what changes in the output.
Semantic induction heads were defined in the first way; we audit them
with the tools of the second.

## Induction heads and semantic induction heads {#sec:sih}

#### Induction heads.

An attention head can be decomposed into a QK circuit that selects where
to attend and an OV circuit that determines what to write
[@elhage2021mathematical]. On a repeated-token sequence, an *induction
head* attends from the second occurrence of $[A]$ to the token $[B]$
that followed its first occurrence and raises the logit of $[B]$. Its
importance for in-context learning was established by ablation
[@olsson2022induction]; Appendix [8](#app:ih){reference-type="ref"
reference="app:ih"} covers the broader literature, including retrieval
heads [@wu2024retrieval].

#### Semantic induction heads.

@ren2024semantic generalize this template by allowing the attended and
written tokens to differ. For a relation triplet $(t_s,r,t_o)$ in
context, a *semantic induction head* (SIH) attends from a later position
$t_j$ to the head token $t_s$ and raises the logit of the tail token
$t_o$ (Figure [1](#fig:ri){reference-type="ref" reference="fig:ri"}).
The *relation index* (RI) scores this behavior using $$\begin{equation}
\label{eq:ri}
\mathrm{RI}_h
=\operatorname*{mean}_{(T,j):\,h\text{ attends to }t_s}
\frac{q^{h,j}_{t_o}}{\sum_{k\le j}q^{h,j}_{t_k}},
\end{equation}$$ where $p^{h,j}=\mathrm{softmax}(x_jW^h_{OV}W_U)$,
$q^{h,j}_t=\max(0,p^{h,j}_t-\bar p^{h,j})$, and $x_j$ is the current
token's raw input embedding. The mean $\bar p^{h,j}$ and the denominator
use distinct visible tokens. Thus RI measures the share of positive
centred projection mass assigned to $t_o$, averaged over positions whose
attention argmax selects $t_s$ by a fixed margin. Heads with high
relation-specific RI are called SIHs. The original study also reports
that some indices rise when few-shot pattern discovery emerges during
training, interpreting this co-timing as evidence that SIHs support
in-context learning. Appendix [7](#app:ri){reference-type="ref"
reference="app:ri"} gives the exact definition and selection procedure.

<figure id="fig:ri" data-latex-placement="t">

<figcaption>The relation-index template. At a later position <span
class="math inline"><em>t</em><sub><em>j</em></sub></span>, a qualifying
head attends to the relation’s head token <span
class="math inline"><em>t</em><sub><em>s</em></sub></span>, while its
raw-embedding OV projection assigns mass to the tail token <span
class="math inline"><em>t</em><sub><em>o</em></sub></span>. The
projection is computed at <span
class="math inline"><em>t</em><sub><em>j</em></sub></span>, not from the
attended position.</figcaption>
</figure>

#### What the original study does not establish.

The relation index of @ren2024semantic is descriptive: a high value does
not establish that a head affects relational behavior, and an important
head need not score highly. Its projection uses the current token's raw
embedding rather than the head's contextual output, and the study does
not test matched alternative entities or a calibrated null. The authors
also report co-timing of RI and few-shot performance, which establishes
neither causal responsibility nor continuity between early and mature
high-RI heads. Appendix [7](#app:ri){reference-type="ref"
reference="app:ri"} details these limitations of the original
formulation.

## Causal interventions on heads {#sec:rw-patching}

*Activation patching* replaces the activation of one component on a
clean input with its activation on a minimally different corrupted input
and measures the change in the model's output
[@wang2023ioi; @heimersheim2024patching]. Run in the *noising* direction
(corrupted activation into the clean run) it measures how much of the
output depends on the component; run in the *denoising* direction (clean
activation into the corrupted run) it measures how much the component
alone restores. @heimersheim2024patching note that the two are not
symmetric and recommend reporting both. The choice of metric matters as
much as the intervention: @zhang2024bestpractices show that logit
difference between the correct and the corrupted answer is more
informative and less brittle than probability or accuracy, and that
conclusions about which components matter can flip between metrics. The
indirect-object-identification (IOI) study of @wang2023ioi established
the per-head, per-position form of these tools on GPT-2 small and
introduced the vocabulary that later work inherited: *name movers* that
copy the answer, *negative name movers* that suppress it, and *backup*
heads that activate only when others are removed. We use noising and
denoising head patching with the logit-difference metric throughout
(§[3](#sec:setting){reference-type="ref" reference="sec:setting"}), at
all prompt positions and at the answer position separately, and we split
a head's patched output into the attention pattern and the values it
reads, as in the QK/OV decomposition of
§[2.1](#sec:sih){reference-type="ref" reference="sec:sih"}.

## Reading what a head writes {#sec:rw-readout}

Intervention tells us that a head matters; a second family of tools asks
what it contributes. *Direct logit attribution* projects a head's output
vector onto the unembedding to see which tokens it promotes
[@elhage2021mathematical; @wang2023ioi]; the relation index of
§[2.1](#sec:sih){reference-type="ref" reference="sec:sih"} is a
weight-only version of this projection that uses the current token's
embedding instead of the head's actual output. Behavioural
*fingerprints* test a head on controlled synthetic inputs:
prefix-matching and copying scores on repeated random sequences
[@olsson2022induction] and needle-retrieval scores on key--value
contexts [@wu2024retrieval] identify induction and retrieval heads
independently of any natural-language task. Finally, *Patchscopes*
[@ghandeharioun2024patchscopes] decode a hidden representation by
injecting it into a separate prompt whose continuation exposes its
content; the method reads representations, not head contributions, so
head attribution requires comparing intact and head-intervened
representations under the same readout. We use all three: the contextual
output projection and its raw-embedding counterpart side by side, both
synthetic fingerprints on every head in our inventory, and a controlled
Patchscopes-style readout of the sites where our strongest head writes.

## From heads to circuits {#sec:rw-circuits}

A list of important heads is not a mechanism. *Path patching*
[@wang2023ioi] isolates a single edge: the output of a source component
is patched only along its path into a chosen receiver (for a head, its
query, key or value input), so an effect on the output shows that the
receiver uses what the source wrote. @conmy2023acdc automate this
search, pruning edges whose removal does not change a task metric, and
@haklay2025position show that edges must be indexed by token position:
the same head can be essential at one position and inert at another, and
position-agnostic circuits miss this. How a discovered circuit is judged
also matters. @hanna2024faithfulness argue that overlap with a
hand-found reference is the wrong target and that a circuit should be
evaluated by *faithfulness*: how much of the full model's behaviour it
reproduces when everything outside it is ablated, checked with several
ablation schemes and metrics. Our Stage-4 design follows these three
points: edges are tested by path patching into keys and values at the
source's fact positions, routes are indexed by position, and the
resulting head mechanism is evaluated by faithfulness on families held
out from every earlier stage.

## The gap {#sec:rw-gap}

@ren2024semantic identify SIHs through an attention-and-weight
signature, whereas intervention-based circuit analyses test components
through their effects on behavior [@wang2023ioi; @conmy2023acdc]. The
original SIH study did not compare RI with head-level interventions on
the same heads. We compare RI with patching-based importance in
OLMo-2-7B, trace how selected heads participate in relation retrieval,
and test whether early high-RI or routing changes involve the heads that
score highly later.

# Experimental Setting {#sec:setting}

#### Model.

All experiments use the base OLMo-2-1124-7B [@olmo2024olmo2] (32 layers,
32 heads per layer, $d_{\text{model}}{=}4096$), revision `7df9a82`,
without fine-tuning, in fp16. We chose OLMo-2 because its training
checkpoints are public, which the developmental analysis needs, and
because at 7B it is the largest model on which our per-head,
per-position interventions were affordable on the available GPUs
(Appendix [9](#app:compute){reference-type="ref"
reference="app:compute"}).

#### Task.

The model must retrieve a relation stated in its context. A prompt
contains four in-context demonstrations followed by a test block of four
facts of the form "$\langle$mother$\rangle$ is the mother of
$\langle$child$\rangle$." and the question "Question: Who is the mother
of $\langle$child$\rangle$? Answer:". The answer is the mother named in
the *query fact*, the fact whose child is asked about. All names are
invented, so the answer cannot come from parametric knowledge, and the
test block always contains a second, structurally identical fact whose
mother is the *distractor*. This is the SIH setting of
§[2.1](#sec:sih){reference-type="ref" reference="sec:sih"}---a head
token (the child), a relation (mother-of) and a tail token (the
mother)---with the relation annotated at construction time rather than
by a parser.

#### Data.

We generated 200 *families* of invented names. Each family yields a
*clean* prompt, a *corrupted* twin in which the two answer-side mothers
are swapped inside the fact lines (so the distractor becomes the answer
while every token position and mention count is unchanged, as activation
patching requires), and a *reorder* variant with shuffled fact lines;
each in two fact orders. Appendix [10](#app:data){reference-type="ref"
reference="app:data"} shows a complete example. Before any intervention
we verified that the model performs the task: 4-shot accuracy on clean,
corrupted and reordered prompts, in both orders, is 93%, 94% and 89% of
families. The 176 families correct on clean and corrupted prompts in
both orders are split by family id: 89 *discovery* families (178
clean/corrupted pairs) for all scoring and selection, and 87 *held-out*
families used only for the final frozen behavioural and
replacement-sensitivity evaluation. A fixed subset of 20 discovery
families (40 pairs) is used where a measurement must cover all 1,024
heads.

#### Metric and intervention scopes.

Following @zhang2024bestpractices, the outcome of every intervention is
the logit difference between the clean answer $y_c$ and the corrupted
answer $y_r$, $$\begin{equation}
\label{eq:metric}
M(x)=\operatorname{logit}(y_c\mid x)-\operatorname{logit}(y_r\mid x),
\end{equation}$$ evaluated at the first token at which the two answers
differ (a shared prefix, when one exists, is teacher-forced). The sign
is fixed to the clean answer on both inputs, so $M(x_c)>0$ and
$M(x_r)<0$ mean that the model follows the stated fact. The *importance*
of head $h$ is the drop in $M$ on the clean prompt when the head's
output is replaced by its output on the corrupted twin,
$I_h=M(x_c)-M(x_c;a_h
\leftarrow a^r_h)$ (noising); the *recovery* $J_h$ is the rise in $M$ on
the corrupted prompt when the clean output is inserted (denoising). Two
scopes are used throughout: *Scope P* replaces the head at all original
prompt positions, *Scope F* only at the final colon of "Answer:".
Effects are averaged over the two orders within a family and then over
families; the family is the unit of analysis, and we report signed means
alongside mean absolute family effects, since large effects of opposite
sign can cancel.

#### Developmental setting.

The training-time analysis (§[4.5](#sec:method-dev){reference-type="ref"
reference="sec:method-dev"}) uses 17 public checkpoints of the same
model's first pre-training stage, densely sampled between steps 600 and
1,000 (about 3--5B tokens). In-context learning is measured, as in
@ren2024semantic, by few-shot accuracy on four synthetic classification
tasks at 0--20 shots with 50 frozen prompts per condition; the relation
index is computed for all 1,024 heads on a balanced set of 700
AGENDA [@koncelkedziorski2019text] relation triplets

#### Reproducibility.

Every run is driven by a frozen plan file with content hashes of the
model revision, data and code; each run repeats a gate that reproduces
reference logits and saved importances before measuring; results are
written with per-record provenance. Code, plans and analysis tables are
released with the paper (Appendix [9](#app:compute){reference-type="ref"
reference="app:compute"}).

# Method {#sec:method}

We compare the SIH criterion with causal importance across all heads
(§[4.1](#sec:method-ri){reference-type="ref"
reference="sec:method-ri"}--[4.2](#sec:method-causal){reference-type="ref"
reference="sec:method-causal"}), profile causally important and
RI-selected heads (§[4.3](#sec:method-char){reference-type="ref"
reference="sec:method-char"}), and test RI participation in
communication and task behaviour
(§[4.4](#sec:method-circuits){reference-type="ref"
reference="sec:method-circuits"}). A separate checkpoint analysis tests
the proposed link between RI and the emergence of in-context learning
(§[4.5](#sec:method-dev){reference-type="ref"
reference="sec:method-dev"}).

## Reproducing the SIH criterion {#sec:method-ri}

#### Scoring.

We apply Eq. [\[eq:ri\]](#eq:ri){reference-type="ref" reference="eq:ri"}
to every head on all discovery prompts. An event qualifies when the
head's actual attention selects the child with the published dominance
margin ($\tau=2.2$). Its score is the mother's share of the clipped,
centred probability mass obtained from $x_jW^h_{OV}W_U$, using the
*current token's raw embedding* and both first- and last-token mother
anchors. Scores are averaged within family, then across families with
events. We report the pooled index (including demonstrations) and
test-only indices separating all versus query facts and all test
positions versus the answer position.

#### Controls.

Subtracting the mean score of other visible test-fact mothers or sampled
non-name words yields the *name gap* and *word gap*. These measure
relative target preference; static token preferences can survive the
controls.

#### Calibration and selection.

On a length-matched subset, family-consistent target/control swaps test
target preference, with Holm correction across all heads. Separately, a
fixed rule selects heads with the highest positive target or gap scores
within each anchor and test-only population, subject to minimum support.
These candidates and the pooled-index selection form the *RI-selected*
set for the causal audit; membership is not a significance claim. Exact
definitions, selection thresholds and sensitivity analyses are in
Appendix [11](#app:stage1){reference-type="ref" reference="app:stage1"}.

## Causal importance of every head {#sec:method-causal}

#### Intervention.

Figure [2](#fig:patch){reference-type="ref" reference="fig:patch"} shows
how one head's corrupted-run activations replace its clean-run
activations before the output projection. Importance $I_h$ is the
resulting drop in the answer contrast $M$
(§[3](#sec:setting){reference-type="ref" reference="sec:setting"}):
positive for a shift toward the corrupted answer, negative for the
reverse. Comparing the two patching scopes tests whether the effect is
confined to the answer position.

<figure id="fig:patch" data-latex-placement="t">

<figcaption>Single-head activation patching. Scope P replaces all
original prompt positions; Scope F only the final colon. Grey positions
have identical activations, so replacing them has no effect. Shared
answer-prefix tokens remain unpatched.</figcaption>
</figure>

#### Coverage.

Attribution patching [@syed2023attribution] provides a first-order
estimate for all heads from clean and corrupted forwards and a clean
backward pass: $$\begin{equation}
\label{eq:screen}
\widehat I_h=-\big\langle\nabla_{a_h}M(x_c),\,a_h^r-a_h^c\big\rangle .
\end{equation}$$ Exact patching of every head on a common panel provides
a causal ranking independent of the RI selection and calibrates this
approximation. An extension set receives exact measurements across all
discovery families (Table [1](#tab:coverage){reference-type="ref"
reference="tab:coverage"}). All-head comparisons use the common panel;
rankings never mix effects measured on different family sets.

::: {#tab:coverage}
  Measurement                      Heads   Pairs
  ------------------------------ ------- -------
  Screen, Scope P                  1,024     178
  Exact, Scope P: common panel     1,024      40
  Exact, Scope P: extension          148     178
  Exact, Scope F                      69     178

  : Causal-map coverage. Each family contributes two fact orders.
:::

#### Comparisons.

We compare RI with exact importance through rank correlations (overall
and within layer), top-decile omissions and their causes, and
selected-head effects relative to layer-matched non-selected heads and
random controls. Family-bootstrap intervals describe uncertainty within
the discovery sample, without correcting for selection. Head sets,
calibration thresholds, intervention details and audit criteria are in
Appendix [12](#app:stage2){reference-type="ref" reference="app:stage2"}.

## Characterizing important heads {#sec:method-char}

We profile strong and RI-selected heads, moderate-effect candidates and
random controls to propose components and connections for causal testing
(Table [2](#tab:profile){reference-type="ref" reference="tab:profile"}).
The inventory and measurement coverage are in
Appendix [13](#app:stage3){reference-type="ref" reference="app:stage3"}.

#### Where does the head act?

Comparing Scope F with Scope P tests how much of a head's effect is
expressed at the answer colon. Patching individual test positions then
locates earlier contributions. Colon and earlier effects suggest reader
and writer roles, respectively; neither establishes a connection.
Single-position effects need not add up. Reverse patching checks
symmetry by restoring clean activations in the corrupted run.

#### What does it read and promote?

These diagnostics ask whether a head follows the queried fact and
promotes its answer, or instead reflects copying or positional
preferences. We track attention from the colon to fact entities and to
the colon itself across query changes, fact reordering and mother swaps,
and project actual head outputs onto the vocabulary alongside the
criterion's raw-embedding projection. To isolate the embedding choice,
we also recompute RI with contextual inputs at the same positions,
keeping events and controls fixed. Copying and synthetic
induction/retrieval probes assess alternative roles. For selected
writers, residual transfer into diagnostic prompts compares intact and
head-replaced sources; a full source swap checks whether the readout can
detect the changed content. Projections and readouts diagnose content
rather than establish a causal route.

#### What carries the effect?

The head forms $A^hV^h$: its attention pattern $A^h$ weights value
vectors $V^h$ from attended positions. At the colon, we replace either
factor with its corrupted-run counterpart while keeping the other clean,
then replace both. Changes in the answer contrast $M$
(Eq. [\[eq:metric\]](#eq:metric){reference-type="ref"
reference="eq:metric"}) distinguish routing from retrieved content. The
joint effect minus the two separate effects measures their interaction,
reported per family.

::: {#tab:profile}
  Question          Main diagnostic                    Use in route tests
  ----------------- ---------------------------------- --------------------------
  Where?            Scope/position patches             Source sites
  Read / promote?   Attention, controls, projections   Receiver/role hypotheses
  How carried?      Pattern--value swaps               Channel hypotheses

  : Head profiles guide the causal tests of
  §[4.4](#sec:method-circuits){reference-type="ref"
  reference="sec:method-circuits"}; they do not establish connections.
:::

## From heads to a mechanism {#sec:method-circuits}

We test whether the profiled heads communicate, depend on one another,
and include RI-selected participants in the model's use of the stated
relation.

#### Communication.

We first test whether one head affects the answer through another. A
route links a *source*, whose output we replace at a specified site, to
a later *receiver* through its query, key or value input. The two-run
intervention in Figure [3](#fig:route){reference-type="ref"
reference="fig:route"} isolates a source-induced channel change against
frozen intermediate components, then measures its effect on $M$ through
the receiver's colon output in a fresh run. Retained routes are extended
across discovery families in both donor directions. Separately measured
routes need not compose: bounded tests release selected intermediate
heads, MLPs or blocks to examine composed chains and mediated effects
(Appendix [14](#app:stage4){reference-type="ref"
reference="app:stage4"}).

<figure id="fig:route" data-latex-placement="t">

<figcaption>Direct-route measurement. The hybrid isolates a
source-induced change in one receiver channel; a fresh clean run
measures its effect through the receiver’s colon output. Shared
normalization remains live (Appendix <a href="#app:stage4"
data-reference-type="ref"
data-reference="app:stage4">14</a>).</figcaption>
</figure>

#### Joint and conditional effects.

We next ask whether connected heads make overlapping or complementary
contributions, grouping heads that share a source or receiver in the
measured routes. Whole-group and group-minus-member patches measure
nonadditivity and conditional contributions. Receiver blocking restores
a source's clean activation while clamping its receivers to corrupted
outputs; lost recovery measures dependence on their response, not an
additive mediation share. To expose contributions hidden by redundancy,
RI candidates are tested with and without a strong reader replaced, and
those without strong standalone effects collectively against
layer-matched non-RI cohorts. A name copier with little standalone
effect is tested for backup behaviour. Selected contrasts are repeated
with mean replacement to assess baseline sensitivity.

#### RI participation in retained mechanisms.

The final experiment tests RI participation in these structures' use of
the queried relation. It tests fixed small structures and a broad head
set, replacing excluded test-block head outputs by role means while
leaving the rest of the model live. We cross a swap of two mothers with
a query change to the other affected child. The original condition,
facts changed, query changed, and both changed have correct answers
$(a,b,b,a)$: $a$ is the original mother and $b$ the mother exchanged
with her. This tests dependence on the queried binding. Fidelity checks
against the full model and an all-heads-mean baseline select a broad
background $C$ from two fixed sets; failure leaves it explicitly
partial.

For candidate $h$, compare $C_h^-=C\setminus\{h\}$ with
$C_h^+=C\cup\{h\}$: $$\begin{align}
\Delta B_h &= B_{C_h^+}-B_{C_h^-}, \label{eq:ri-functional}\\
\Gamma_{h,t} &= I_t(C_h^+)-I_t(C_h^-). \label{eq:ri-route}
\end{align}$$ Here $B_D$ is the four-cell binding contrast
(Eq. [\[eq:bd\]](#eq:bd){reference-type="ref" reference="eq:bd"}),
reported with cell-wise performance, and $I_t(D)$ is route $t$'s noising
effect in background $D$. The two contrasts separate behavioural
contribution from route modulation. Collective RI removal tests group
dependence. Candidates are screened against every fixed route; selected
associations receive reverse patching and a predeclared connection test
with a matched receiver control. Behavioural follow-up checks
donor-replacement sensitivity before freezing comparisons for held-out
evaluation. Modulation alone does not place a head on a route; any
semantic interpretation remains specific to this task and background.
Exact rosters, fidelity guards and measurement coverage are in
Appendix [14.1](#app:final-ri){reference-type="ref"
reference="app:final-ri"}.

## Developmental check {#sec:method-dev}

To address the training-time gap of
§[2.1](#sec:sih){reference-type="ref" reference="sec:sih"}, we adapt the
analysis of @ren2024semantic to OLMo-2's public checkpoints, asking
whether in-context learning emerges alongside RI changes and whether the
same heads remain high-scoring. This tests temporal association, not
causation.

#### Design.

We reuse frozen prompts from the original four synthetic classification
task families across checkpoints at each shot count, measuring accuracy
among legal labels separately from output-format validity. All heads are
scored on a fixed AGENDA relation set, primarily requiring the source
token to receive maximum attention. To distinguish routing frequency
from selectivity, we track the gate pass rate, RI conditional on
passing, and RI per opportunity, with failed gates contributing zero to
the last measure.

Rather than selecting rising trajectories, we trace the final
checkpoint's highest-RI heads backwards and compare each checkpoint's
top heads with that final set. This tests whether early and mature high
scores belong to the same heads. The broad behavioural window preceded
analysis of RI timing; denser behavioural sampling within it followed an
observed routing peak and is treated as targeted follow-up. Coverage,
preprocessing differences and statistical tests are detailed in
Appendix [15](#app:dev){reference-type="ref" reference="app:dev"}.

# Results {#sec:results}

We report the four audit stages of
§[4](#sec:method){reference-type="ref" reference="sec:method"}, followed
by frozen held-out evaluation of the final selected configurations.
Aggregation, per-head estimates and sensitivity analyses are detailed in
Appendices [11](#app:stage1){reference-type="ref"
reference="app:stage1"}--[14](#app:stage4){reference-type="ref"
reference="app:stage4"}.

## RI poorly identifies causal contributors {#sec:results-why}

#### What RI selects.

Under our percentile rule, pooled RI selects two heads whose qualifying
attention is exclusively self- or previous-token attention;
demonstrations contribute over 99% of their scores
(Table [6](#tab:pooled){reference-type="ref" reference="tab:pooled"}),
consistent with static token preferences rather than retrieval. No
eligible head passes the matched-target test after Holm correction. The
test-only rule selects 59 heads, but several are sensitive to the
current token, target anchor or fact order
(Appendix [11](#app:stage1){reference-type="ref"
reference="app:stage1"}).

<figure id="fig:ri-vs-imp" data-latex-placement="t">
<span class="image placeholder"
data-original-image-src="fig_ri_vs_importance"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>RI versus exact signed Scope-P importance on 40 common pairs
(symmetric-log axis). Orange marks the 61 RI-selected heads; shading
marks <span
class="math inline">|<em>I</em><sub><em>h</em></sub>| &lt; 0.3</span>.
(a) Pooled RI for all 1,024 heads. (b) First-token test-only name gap
for 167 supported heads.</figcaption>
</figure>

#### Against the causal map.

Pooled RI correlates weakly with signed importance ($\rho=0.13$),
whereas the test-only name gap correlates negatively ($\rho=-0.25$, or
$-0.30$ within layer; Figure [4](#fig:ri-vs-imp){reference-type="ref"
reference="fig:ri-vs-imp"}). The two pooled-index heads have near-zero
effects, and 48 of the combined 61 selected heads have $|I_h|<0.1$, less
than 1% of the clean--corrupted gap. Conversely, among the 148 heads
measured on all 178 pairs, only eight of the 25 heads with
$|I_h|\geq0.3$ are selected. Both selections are therefore incomplete,
and the pooled selection also meets our pre-registered misleading
criterion (Appendix [12](#app:stage2){reference-type="ref"
reference="app:stage2"}).

#### Why RI can miss retrieval.

At the answer position the current token is always a colon, so when the
visible token IDs are unchanged, raw-embedding RI cannot adapt its name
preferences when mothers are reassigned
(Appendix [11](#app:stage1){reference-type="ref"
reference="app:stage1"}). Copying a current non-target name may also
raise the name-control score and penalize a useful head, as examined
through the current-token and name-gap diagnostics in
Appendix [13](#app:stage3){reference-type="ref" reference="app:stage3"}.
These limitations concern the index, not the model's capacity for
semantic computation.

## The causal map {#sec:results-map}

#### A few heads carry most of the effect.

The gradient screen closely reproduces the exact ranking ($\rho=0.945$;
Appendix [12](#app:stage2){reference-type="ref"
reference="app:stage2"}), and all effects reported below use exact
patching. Against a mean clean--corrupted gap of 11.2 logits, replacing
L17H1 changes the metric by 5.7 logits, about half the gap, while three
further heads each change it by roughly a quarter
(Figure [5](#fig:causal-map){reference-type="ref"
reference="fig:causal-map"}). Twenty-five heads reach $|I_h|\geq0.3$: 17
positive and eight negative, with negative effects shifting the model
toward the stated fact under corrupted-output insertion. These are not
independent shares because head effects can interact. The same four
heads lead the exact common-subset scan, indicating that the leading
heads are not an artefact of the extension set
(Appendix [12](#app:stage2){reference-type="ref"
reference="app:stage2"}).

<figure id="fig:causal-map" data-latex-placement="t">
<span class="image placeholder" data-original-image-src="fig_causal_map"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>The causal map: signed Scope-P importance of the 25 strong
heads on 178 pairs, with family-bootstrap intervals; orange marks
RI-selected heads.</figcaption>
</figure>

#### Two positions, two kinds of head.

Restricting the patch to the final colon splits the map by depth
(Figure [13](#fig:scopeF-ratio){reference-type="ref"
reference="fig:scopeF-ratio"},
Appendix [12](#app:stage2){reference-type="ref"
reference="app:stage2"}). In the Scope-F set, measurable heads in layers
16--26 retain nearly all their Scope-P effect under Scope F, indicating
answer-position contributions, whereas heads in layers 6--11 lose their
effects and therefore write information at earlier positions for later
components to read; two layer-14--15 heads are intermediate. At the
colon, positive heads raise the donor-retrieved name and negative heads
lower it (Table [9](#tab:scopeF-app){reference-type="ref"
reference="tab:scopeF-app"});
§[5.3](#sec:results-char){reference-type="ref"
reference="sec:results-char"} extends localization to all strong heads
and tests how their effects arise.

Together, these positional differences suggest a writer--reader
organization whose computations and connections are tested in
§§[5.3](#sec:results-char){reference-type="ref"
reference="sec:results-char"}--[5.4](#sec:results-routes){reference-type="ref"
reference="sec:results-routes"}.

## What the important heads do {#sec:results-char}

#### Writers and readers.

Twenty of the 25 strong heads deliver nearly all their effect at the
colon. The strongest, L17H1, instead acts within the query fact, at
mother tokens and shared syntax positions
(Figure [6](#fig:profiles){reference-type="ref"
reference="fig:profiles"}). A controlled readout shows that its write
carries mother-identity information
(Table [13](#tab:readout-app){reference-type="ref"
reference="tab:readout-app"}). L15H25 acts at the query child's final
token, but its written content remains unresolved. The strongest writer
thus shares the readers' layer band rather than belonging to the shallow
group of §[5.2](#sec:results-map){reference-type="ref"
reference="sec:results-map"}.

<figure id="fig:profiles" data-latex-placement="t">
<span class="image placeholder"
data-original-image-src="fig_head_profiles" data-original-image-title=""
width="\columnwidth"></span>
<figcaption>Single-position effects of three fact-position heads by
token role (M: mother, C: child, oth.: other test facts, Q: question);
mean and SD over 20 families.</figcaption>
</figure>

#### Question-guided selection, value-carried effects.

At the colon, heads attend to mothers, children or the colon itself
(Figure [14](#fig:colon-attention){reference-type="ref"
reference="fig:colon-attention"}). Mother- and child-attenders follow a
changed query to the newly relevant fact in all 20
families(Figure [15](#fig:stage3-app){reference-type="ref"
reference="fig:stage3-app"}(a)); reordering or swapping mothers has
little effect on family-mean attention. Values carry most of the leading
heads' mean effects(Figure [15](#fig:stage3-app){reference-type="ref"
reference="fig:stage3-app"}(b)). Together, these findings suggest
question-guided selection of positions whose content changes with the
mother assignment. L27H6's output directly favors the answer, whereas
several strong heads have output projections much smaller than their
causal effects, motivating tests of downstream computation.

#### Copying does not determine causal role.

Attended-name labels depend on the token anchor, and weight copying
alone does not determine whether a head supports the correct answer.
L18H18 has a positive causal effect ($I_h=+2.70$) despite a negative
weight-copying score ($-0.08$); L30H18 has the highest weight-copying
score ($+0.60$) but attends to a distractor and has a negative effect
($I_h=-1.06$;
Table [\[tab:profiles-app\]](#tab:profiles-app){reference-type="ref"
reference="tab:profiles-app"}).
Appendix [13](#app:stage3){reference-type="ref" reference="app:stage3"}
reports the event counts, anchor checks, control-name gaps, and
synthetic-probe fingerprints.

## Measured routes and RI participation {#sec:results-routes}

#### From profiles to connected paths.

On 89 families, L17H1 reaches L18H18/L18H19 through V with effects of
3.40/2.07 logits; clamping its seven tested receivers removes 89.4% of
writer-union restoration. Selectively releasing intermediate heads
establishes the composed writer--reader paths in
Figure [7](#fig:stage4-chains){reference-type="ref"
reference="fig:stage4-chains"}, placing RI heads between fact-site
writers and later readers. L8H15 also has a direct V route despite its
weaker standalone effect; releasing MLP9 attenuates that positive route
(Appendix [14.0.0.9](#app:s43-results){reference-type="ref"
reference="app:s43-results"}).

<figure id="fig:stage4-chains" data-latex-placement="t">
<span class="image placeholder"
data-original-image-src="fig_stage4_chains" data-original-image-title=""
width="\columnwidth"></span>
<figcaption>Measured paths, 89 families. Orange: RI heads. Numbers:
whole-path noising/restoration effects. Intermediate heads are released
jointly; Q/V denotes the final receiver channel, with output evaluated
at the colon.</figcaption>
</figure>

#### Joint effects and weaker RI heads.

Replacing L18H18 and L18H19 together reduces the answer margin by 5.63
logits. Group-minus-member tests show that a head's contribution changes
when its partners are also replaced; singleton effects therefore do not
predict group behaviour. Separately, we test 31 RI-selected heads
prioritized by earlier RI rankings or causal effects (RI31). Excluding
their eight strong members leaves RI23. Replacing these 23 together
changes family margins by 0.594 logits in mean absolute magnitude,
versus 0.215 for layer-matched non-RI groups. Thus weaker RI heads
matter collectively, although the signed effect reverses between
replacement baselines
(Appendix [14.0.0.7](#app:s42-residual-results){reference-type="ref"
reference="app:s42-residual-results"}).

#### Additional participation depends on the background.

The final experiment retains a 33-head set assembled from causal and
route evidence (C33), then adds 17 RI heads (C50). Neither preserves the
task: on 20 families, C50 retains 11.5% of the fact-swap gap, prefers
the first chain's mother in 85% of cases and achieves 50% candidate
accuracy.

Within this partial background, six candidates with weak standalone
effects, chosen from existing RI rankings at different token anchors,
are tested against five previously measured routes, including those in
Figure [7](#fig:stage4-chains){reference-type="ref"
reference="fig:stage4-chains"} (rosters:
Appendix [14.1](#app:final-ri){reference-type="ref"
reference="app:final-ri"}). L9H16 and L1H27 improve the four-cell
binding contrast by 0.601 and 0.260 logits, positive in every family.
Three routes remain retained, but no candidate--route interaction passes
the effect rule, leaving the candidates' connections unresolved
(Appendix [14.2](#app:s44-results){reference-type="ref"
reference="app:s44-results"}).

On 87 held-out families, L9H16 and L1H27 reproduce the four-cell binding
gains (0.585 and 0.261 logits), each positive in every family. Their
original fact-swap effects are positive with mean replacement but nearly
vanish with paired donors; RI31's collective effect persists under both
baselines, though the group includes eight already-strong heads. Under
mean replacement, C50 retains only 12.8% of the full fact-swap gap.
Validation thus supports the binding effects in this partial background,
not a sufficient circuit or new route assignments
(Appendix [14.3](#app:s44-validation){reference-type="ref"
reference="app:s44-validation"}).

# Conclusions {#sec:conclusions}

On synthetic single-hop mother-of retrieval, RI poorly identifies
causally important heads: it misses major contributors and selects many
with weak standalone effects
(Section [5.1](#sec:results-why){reference-type="ref"
reference="sec:results-why"}). Path interventions reveal interacting
fact-position writers and answer-position readers, including RI-selected
participants; copying alone does not explain their roles
(Sections [5.3](#sec:results-char){reference-type="ref"
reference="sec:results-char"}--[5.4](#sec:results-routes){reference-type="ref"
reference="sec:results-routes"};
Appendix [13](#app:stage3){reference-type="ref"
reference="app:stage3"}). This establishes task-specific participation,
not that individual heads encode the relation.

Weaker RI heads contribute collectively, but their signed group effect
depends on the replacement baseline
(Section [5.4](#sec:results-routes){reference-type="ref"
reference="sec:results-routes"}). L9H16 and L1H27 reproduce binding
gains on held-out families, though their fact-swap effects nearly vanish
with paired donors
(Appendix [14.3](#app:s44-validation){reference-type="ref"
reference="app:s44-validation"}). The retained 50-head set does not
preserve the full task, and these weaker heads' routes remain unresolved
(Appendix [14.2](#app:s44-results){reference-type="ref"
reference="app:s44-results"}). Early routing changes during training do
not identify mature high-RI heads
(Appendix [15](#app:dev){reference-type="ref" reference="app:dev"}).
Thus, neither RI nor training-time association identifies a sufficient
causal circuit.

# Limitations {#limitations .unnumbered}

Our interventions establish task-specific causal effects, but do not
uniquely identify semantic representations or show that the tested
circuits are exhaustive or dominant. Untested or redundant routes may
contribute, and retained heads depend on live MLPs, embeddings and
normalization (§[4.4](#sec:method-circuits){reference-type="ref"
reference="sec:method-circuits"}).

Computational constraints limited measurement coverage and intervention
types. We study one base model and one relation, mother-of, in synthetic
single-hop prompts (§[3](#sec:setting){reference-type="ref"
reference="sec:setting"}), without testing two-hop composition or
transfer to other relations or models. These findings do not establish
general semantic ability. Training-time associations are observational
(Appendix [15](#app:dev){reference-type="ref" reference="app:dev"}).

# AI Disclosure and Reflection {#ai-disclosure-and-reflection .unnumbered}

We used Claude Opus 5 and Fable 5 for coding, documentation,
brainstorming, formalizing our experiment plans, collecting and
organizing experimental outputs, and retrieving information from our
paper collection using Andrej Karpathy's *LLM Wiki*.[^2] We used Codex
for manuscript drafting and editing. These tools supported organization
and idea development, but accurate, concise claims required iterative
human review. We directed the research and remain responsible for its
methods, interpretations and final content.

# The relation index of @ren2024semantic {#app:ri}

This appendix restates the SIH criterion as defined in the original
paper (their §3, Eq. 2--3), so that the main text can stay short, and
lists the evidence behind the limitations noted in
§[2.1](#sec:sih){reference-type="ref" reference="sec:sih"}. Our own
re-implementation, with its controls and calibration, is described in
§[4.1](#sec:method-ri){reference-type="ref" reference="sec:method-ri"}.

#### Notation.

The model has $L$ layers of $H$ heads. For head $h$,
$W^h_{OV}=W^h_v W^h_o$ is the OV matrix, $W^h_{QK}=W^h_q (W^h_k)^{\top}$
the QK matrix, $W_e$ the input embedding and $W_U$ the unembedding. For
an input $[t_1,\ldots,t_n]$, $x_i=t_i W_e$ is the *input embedding* of
token $t_i$. A relation is a triplet $T=(t_s,r,t_o)$ with head token at
position $s$ and tail token at position $o$.

#### Step 1: attention condition (QK).

At a current position $j\ge\max(s,o)$, let $A^h_j$ be head $h$'s
attention over positions $1..j$, written in the original paper as
$\mathrm{softmax}(x_j W^h_{QK}
x^{\top}/\sqrt{d_h})$ in the notation of Eq. 1 there. The head qualifies
for $(T,j)$ if $$\begin{equation*}
s=\arg\max_{k\le j} A^h_{j,k}
\quad\text{and}\quad
\frac{A^h_{j,s}}{\max_{k\ne s} A^h_{j,k}}>\tau,\qquad \tau=2.2 .
\end{equation*}$$ The paper's formula writes the attention in terms of
the embedding matrix $x$; in our implementation $A^h_j$ is the attention
pattern actually produced by the model's forward pass at the head's
layer (captured from the attention kernel), and only the OV projection
of Step 2 uses the raw embedding.

#### Step 2: output condition (OV).

The head's output at $j$ is projected onto the vocabulary from the input
embedding of the current token, $$\begin{equation*}
p^{h,j}=\mathrm{softmax}\!\left(x_j W^h_{OV} W_U\right)\in\mathbb{R}^{|V|}.
\end{equation*}$$ Only the tokens $t_k$, $k\le j$, that appear in the
context are kept (duplicates counted once). Their probabilities are
centred and clipped,
$q^{h,j}_{t_k}=\max\!\big(0,\;p^{h,j}_{t_k}-\mathbb{E}_{k'\le j}[p^{h,j}_{t_{k'}}]\big)$,
and the score for $(T,j)$ is the share of the clipped mass on the tail
token, $$\begin{equation*}
a^{h,j}_T=\frac{q^{h,j}_{t_o}}{\sum_{k\le j} q^{h,j}_{t_k}} .
\end{equation*}$$

#### Worked example.

Table [3](#tab:ri-example){reference-type="ref"
reference="tab:ri-example"} carries the computation through for one
event, with illustrative numbers. The context is "Anna is the mother of
Ben ." and the current token is the period ($j{=}7$), so seven distinct
tokens are visible. Suppose head $h$ attends most to *Anna* with a
margin above $\tau$, so the event qualifies with $t_s{=}$Anna. The
raw-embedding projection of the period is pushed through $W^h_{OV}W_U$
and the softmax is restricted to the visible tokens; the mean of the
seven values, $\bar p{=}0.143$, is subtracted and negatives are clipped.
The tail token *Ben* holds $0.52$ of the clipped mass, which is the
event score. Two properties of the score are visible here: it depends
only on the current token's embedding and on which tokens are visible,
so every event of this head at a period with the same visible set
receives the same vector $p$; and the tail competes with all visible
tokens, including *is* and *the*, not with other names.

::: {#tab:ri-example}
  $t_k$                Anna     is    the   mother     of     Ben      .
  ----------------- ------- ------ ------ -------- ------ ------- ------
  $p^{h,j}_{t_k}$      0.30   0.05   0.05     0.15   0.05    0.32   0.08
  $q^{h,j}_{t_k}$     0.157      0      0    0.007      0   0.177      0

  : Illustrative event score. $q=\max(0,p-\bar p)$ with $\bar p=0.143$;
  $\sum_k q_k=0.341$;
  $a^{h,j}_T=q_{\text{Ben}}/\sum_k q_k=0.177/0.341=0.52$. The numbers
  are chosen for the example, not measured.
:::

#### Step 3: aggregation.

$\mathrm{RI}_h$ is the mean of $a^{h,j}_T$ over all qualifying $(T,j)$
for a given relation type. There is no threshold that turns an index
into a label: the paper reports heatmaps of $\mathrm{RI}_h$ per relation
and calls heads with salient values SIHs. An appendix of the original
paper shows that $\tau\in\{2.0,2.5\}$ yields a similar set of salient
heads.

#### Setting of the original study.

InternLM2-1.8B (24 layers, 16 heads); the AGENDA test set (1,000
abstracts with annotated knowledge graphs); three syntactic dependencies
(subject, object, modifier) and seven relation types; multi-token
entities replaced by single capital letters and frequent function words
removed before scoring.

#### Evidence for the limitations in §[2.1](#sec:sih){reference-type="ref" reference="sec:sih"}. {#evidence-for-the-limitations-in-secsih.}

- The word "ablation" occurs once in the original paper and refers to
  varying $\tau$ (their Appendix A). No activation, weight or attention
  intervention is reported.

- The training analysis (their §4.2) uses a model the authors train
  themselves ($d{=}2048$, 40k steps on SlimPajama, checkpoints every 200
  steps), not the released InternLM2-1.8B used for the SIH
  identification in their §3. In that analysis the margin condition on
  $\tau$ is dropped ("it is too challenging to find attention heads
  having an extremely high attention probability"), and the plotted
  heads are "sampled ...with an increasing relation index". The
  inference is stated as "we infer that the formation of semantic
  induction heads plays a crucial role in the development of the ICL
  ability" [@ren2024semantic Sec. 4.2].

- The projection $x_j W^h_{OV} W_U$ uses $x_j=t_j W_e$. The paper
  justifies this by quoting @elhage2021mathematical: "both the QK
  circuit and OV circuit are directly performed on input embeddings."
  That statement describes attention-only models without MLPs and
  normalization, where the residual at every layer is a sum of head
  outputs added to the embedding; in a 24- or 32-layer model with MLPs
  and layer normalization the value input of head $h$ at position $j$ is
  the normalized residual $\mathrm{LN}(r^{\ell}_j)$, and the head's
  output at $j$ is $\sum_k A^h_{j,k}\,\mathrm{LN}(r^{\ell}_k)W^h_{OV}$,
  a function of the attended positions $k$.

- In $a^{h,j}_T$ the tail competes with every earlier distinct token,
  not with tokens of the same type (other entities), and no reference
  distribution for $\mathrm{RI}_h$ is reported.

#### Connection to our audit.

These limitations correspond to four parts of our analysis. The absence
of causal evidence motivates the all-head intervention map of
§[4.2](#sec:method-causal){reference-type="ref"
reference="sec:method-causal"}; the developmental ambiguity motivates
the checkpoint analysis of §[4.5](#sec:method-dev){reference-type="ref"
reference="sec:method-dev"}; the raw-embedding approximation motivates
the contextual diagnostics of
§[4.3](#sec:method-char){reference-type="ref"
reference="sec:method-char"}; and the lack of controls and calibration
motivates the controlled RI variants of
§[4.1](#sec:method-ri){reference-type="ref" reference="sec:method-ri"}.

# Induction heads: background {#app:ih}

@elhage2021mathematical decompose an attention head into a query--key
(QK) circuit, which decides where the head attends, and an output--value
(OV) circuit, which decides what the head writes. In one- and two-layer
attention-only models they identify *induction heads*: on a sequence
$[A][B]\ldots[A]$, the QK circuit attends from the second $[A]$ to $[B]$
(prefix matching) and the OV circuit raises the logit of $[B]$
(copying). @olsson2022induction define both properties behaviourally, as
a prefix-matching score and a copying score measured on repeated random
token sequences, and show that ablating induction heads removes most of
the in-context loss reduction of small models; they also observe that
induction heads form in a narrow window of training that coincides with
a sharp improvement in in-context learning.

<figure id="fig:ih" data-latex-placement="h">

<figcaption>The induction-head template. The QK circuit attends from the
second <span class="math inline">[<em>A</em>]</span> to the token that
followed the first, and the OV circuit raises that token’s logit. The
SIH criterion keeps the template but lets the written token differ from
the attended one: it attends to <span
class="math inline"><em>t</em><sub><em>s</em></sub></span> and writes
<span class="math inline"><em>t</em><sub><em>o</em></sub></span>
(Figure <a href="#fig:ri" data-reference-type="ref"
data-reference="fig:ri">1</a>).</figcaption>
</figure>

Two later developments are relevant here. @wu2024retrieval identify
*retrieval heads*, heads with the same behavioural signature that copy a
needle from a long context; they show that a small set of such heads is
largely universal across models and that masking them destroys
long-context factual recall, whereas masking random heads does not.
@edelman2024evolution study how statistical induction heads emerge
during training on Markov-chain data and show that in-context learning
develops through phases in which simpler statistical strategies precede
the induction mechanism.

#### Causal head roles in IOI.

The indirect-object-identification analysis of @wang2023ioi introduced
several influential functional labels: *name-mover heads* attend to and
promote the relevant name, *negative name-mover heads* reduce its
relative logit, and *backup heads* acquire a substantial contribution
when primary heads are removed. We use these terms only as functional
shorthand: the name-mover classifications in
Appendix [13](#app:stage3){reference-type="ref" reference="app:stage3"}
do not imply that a head instantiates the complete IOI circuit, and a
weak head is treated as a backup candidate unless the
weakened-background test of
Appendix [14](#app:stage4){reference-type="ref" reference="app:stage4"}
exposes a conditional causal contribution.

Two features of this literature provide the relevant contrast for
semantic induction heads. First, the head type is defined by a behaviour
the head produces on a controlled input, so a head either performs the
behaviour or does not. Second, the importance of the head type is
established by intervention on the model, not by the score alone. In our
setting the repeated-sequence and key--value retrieval probes of
§[4.3](#sec:method-char){reference-type="ref"
reference="sec:method-char"} apply these behavioural tests to every head
in our inventory.

# Compute, runs and released artifacts {#app:compute}

The cluster-based RI and intervention runs used the Tel Aviv University
Slurm cluster (partition `studentkillable`) using NVIDIA GPUs with
11--12 GB of memory. Each worker sharded the base OLMo-2-1124-7B model
in fp16 across two GPUs. The mature-model audit used the model without
fine-tuning at revision `7df9a82518afdecae4e8c026b27adccc8c1f0032`. S4.3
also used Colab runtimes with A100 GPUs, including 40 GB and 80 GB
devices; the developmental sweep instead uses the checkpoints of
Appendix [15](#app:dev){reference-type="ref" reference="app:dev"}.

Table [4](#tab:compute){reference-type="ref" reference="tab:compute"}
lists the production runs. Cluster intervention runs use a frozen plan.
The plan's SHA-256, the hashes of the relevant code and data files, and
the model revision are recorded in a manifest. Before measurement, a
validation gate reproduces reference candidate logits and, where
applicable, previously saved importance values within a fixed tolerance;
the run aborts if the gate fails. Output records retain per-item
provenance, and interrupted runs resume from checksummed checkpoints.
Code, plans, manifests and analysis tables are available at
<https://github.com/Maxgolu/NLP_proj>.

<figure id="fig:pipeline" data-latex-placement="h">

<figcaption>Cluster intervention workflow. A run starts from a frozen
plan, passes a gate that reproduces reference quantities, writes
checksummed records, and its analysis tables are recomputed on CPU from
those records and re-checked independently.</figcaption>
</figure>

::: {#tab:compute}
  Run                          Job                                     Wall-clock Content
  ---------------------------- -------- ----------------------------------------- ------------------------------------------
  Stage 1, RI scan             TODO                                          TODO all 1,024 heads, 534 prompts
  Stage 1, test-only RI        905839                                        TODO name/word controls
  Stage 2, causal map          888691                              1h15m (6 GPUs) all heads, 40 pairs; 92 heads, 178 pairs
  Stage 2, extension           912879                                         33m Scope P/F, 178 pairs
  Stage 3, characterization    TODO       $\approx$`<!-- -->`{=html}3.7h (6 GPUs) 263,328 patched forwards
  Stage 3, readout             918689                                         33m 1,920 readout records
  Stage 4, routes and groups   TODO                                          TODO TODO
  Developmental sweep          TODO                                          TODO 17 checkpoints

  : Production runs. For cluster jobs, wall-clock is the Slurm elapsed
  time.
:::

# Prompt format and variants {#app:data}

Figure [10](#fig:data){reference-type="ref" reference="fig:data"} shows
the test block of one discovery family in its variants. Every prompt
begins with four demonstrations of the same form (four facts, a question
and its answer), identical across all variants and orders of a family,
so that twins differ only in the test block. A test block holds two
chains of two facts each: the *query chain*, whose middle name is asked
about, and a *distractor chain* of the same shape. The answer is the
mother of the queried name, i.e. the top of the query chain; the
distractor is the top of the other chain. The queried name therefore
occurs three times (as a mother, as a child, and in the question), and
the two candidate answers occur once each. Each invented name spans
multiple OLMo-2 tokens, and candidate names are matched in token length
so that clean and corrupted prompts remain positionally aligned.
Relation annotations (head, tail, character spans) are produced at
construction time for every fact, including demonstrations.

<figure id="fig:data" data-latex-placement="t">
<table>
<thead>
<tr>
<th style="text-align: left;"><strong>Clean</strong> (answer:
Rarnia)</th>
<th style="text-align: left;"><strong>Corrupted</strong> (answer:
Dirlna)</th>
</tr>
</thead>
<tbody>
<tr>
<td
style="text-align: left;"><code>Jarlia is the mother of Kernna.</code></td>
<td
style="text-align: left;"><code>Jarlia is the mother of Kernna.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><strong><code>Rarnia</code></strong><code> is the mother of Jarlia.</code></td>
<td
style="text-align: left;"><strong><code>Dirlna</code></strong><code> is the mother of Jarlia.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Kulnra is the mother of Tilnia.</code></td>
<td
style="text-align: left;"><code>Kulnra is the mother of Tilnia.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><strong><code>Dirlna</code></strong><code> is the mother of Kulnra.</code></td>
<td
style="text-align: left;"><strong><code>Rarnia</code></strong><code> is the mother of Kulnra.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Question: Who is the mother of Jarlia?</code></td>
<td
style="text-align: left;"><code>Question: Who is the mother of Jarlia?</code></td>
</tr>
<tr>
<td style="text-align: left;"><code>Answer:</code></td>
<td style="text-align: left;"><code>Answer:</code></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Reorder</strong> (answer:
Rarnia)</td>
<td style="text-align: left;"><strong>Other fact order</strong> (answer:
Rarnia)</td>
</tr>
<tr>
<td
style="text-align: left;"><code>Kulnra is the mother of Tilnia.</code></td>
<td
style="text-align: left;"><code>Kulnra is the mother of Tilnia.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Jarlia is the mother of Kernna.</code></td>
<td
style="text-align: left;"><code>Dirlna is the mother of Kulnra.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Rarnia is the mother of Jarlia.</code></td>
<td
style="text-align: left;"><code>Jarlia is the mother of Kernna.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Dirlna is the mother of Kulnra.</code></td>
<td
style="text-align: left;"><code>Rarnia is the mother of Jarlia.</code></td>
</tr>
<tr>
<td
style="text-align: left;"><code>Question: Who is the mother of Jarlia?</code></td>
<td
style="text-align: left;"><code>Question: Who is the mother of Jarlia?</code></td>
</tr>
<tr>
<td style="text-align: left;"><code>Answer:</code></td>
<td style="text-align: left;"><code>Answer:</code></td>
</tr>
</tbody>
</table>
<figcaption>Test block of discovery family 0 (the four demonstrations
that precede it are omitted). Top: the clean prompt and its corrupted
twin; only the two bold names are swapped, so the two prompts are
token-aligned and the distractor becomes the answer. Bottom left: the
reorder variant, with the same facts in a different order. Bottom right:
the second fact order of the clean prompt, in which the distractor chain
precedes the query chain. The query fact is the one whose child is the
queried name (“Rarnia is the mother of Jarlia.”); the final colon of
“Answer:” is the answer position at which Scope-F interventions are
applied.</figcaption>
</figure>

#### Variants.

The clean variant leaves both candidate mothers in their original fact
lines. The corrupted variant swaps only the two answer-side mother
spans, so the distractor becomes correct; because the candidate names
are matched in token length, all token positions and mention counts
remain unchanged. The reordered variant permutes the four test facts
without changing their contents. Each variant is instantiated in two
fact orders. In the released data, the clean and reordered variants are
named `base` and `reorder`, respectively.

::: {#tab:data}
  Quantity                                                                             Value
  ----------------------------------------------------------------- ------------------------
  Generated families                                                                     200
  4-shot accuracy, clean / corrupted / reorder (both orders)                 93% / 94% / 89%
  Families correct on clean and corrupted, both orders                                   176
  discovery / held-out (split by family id)                                          89 / 87
  Clean--corrupted pairs, discovery                                                      178
  Common subset (all-head measurements)                               20 families / 40 pairs
  Prompts per family (base, corrupted, reorder $\times$ 2 orders)                          6
  Demonstrations per prompt                                                                4
  Prompt length, tokens (mean)                                                           282
  Positions differing between twins (mean)                                                44
  Pairs with a shared answer prefix                                                        4

  : Data set and split.
:::

# Relation-index scan: details and sensitivity {#app:stage1}

#### Run.

The scan (`stage1_v4_calibrated`) covered all 1,024 heads on 534 prompts
(89 discovery families $\times$ base/corrupted/reorder $\times$ two
orders, four demonstrations each). Preflight checks reproduced the
behavioural candidate scores and showed zero drift under inert hooks and
self-patching; attention capture had a maximum vocabulary-logit drift of
0.039 on one prompt. The test-only extension (`ri_test_v2`) scored all
46,634 saved test-block events; 237,685 replication checks of saved
scores passed. Independent recomputation reproduced all reported means,
contrasts and support counts.

#### Attention condition and populations.

Every annotated fact, demonstrations included, supplies a child source
at its last token $s$ and a mother target anchored at its first token
(primary) or last token (sensitivity). At a current position $j$ where
both names are fully visible, the model's actual attention must satisfy
$$\begin{equation}
\label{eq:qkgate}
s=\arg\max_{k\le j}A^h_{j,k}
\quad\text{and}\quad
A^h_{j,s}>\tau\max_{\substack{k\le j\\k\ne s}}A^h_{j,k},
\end{equation}$$ with the published $\tau=2.2$; the raw embedding is
used only for the OV score. The pooled index includes demonstration and
test events. The test-only extension requires both the fact and the
current position to lie in the test block, and crosses two fact scopes
(all test facts or only the query fact) with two position scopes (all
test positions or only the final colon), giving four test-only
populations in addition to the pooled index. Each test-only population
uses both mother anchors and pools the base, corrupted and reorder
variants and both orders; the original full visible context, including
demonstrations, is retained for score normalization.

#### Method details.

*Event score.* For a qualifying $(T,j)$ with anchor token $t_o$, the
score is $a^{h,j}_T=q^{h,j}_{t_o}/\sum_{k\le j}q^{h,j}_{t_k}$ with
$q^{h,j}_t=\max(0,p^{h,j}_t-\bar p^{h,j})$ and
$p^{h,j}=\mathrm{softmax}(x_jW^h_{OV}W_U)$ restricted to the distinct
token ids visible at $j$ (Appendix [7](#app:ri){reference-type="ref"
reference="app:ri"}); anchor-token collisions between target and control
names are treated as ties. *Aggregation.* Scores are averaged within
family and then with equal weight across the families that have at least
one scored event for that head; families without events do not enter the
mean. *Name controls.* The other test-fact mothers whose full name is
visible at $j$; token length need not match. *Word controls.* Up to
three word types sampled from fully visible non-name words in the test
block (mostly template words); sampling is shared across heads and facts
at a given prompt position. They provide a weaker specificity check. All
reported test-only means and counts use the name-control-eligible subset
of events; events without an available control are excluded from the gap
statistics but retained in the pooled index. *Calibration population.*
Test-block events in which both the target and a control mention are
fully visible, of equal token length and with distinct first tokens;
19,901 of 278,134 QK events qualify. Heads need at least 50 such events
across ten families (80 heads). The null swaps target and control labels
consistently within each family (100,000 replicates) and assumes the two
labels are exchangeable under the null; heads are tested with Holm
correction over 1,024. *Selection rule.* From the name-control-eligible
events, rank heads by the family-weighted target score, name gap and
word gap separately for each of the four test-only populations and both
anchors. Retain the ten largest positive means per ranking, requiring at
least 20 scored events in ten families for the corresponding statistic,
and take their union. The pooled-index selection uses the separate
descriptive 99.9th-percentile rule. The causal audit includes the union
of these test-only and pooled candidates; references to the test-only
set exclude the pooled-only additions. The 50-event threshold of the
calibration and the 20-event threshold of the selection are different
requirements for different purposes and yield different head
populations.

#### Why the pooled index is dominated by demonstrations.

Under the original pooled index and the descriptive 99.9th-percentile
rule the two top heads are L3H11 and L9H22.
Table [6](#tab:pooled){reference-type="ref" reference="tab:pooled"}
splits their events by block: demonstrations contribute over 99% of each
head's summed target score, and neither head has a single event from a
test position attending to a demonstration source. Inspection of the
events shows why: all 86 passes of L9H22 attend from a position to
itself, and all 149 passes of L3H11 attend one token back, from a period
or newline to the end of the preceding name. In both cases the OV
projection is computed from the same current-token embedding for every
event, so the reported index varies only with which target token is
selected and how the visible context normalizes it. A high pooled index
can thus be obtained from local attention with a static token
preference, without any event that demonstrates retrieval of a test
relation. This does not show that these heads take no part in relational
retrieval; it shows that the pooled score does not require it, which
motivates the test-block restriction of
§[4.1](#sec:method-ri){reference-type="ref" reference="sec:method-ri"}.

::: {#tab:pooled}
  Head      Pooled RI   QK freq. (%)   Demo: $n$ / RI   Test: $n$ / RI
  ------- ----------- -------------- ---------------- ----------------
  L3H11         0.243         0.0097      144 / 0.250        5 / 0.038
  L9H22         0.374         0.0056       72 / 0.445       14 / 0.007

  : The two heads selected by the pooled index and the descriptive
  99.9th-percentile rule, split by whether the annotated fact and the
  current position lie in a demonstration or in the test block.
  Frequency is the share of eligible fact--position evaluations that
  pass the QK gate.
:::

#### Matched-target calibration.

The matched subset retained 19,901 of 278,134 QK events; 80 heads had at
least 50 matched events across ten families. Under 100,000
family-consistent target/control swaps, no head passed Holm correction
at $\alpha{=}0.05$, over 1,024 or over the 80 supported heads. The three
smallest raw $p$ values are L16H4 ($p{=}0.006$, target 0.0147
vs. control 0.0093, 87 events, 40 families), L9H18 ($p{=}0.023$) and
L13H13 ($p{=}0.025$). Neither pooled-index head had a matched event.

#### Test-only candidates.

Table [7](#tab:ricands){reference-type="ref" reference="tab:ricands"}
lists the leading heads under the test-only index with name controls
(all test facts, all test positions). The frozen rule
(§[4.1](#sec:method-ri){reference-type="ref" reference="sec:method-ri"})
selects 59 heads in total, 36 of which enter at least one name-gap
ranking. Two patterns recur among high scorers. Some heads owe much of
their score to events in which the current token is part of the target
name itself (L23H10: 31 of 60 comparisons; L15H18: 104 of 110); such
events can be scored without any retrieval across positions, so they do
not demonstrate it, although they do not exclude it. Excluding those
events reduces but does not remove L23H10's gap. Others depend on one
current token (for L17H5 the token *mother* contributes 66% of the
family-weighted target score) or on the reorder variant (299 of L9H16's
343 comparisons). Paired base/corrupted events that follow the changed
fact are rare: L11H4 has 21 jointly passing pairs in 18 families, with a
positive name gap in both versions for only five families.

::: {#tab:ricands}
  Head     Anchor     Events   Fam.   Target   Control         Gap
  -------- -------- -------- ------ -------- --------- -----------
  L11H4    first          94     39   0.0147    0.0081   $+0.0065$
  L17H5    first          65     34   0.0229    0.0168   $+0.0061$
  L9H16    last          343     87   0.0209    0.0135   $+0.0074$
  L12H2    first          52     32   0.0201    0.0129   $+0.0072$
  L25H18   last           46     17   0.0287    0.0156   $+0.0131$
  L23H10   first          60     24   0.0283    0.0068   $+0.0215$

  : Leading test-only candidates (all test facts and positions). Events
  are target/control comparisons; family-weighted means over the
  name-control-eligible subset.
:::

#### The answer position and the raw-embedding invariance.

At the final colon, 20 heads have qualifying events (748 in total, 744
to test sources); L16H21 and L16H1 attend to the query fact in over 220
prompts each (41% of prompts). Their name gaps at that position are near
zero or negative (Table [8](#tab:final){reference-type="ref"
reference="tab:final"}). Part of the reason is structural. The current
token at the answer position is always the colon, so for a fixed head
the raw-embedding projection vector is the same in every prompt, and the
normalized score is the same whenever the set of visible token ids is
the same; only the attention condition can differ between such prompts.
In all 49 jointly passing base/corrupted pairs of L16H21 the sign of the
old-versus-new-mother contrast flips exactly when the target label
flips, at both anchors: once the head attends, the scoring component
cannot follow a reassignment of mothers at this position. The same holds
at any earlier position whose current token and visible token set are
unchanged. This is a limit of the score, not a statement about what the
head's actual output carries.

::: {#tab:final}
  Head       Query events   Fam.   QK (%)   First gap    Last gap
  -------- -------------- ------ -------- ----------- -----------
  L16H21              222     77     41.6   $-0.0004$   $+0.0034$
  L16H1               223     76     41.8   $-0.0025$   $-0.0034$
  L16H24               68     35     12.7   $-0.0028$   $+0.0036$
  L16H4                63     32     11.8   $+0.0004$   $-0.0072$

  : Heads whose attention at the answer position selects the query fact
  (denominator 534 prompts), with their name gaps at the two anchors.
:::

<figure id="fig:ri-dist" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figE_ri_distribution"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>The pooled index over heads. (a) Distribution of the
first-anchor index for the 342 heads with at least one qualifying event
(log count); the two heads selected by the percentile rule are marked.
(b) First- against last-anchor index for all heads; the anchor changes
which heads lead (L29H14 tops the last-anchor ranking).</figcaption>
</figure>

#### Dominance margin.

Over all 1,455,940 eligible dominance ratios (event-weighted,
demonstrations included), 80.9% are at or below the published
$\tau{=}2.2$ and the 95th percentile is 4.48. Refiltering the saved
passes at 4.48 keeps 72,797 events; the two pooled heads keep 30 and 49
passes respectively. The anchor also matters: under the last token,
L29H14 tops the conditional-strength ranking
(Figure [11](#fig:ri-dist){reference-type="ref"
reference="fig:ri-dist"}b). All such variants are reported as
sensitivity analyses; the audit itself keeps the published parameters.

# Causal map: details {#app:stage2}

#### Intervention and readout.

The patched activation $a_h$ is head $h$'s slice of the input to the
attention output projection. For an aligned clean/corrupted pair
$(x_c,x_r)$, replace it on the clean run by the corrupted-run value
$a_h^r$ and measure $$\begin{equation}
\label{eq:importance}
I_h=M(x_c)-M\!\left(x_c;\,a_h\leftarrow a_h^r\right).
\end{equation}$$ Scope P replaces all original prompt positions; Scope F
replaces only the final colon
(Figure [2](#fig:patch){reference-type="ref" reference="fig:patch"}).
The demonstrations and first test fact are identical, so replacements
before the first differing token have no effect under causal attention;
on average, only the last 44 of 282 prompt tokens can be affected. Both
scopes read the first divergent answer token, leaving any teacher-forced
shared prefix unpatched. Scope-F runs store both answer logits
separately: with $\Delta$ denoting patched minus clean logits,
$I_h^F=\Delta\operatorname{logit}(y_r)-\Delta\operatorname{logit}(y_c)$,
separating promotion of the corrupted answer from suppression of the
clean answer.

#### Coverage and head sets.

The screen was computed for all heads on all 178 discovery pairs. Exact
Scope-P patching covered all 1,024 heads on a common subset of 20
randomly chosen families (40 pairs). Exact Scope-P patching on all 178
pairs covered 148 heads: a 92-head set selected on discovery data before
the extension measurements (the top 25 absolute screen estimates; the
top 25 supported pooled-RI scores at each mother anchor, requiring at
least 50 events; the fixed references L3H11, L9H22 and L16H4; and 25
random controls sampled outside that union), the 46 test-only RI
candidates not already in that set, and ten further heads completing the
set of 19 heads with at least one answer-position QK event to a
test-fact source. Exact Scope-F patching on all 178 pairs covered the
union of the 59 RI candidates and the 19 answer-position heads (69
heads). The extension heads were chosen after the common-subset results
were available and before any measurement on the remaining 138 pairs;
their selection uses discovery data only, so intervals for selected
heads describe stability within the run and are not selection-adjusted.
Twenty-five heads with the largest common-subset effects were all
already covered, since the 92-head set contained the screen's top 25.
The common-subset values of the extension heads were taken from the
existing all-head run, not recomputed; a reproduction check on six heads
of one family matched the saved effects exactly.

All-head RI comparisons and layer-matched contrasts use exact importance
on the 40 common pairs; selected-versus-random median comparisons use
the 178-pair measurements available for both groups. RI rank
correlations are reported both overall and after ranking within layer,
using the statistic's eligible head population. Effects are averaged
over orders within family before aggregation; family-bootstrap intervals
are descriptive and are not selection-adjusted.

#### Gates.

Before measurement, each worker verified zero drift of the behavioural
candidate scores under hooks, zero effect of self-patching (at all
positions and at the final position), zero self-attribution, and zero
effect of patching positions before the first differing token, whose
activations are identical under causal attention.

#### Screen calibration.

On the 40 common pairs, with the same pairs for both, the signed
head-ranking Spearman correlation between $\widehat I_h$ and exact $I_h$
is 0.945 (0.921 for absolute means, 0.903 pair-by-head), above the
pre-registered 0.7, so the integrated-gradients fallback was not used.
Among the 148 extension heads on the identical 178 pairs the correlation
is 0.990; comparing the 178-pair screen with the 40-pair exact values
gives 0.698, a difference that mixes approximation error with family
sampling. Rank agreement does not imply per-head accuracy: the screen
under-shoots the strongest heads (same-pair residuals L17H1 2.24, L27H6
0.51, L18H19 0.28 logits), as expected of a local linear approximation.

<figure id="fig:map-app" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figF_screen_layers"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>The causal map. (a) Gradient screen against exact importance
on the same 40 pairs for the 148 heads with full coverage (symmetric-log
axes); the screen under-shoots the strongest head. (b) Exact importance
of all 1,024 heads by layer; orange marks the 25 heads with <span
class="math inline">|<em>I</em><sub><em>h</em></sub>| ≥ 0.3</span> on
the 178 pairs. Strong heads of both signs lie in layers
13–30.</figcaption>
</figure>

#### Behavioural contrast.

Over the 89 discovery families (orders averaged) the clean metric is
5.68 \[5.34, 6.03\], the corrupted metric with the clean sign fixed
$-5.49$ \[$-5.85$, $-5.14$\], and the gap 11.17 \[10.63, 11.73\]; every
clean pair has $M>0$ and every corrupted pair $M<0$. Four of the 178
pairs have a non-empty shared answer prefix. Intervals are percentile
family bootstraps (20,000 resamples), computed after the results were
seen.

#### Stability across subsets and orders.

The all-head exact scan on the 20-family common subset ranks the same
four heads first, with effects 5.742 (L17H1), 3.222 (L27H6), 3.080
(L18H19) and 2.768 (L18H18) against 5.701, 3.071, 3.072 and 2.704 on all
178 pairs.

These differences are all below 5%; after averaging the two fact orders
within each family, all four heads have the expected positive sign in
every discovery family. Over all 1,024 heads the Spearman correlation
between the order-0 and order-1 exact rankings on the common subset is
0.46, a value dominated by the many heads with near-zero effect; within
the 92-head extension set it is 0.73 and among the top 25 it is 0.66,
with 17 heads in both order-specific top-25 lists. All ten strongest
heads have positive mean importance in both orders.

#### Scope F: how the answer-position heads act.

Table [9](#tab:scopeF-app){reference-type="ref"
reference="tab:scopeF-app"} splits the Scope-F effect of the strongest
heads of the 69-head Scope-F set into the change of the clean-answer,
corrupted-answer and source-name logits at the first divergent token.
For the 69 heads the two scopes correlate at 0.79, but the ratio
$I^F_h/I^P_h$ is bimodal: between 0.91 and 1.09 for every head in layers
16--26 with $|I_h|>0.04$, indistinguishable from zero for every such
head in layers 6--11 (L6H24, L8H8, L8H15, L9H16, L10H16, L11H15), and
intermediate for L14H23 (0.12) and L15H3 (0.46). The promotion is not
perfectly specific to the answer: for L18H19, L16H21 and L16H1 the logit
of the source (child) name also rises by 0.3--0.4, so a Scope-F effect
shows that the head raises the name retrieved on the donor run, not that
it retrieves only the correct one.
Figure [13](#fig:scopeF-ratio){reference-type="ref"
reference="fig:scopeF-ratio"} plots the ratio by layer.

<figure id="fig:scopeF-ratio" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figF_scopeF_ratio" data-original-image-title=""
width="\columnwidth"></span>
<figcaption>Share of each head’s effect delivered at the colon:
Scope-F/Scope-P ratio by layer for the 20 heads in the Scope-F set with
<span
class="math inline">|<em>I</em><sub><em>h</em></sub>| &gt; 0.04</span>;
marker size denotes <span
class="math inline">|<em>I</em><sub><em>h</em></sub>|</span>. Heads in
layers 16–26 keep essentially all of their effect under the colon-only
patch (answer-position heads); heads in layers 6–11 lose it (writers at
earlier positions); L14H23 and L15H3 are intermediate.</figcaption>
</figure>

::: {#tab:scopeF-app}
  Head       $I^P_h$   $I^F_h$   $\Delta$ clean   $\Delta$ corr.   $\Delta$ source
  -------- --------- --------- ---------------- ---------------- -----------------
  L18H19        3.07      3.08          $-0.20$          $+2.88$           $+0.37$
  L22H5         0.98      0.98          $-0.06$          $+0.92$           $+0.02$
  L16H21        0.93      0.89          $+0.01$          $+0.90$           $+0.32$
  L16H1         0.87      0.86          $-0.05$          $+0.81$           $+0.33$
  L17H24        0.43      0.43          $-0.03$          $+0.40$           $+0.06$
  L26H23     $-0.37$   $-0.37$          $+0.08$          $-0.29$           $-0.06$
  L19H16     $-0.82$   $-0.83$           $0.00$          $-0.83$           $-0.16$
  L23H15     $-0.95$   $-0.95$          $+0.01$          $-0.94$           $-0.19$

  : Scope-F patch of the eight strongest heads in the Scope-F set:
  importance under both scopes and the mean change of the clean-answer,
  corrupted-answer and source (child) name logits, 89 families, orders
  averaged. Positive heads promote the name retrieved on the donor run
  rather than suppressing the alternative; negative heads lower it.
:::

#### Pre-registered audit criteria.

The selection is judged *sound* if selected heads have systematically
larger importance than layer-matched non-selected heads; *incomplete* if
heads in the top decile of exact importance lie outside the selection,
with the reason recorded (no qualifying QK event, insufficient support,
or a supported but non-positive name gap at both anchors); and
*misleading* if the median importance of selected heads does not exceed
that of the 25 random controls (family bootstrap of the median
difference). The criteria are not mutually exclusive. They were written
for the pooled index and re-applied to the test-only selection, whose
membership was decided after Stage-1 results were inspected; the second
application is exploratory.

#### Rank correlations and verdicts.

Table [10](#tab:corr-app){reference-type="ref" reference="tab:corr-app"}
lists the Spearman correlations between the Stage-1 statistics and exact
importance on the 40 common pairs; support means at least 20 comparisons
in ten families. The two pooled-index heads have $I_h=-0.002$ logits on
all 178 pairs, with the expected sign in 47% and 42% of families,
against a median of 0.0002 for the 25 random controls. Top-25 overlap
with the exact ranking is 0.02 by Jaccard for both RI versions, against
0.52 for the gradient screen. Of the 102 heads in the top decile of
importance, 88 lie outside the test-only selection: 23 have no
qualifying attention event, 22 fall below the support rule, 18 pass with
a non-positive name gap at both anchors, and 25 have a positive gap that
did not reach any top ten. The two strongest misses illustrate two of
these routes: L17H1 ($I_h=5.70$) has only two qualifying attention
events in the whole scan, and L27H6 ($I_h=3.07$) has 85 qualifying
events but a pooled index of zero, so its raw-embedding projection does
not favour the target at those events. Under the pre-registered criteria
the pooled index is *incomplete* (101 of the 102 top-decile heads fall
below even the lenient control threshold of 0.137; the second-strongest
head has index zero) and *misleading* (median importance of its two
heads $-0.0021$ against 0.0002 for the random controls); soundness is
not supported. Re-applied to the test-only selection: incomplete (88 of
102 top-decile heads outside it); the median importance of selected
heads exceeds the controls by 0.003 logits (bootstrap interval \[0.0009,
0.0067\]), 0.03% of the gap, with 56% positive signs against 60% for the
controls; the pooled layer-matched difference is $+0.035$ logits
\[0.023, 0.047\] but positive in only 12 of 28 layers and carried by
four heads in layers 16--22 that the gradient screen had already
selected.

::: {#tab:corr-app}
  Statistic                                                                                           $n$   $\rho$ vs. $I_h$
  ----------------------------------------------------------------------------------------------- ------- ------------------
  Pooled index, first anchor (all heads)                                                            1,024           $+0.135$
  Test-only target, first anchor                                                                      172           $-0.013$
  Test-only name gap, first anchor                                                                    167           $-0.252$
  ranks within layer                                                                                  158           $-0.302$
  without layers 16--18                                                                               144           $-0.278$
  without the 25 strongest heads                                                                      153           $-0.214$
  positive gaps only                                                                                   78           $-0.102$
  self-attention events only                                                                           39           $-0.537$
  Test-only name gap, query fact, first                                                               103           $-0.335$
  Test-only name gap, last anchor                                                                     167           $-0.091$
  Control-name mean (other visible names)                                                             167           $+0.143$
  Gradient screen (Eq. [\[eq:screen\]](#eq:screen){reference-type="ref" reference="eq:screen"})     1,024           $+0.945$

  : Spearman rank correlations with exact Scope-P importance on the 40
  common pairs. Family-bootstrap interval for the main name-gap value:
  $[-0.289,-0.169]$.
:::

# Head characterization: details {#app:stage3}

#### Inventory.

The 105 heads comprise: 48 RI-selected heads with small causal effect;
12 strong positive heads outside the RI selection; 5 strong positive
RI-selected heads; 8 strong negative heads; 7 moderate heads
($0.1\le|I_h|<0.3$); and the 25 random controls of the causal map.
"RI-selected" here means the 59 test-only candidates plus the two heads
of the original pooled index (61 heads). "Strong" means $|I_h|\ge0.3$
logits on the 178-pair Scope-P mean; the thresholds were applied within
the 148 heads that have full exact coverage, so heads outside that set
are not thereby shown to be weak. Membership
(Table [11](#tab:inventory){reference-type="ref"
reference="tab:inventory"}) was fixed from the saved causal map before
any Stage-3 measurement. Each run began with a gate that reproduced
saved Stage-2 effects on probe heads, verified that the reconstructed
attention patterns match the model's attention output within tolerance,
and verified that the joint attention+value replacement equals the
ordinary head patch.

::: {#tab:inventory}
  Group ($n$)                             Heads
  --------------------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Strong positive, not RI-selected (12)   L14H26, L15H25, L17H1, L17H17, L18H18, L19H22, L21H6, L21H18, L21H23, L25H17, L27H6, L30H13
  Strong positive, RI-selected (5)        L16H1, L16H21, L17H24, L18H19, L22H5
  Strong negative (8)                     L13H10, L17H3, L18H24, L19H16, L20H1, L23H15, L26H23, L30H18
  Moderate ($0.1\le|I_h|<0.3$) (7)        L6H24, L7H1, L8H15, L12H17, L14H23, L15H3, L16H31
  RI-selected, small effect (48)          L1H27, L2H21, L2H22, L3H11, L4H0, L5H18, L6H2, L7H18, L7H25, L8H8, L8H16, L9H6, L9H16, L9H17, L9H22, L10H16, L11H3, L11H4, L11H7, L11H15, L12H2, L12H9, L13H1, L13H22, L14H10, L15H0, L15H2, L15H18, L15H28, L15H29, L16H2, L16H4, L16H16, L16H24, L17H5, L18H15, L20H7, L20H14, L21H24, L21H25, L21H27, L23H10, L24H17, L25H18, L26H31, L27H23, L30H3, L31H1
  Random controls (25)                    L0H1, L2H1, L6H12, L9H23, L9H26, L11H20, L11H27, L13H30, L15H27, L18H5, L18H22, L19H20, L21H11, L21H22, L22H15, L22H31, L23H17, L25H11, L25H13, L25H15, L25H20, L25H21, L28H1, L28H15, L29H5

  : The 105-head inventory, fixed from the saved causal map before any
  Stage-3 measurement.
:::

::: {#tab:stage3cov}
  Measurement                            Heads            Inputs
  -------------------------------------- ---------------- ----------------------------
  Scope F vs. P, promotion/suppression   105              178 pairs
  Single-position scan                   105              40 pairs
  Reverse patching                       top 10 $|I_h|$   178 pairs
  Attention profile + controls           105              20 fam. $\times$ 8 prompts
  Output projections                     105              20 families, colon
  Attention--value separation            105              178 pairs
  Weight copying score                   105              314 name-token ids
  Attended-name contrast                 105              160 prompts
  Matched contextual RI                  70               saved Stage-1 events
  Synthetic probes                       105              32 + 32 prompts
  Controlled readout                     2                4 sites, 40 pairs

  : Coverage of the Stage-3 measurements.
:::

#### Single-position scan and reverse patching.

For each common pair and inventory head, the corrupted activation is
inserted at one test-block position at a time, from the first differing
token to the colon; query-fact positions are labelled (mother tokens,
"is", "of", child tokens, period). Reverse patching inserts the clean
activation into the corrupted run for the ten heads with the largest
$|I_h|$, at Scope P and Scope F, to test the intervention in the
opposite direction and its symmetry. A colon-localized effect suggests a
reader role, but does not identify what is read; an earlier-site effect
suggests a writer role. The causal site need not coincide with a
qualifying RI event. Single-position effects need not sum to the
all-position effect, and restoration measures recovery conditional on
the corrupted model, not sufficiency of the restored head alone.

#### Attention profile and controls.

On the 20 common families, both orders, four prompt types are captured:
base, corrupted, reorder and changed-query. Attention at the colon is
summed over the token spans of the query mother, the query child, each
distractor mother and the colon, for every head on every prompt,
independently of the criterion's gate. The changed query asks about the
other answer-side child while leaving the facts unchanged; reorder
changes fact positions, and corruption swaps the two answer-side
mothers. Together these controls distinguish attention that follows the
question from sensitivity to position or mother assignment. Controls are
paired within family. Family means and per-prompt absolute changes are
both reported, since signed means can hide opposite-sign changes.

#### Output projections.

The contextual projection is $o^h_jW_U$ with $o^h_j$ the head's actual
output vector at the colon (before the shared post-attention
normalization of OLMo-2, which is not assigned to heads); the contrast
is the query mother's first-token logit minus the mean over the
distractor mothers' first tokens. The raw-embedding projection uses
$x_jW^h_{OV}W_U$ with $x_j$ the colon's embedding, as in the criterion.
Both are diagnostics of one vector, not attributions of the causal
effect.

#### Attention--value separation.

For each pair, recompute the head's pre-projection activation at the
colon using the attention pattern and values independently from the
clean ($c$) or corrupted ($r$) run. Let $m_{uv}$ be the clean
recipient's answer contrast after inserting $A_u^hV_v^h$ into that
head's colon slice; all subsequent computation stays live. The four
conditions define $$\begin{align*}
I_h^{\mathrm{pattern}}&=m_{cc}-m_{rc},\\
I_h^{\mathrm{value}}&=m_{cc}-m_{cr},\\
I_h^{\mathrm{joint}}&=m_{cc}-m_{rr},\\
\eta_h&=I_h^{\mathrm{joint}}-I_h^{\mathrm{pattern}}-I_h^{\mathrm{value}}.
\end{align*}$$ For every measured head and pair, clean reconstruction
must reproduce the intact readout and joint replacement the ordinary
Scope-F patch. Report signed family means and mean absolute family
interactions; a small mean interaction does not establish separability
if effects cancel across families.

#### Weight copying and attended-name contrast.

The weight score projects each of the 314 first/last test-name token ids
through $W^h_{OV}W_U$ and takes the token's own logit minus the mean
over the other token ids in the set; a 128-token ordinary-word sample
serves as reference. The attended-name contrast uses every test-block
event of the 160 core prompts in which the head's attention argmax lies
inside a visible test name: the contextual logit of that name's first
token minus the mean over the other visible distinct names.
Classification rule, fixed in advance, with four outcomes: with at least
20 events in ten families, *name mover* if the contrast is positive in
$\ge80\%$ of events, *negative name mover* if negative in $\ge80\%$, and
*neither* otherwise; with fewer events or families, *insufficient*.
Variants: last-token anchor, same-role controls, non-self events,
colon-site events, base prompts only.

#### Matched contextual RI.

For every saved Stage-1 event of an inventory head, the raw embedding in
the OV projection is replaced by the head's actual normalized value
input at the same position, with events, controls, anchors and family
weights unchanged. Seventy heads have valid events, of which 68 have
supported name-gap estimates and enter the correlation analysis. This
isolates the embedding choice from every other ingredient of the
criterion. It differs from the actual-output projection above: it still
projects the current position's own input rather than the
attention-weighted values from attended positions.

#### Synthetic probes.

The repeated-sequence test uses 32 prompts, each repeating 32 distinct
random tokens, with 31 next-token predictions in the second copy (992
scored positions). The retrieval test uses 32 prompts containing 16
key--value pairs each and one queried key (32 scored positions). Tokens
come from the 128-token non-name pool. For each head, record attention
mass and argmax at the copy source, and the projected gold-token logit
relative to the other candidate tokens. Report the model's
vocabulary-wide accuracy separately; these probes characterize induction
and retrieval behaviour, not causal participation in the mother-of task.

#### Controlled readout.

For the two heads whose effect localizes to positions inside the query
fact and is not explained by the output projections, the residual after
the head's block is captured at those positions on the 40 common pairs
and injected at a placeholder in one of two readout prompts at the same
layer: an identity completion ("cat $\to$ cat; ...; x $\to$") and a
prompt that asks for the mother of the injected entity ("Fact about a
person: x. Question: Who is the mother of this person? Answer:").
Conditions: intact clean source; source with the head's activation
replaced by its corrupted-run value at the site; the same replacement
from the first differing token to the site; intact corrupted source; the
same token role in the other answer-side fact; no injection. The
statistic is $C=\log p(t_q)-\log p(t_o)$ for the first tokens of the
query mother and the other candidate, with paired shifts from the intact
condition. The source-swap condition is essential: if the readout does
not respond to the full entity swap, a null head effect is
uninformative. A gate verified that injecting a prompt's own residual
leaves the readout unchanged and that a self patch leaves the residual
unchanged. Head dependence is read only from the intact-versus-replaced
contrast of a multi-component residual.

#### Results: where the effect enters.

Of the 25 strong heads, 20 have $I^F_h/I_h\in[0.96,1.07]$, with $I^F_h$
of the expected sign in $\ge97.8\%$ of families for every strong
positive head and $\le4.5\%$ for every strong negative head. The five
others: L17H1 ($I_h=+5.70$, $I^F_h=-0.06$), L15H25 ($+0.68$, $+0.17$),
L14H26 ($+0.40$, $+0.01$; diffuse over template words), L13H10 ($-0.68$,
$-0.03$) and L17H3 ($-0.55$, $-0.40$; mixed). Random controls have
$|I^F_h|\le0.019$ except L18H22 ($+0.089$, equal to its Scope-P effect).
Reverse patching of the ten largest heads recovers the metric by close
to the noising effect at both scopes (L17H1 $+5.21$ vs. 5.70; L18H19
$+3.22$ vs. 3.07; L27H6 $+3.06$; L18H18 $+2.85$; L21H18 $+1.75$; L25H17
$+1.27$; L22H5 $+1.01$; L20H1 $-1.63$; L30H18 $-1.20$; L23H15 $-0.93$);
recovery is conditional on the rest of the corrupted model.
Single-position sums track the all-position effect in the aggregate
(Spearman 0.91 over 105 heads) but with large cancelling residuals for
L17H1 (mean absolute residual per family 0.89 logits, maximum 2.33), so
the profile is not an additive partition. L13H10 mirrors L17H1 with the
opposite sign ("is" $-0.22$, "of" $-0.22$, mother last $-0.14$). L17H1's
family-mean effects by role: mother tokens $+0.05/+1.15/
+1.67$, "is" $+1.77$, "mother" $+0.33$, "of" $+0.36$, child $\approx0$,
period $+1.38$, other facts $-0.96$ in total, question and colon
$\approx0$. The Scope-F effect of every strong colon head is dominated
by the corrupted-answer logit (88--98% of $I^F_h$); L27H6 alone also
lowers the clean answer ($+1.96$ and $+1.10$).

#### Results: attention and output.

Figure [14](#fig:colon-attention){reference-type="ref"
reference="fig:colon-attention"} shows the colon attention of the strong
answer-position heads. Mother-attenders (mass on the query mother):
L27H6 0.58, L21H18 0.60, L23H15 0.51, L22H5 0.34, L19H22 0.31, L21H6
0.30, L19H16 0.21, L26H23 0.21, L20H1 0.17, L18H18 0.14, L18H19 0.13,
L17H24 0.13. Child-attenders: L16H21 0.40, L16H1 0.36. Self-attenders:
L25H17 0.57, L13H10 0.42, L21H23 0.20, L30H13 0.19, L17H17 0.19. Random
controls: mean 0.017 on the query mother. Under a changed query the
shift toward the newly queried fact is positive in 20 of 20 families for
every mother- and child-attender (L21H18 $+1.07$, L23H15 $+0.91$, L27H6
$+0.84$, L16H21 $+0.83$, L16H1 $+0.68$; controls $\le+0.18$); L30H18
moves away ($-0.19$). Reordering and corruption change the family-mean
mass on the query mother by at most 0.11 and 0.085, but per-prompt
absolute changes reach 0.49 (L21H18), so invariance holds on average
only; the model answers 38 of the 40 changed-query prompts. Contextual
projection (query mother minus distractor mothers, first token): L27H6
$+2.99$, L30H13 $+0.76$, L22H5 $+0.41$, L21H6 $+0.33$, L21H18 $+0.25$,
L30H18 $-0.88$; L18H19 $+0.16$, L18H18 $+0.11$, L25H17 $+0.16$, L20H1
$-0.09$, L23H15 $+0.04$, L19H16 $-0.04$, L16H21 and L16H1 within
$\pm0.02$. Raw-embedding projection: all within $\pm0.008$ (Spearman
with $I^F_h$ $-0.07$). Matched contextual RI: Spearman correlation of
the main name gap with importance changes from $-0.26$ to $-0.17$ over
68 supported heads; the raw target statistic changes from $+0.01$ to
$+0.02$ over 70 heads. Weight copying score: Spearman $+0.37$ with
signed $I_h$, $+0.02$ with $|I_h|$; L30H18 $+0.60$, L21H6 $+0.35$,
L26H23 $-0.20$, L27H6 $+0.04$, L18H18 $-0.08$, L17H1 $+0.01$.
Current-token partition of the saved RI events: for L18H19, L21H18 and
L21H6 the name gap is $-0.023$, $-0.017$ and $-0.025$ when the current
token is a control name and within $\pm0.004$ otherwise.

<figure id="fig:colon-attention" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figG_colon_attention"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>Colon attention of the strong answer-position heads over the
query mother, query child, the three distractor mothers and the colon
itself (base prompts, 20 common families); labels report Scope-F
importance. Mother-attenders occur with both effect signs, the two
child-attenders select the queried child’s second mention, and
colon-attenders put most mass on the colon.</figcaption>
</figure>

<figure id="fig:stage3-app" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figG_attention_values"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>(a) Changed-query control: family-mean shift of attention
toward the newly queried mother (<span
class="math inline"><em>x</em></span>) and child (<span
class="math inline"><em>y</em></span>) for all 105 heads.
Mother-attenders move along <span class="math inline"><em>x</em></span>,
child-attenders along <span class="math inline"><em>y</em></span>,
random controls (crosses) barely move; L30H18 moves away. (b)
Attention–value separation at the colon for the twelve strongest
answer-position heads: the ordinary Scope-F patch beside the values-only
and pattern-only replacements (178 pairs).</figcaption>
</figure>

#### Results: attended-name labels and fingerprints.

At the last-token anchor, none of the six first-token name movers or the
negative name mover retains its label. Name movers: L27H6 (96% of 768
events), L17H24 (94%), L26H31 (91%), L18H19 (84%), L22H5 (82%), L21H6
(82%); negative name mover: L20H1 (17% positive). L21H18 (79%), L19H22
(78%), L19H16 (77% negative) and L26H23 (79% negative) miss the rule;
L23H15 favours the attended name in 64% of 2,428 events; L30H18 in 67%
(mean $+0.60$) while attending to the distractor. L18H19's label rests
on earlier-site events (its colon-site subset is insufficient). L26H31
puts 0.09 of its colon mass on the query mother and selects the answer
in 13% of its colon-site name events. On the synthetic probes (model
accuracy 100% and 96%), retrieval-type attention is shown by L19H16
(0.97), L21H18 (0.88), L23H15 (0.84), L18H18 (0.56), L16H1 (0.47) and
induction-type attention by L23H15 (0.66), L21H6 (0.64), L16H1 (0.45);
L19H16, L26H23 and L23H15 attend to the source without promoting it.
L27H6, L17H1, L20H1 and L30H18 show no fingerprint. Copying is neither
necessary nor sufficient for a strong effect: L18H18 ($I_h=2.70$) and
L27H6 are not copiers by weight ($-0.08$, $+0.04$); L30H18, the
strongest weight copier ($+0.60$), attends to the distractor and hurts
the answer; L26H31 promotes the attended name as strongly as L27H6 (mean
contrast $+2.09$) but rarely attends to the answer and has
$I^F_h=+0.027$.

#### Results: attention--value separation and readout.

Values: L18H19 $+3.16$ (pattern $+0.08$), L27H6 $+2.59$ ($+0.27$),
L18H18 $+2.88$ ($+0.11$), L21H18 $+1.70$ ($+0.01$), L25H17 $+1.29$
($-0.02$); routing terms for the negative heads L20H1 $-0.22$, L30H18
$-0.18$, L19H16 $-0.16$. Mean absolute family-level interaction: 1.25
logits for L27H6 (maximum 4.58), 0.81 for L18H18, 0.54 for L18H19.
Readout (Table [13](#tab:readout-app){reference-type="ref"
reference="tab:readout-app"}): at L17H1/"is" the identity contrast falls
from $+0.515$ to $+0.056$ after head replacement ($-0.006$ for the
corrupted source, $+0.002$ for the other-fact control), negative in
every family (median $-0.459$, range $[-0.70,-0.23]$); at the period
$-0.041$ vs. $-0.046$. The head-site and head-span conditions are
identical by construction at this capture layer. Absolute decoding
probabilities are low (query-mother first token 0.037% at "is" under
identity readout, against 0.004% without injection).

::: {#tab:readout-app}
  Writer   Site         Readout            $C$       swap       head   neg. 
  -------- ------------ ----------- ---------- ---------- ---------- -------
  L17H1    is           identity      $+0.515$   $-0.521$   $-0.458$   20/20
  L17H1    period       identity      $+0.288$   $-0.046$   $-0.041$   20/20
  L17H1    is           mother-of     $+0.255$   $-0.009$   $-0.010$   17/20
  L17H1    period       mother-of     $+0.265$   $-0.027$   $-0.023$   20/20
  L15H25   child last   identity      $+0.182$   $+0.020$   $+0.006$    8/20
  L15H25   child last   mother-of     $+0.243$   $+0.002$   $-0.000$    9/20
  L15H25   is           identity      $+0.294$   $-0.151$   $+0.003$    7/20

  : Controlled readout, 20 families, orders averaged. $C$: intact-clean
  contrast $\log p(t_q)-\log p(t_o)$; swap and head: paired change under
  the intact corrupted source and under head replacement at the site;
  neg.: families with a negative head change.
:::

For L15H25 at the child-last site, neither readout responds materially
to the full source swap (identity: $+0.020$; mother-of: $+0.002$), so
its null response to head replacement is uninformative and the content
written at this site remains unresolved.

::: table*
  -------- -------- --------- ----------- -------------------------------- ----------------- --------- ---------- ------------ --------------------------------------
  Head     Group        $I_h$   $I_h^{F}$ Attention at colon                       Ctx. out.      Copy         KV         Rep. Attended-name label
                                          Q-mother/Q-child/D-mother/self     $\Delta$ mother   weights   src att.   src argmax (fraction positive; events/families)
  L17H1    S+         $+5.70$     $-0.06$ 0.05/0.06/0.02/0.04                        $-0.00$   $+0.01$       0.00         0.00 neither (77%; 3469/20)
  L18H19   S+RI       $+3.07$     $+3.08$ 0.13/0.05/0.03/0.00                        $+0.16$   $+0.12$       0.16         0.02 NM (84%; 646/20)
  L27H6    S+         $+3.07$     $+3.06$ 0.58/0.00/0.05/0.00                        $+2.99$   $+0.04$       0.05         0.01 NM (96%; 768/20)
  L18H18   S+         $+2.70$     $+2.72$ 0.14/0.03/0.03/0.00                        $+0.11$   $-0.08$       0.27         0.06 neither (73%; 332/20)
  L21H18   S+         $+1.70$     $+1.70$ 0.60/0.00/0.10/0.00                        $+0.25$   $+0.11$       0.46         0.39 neither (80%; 1656/20)
  L25H17   S+         $+1.27$     $+1.27$ 0.00/0.00/0.00/0.57                        $+0.15$   $-0.01$       0.01         0.00 neither (43%; 625/20)
  L22H5    S+RI       $+0.98$     $+0.98$ 0.34/0.01/0.08/0.01                        $+0.41$   $+0.04$       0.18         0.22 NM (82%; 730/20)
  L16H21   S+RI       $+0.92$     $+0.89$ 0.01/0.40/0.01/0.00                        $-0.01$   $-0.02$       0.04         0.00 neither (60%; 551/20)
  L16H1    S+RI       $+0.87$     $+0.86$ 0.01/0.36/0.02/0.01                        $-0.00$   $+0.01$       0.28         0.45 neither (50%; 669/20)
  L21H6    S+         $+0.70$     $+0.69$ 0.29/0.01/0.04/0.00                        $+0.33$   $+0.35$       0.12         0.64 NM (81%; 858/20)
  L15H25   S+         $+0.68$     $+0.17$ 0.03/0.06/0.02/0.01                        $-0.00$   $+0.04$       0.00         0.00 neither (42%; 1797/20)
  L30H13   S+         $+0.63$     $+0.63$ 0.02/0.01/0.01/0.19                        $+0.76$   $-0.03$       0.03         0.00 neither (37%; 917/20)
  L21H23   S+         $+0.61$     $+0.60$ 0.05/0.01/0.02/0.20                        $+0.07$   $-0.00$       0.04         0.00 insuff. (25%; 20/8)
  L17H17   S+         $+0.53$     $+0.51$ 0.04/0.05/0.03/0.19                        $+0.00$   $+0.01$       0.01         0.00 neither (39%; 103/19)
  L17H24   S+RI       $+0.43$     $+0.43$ 0.13/0.03/0.05/0.02                        $+0.03$   $+0.18$       0.02         0.00 NM (94%; 1073/20)
  L19H22   S+         $+0.43$     $+0.42$ 0.31/0.06/0.07/0.01                        $+0.07$   $+0.17$       0.04         0.01 neither (78%; 1127/20)
  L14H26   S+         $+0.40$     $+0.01$ 0.01/0.05/0.00/0.01                        $+0.01$   $-0.00$       0.00         0.00 neither (54%; 959/20)
  L26H23   S$-$RI     $-0.37$     $-0.37$ 0.21/0.02/0.03/0.01                        $-0.04$   $-0.20$       0.16         0.15 neither (21%; 1670/20)
  L18H24   S$-$       $-0.53$     $-0.57$ 0.06/0.06/0.04/0.02                        $+0.00$   $-0.06$       0.02         0.00 neither (34%; 70/15)
  L17H3    S$-$       $-0.55$     $-0.40$ 0.06/0.10/0.02/0.04                        $+0.00$   $-0.03$       0.08         0.00 neither (44%; 319/20)
  L13H10   S$-$       $-0.68$     $-0.03$ 0.00/0.01/0.00/0.42                        $+0.00$   $-0.03$       0.00         0.00 neither (43%; 700/20)
  L19H16   S$-$RI     $-0.82$     $-0.83$ 0.21/0.04/0.03/0.01                        $-0.04$   $-0.09$       0.55         0.36 neither (23%; 1906/20)
  L23H15   S$-$RI     $-0.95$     $-0.95$ 0.51/0.01/0.05/0.00                        $+0.04$   $+0.09$       0.49         0.66 neither (64%; 2428/20)
  L30H18   S$-$       $-1.06$     $-1.06$ 0.05/0.01/0.11/0.00                        $-0.88$   $+0.60$       0.05         0.00 neither (67%; 2528/20)
  L20H1    S$-$       $-1.56$     $-1.53$ 0.17/0.05/0.01/0.02                        $-0.09$   $-0.17$       0.02         0.00 neg. NM (17%; 437/20)
  -------- -------- --------- ----------- -------------------------------- ----------------- --------- ---------- ------------ --------------------------------------
:::

# Stage 4: rules, rosters and configurations {#app:stage4}

S4.1--S4.3 and both discovery and held-out evaluation of the revised
final S4.4 experiment (S45 in the implementation) have been executed.
The latter replaces the earlier S4.4--S4.5 retention/reduction design.
The design is in Appendix [14.1](#app:final-ri){reference-type="ref"
reference="app:final-ri"}; discovery results and frozen 87-family
validation are reported separately below with their actual populations.

#### Route semantics.

The source's pre-projection slice $z=AV$ is replaced by the donor value;
the shared attention-output normalization stays live; all other
residual-branch increments between source and receiver are frozen after
their normalization in the hybrid run. The receiver's channel is
recaptured from the hybrid residual and injected into a fresh recipient
run: for fact-site K/V routes at the fact positions, with only the
receiver's original colon output row released; for Q routes at the
colon. In the recipient run all later computation is live. A retained
route supports transmission through the declared source, site and
receiver channel in this frozen background; its effect is not an
additive share of the source's full-model importance. Separately
measured routes do not establish their composition into a chain. Effects
are averaged over orders within family and then over families;
20,000-draw family-bootstrap intervals are descriptive after selection.
*Retention rule*, fixed in advance with $\tau=0.10$ logits: coherent if
$|\bar I|\ge\tau$ with the same sign in at least 70% of families;
heterogeneous if $\mathbb{E}_f|I_f|\ge\tau$ and $|I_f|\ge\tau$ in at
least 20% of families. The rule orders measurements; it is not a
significance test.

#### S4.1 sources, receivers, controls.

Sources with fact-site masks: L17H1 (query-fact mother tokens, "is",
period, and their union), L15H25 (child-last, query sentence), L13H10
and L17H3 (sentence masks), and two coverage candidates outside the
inventory, L13H18 and L24H19, added after a coverage check of the exact
common-subset map. Colon sources: the strong answer-position heads.
Receivers are a bounded candidate list built from the Stage-3 attention
profiles and coverage rules, not every later head: for fact-site
sources, later heads attending to the site, channels K and V; for colon
sources, later heads on Q and a residual bypass that tests whether the
source's write reaches the answer without passing through a tested head.
Initial measurement on the 40 common pairs; retained routes extended to
all 178 pairs in both donor directions under a cap of 96 extended
configurations. Controls by intervention type: random receivers for head
routes; wrong-site masks for fact-site routes; self patches;
reverse-direction routes; the ordinary single-head patch as comparator;
order and prefix sensitivity per route. Counts of retained routes are
reported with the results.

#### S4.2 groups.

::: center
  ID   Source / site         Members (all replaced at the colon)
  ---- --------------------- -----------------------------------------------------
  O1   L13H10 / sentence     L16H1, L17H3, L18H18, L18H19
  O2   L15H25 / child-last   L16H1, L16H21
  O3   L17H1 / mother        L18H18, L18H19, L18H24, L19H16, L20H1, L22H5, L27H6
  O4   L17H3 / colon         L18H18, L18H19
  I1   L27H6 / Q             L18H18, L18H19, L19H16, L20H1, L23H15, L25H17
  I2   L21H23 / KV           L17H24, L18H18, L18H19, L19H16, L20H1
  I3   L30H18 / Q            L18H18, L18H19, L20H1, L23H15, L25H17
  I4   L18H18 / Q            L16H1, L16H21, L17H3
:::

Outgoing groups (O) collect the receivers of one source; incoming groups
(I) the sources of one receiver. With $I(S)$ the importance under
simultaneous replacement of group $S$, whole-group, singleton and
group-minus-member patches give $$\begin{align*}
N(S)&=I(S)-\sum_{h\in S}I(h),\\
\delta_h(S)&=I(S)-I(S\setminus\{h\}).
\end{align*}$$ $N(S)$ measures nonadditivity and $\delta_h(S)$ the
member's contribution conditional on the other group members being
replaced. Singles, whole groups and group-minus-one masks deduplicate to
49 base masks under a 128-configuration cap; selected group contrasts
and two blocking contrasts were extended to all 89 families. Receiver
blocking: L17H1's clean activation restored in the corrupted run at its
source masks with the O3 members clamped to their corrupted colon
outputs; L15H25 restored with the O2 pair clamped; self-clamp and
reverse controls. The loss of source restoration under receiver clamping
measures dependence on receiver response; overlapping paths and
interactions prevent an exclusive partition into mediation shares.
Mean-replacement sensitivity was run for the extended contrasts, not for
every configuration.

#### S4.2 RI participation.

RI31: the RI-selected heads retained by the rule "best original RI rank
$\le3$ in any ranking, or $\max(|I_h|,|I^F_h|)\ge0.1$" (23 by rank, 8 by
effect). It contains the eight strong RI heads (positive: L16H1, L16H21,
L17H24, L18H19, L22H5; negative: L19H16, L23H15, L26H23) and 23 residual
heads. Reference cohorts: four sets of 23 non-selected heads matched to
the residual set in size and layer composition; they are compared with
the residual 23, not with all 31. Weakened background $K$: the smallest
prefix of the strongest colon readers whose replacement leaves the
symmetric clean-minus-corrupted gap inside a pre-specified range, with a
fallback rule if no prefix qualifies; the rule selected $K=\{$L21H18$\}$
(66% of the gap left). All 31 candidates were measured on the common
panel; 14 were extended (the eight strong heads and six retained
conditional candidates: L6H24, L8H15, L14H23, L15H3, L16H31, L20H7).
Means are indexed by head, fact slot, token role and within-name bucket,
computed over the 89 families with the evaluated family excluded. Backup
test: L26H31, a name mover with negligible standalone effect, in five
backgrounds with upstream or downstream members replaced; the same
retention rule applies.

#### S4.1 results: retained routes.

Of the 96 configurations extended to all 89 families, 95 retain under
noising and 93 under restoration; 94 and 92 retain on the additional 69
families alone, and all 96 mean signs agree between the two donor
directions (Figure [16](#fig:routes-app){reference-type="ref"
reference="fig:routes-app"}a).
Table [14](#tab:routes-app){reference-type="ref"
reference="tab:routes-app"} lists the main retained routes. All 55
controls are smaller than their target in mean absolute family effect
(Figure [16](#fig:routes-app){reference-type="ref"
reference="fig:routes-app"}b); 7 of 17 other-site controls nevertheless
retain under the rule, and 0 of 38 random receivers. Controls are not
norm-matched. Nineteen retained core configurations were not extended
(cap), and L13H18 has no retained route among 138 localized direct
configurations.

<figure id="fig:routes-app" data-latex-placement="h">
<span class="image placeholder" data-original-image-src="figH_routes"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>Route mapping on the 89 discovery families. (a) Noising
against restoration effect of the 96 extended configurations, by
receiver type. (b) Mean absolute family effect of each control against
that of its target route (log axes); every control lies below the
diagonal.</figcaption>
</figure>

::: {#tab:routes-app}
  Source / site         Receiver / channel     $I_{89}$   $J_{89}$   $I_{69}$
  --------------------- -------------------- ---------- ---------- ----------
  L17H1 / union         L18H18 / V             $+3.400$   $+3.335$   $+3.391$
  L17H1 / union         L18H19 / V             $+2.074$   $+2.054$   $+2.057$
  L17H1 / union         L18H24 / V             $-1.004$   $-1.070$   $-0.988$
  L17H1 / union         L22H5 / V              $+0.898$   $+0.865$   $+0.882$
  L17H1 / union         L20H1 / V              $-0.582$   $-0.576$   $-0.567$
  L17H1 / union         L19H16 / V             $-0.247$   $-0.225$   $-0.243$
  L15H25 / child-last   L16H1 / V              $+0.182$   $+0.187$   $+0.182$
  L15H25 / child-last   L16H21 / V             $+0.210$   $+0.223$   $+0.215$
  L16H1 / colon         L18H18 / Q             $+0.372$   $+0.370$   $+0.367$
  L16H21 / colon        L18H18 / Q             $+0.349$   $+0.407$   $+0.348$
  L18H18 / colon        L27H6 / Q              $+0.662$   $+0.663$   $+0.674$
  L18H19 / colon        L27H6 / Q              $+0.731$   $+0.742$   $+0.736$
  L13H10 / sentence     L16H1 / V              $-0.175$   $-0.175$   $-0.172$
  L13H10 / sentence     L17H3 / V              $+0.150$   $+0.139$   $+0.151$
  L14H26 / sentence     L16H1 / V              $+0.136$   $+0.140$   $+0.139$
  L17H3 / colon         L18H18 / Q             $-0.158$   $-0.172$   $-0.155$
  L18H19 / colon        MLP 19                 $+0.532$   $+0.554$   $+0.514$
  L18H19 / colon        bypass                 $+0.754$   $+0.770$   $+0.748$
  L18H18 / colon        bypass                 $+0.547$   $+0.562$   $+0.546$
  L24H19 / colon        bypass                 $+0.165$   $+0.165$   $+0.160$

  : Main retained routes: noising ($I$) and restoration ($J$) effects on
  the 89 discovery families and noising on the additional 69 families
  outside the common subset. "Union" is the union of L17H1's
  mother-token, "is" and period masks; "bypass" tests the source's write
  reaching the answer without passing through a tested head.
:::

#### S4.2 results: blocking, RI participation, backup.

Table [15](#tab:blocking-app){reference-type="ref"
reference="tab:blocking-app"} gives the receiver-blocking results and
Table [16](#tab:ri31-app){reference-type="ref" reference="tab:ri31-app"}
the RI-participation audit for the 14 extended candidates. The
whole-group effects of the eight G1 groups overlap heavily because the
L18H18/L18H19 pair belongs to most of them (O1, O3, O4, I1, I2, I3);
group differences persist after same-order normalization, and the
full-model gap itself depends on fact order (14.6 vs. 7.8 logits). Donor
and mean replacement can give very different candidate accuracies (O3:
36.0% vs. 97.8% clean). In the backup test, L26H31 stays below the
retention rule in all five backgrounds (intact, $-$L21H18, $-\{$L21H18,
L22H5, L21H6$\}$, the same plus L27H6, and $-$L27H6: mean effect at most
$+0.024$ logits, mean absolute at most $0.040$).

::: {#tab:blocking-app}
  Source   Mask             $n_f$      $J_W$        $B$   $B/J_W$
  -------- -------------- ------- ---------- ---------- ---------
  L17H1    writer-union        89   $+5.358$   $+4.790$     89.4%
  L17H1    sentence            89   $+5.972$   $+5.351$     89.6%
  L15H25   child-last          20   $+0.457$   $+0.442$     96.7%
  L15H25   sentence            20   $+0.451$   $+0.442$     97.8%
  L13H10   sentence            20   $-0.330$   $-0.429$    129.9%
  L17H3    colon               20   $-0.417$   $-0.304$     72.9%

  : Receiver blocking. $J_W$: restoration obtained by restoring the
  source's clean activation at its mask in the corrupted run; $B$: the
  part of it lost when the source's receivers (O3 for L17H1, O2 for
  L15H25, O1 for L13H10, O4 for L17H3) are clamped to their corrupted
  colon outputs. Ratios are not exclusive mediation shares.
:::

::: {#tab:ri31-app}
  Head        $I_{K}$    $J_{K}$   $I_{K,\mu}$   $\mathbb{E}_f|I_{K,f}|$   $I_K-I_0$
  -------- ---------- ---------- ------------- ------------------------- -----------
  L18H19     $+3.006$   $+3.129$      $+1.689$                     3.006    $-0.066$
  L22H5      $+1.088$   $+1.115$      $+0.304$                     1.088    $+0.105$
  L16H21     $+0.893$   $+0.999$      $+0.999$                     0.899    $-0.032$
  L16H1      $+0.798$   $+0.769$      $+1.688$                     0.803    $-0.070$
  L17H24     $+0.413$   $+0.413$      $+0.324$                     0.413    $-0.020$
  L19H16     $-0.874$   $-0.893$      $-0.448$                     0.874    $-0.054$
  L23H15     $-0.933$   $-0.902$      $-0.394$                     0.933    $+0.014$
  L26H23     $-0.367$   $-0.354$      $-0.167$                     0.376    $+0.001$
  L8H15      $+0.284$   $+0.316$      $+0.130$                     0.379    $+0.005$
  L14H23     $-0.215$   $-0.201$      $-0.134$                     0.246    $-0.003$
  L16H31     $-0.173$   $-0.167$      $-0.261$                     0.174    $+0.014$
  L15H3      $-0.104$   $-0.086$      $-0.154$                     0.115    $+0.002$
  L6H24      $+0.096$   $+0.103$      $-0.065$                     0.241    $-0.005$
  L20H7      $+0.032$   $+0.099$      $-0.177$                     0.318    $+0.058$

  : RI participation on the weakened background $K=\{$L21H18$\}$, 89
  families: noising ($I_K$) and restoration ($J_K$) with the corrupted
  donor, noising with mean replacement ($I_{K,\mu}$), mean absolute
  family effect, and the change from the intact-model single
  ($I_K-I_0$). Top: the eight strong RI heads; bottom: the six
  conditional candidates retained for extension. Conditional
  participation is not a newly unmasked backup for any head.
:::

#### S4.2 results: residual RI group and nonadditivity. {#app:s42-residual-results}

RI23 is RI31 minus its eight strong members, not the 23 heads originally
selected by rank. On 89 families, its intact-background donor-noising
mean is $+0.116$, with mean absolute family effect $0.594$; the four
layer-matched reference cohorts average $0.215$ in absolute effect. The
paired magnitude excess is $0.379$ (95% descriptive interval
$[0.280,0.488]$). Under the weakened background these values are
$+0.185$, $0.656$ and $0.197$, respectively. Mean replacement gives
signed/absolute effects of $-0.633/0.954$ intact and $-0.636/0.979$
weakened. References are non-null and not matched for perturbation norm;
the comparison supports collective influence of the RI remainder, not
RI-specific semantic selectivity.

The L18H18/L18H19 pair has joint noising effect $5.630$ on 89 families.
Incoming groups I1--I3 have nonadditivity $+1.575,+1.585,+0.981$,
whereas the pair's mean is $-0.173$ with mean absolute family
interaction $0.564$. On the core panel, interactions between positive
and negative partitions are $2.285,1.754,1.831$ for I1--I3. These
overlapping groups therefore cannot be summed as independent circuit
contributions. For example, L16H1's conditional effect is $0.961$ in the
child-reader pair but $0.197$ in O1, where downstream readers are also
replaced. Such context dependence motivates the retained-set test rather
than identifying a sufficient group from large singleton effects.

#### S4.3 local expansion.

Local mediator tests release only specified MLP branches or whole blocks
from the frozen hybrid. Releasing a block frees both its normalized
attention and MLP branches; a block effect is not attributed to every
constituent head. Composed chains instead release named intermediate
head slices while clamping the other heads in their attention layers,
then recompute the shared output projection and normalization. Their MLP
branches remain frozen unless explicitly released. The resulting
receiver channel is tested in a fresh recipient as in S4.1. Sources:
L13H18 (three retained slot masks; receivers L16H1, L16H21, L18H18,
L18H19; K and V; MLP13, MLP14 or both released), L8H15 (two sentence
masks; same receivers; MLP8/MLP9), L16H31, L20H7 and L26H23 (colon
sources; Q receivers in later layers and residual continuation; local
MLPs or blocks released), and two chains (L15H25 $\to$ {L16H1, L16H21}
$\to$ Q of L18H18/L18H19; L17H1 $\to$ {L18H18, L18H19} $\to$ Q of the
late readers). The initial design comprised 249 configurations plus 56
direct comparators before deduplication, on the 40 common pairs. Sixteen
selected contrasts and 11 direct prerequisites were extended to 178
pairs in both directions. For composed fact-to-colon-Q chains, the
no-intermediate comparator is structurally zero: a fact-site change
cannot reach the later colon query when every intervening component is
frozen. Their route effects are therefore reported directly; the zero
subtraction supplies no additional evidence of mediation.

#### S4.3 results: composed chains and local processing. {#app:s43-results}

All 16 selected contrasts retain on 89 families in both donor
directions; 14 are coherent and two retain only through the
heterogeneous rule. The same classification holds on the additional 69
discovery families. The principal composed effects are in
Table [17](#tab:s43-chains){reference-type="ref"
reference="tab:s43-chains"}. These are measurements of complete
selectively released paths, not concatenations of independent edges. For
these chains the matched no-intermediate comparator is zero on every
pair, so the reported increment equals the raw effect.

::: {#tab:s43-chains}
  Composed path / source mask                                   $I$     $J$
  --------------------------------------------------------- ------- -------
  L17H1 $\to$ {L18H18,L18H19} $\to$ L27H6 Q / union           1.635   1.511
  L17H1 $\to$ L18H18 $\to$ L27H6 Q / union                    0.876   0.809
  L17H1 $\to$ L18H19 $\to$ L27H6 Q / union                    0.430   0.409
  L17H1 $\to$ pair $\to$ L21H6 Q / union                      0.208   0.194
  L17H1 $\to$ pair $\to$ L22H5 Q / union                      0.194   0.170
  L17H1 $\to$ pair $\to$ L21H18 Q / union                     0.159   0.174
  L15H25 $\to$ {L16H1,L16H21} $\to$ L18H18 Q / child-last     0.172   0.179
  L15H25 $\to$ child pair $\to$ L18H18 Q / sentence           0.217   0.222
  L15H25 $\to$ child pair $\to$ L18H19 Q / sentence           0.151   0.149

  : S4.3 composed routes, 89 families. $I$: noising; $J$: restoration.
  "Pair" denotes the layer-18 pair and "child pair" the layer-16 pair.
  Union is the query mother's tokens, "is" and period.
:::

The leading L17H1 chain has 98.9% family-level agreement between
donor-direction signs. Its noising effect differs by fact order ($2.220$
versus $1.050$). The child-last L15H25 chain has 96.6% directional sign
agreement. Both chains include RI intermediates, while their fact-site
sources are non-RI.

The 11 extended direct prerequisites also reveal L8H15
sentence-to-L18H18/L18H19 V effects of $+0.465/+0.460$ in noising, with
coherent effects in both directions. Releasing MLP9 reduces these
positive routes: the noising increments are $-0.145/-0.147$, leaving raw
effects approximately $+0.321/+0.313$. Releasing MLP8 and MLP9 gives
increments $-0.160/-0.131$ and raw effects approximately
$+0.306/+0.329$. Thus these negative increments describe attenuation,
not negative raw routes. Restoration increments have the same signs.

For L26H23, releasing MLP26+block27+block28 changes its residual
continuation by $-0.183/-0.166$ (noising/restoration), with 91.0%
directional sign agreement. Block27 alone gives increments
$-0.089/-0.084$ and retains only as heterogeneous; it includes L27H6, a
candidate component not isolated by this whole-block test. L20H7's
full-release increments are $-0.069/-0.061$, with only 47.2% directional
sign agreement: heterogeneous sensitivity rather than a stable negative
route. No tested local route for L13H18 or L16H31 passed the initial
rule. Six retained contrasts omitted by the extension cap remain
core-panel findings.

## Final experiment: RI participation in measured structures {#app:final-ri}

The design separates three questions: whether a retained set preserves
the task, whether an RI head changes that behaviour, and whether it
changes communication along a measured route. All rosters below are
fixed from discovery evidence before the new measurements. There is no
greedy reduction or expansion beyond the declared fallback.
Table [20](#tab:final-coverage){reference-type="ref"
reference="tab:final-coverage"} gives the measurement order and
populations; every family contributes both fact orders.

#### Structures and broad backgrounds.

Table [\[tab:final-structures\]](#tab:final-structures){reference-type="ref"
reference="tab:final-structures"} distinguishes each retained head set
from its primary route anchor. Node retention keeps those heads live at
every original test-block position; route interventions use the narrower
historical source mask and receiver channel. All five sets lie inside
$C_{33}$. $T_3\subset T_1$ is a nested comparison, not independent
replication; $T_3$ and $T_5$ contain no head from the original 59-head
RI selection. $T_2$ includes both layer-18 receivers, but its anchor
tests only the L18H18 branch. $T_4$ and $T_5$ are direct two-head
routes; $T_5$ is an opposing route and need not solve the task alone.

::: table*
  ID      Retained heads                          Primary route anchor
  ------- --------------------------------------- -----------------------------------------------------------------------
  $T_1$   L17H1, L18H18, L18H19, L27H6            L17H1 / writer-union $\to$ live {L18H18, L18H19} $\to$ L27H6 / Q
  $T_2$   L15H25, L16H1, L16H21, L18H18, L18H19   L15H25 / query-child-last $\to$ live {L16H1, L16H21} $\to$ L18H18 / Q
  $T_3$   L17H1, L18H18, L27H6                    L17H1 / writer-union $\to$ live {L18H18} $\to$ L27H6 / Q
  $T_4$   L8H15, L18H18                           L8H15 / query sentence $\to$ L18H18 / V (no MLP release)
  $T_5$   L20H1, L27H6                            L20H1 / colon $\to$ L27H6 / Q
:::

$C_{33}$ inherits the earlier broad candidate set: L6H24, L8H15, L13H10,
L13H18, L14H23, L14H26, L15H3, L15H25, L16H1, L16H21, L16H31, L17H1,
L17H3, L17H17, L17H24, L18H18, L18H19, L18H24, L19H16, L19H22, L20H1,
L20H7, L21H6, L21H18, L21H23, L22H5, L23H15, L24H19, L25H17, L26H23,
L27H6, L30H13 and L30H18. The only fallback, $C_{50}$, adds the
remaining 17 RI31 heads: L1H27, L8H8, L9H6, L9H17, L11H4, L11H7, L12H2,
L16H4, L16H16, L16H24, L17H5, L21H25, L23H10, L25H18, L26H31, L27H23 and
L30H3. The chosen broad background is denoted $C$ throughout this
appendix.

#### RI candidates and references.

The six candidates in
Table [18](#tab:final-candidates){reference-type="ref"
reference="tab:final-candidates"} span existing RI rankings and
diagnostics; they are not selected using the new outcomes. Five (all
except L9H16) failed the S4.2 G2 retention rule on its weakened
background, so the new test concerns different structures and
backgrounds. L18H19 and L8H15 serve as functional references on the core
panel, without entering the six-candidate route screen. Define $R(C)$ as
the members of $C$ in the original 59-head test-only RI selection; the
two pooled-only heads are not added.

::: {#tab:final-candidates}
  Head     Selection reason and qualification
  -------- -----------------------------------------------------------------------------------------------
  L1H27    Query-fact first-anchor target RI rank 1; broad support.
  L11H4    Query-fact first-anchor name-gap rank 2; survives concentration checks.
  L17H5    Query-fact first-anchor name-gap rank 1; nonlocal attention, template dependence.
  L23H10   All-facts first-anchor name-gap rank 1; current-target alternative.
  L25H18   All-facts last-anchor name-gap rank 2; limited query-fact support.
  L9H16    Previously prioritized last-anchor candidate; broad support, order sensitivity; outside RI31.

  : Additional RI candidates fixed before the final experiment.
:::

#### Replacement background.

Excluded head-output slices are replaced at every original test-block
position; embeddings, MLPs, shared normalization, demonstrations and
answer-prefix computation remain live
(Figure [17](#fig:retained){reference-type="ref"
reference="fig:retained"}). The empty mask replaces all 1,024 heads at
those positions. Role means use syntactic slot, role and token bucket
(single, first, interior, last), never identity, query relevance or the
answer. The reference bank covers all heads and is family-balanced over
the 89 discovery families, original queries, both fact assignments and
orders. Discovery uses leave-one-family-out means; held-out uses the
fixed full discovery bank. Changed-query and held-out examples never
update it.

<figure id="fig:retained" data-latex-placement="t">

<figcaption>Candidate toggle in a fixed retained background. White:
live; grey: role-mean replacement; orange: the candidate restored to
live computation. The two retained configurations differ only in <span
class="math inline"><em>h</em></span>. One is the chosen <span
class="math inline"><em>C</em></span>. Non-head computation,
demonstrations and answer prefixes remain live.</figcaption>
</figure>

#### A. Behaviour and background selection.

On the 20 common families, cross original/swapped mother assignments
with the original/alternative queried child. The cells
$x_{00},x_{10},x_{01},x_{11}$ have gold answers $a,b,b,a$ and retain the
fixed $a-b$ scoring sign. Rebuild tokenization, offsets, masks and the
shared answer prefix for changed queries; do not filter out new-cell
errors. For configuration $D$, define $$\begin{align}
b_f(D)&=\tfrac14\,\mathbb{E}_o\bigl[
 M_D(x_{00})-M_D(x_{10}) \notag\\
&\hspace{3.7em}{}-M_D(x_{01})+M_D(x_{11})\bigr], \notag\\
B_D&=\mathbb{E}_f b_f(D). \label{eq:bd}
\end{align}$$ Report the four margins and accuracies, both fact-axis and
both query-axis contrasts, and full-vocabulary first-token outcomes.
$B_D$ alone does not establish correct behaviour in every cell; ratios
to the full-model contrast require a stable denominator.

Evaluate full, empty, $T_1$--$T_5$ and $C_{33}$; evaluate $C_{50}$ only
if $C_{33}$ fails. With $g_f(D)=\mathbb{E}_o[M_D(x_{00})-M_D(x_{10})]$,
the fidelity guards are $$\begin{align*}
F(D)&=\frac{\mathbb{E}_f g_f(D)}{\mathbb{E}_f g_f(\mathrm{full})}
       \in[0.8,1.2],\\
L(D)&=\frac{\mathbb{E}_f|g_f(D)-g_f(\mathrm{full})|}
           {\mathbb{E}_f|g_f(\mathrm{full})|}\le0.2,
\end{align*}$$ with candidate accuracy at most five percentage points
below full in every cell and order separately. Use $C=C_{33}$ if it
passes, otherwise $C=C_{50}$; if both fail, retain $C_{50}$ explicitly
as a partial background. If the empty mask passes, do not claim that the
retained heads explain the behaviour. A failed small-set mask does not
disqualify its route from testing inside $C$.

#### B. Functional participation.

Evaluate $C_h^-$ and $C_h^+$ for all six candidates on the core
four-cell panel, plus $C\setminus\{\mathrm{L18H19}\}$,
$C\setminus\{\mathrm{L8H15}\}$ and $C\setminus R(C)$. One state per
candidate equals $C$, leaving at most seven distinct candidate
backgrounds. Adding L9H16 to $C_{50}$ gives a 51-head test
configuration, not a new fallback. Record family-level
$d_{h,f}=b_f(C_h^+)-b_f(C_h^-)$, fixed-sign and gold-oriented cell
margins, accuracies and changes in $F,L$. Apply the inherited
coherent/heterogeneous rule separately to $d_{h,f}$ and original-cell
gold-oriented margin effects. These tests assess functional
participation in $C$, not membership in a particular route.

#### C. Association with measured routes.

Screen all six candidates against all five anchors on the original 40
fact-swap pairs, in the noising direction. For each state $D=C_h^\pm$,
$$\begin{align*}
I_{t,f}(D)&=\mathbb{E}_o\bigl[M_D(x_{00})\\
 &\quad{}-M_D(x_{00};t\leftarrow x_{10})\bigr],\\
\gamma_{h,t,f}&=I_{t,f}(C_h^+)-I_{t,f}(C_h^-),
\end{align*}$$ so $\Gamma_{h,t}=\mathbb{E}_f\gamma_{h,t,f}$ in
Eq. [\[eq:ri-route\]](#eq:ri-route){reference-type="ref"
reference="eq:ri-route"}. Preserve historical route masks, channels and
selective releases. Source donors, frozen increments, receiver captures
and fresh recipients all belong to the same $D$; full-model donors are
not reused in retained backgrounds. Both raw route effects accompany
$\Gamma$: a negative difference can attenuate a positive route or
strengthen a negative one. Downstream amplification can change $\Gamma$
without placing $h$ on the serial path.

A pair is eligible only if the route itself passes the inherited effect
rule in at least one state and its family-level $\gamma$ also passes.
Select at most three pairs: coherent before heterogeneous, then
decreasing $\mathbb{E}_f|\gamma|$, then layer/head and structure ID. A
first pass allows one pair per candidate; a second fills remaining slots
with at most two per candidate. Fill otherwise unused slots only with
candidates passing the functional $d$ rule but having no selected pair,
ordered by class, $\mathbb{E}_f|d|$ and head ID. These receive
behavioural follow-up only. Fewer than three, including zero, is valid;
at most three distinct candidates are selected. Record whether each
route-modulation candidate also has a functional effect. Selection is
not revised after reverse patching.

#### D. Direction and named-connection checks.

For selected route pairs, repeat the comparison with restoration
$J_t(D)=M_D(x_{10};t\leftarrow x_{00})-M_D(x_{10})$ on the same 40
pairs. Test one predeclared direct connection per pair
(Table [19](#tab:final-attachments){reference-type="ref"
reference="tab:final-attachments"}) in both directions, alongside a
matched alternative receiver. For each unique connection, sample a
receiver from the same layer, excluding $C_{50}$, all structure members,
the original RI set and the candidate. Freeze controls for every
possible connection before new measurements, using seed 20260926, sorted
eligible IDs and lexically ordered connection keys.

Both the target and control receiver must be live in both compared runs.
Thus these diagnostics use $C_h^+$ plus the control receiver, with the
target connection recomputed in that same background; this differs from
the primary modulation background. V connections use all four fact
sentences, excluding question and colon, without an oracle query-fact
restriction; Q connections originate and end at the colon. Sources
precede receivers in layer order; only the receiver's colon output row
is released.

::: {#tab:final-attachments}
  Structure       Candidate               Direct connection
  --------------- ----------------------- ----------------------------------
  $T_1,T_3,T_4$   $h\in E$                $h$ / all facts $\to$ L18H18 / V
  $T_2$           $h\in E$ except L17H5   $h$ / all facts $\to$ L16H1 / V
  $T_2$           L17H5                   $h$ / colon $\to$ L18H18 / Q
  $T_1$--$T_4$    $h\in A$                L18H18 / colon $\to$ $h$ / Q
  $T_5$           $h\in E$                $h$ / all facts $\to$ L20H1 / V
  $T_5$           $h\in A$                $h$ / colon $\to$ L27H6 / Q

  : Frozen direct-connection hypotheses.
  $E=\{\mathrm{L1H27,L9H16,L11H4,L17H5}\}$ and
  $A=\{\mathrm{L23H10,L25H18}\}$. Shared endpoints do not constitute
  independent discoveries for each structure.
:::

#### Extension, sensitivity and held-out evaluation.

On all 89 discovery families, extend only original-query behaviour:
full, empty, $C$, $C\setminus R(C)$ and the additional state of each
selected candidate (at most seven configurations). Repeat the latter
retained configurations (at most five) using the paired opposite-fact
donor instead of means, symmetrically on original and swapped inputs.
Report baseline disagreement rather than selecting the favourable
replacement. A fidelity failure leaves $C$ fixed and labels it partial;
it does not trigger another search. New route modulation and connection
checks remain on the 20 common families until validation.

Before the 87 held-out families are opened, freeze $C$, selected
candidates and pairs, controls, expected signs, mean bank, thresholds
and input construction. Validation evaluates the four-cell panel for the
same at-most-seven configurations, selected route modulation in both
states and both directions, selected connections and controls in both
directions, and paired-donor sensitivity. Route and donor tests use the
original fact axis only. Unselected candidates and standalone
$T_1$--$T_5$ behaviour remain discovery-only. Failed validation never
triggers replacements or new selection.

::: {#tab:final-coverage}
  Step         Families   Measurements
  ------------ ---------- ----------------------------------------------------------------------------------------------------------
  A            20         Four cells: full, empty, five structures, $C_{33}$; $C_{50}$ only on failure.
  B            20         Four cells: six candidate toggles, two references, collective RI removal.
  C            20         All 30 candidate--route pairs; noising on the original fact axis.
  D            20         At most three selected pairs: restoration; direct connections and controls, both directions.
  Extension    89         Original-query behaviour and paired-donor sensitivity for selected candidates and collective RI removal.
  Validation   87         Frozen four-cell behaviour; selected modulation, connections/controls and donor sensitivity.

  : Logical order and coverage of the final experiment. Each family has
  two fact orders. Functional-only selections omit route and connection
  tests.
:::

#### Inference and interpretation.

Average orders within family before aggregating. Report signed and
absolute family effects, each order separately, and mean absolute
within-family order differences. Use 20,000 paired family-bootstrap
resamples (seed 20260926); discovery intervals are descriptive, not
selection-adjusted. Apply the inherited coherent/heterogeneous rule
separately to modulation and direct connections in each direction. A
coherent bidirectional result requires coherence in both directions, the
same mean sign, and at least 70% family sign agreement between
directions. Otherwise report heterogeneity, asymmetry or failure to
reproduce. Report the paired difference in absolute family effects
between target connection and receiver control; without an excess over
control, do not claim receiver selectivity. Its interval is descriptive
rather than a new significance test.

Reproduced functional effects and communication support participation in
this task under the declared live background, possibly with an opposing
sign. A functional effect without a retained connection leaves the
connection unresolved; modulation alone leaves overall functional
contribution unresolved. A null is limited to the tested backgrounds,
channels and sites. A named connection supports a link to a member or
branch, not exclusive mediation through the whole chain, and collective
RI dependence does not assign a relational role to every member.

#### Validity checks.

All-live retention must reproduce the full model, self-donor patches
must be identity interventions in each background, and historical
full-model route anchors must reproduce before new comparisons. Verify
structurally zero no-intermediate fact-to-colon-Q comparators, unchanged
earlier fact activations under query-only edits, identity-free role
keys, and live control receivers in both diagnostic contexts. Input,
background, mean-bank population, source/receiver masks, direction and
selective releases must match within every paired contrast.

## Final-experiment discovery results {#app:s44-results}

The completed run uses the frozen design above: 20 core families, with
two orders and four fact--query cells, followed by the original fact
axis on all 89 discovery families. Its implementation is S45 discovery
v2. All numbers here precede the 87-family held-out evaluation reported
in Appendix [14.3](#app:s44-validation){reference-type="ref"
reference="app:s44-validation"}; the expansion includes the 20 core
families.

#### Retention and order bias.

All five small sets and both broad sets fail the fidelity guards. Core
full-model gap $g=11.200$ falls to $0.582$ for C33 and $1.283$ for C50;
C50 is therefore the predeclared partial background $C$. Its four-cell
contrast is $B_C=0.752$ versus $5.562$ in full. C50 prefers the first
chain's answer candidate in 85% of the 160 cell--order examples, versus
51.9% in full, while candidate accuracy is 50% versus 98.1%. The
original-facts query contrast is $1.497$ versus $10.936$. The empty-mask
core contrasts are zero, although margins vary between families; its
89-family gap is $0.018$. Retention failure does not locate the missing
computation outside C50 or invalidate routes measured within a richer
background.

#### Individual and collective functional effects.

Table [21](#tab:s44-functional){reference-type="ref"
reference="tab:s44-functional"} reports core changes in the four-cell
contrast. L9H16 and L1H27 pass coherently, positive in all 20 families.
No candidate changes two-candidate accuracy, although L1H27 increases
full-vocabulary top-token accuracy by five percentage points in two
cell--order conditions. The 31 RI members include eight strong heads;
their collective effect does not isolate the weaker remainder, unlike
the S4.2 RI23 comparison.

::: {#tab:s44-functional}
  Head/group             $\Delta B$ 95% family interval
  -------------------- ------------ ---------------------
  L9H16                    $+0.601$ $[0.518,0.688]$
  L1H27                    $+0.260$ $[0.215,0.306]$
  L11H4                    $+0.004$ $[0.002,0.006]$
  L17H5                    $-0.009$ $[-0.015,-0.003]$
  L23H10                   $+0.004$ $[0.001,0.007]$
  L25H18                  $+0.0003$ $[-0.004,0.004]$
  L18H19 (reference)       $+0.247$ $[0.197,0.303]$
  L8H15 (reference)        $+0.133$ $[0.103,0.164]$
  RI31 jointly             $+0.759$ $[0.643,0.882]$

  : Core functional contributions, 20 families. Intervals are
  descriptive; selection follows the predeclared effect rule rather than
  interval exclusion of zero.
:::

#### Routes and candidate association.

In C50, T1, T3 and T4 have effects $0.618,0.393,0.255$, retaining 37.2%,
45.7% and 53.4% of their corresponding full-background core means. T2
and T5 fall below the rule ($0.042,-0.019$). All 30 candidate--route
combinations were measured; 18 have a retained route in at least one
candidate state. None passes the coherent or heterogeneous modulation
rule. The largest mean $\Gamma$ is L9H16--T1: $+0.0303$
($[0.0159,0.0464]$); its mean absolute family effect is $0.0341$, below
$0.1$. A small directional interaction is therefore observed, but no new
route association is selected. Only the two functional candidates
advance; the conditional reverse-modulation and attachment tests are not
triggered.

A post hoc paired comparison of T1 and T3 isolates the added release of
L18H19 alongside L18H18: $+0.225$ ($[0.168,0.283]$) in C50 and $+0.805$
($[0.573,1.055]$) in full, positive in all 20 families. This supports a
conditional contribution of a known RI participant, not an additive
share or new candidate assignment. Below-threshold modulation does not
establish that the additional candidates are outside these routes or act
in parallel.

#### Extension and replacement sensitivity.

On 89 families, full-model $g=11.170$; C50 retains $1.342$ under means
and $2.272$ under opposite-fact donors. Adding L9H16 changes $g$ by
$1.210$ or $0.003$; retaining L1H27 changes it by $0.552$ or $0.006$.
Retaining RI31 changes it by $1.448$ or $4.615$, respectively. These
original-axis changes are not the core four-cell $\Delta B$ values.
Under means, C50 gaps are $9.937/-7.253$ by order; under donors they are
$3.141/1.403$. The non-RI remainder under means has gaps $5.556/-5.767$,
cancelling to $-0.106$, rather than being inert. Changing the baseline
changes both the surrounding background and the absent candidate's
replacement; sensitivity alone does not identify redundancy or
compensation.

#### Verification and validation status.

Independent recomputation reproduced core behaviour, candidate effects
and all 1,600 pair-level route records; 1,035 arithmetic and consistency
checks passed. The local extension export contains only the 20 core
families, so 89-family statistics are taken from the frozen summary;
their paired intervals were not independently reconstructed.
Figures [18](#fig:s44-retention-app){reference-type="ref"
reference="fig:s44-retention-app"} and
[19](#fig:s44-effects-app){reference-type="ref"
reference="fig:s44-effects-app"} show the main diagnostics. The actual
frozen validation suite contains six mean configurations on all four
cells and four donor configurations on the original axis, with no
selected route or attachment jobs. Its completed results are reported
below.

<figure id="fig:s44-retention-app" data-latex-placement="t">
<span class="image placeholder"
data-original-image-src="figH_final_retention"
data-original-image-title="" width="95%"></span>
<figcaption>Final discovery retention and four-cell behaviour on 20
families. Left: fact-swap gaps with family-bootstrap intervals. Right:
fixed-sign margins by cell and order; correct answers are <span
class="math inline"><em>a</em>, <em>b</em>, <em>b</em>, <em>a</em></span>.</figcaption>
</figure>

<figure id="fig:s44-effects-app" data-latex-placement="t">
<span class="image placeholder"
data-original-image-src="figH_final_effects"
data-original-image-title="" width="95%"></span>
<figcaption>Core functional contributions and candidate–route
modulation. Intervals are descriptive family bootstraps; stars mark
routes below retention in both candidate states. No candidate–route
combination passes the effect rule.</figcaption>
</figure>

## Final-experiment held-out evaluation {#app:s44-validation}

#### Frozen scope and verification.

The completed evaluation uses 87 families disjoint from the 89 discovery
families, with two orders each. The discovery freeze fixes C50 as a
partial background, RI31 and two functional-only candidates, L9H16 and
L1H27. Six mean-replacement configurations cover all four cells; four
paired-donor configurations cover the original fact axis. No
candidate--route pair was selected, so no scientific Gamma or attachment
validation was scheduled. Route probes in the implementation gate are
not a route-validation panel.

The completion marker, canonical freeze signature, freeze-file and
held-out-plan hashes, discovery-plan linkage, rosters and code hashes
match the frozen run. Independent recomputation from 522 mean-baseline
family records reproduced state contrasts, fidelity, candidate accuracy,
paired effects and bootstrap intervals; all 226 consistency checks
passed. This export includes all eight cell--order margins for each mean
configuration but no donor family vectors, worker records or mean-bank
files. Donor results below therefore use the supplied 87-family summary;
no new paired donor intervals are reconstructed. The recorded gate
passed, with zero inert/all-live and earlier-prefix-invariance errors;
maximum clean/corrupted attention reconstruction error was
$2.44\times10^{-4}$.

#### Functional effects reproduce.

Table [22](#tab:s44-heldout-functional){reference-type="ref"
reference="tab:s44-heldout-functional"} contrasts the core discovery
effects with the held-out results. Both candidates pass the frozen
coherent rule and improve the four-cell contrast in every held-out
family after averaging orders. The mean changes by order are
$0.592/0.577$ for L9H16 and $0.263/0.259$ for L1H27; mean gold-oriented
margin changes are positive in every cell. The claim of 87/87 positive
families does not mean every family--order effect is positive. L9H16
raises average two-candidate accuracy from 51.01% to 53.30% (16 net
correct decisions over 696 cell--order examples). Retaining L1H27 raises
it from 50.57% to 51.01% (three net decisions). These accuracy
differences are descriptive; the frozen selection criterion concerns
margin contrasts.

::: {#tab:s44-heldout-functional}
  Head/group     Disc. $\Delta B$   Held-out $\Delta B$ 95% interval
  ------------ ------------------ --------------------- -----------------
  L9H16                     0.601                 0.585 $[0.525,0.656]$
  L1H27                     0.260                 0.261 $[0.229,0.301]$
  RI31                      0.759                 0.727 $[0.656,0.818]$

  : Four-cell contributions under mean replacement: 20 discovery versus
  87 held-out families. All three held-out effects are positive in 87/87
  families. Intervals use the frozen 20,000-draw paired family
  bootstrap.
:::

#### Replacement dependence also reproduces.

The original-axis comparisons in
Table [23](#tab:s44-heldout-baselines){reference-type="ref"
reference="tab:s44-heldout-baselines"} use $\Delta g$, not the four-cell
$\Delta B$. Individual candidate effects remain much smaller under
paired-donor replacement. L9H16's mean is near zero and slightly
negative; L1H27's is positive but small. The missing donor family
vectors preclude an independent assessment of donor-effect
heterogeneity. Collective RI31 dependence persists, but includes the
eight previously strong heads and must not be read as validation of RI23
alone.

::: {#tab:s44-heldout-baselines}
  Contribution     Mean $\Delta g$   Donor $\Delta g$ Mean 95% interval
  -------------- ----------------- ------------------ -------------------
  L9H16                      1.170           $-0.002$ $[1.028,1.345]$
  L1H27                      0.532              0.029 $[0.449,0.633]$
  RI31                       1.404              4.427 $[1.201,1.628]$

  : Held-out fact-swap contributions, 87 families. Donor differences are
  computed from the supplied configuration means; intervals are
  reconstructed only for the mean-baseline paired differences.
:::

#### The retained set remains partial.

Held-out full-model $g=10.918$ and $B=5.433$, versus $g=1.397$ and
$B=0.729$ for C50 under means. Candidate accuracy is 98.71% versus
51.01%; first-chain preference is 51.29% versus 85.49%. The
original-facts query contrast is $10.899$ versus $1.499$. C50 retains
12.8% of the fact-swap gap, increasing to 23.5% with L9H16, still below
the fidelity guard. The empty-mask gap is $-0.013$ and its four-cell
contrast $0.002$.

C50 order-specific gaps are $9.820/-7.026$ under means and $2.634/1.213$
under donors; donor fidelity is 17.6%. Without RI31, its mean-baseline
gap is $-0.007$, but the order-specific values are $5.454/-5.469$:
cancellation, not absence of fact sensitivity. Its four-cell contrast
falls to $0.002$. The corresponding donor gap is $-2.504$ versus $1.923$
with RI31 live.

#### Outcome labels and interpretation.

The supplied analyzer labels both candidates "functional participant,
attachment unresolved". This label uses mean-baseline functional
reproduction; the label assignment does not incorporate the donor
comparisons. We therefore report reproduced functional participation
together with baseline sensitivity, rather than treating the automatic
label as evidence of robustness across replacement methods. No new
threshold or candidate selection is introduced after validation. The
complete-chain measurements, the 30 discovery modulation contrasts and
the post hoc nested-route comparison remain discovery evidence; the
held-out result validates the frozen behavioural comparisons, not new
connections or a sufficient retained circuit.

# Developmental check: details {#app:dev}

#### Correspondence with @ren2024semantic.

Same: the four task families (fruit/month and furniture/profession
binary tasks, a four-class fruit/month task, a nine-class
fruit/animal/month task); the shot ladder $\{0,1,2,3,4,5,10,20\}$ with
class coverage enforced in the demonstrations when the number of shots
permits; the format-versus-pattern distinction; the AGENDA relation set
with seven relation types and entities reduced to single letters; the
developmental QK gate (argmax only) with $\tau{=}2.2$ as sensitivity;
the raw-embedding OV score. Adapted: accuracy is constrained to the
legal task labels, with vocabulary-wide format validity and legal-label
probability mass reported separately; inputs are relation-bearing
sentences extracted from the corpus rather than whole abstracts; rows
are filtered for unambiguous single occurrences and tokenizer-safe
single-letter anchors under the OLMo-2 tokenizer, 60 self-relation rows
(source and target collapsing to the same entity) are excluded and
listed, and the assessment is balanced to 100 triplets per relation
(700); the QK gate uses the model's actual attention probabilities; the
OV score is computed exactly on the visible-token subset (the
full-vocabulary softmax normalization cancels after centring and in the
ratio; verified numerically); the original spaCy function-word removal
is not applied (documented deviation); only the forward relation
direction is run. Per relation type, the 15 heads with the highest index
are reported, with a support requirement of at least ten scored events.

#### Checkpoints.

Behaviour: 17 Stage-1 checkpoints of OLMo-2-1124-7B (steps 150, 600,
700, 850, 900, 1,000, 2,000, 3,000, 4,000, 6,000, 9,000, 19,000, 38,000,
76,000, 152,000, 357,000, 928,646; 1B to 3.9T tokens). Relation index:
ten of these (steps 150, 600, 700, 850, 900, 1,000, 2,000, 3,000, 9,000,
928,646). The broad behavioural window (steps 600--1,000) was
established before the index timing was analysed; steps 700, 850 and 900
were added to the behavioural measurement after the index sweep showed a
routing maximum near step 850. The 850/900 localization is therefore
targeted follow-up evidence rather than an independent temporal
prediction. Checkpoint identity is the exact optimizer step, since
OLMo's rounded token labels coincide for steps 600/700 and 850/900.

#### Behavioural metrics.

Fifty frozen prompts per task and shot count are reused at every
checkpoint (1,600 assessments per checkpoint), enabling paired
comparisons. Per task, shot count and checkpoint: constrained accuracy;
gold-versus-best-wrong margin; format validity (vocabulary-wide top
token is a legal label); legal-label probability mass; the 20-shot minus
1-shot accuracy gain; Wilson 95% intervals against task-specific chance;
exact paired McNemar/binomial tests between adjacent checkpoints on the
reused prompt ids. The paired tests span 64 comparisons; raw
significance of some comparisons does not survive correction over that
family, and the report treats them as descriptive.

<figure id="fig:dev-app" data-latex-placement="h">
<span class="image placeholder"
data-original-image-src="figI_developmental"
data-original-image-title="" width="\columnwidth"></span>
<figcaption>Developmental measurements (shaded: steps 700–900). (a)
20-shot constrained accuracy per task with Wilson intervals; dotted
lines mark chance. (b) The three index statistics averaged over all
heads and relations at the ten RI checkpoints: the argmax gate pass rate
peaks near step 850 and then falls, the index per opportunity follows
it, and the conditional index rises slowly and monotonically. (c)
Overlap of each checkpoint’s top-15 heads with the final checkpoint’s
(Jaccard) and all-head rank correlation with the final
checkpoint.</figcaption>
</figure>

#### Index statistics.

All 1,024 heads are scored on the same 700 assessment triplets across
checkpoints. Per head and checkpoint: the index over gated events
(conditional on the gate); the gate pass rate under the argmax gate and
under $\tau{=}2.2$; the index per measurement opportunity (failed gates
contribute zero), which combines frequency and strength; and two
null-target controls. Developmental changes in the conditional index are
tested on a fixed-support population (relation--head cells with at least
ten scored events at both endpoints) with a paired Bayesian cluster
bootstrap whose resampling unit is a cluster of text groups, not heads
or events (Dirichlet weights, 5,000 replicates); changes in pass rate
use a paired cluster bootstrap over the same units. Head stability: the
top-$k$ heads at the final checkpoint are traced backwards;
per-checkpoint top-$k$ Jaccard overlap with the final set and all-head
rank correlation with the final checkpoint are reported. Absence of a
clear change in the conditional index is not evidence of no change, and
stability or instability of head identity is not a causal test of the
heads' role.

[^1]: Code, frozen inputs, run manifests and analysis outputs:
    <https://github.com/Maxgolu/NLP_proj>

[^2]: <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
