# Phase-A single-hop behavioral baseline — OLMo-2-7B (singlehop_v1)

Runs: olmo2_singlehop_{0,4,12}shot (OLMo-2-1124-7B @ 7df9a825, kinship families
000-199, dataset singlehop_v1). Candidate metric; "both" = correct in both fact
orders (chance 25%).

| shots | variant   | acc order0 | acc order1 | both | 95% CI | free |
|-------|-----------|-----------:|-----------:|-----:|--------|-----:|
| 0     | base      | .76 | .99 | .76 | [.70,.81] | .85 |
| 0     | corrupted | .72 | 1.00 | .72 | [.66,.78] | .84 |
| 0     | reorder   | .78 | .75 | .61 | [.54,.67] | .73 |
| 4     | base      | 1.00 | .93 | .93 | [.88,.95] | .96 |
| 4     | corrupted | 1.00 | .94 | .94 | [.90,.97] | .97 |
| 4     | reorder   | .94 | .94 | .89 | [.84,.93] | .92 |
| 12    | base      | 1.00 | .93 | .93 | [.88,.95] | .96 |
| 12    | corrupted | 1.00 | .93 | .93 | [.88,.95] | .96 |
| 12    | reorder   | .86 | .89 | .79 | [.73,.84] | .85 |

Findings:
1. The corrupted (answer-swap) twin is tracked at parity with base (.94 vs .93
   both-orders at 4-shot; free generation names the swapped mother 388/400).
   Clean/corrupted behavioral contrast exists at near-ceiling — patching
   preconditions met.
2. Working set (correct on base AND corrupted, both orders): 4-shot 176/200
   families (89 discovery / 87 validation); 0-shot 102/200; 12-shot 176/200.
   Adding reorder: 162/200 at 4-shot.
3. Candidate log-prob margins on correct items: ~2.6 nats at 0-shot, ~5.4-5.5
   at 12-shot — strong intervention signal.
4. Single-hop IS layout-robust (.89 both-orders under full fact shuffling at
   4-shot) — vs .37 for two-hop on v3.1. The two-hop failure is
   composition-specific, not fact-reading. Note: 12-shot reorder dips to .79
   (another small demos-induce-pattern effect; consistent with prior findings).
5. 0-shot shows the raw recency prior again (.76 far vs .99 adjacent).

Recommended Phase-A working condition: 4-shot (single-hop-only demos), 89
discovery families; 0-shot as secondary contrast. Next: RI head scan + costing
dry-run on these prompts.
