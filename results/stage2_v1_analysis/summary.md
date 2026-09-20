# Stage 2 validation and initial analysis

## Run integrity

- `summary.json` declares `complete: true`; Slurm job 888691 separately exited 0:0.
- 178/178 screen files and 178/178 extension files are present and finite.
- All three replica gates passed. Raw per-pair arrays reproduce `head_effects.csv`.
- Gate maxima: runner drift 0; self-patch drift 0; corrupted-embedding endpoint drift 0.
- 4/178 pairs use a non-empty shared answer prefix.

## Behavioral contrast at the intervention token

Across 89 families (two orders averaged), mean clean metric = 5.6796 (95% family bootstrap CI [5.3382, 6.0302]); mean corrupted metric with the clean sign fixed = -5.4909 (CI [-5.8496, -5.1410]).
The mean clean-corrupted gap is 11.1705 (CI [10.6323, 11.7250]). The clean metric is positive on 100.0% of pairs and the fixed-sign corrupted metric is negative on 100.0%.

## Approximation quality and RI audit

First-order attribution agrees strongly with the 40-pair all-head exact map: signed Spearman = 0.945, absolute Spearman = 0.921, and pair-by-head Spearman = 0.903. The pre-registered 0.7 threshold was passed, so IG was not used.
When the 178-pair attribution mean is compared with the 40-pair exact mean, rather than matching both methods on the same 40 pairs, Spearman is 0.698; this lower value reflects the different family samples as well as approximation error. On the same 178 pairs and within the selected 92 heads, attribution-versus-exact Spearman is 0.990.
RI-first has only a weak positive association with exact causal importance on the random 40-pair subset: Spearman = 0.135, family-bootstrap 95% interval [0.063, 0.168]. RI-last correlation is 0.132.
The exact rankings from order 0 and order 1 correlate at 0.461 across all heads; this statistic is dominated by the many near-zero heads, while the strongest heads below retain the same direction across almost every family.

Importance below is `- exact_delta`, so positive values mean that the corrupted-head patch moves the metric toward the corrupted answer. CIs resample whole families; orders are averaged within family.

## Top exact heads on the random 20-family subset

| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |
|---|---:|---:|---:|---:|---:|---:|
| L17H1 | 0.0094 | 3.3619 | 5.7419 | [5.0345, 6.4580] | 1.00 | 0.514 |
| L27H6 | 0.0000 | 2.6401 | 3.2216 | [2.4705, 3.9670] | 1.00 | 0.288 |
| L18H19 | 0.0182 | 2.7847 | 3.0804 | [2.6698, 3.5044] | 1.00 | 0.276 |
| L18H18 | 0.0151 | 2.2260 | 2.7676 | [2.3876, 3.1686] | 1.00 | 0.248 |
| L21H18 | 0.0096 | 1.4799 | 1.8498 | [1.6293, 2.0735] | 1.00 | 0.166 |
| L25H17 | 0.0116 | 1.1830 | 1.3193 | [1.1778, 1.4678] | 1.00 | 0.118 |
| L22H5 | 0.0155 | 0.9502 | 1.0683 | [0.9141, 1.2246] | 1.00 | 0.096 |
| L16H21 | 0.0240 | 0.6954 | 0.9032 | [0.6818, 1.1358] | 1.00 | 0.081 |
| L16H1 | 0.0137 | 0.8122 | 0.8772 | [0.7047, 1.0625] | 1.00 | 0.079 |
| L21H6 | 0.0067 | 0.6623 | 0.7522 | [0.5802, 0.9286] | 1.00 | 0.067 |

## Top exact heads among the 92-head all-family extension

| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |
|---|---:|---:|---:|---:|---:|---:|
| L17H1 | 0.0094 | 3.3619 | 5.7013 | [5.2942, 6.1124] | 1.00 | 0.510 |
| L18H19 | 0.0182 | 2.7847 | 3.0721 | [2.8666, 3.2756] | 1.00 | 0.275 |
| L27H6 | 0.0000 | 2.6401 | 3.0709 | [2.7487, 3.3902] | 1.00 | 0.275 |
| L18H18 | 0.0151 | 2.2260 | 2.7037 | [2.5026, 2.9065] | 1.00 | 0.242 |
| L21H18 | 0.0096 | 1.4799 | 1.7040 | [1.5881, 1.8217] | 0.99 | 0.153 |
| L25H17 | 0.0116 | 1.1830 | 1.2672 | [1.1902, 1.3418] | 1.00 | 0.113 |
| L22H5 | 0.0155 | 0.9502 | 0.9825 | [0.9036, 1.0632] | 1.00 | 0.088 |
| L16H21 | 0.0240 | 0.6954 | 0.9245 | [0.8082, 1.0491] | 0.99 | 0.083 |
| L16H1 | 0.0137 | 0.8122 | 0.8679 | [0.7664, 0.9717] | 0.98 | 0.078 |
| L21H6 | 0.0067 | 0.6623 | 0.7023 | [0.6361, 0.7697] | 0.98 | 0.063 |

## Pre-specified and Stage-1 anatomy heads

| Head | RI first | Attribution | Exact importance | 95% CI | Positive families | Gap fraction |
|---|---:|---:|---:|---:|---:|---:|
| L3H11 | 0.2431 | -0.0017 | -0.0022 | [-0.0067, 0.0024] | 0.47 | -0.000 |
| L9H22 | 0.3739 | -0.0021 | -0.0020 | [-0.0065, 0.0027] | 0.42 | -0.000 |
| L16H4 | 0.0196 | 0.0243 | 0.0257 | [0.0190, 0.0328] | 0.78 | 0.002 |
| L16H1 | 0.0137 | 0.8122 | 0.8679 | [0.7664, 0.9727] | 0.98 | 0.078 |
| L16H21 | 0.0240 | 0.6954 | 0.9245 | [0.8048, 1.0475] | 0.99 | 0.083 |

## Ranking overlap and pre-registered verdicts

Only 1/25 exact top heads overlap the supported RI-first top 25, and 1/25 overlap the RI-last top 25. In contrast, 17/25 overlap the attribution top 25.
All 101/102 heads in the top decile of positive exact importance lie below the lenient RI 99% control-mean threshold (0.137426); 24 have RI exactly zero. L27H6 is rank 2 in the random exact subset and has RI zero.
For reference, the historical 99.9% control-mean threshold that selected L3H11 and L9H22 is 0.182739.
The original RI-selected heads rank 188 (L3H11) and 724 (L9H22) by exact importance on the random subset. On all 89 families their median importance is -0.0021, versus 0.0002 for the 25 random controls.
Under the project's pre-registered language, the result supports `incomplete` and meets the descriptive `misleading` criterion. It does not support `sound`; that criterion was not operationalized beyond 'systematically larger', and only two heads passed the historical pooled rule.

## Random-control reference

The 25 random extension controls have median exact importance 0.0002, range [-0.0089, 0.0832]. This is descriptive: the non-control extension heads were selected using the discovery data, so naive post-selection p-values would not be valid.

## Interpretation boundary

The results establish a high-quality causal head screen for this clean-to-corrupted, all-prompt-position patching experiment. They do not identify edges, semantic specificity, necessity under redundancy, or a complete circuit. Exact all-family effects exist only for the selected 92 heads; comparisons involving that set must account for its data-dependent selection.
