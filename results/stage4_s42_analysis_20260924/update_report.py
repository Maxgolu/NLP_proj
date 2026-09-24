from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent;O=ROOT/'חומר כתוב/Stage4_Overleaf'
original=(H/'prior_report/main_s41.tex').read_text(encoding='utf-8')
before=original.split(r'\section{S4.2: concrete prospective measurements}')[0]
appendix=original[original.index(r'\appendix'):]
before=before.replace('% S4.1 is measured; the S4.2 section is a prospective proposal, not executed results.','% S4.1 and S4.2 are measured; S4.3--S4.5 are prospective and unexecuted.')
before=before.replace('S4.1 results and an evidence-based S4.2 measurement plan','S4.1--S4.2 results and a bounded group-level follow-up plan')
start=before.index(r'\begin{abstract}');end=before.index(r'\end{abstract}')+len(r'\end{abstract}')
abstract=r'''\begin{abstract}
This report combines the verified S4.1 route map with completed S4.2 group and conditional-intervention results in OLMo-2-1124-7B. S4.1 identifies strong fact-site V influence from L17H1 into L18H18/L18H19 and an upstream L15H25--L16H1/L16H21 motif, alongside opposing routes and unresolved sources. S4.2 confirms large joint effects involving the layer-18 pair and live-model dependence on route-supported receiver groups: clamping O3 removes about 88--90\% of L17H1's tested intervention effect. This ratio is not a unique mediation share. The RI audit extends 14 conditional candidates; the residual RI23 group is collectively non-negligible but its signed effect depends on the replacement baseline. L26H31 does not satisfy the backup rule in the tested backgrounds. Group effects remain order-dependent after accounting for the full model's own order-dependent confidence, and donor versus mean replacement can yield very different post-intervention accuracies. All 51,904 S4.2 records and 49 numerical exports reproduce locally. We distinguish measured findings from hypotheses and propose one bounded local S4.3 expansion, omitting broad search, followed by a shared S4.4/S4.5 measurement registry for group-only relational behavior and conditional head-mechanism evaluation. Exact head sets, caps, reuse rules and dependencies are stated. No sufficient circuit, new GPU result, universal semanticity claim or held-out validation is implied.
\end{abstract}'''
before=before[:start]+abstract+before[end:]
before=before.replace(r'\setcounter{tocdepth}{1}', '\\newpage\n'+r'\setcounter{tocdepth}{1}')
before=before.replace('Named multi-step mediation, G1--G3, mean-ablation circuits and held-out validation remain unexecuted.', 'At this S4.1 boundary, named multi-step mediation, G1--G3, mean-ablation circuits and held-out validation were unexecuted; the completed G1--G3 audit is reported in Sections 7--10.')
before=before.replace('This report analyzes \\file{stage4_all_v2}', 'Sections 1--6 document S4.1 as measured before the group audit. They analyze \\file{stage4_all_v2}')
before=before.replace('These observations support a targeted conditional RI audit, not either blanket acceptance or blanket rejection of RI.', 'At the S4.1 boundary, these observations motivated a targeted conditional RI audit rather than blanket acceptance or rejection of RI; the measured audit appears later in this report.')
appendix=appendix.replace('proposed groups, exact RI/reference rosters and masks; not runnable GPU code.','the S4.1-derived rosters and masks subsequently used in S4.2; not runnable GPU code.')
appendix=appendix.replace('For CPU reproduction from the project root, run the two supplied analysis scripts.', 'For S4.1 CPU reproduction from the project root, use the two supplied S4.1 analysis scripts.')
appendix=appendix.replace(r'\appendix',r'\appendix'+'\n'+r'\small')
extra=r'''
\section{S4.2 data, reproducibility and exact follow-up rosters}
\label{sec:future_rosters}
Executed data: \file{results/stage4_s42_all_v1/}; isolated package and frozen inputs: \file{results/stage4_s42_v1/}. Readiness verification is in \file{results/stage4_s42_readiness_20260924/}. Additional saved-data analysis and report builders are in \file{results/stage4_s42_analysis_20260924/}. The GPU results and original analyzer exports are preserved. All new statistics are derived on CPU.

The portable Overleaf package includes:
\begin{itemize}
\item \file{data/s42_contrast_ledger.csv}: all measured contrast summaries, with phase, direction and baseline.
\item \file{data/s42_g1_interactions_cpu_extended.csv}: new matched G1 interactions on the available 20/69/89-family populations; no new endpoints.
\item \file{data/s42_g1_task_performance.csv}, \file{s42_g1_order_effects.csv} and \file{s42_full_model_order_baselines.csv}: behavior and properly scaled order comparisons.
\item \file{data/s42_blocking_paired.csv}: source, blocking and residual effects, plus jointly bootstrapped intervention ratios.
\item \file{data/s42_ri31_audit.csv}, \file{s42_residual_vs_references.csv} and \file{s42_residual_nonadditivity.csv}: conditional RI results and paired group comparisons.
\item \file{data/s42_g3_backup_audit.csv}: backup effects and before-intervention attention/output diagnostics.
\item \file{data/s42_replacement_norms.csv}: descriptive intervention magnitudes; these do not constitute matched-norm controls.
\item \file{data/s42_facts.json}, \file{s42_readiness.json}, and \file{s42_next_stage_plan.json}: numerical provenance, verification and the exact prospective head sets/dependencies.
\end{itemize}

The fixed residual RI23 set used by the six-configuration S4.4 panel is:
\begin{quote}\small
RI23PLACEHOLDER
\end{quote}
The four reference cohorts are unchanged from the original proposal and are explicitly enumerated in both proposal and follow-up JSON files. They are layer-matched descriptive controls, not randomization-test samples.

The added analysis uses the same $0.10$-logit retention rule, 20,000 family bootstrap draws and seed 20260923. An interval or apparent comparison selected on discovery data is not presented as a confirmatory held-out test. The four-condition panel, retained-head mechanisms, local mediation expansion and reduction curves remain prospective. The prior S4.1-only PDF and sources were archived locally before this report was updated.
'''
pp=json.loads((ROOT/'results/stage4_s42_v1/inputs/plan.json').read_text())['proposal']
extra=extra.replace('RI23PLACEHOLDER',', '.join(pp['g2']['residual23'])+'.')
appendix=appendix.replace(r'\end{document}',extra+'\n'+r'\end{document}')
(O/'main.tex').write_text(before+r'\input{s42_results}'+'\n\n'+appendix,encoding='utf-8')
(O/'README.txt').write_text('Experiment Stage 4: verified S4.1 and S4.2 results, with prospective S4.3--S4.5 plan.\nUpload all files to Overleaf. Main document: main.tex. Compiler: pdfLaTeX.\nThe included s42_results.tex contains the S4.2 analysis and revised future design.\nNo model weights, held-out inputs, or new GPU results are included.\n',encoding='utf-8')
print('Updated combined report source')
