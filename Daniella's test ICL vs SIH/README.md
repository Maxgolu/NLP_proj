# ICL / SIH developmental side experiment

This completed side experiment asks whether OLMo-2-1124-7B acquires few-shot
in-context learning during the same training period in which SIH-like relational
structure appears. It complements the mature-model, synthetic mother-of causal
audit in the main project; it is not another Stage-4 intervention experiment.

## Evidence and current conclusion

Start with [the two-page summary](Documentataion/OLMo_ICL_RI_summary.pdf), then
[the complete report](Documentataion/OLMo_ICL_RI_side_experiment_complete.pdf).
The original `Documentataion` directory spelling is preserved.

The supplied analysis covers 17 behavioral checkpoints and 10 RI checkpoints.
Behavior uses four synthetic classification tasks, eight shot counts
(0, 1, 2, 3, 4, 5, 10, 20), and 50 fixed examples per task/shot condition:
1,600 assessed examples per checkpoint. The generated ICL stream contains 3,200
examples; the notebook selects the assessment subset deterministically.
RI uses a balanced 700-triplet assessment (100 per relation, seven relations),
all 1,024 heads, and the same selected examples across checkpoints.

The reported early behavioral transition begins between steps 700 and 850;
step 900 is the first sampled checkpoint at which all four 20-shot Wilson
intervals lie above chance (roughly 4B training tokens). Source-directed QK
routing peaks near step 850. Conditional RI does not show a corresponding clear
population-wide jump: the fixed-support RI intervals include zero.
The final high-RI head population is also not yet stable at that transition.

The supported claim is **temporal association with an early routing change**,
not a causal result or direct replication of population-wide conditional-RI
emergence. The broad step-600–1000 behavioral window preceded interpretation
of RI timing; denser behavioral measurements at 700/850/900 were added after
the RI peak motivated follow-up. The precise timing match is therefore adaptive
follow-up evidence, not an independently preregistered prediction.

## Code and data

The Colab notebooks, in execution order, are:

1. `code/data_generation.ipynb`: synthetic ICL prompts and AGENDA transformations.
2. `code/olmo_icl_checkpoint_sweep.ipynb`: behavioral checkpoint measurements.
3. `code/olmo_icl_results_analysis.ipynb`: behavioral summaries and transition checks.
4. `code/olmo_ri_developmental_sweep.ipynb`: developmental RI measurements and controls.
5. `code/olmo_ri_results_analysis.ipynb`: trajectories, stability and sensitivity analyses.

The notebooks retain their original source cells. Embedded execution outputs and
execution counts were cleared for version control; numerical exports and report
figures are retained separately. Their default `ROOT` is a Colab/Google Drive
location; set it to the experiment directory in your environment. These are
research notebooks with GPU sweep cells, not an automatically executed CPU-only
pipeline. Restoring files or reading summaries does not require rerunning a sweep.

Versioned input streams:

- `data/icl_stream.jsonl`: 3,200 generated classification examples.
- `data/ri_agenda_stream.jsonl`: 3,907 intermediate transformed relation examples.
- `data/ri_agenda_olmo_safe.jsonl`: 3,907 tokenizer-audited examples; self-relations
  remain in the source but are excluded when constructing the RI assessment.

`results/ri/developmental_v1/run_config.json` records the executed choices.
`ri_assessment_manifest.jsonl` records the 700 selected examples, and
`ri_excluded_self_relations.jsonl` records the 60 self-relation exclusions.
Both checkpoint manifests are retained. `repository_manifest.json` lists input
hashes/row counts, clean notebook hashes and external measurement identities.
The tokenizer-safe input's SHA-256 was checked against the executed RI config.

The relation stream derives from the AGENDA test data distributed with
[GraphWriter](https://github.com/rikdz/GraphWriter). The generator records the
download locations for `data/preprocessed.test.tsv` and `data/relations.vocab`.
These are transformed upstream examples, not newly authored synthetic text;
retain their upstream attribution and applicable data terms when reusing them.

The primary developmental gate is source-as-attention-argmax; the stricter
`tau=2.2` gate is a sensitivity check. OV uses raw token embeddings. The frozen
stream uses relevant-sentence extraction, safe-letter replacement and tokenizer
checks, but does **not** exactly reproduce the paper's spaCy function-word
removal. Only forward relations are in the recorded run. Do not describe this
as an exact preprocessing/protocol replication of Ren et al.

## Repository selection and reproduction limits

Committed: all five notebook sources, all three input streams, execution and
assessment manifests, current compact CSV summaries, selected-head trajectories,
analysis figures, the complete report and its concise summary.

Kept locally/on external research storage, excluded from Git:

- per-example ICL checkpoint prediction JSONL files;
- RI `chunks/` arrays, per-checkpoint all-head tables, the concatenated
  `ri_head_trajectory.csv`, and large all-head change tables;
- `old_analysis/` and download copies marked `(1)`;
- `OLMo_ICL_RI_side_experiment_methodology.pdf`, whose extracted text is identical
  to the complete report, despite being a separate PDF file.

The manifest records hashes for the external measurement files; it does not
include their contents. Summary tables and reports can be inspected directly
from Git. Recomputing paired prediction tests, chunk bootstraps, or all-head
analyses requires restoring the excluded files at their original relative paths
or deliberately rerunning the corresponding measurements. The repository
publication check verified file structure, source preservation and input identity;
it did not independently rerun this side experiment's numerical analysis.
