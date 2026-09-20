# Stage 1 saved-event anatomy

Only QK-passing events are characterized. The answer position is the last
token of the bare prompt, before any shared answer prefix used by stage 2.
Distance is current position minus the annotated source anchor.

| Fact | Current position | Distance | QK passes |
|---|---|---|---:|
| demo | answer_prediction | distance_ge_2 | 4 |
| demo | demo | distance_ge_2 | 133452 |
| demo | demo | previous | 25884 |
| demo | demo | self | 57270 |
| demo | test_other | distance_ge_2 | 14890 |
| test | answer_prediction | distance_ge_2 | 744 |
| test | test_other | distance_ge_2 | 24060 |
| test | test_other | previous | 7096 |
| test | test_other | self | 14734 |
