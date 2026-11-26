# Continuing one recording or switching paths

This replay starts with transfer vertical and scores its first 30,000 events. The cutoff fitted on 9,000 calibration photographs requested another observation on 10,683 of the 36,000 test photographs. Four actions share these requests. The classifier can select between the first and second disjoint vertical blocks, pool them into a 60,000-event histogram, or request a prefix from a separate horizontal or original square recording. An incomplete request counts toward workload and retains the initial prediction.

| Action on a requested photograph | Completed requests | All-source top-1 (%) | Mean observed event span per source (ms) |
|---|---:|---:|---:|
| Keep the first scan on every photograph | 0 | 41.73 | 14.87 |
| Continue vertically and select between blocks | 10,358 | 43.99 | 18.83 |
| Continue vertically and pool both blocks | 10,358 | 41.86 | 18.83 |
| Switch to horizontal and select | 10,631 | 42.77 | 19.06 |
| Switch to the original square and select | 10,615 | 45.36 | 19.13 |

Square exceeded vertical continuation by 1.37 percentage points (95% within-class interval 1.12 to 1.65, whole-class 1.05 to 1.73). A continuation block comes from the same recording. Stage travel, settling, waiting for the first event, inference latency, and energy were unmeasured.

```sh
OPENBLAS_NUM_THREADS=1 python extensions/continuation/replay.py --output runs/continuation
```

The prefix, eligibility and timing inputs cover all 45,000 source IDs. In `inputs/scores.npz`, `test_first.class` is the prediction for the first vertical prefix and `test_second.us` is the observed span from the end of that prefix to the end of the next block or recording. Arm names are `first`, `second`, `pooled`, `horizontal`, and `square`. Every arm supplies the observed count (`n`), span in microseconds (`us`), completion flag (`eligible`), predicted class and maximum softmax probability (`msp`). For continuation, horizontal and square, `fused` gives the class obtained by averaging full probability vectors from the first and additional observations. The arrays follow the corresponding `calibration_ids` or `test_ids`.

`results/point.json` gives corrections, losses, completeness, time and request costs, including a matched random allocation. `results/heldout_decisions.csv.gz` gives the candidate terminal classes and both request masks. The replay reproduces fitted rules and numerical cutoffs in 2,000 within-class and 2,000 whole-class resamples, including a sensitivity restricted to sources with complete windows for every arm. `time_accuracy.csv` distinguishes observed event spans from assumed path-switch delays. [Raw-data regeneration](../../regeneration/capture_windows.md)
