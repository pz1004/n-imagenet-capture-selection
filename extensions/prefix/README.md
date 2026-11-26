# First-scan event rate and request allocation

These analyses use the first 20 ms of each recording. They ask whether the first vertical scan’s event rate improves the allocation of a limited number of horizontal rescans and whether a switching threshold fitted at 20 ms carries over better than the threshold fitted at 30,000 events. Both are retrospective comparisons on the frozen source split.

The request models used 9,000 reference calibration images to predict the signed change in correctness under the original terminal switching rule. A correction contributes +1, a loss −1, and all other outcomes zero. The predictors were vertical maximum softmax probability, the logarithm of one plus the vertical event rate (observed prefix count divided by 0.020 seconds), or both. Each predictor supplies standardized linear and quadratic terms. The intercept is unpenalized and the slopes have a unit ridge penalty. The terminal rule is common to all request models.

At exactly 10,800 requests on the 36,000 transfer test images, Event Histogram recovered 434 net corrections with confidence alone, 412 with event rate alone and 516 with both. Adding event rate raised accuracy by 0.23 percentage points (paired 95% within-class interval 0.09–0.35, whole-class interval 0.06–0.37). Transferring the combined model’s numerical cutoff requested 11.41% of images. The fixed-quota result assumes all first scans can be ranked together.

## Replay

All first-scan measurements and classifier scores used by the replay are in `data/`. From the repository root, use Python with NumPy installed.

```sh
python extensions/prefix/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python extensions/prefix/replay.py --output replay-prefix
python extensions/prefix/verify.py --results replay-prefix
```

`results/reference_calibration_fits.json` records the three gate fits, score cutoffs, tie boundaries, and reference switching rules. In the result filenames `old` denotes the 6-mm reference trajectories and `new` the transfer trajectories. Each classifier and pair has a JSON result and a 36,000-row source decision table. Both request modes report realized workload, accuracy, rescues and losses, net corrections per 100 extra scans, and excess accuracy over a random request at matching workload. The separate terminal-rule comparison applies both the 20-ms and 30,000-event reference thresholds to test scores from the first 20 ms.

The 2,000 within-class resamples keep the 900 classes fixed. A second set of 2,000 whole-class resamples varies class composition. Each keeps all policies paired by source and fixes calibration fits. The fixed-quota comparison reranks the test draws to maintain 10,800 requests. Ties are resolved by a stable hash of the source ID.
