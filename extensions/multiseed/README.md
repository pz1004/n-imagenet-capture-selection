# Five calibration-only adaptation runs

Each Event Histogram model adapted on 4,500 unlabeled photographs per trajectory. The other 4,500 calibration photographs supplied labels for the selectors. All six classifier states (released and five adapted) used the same 36,000 evaluation sources and the same 30,000-event anchor crop. Each seed was matched across paths. With a transported reference-pair selector and 10,800 horizontal rescans, the released classifier gained 0.725 percentage points over vertical alone. The five adapted gains were 2.047, 1.858, 1.869, 1.911 and 1.975 points. The increase in gain averaged 1.207 points (within-class 95% interval 0.979 to 1.459, whole-class 0.903 to 1.494).

With selectors fitted on the target pair, the accuracy advantage of a square over a horizontal follow-up fell from 2.76 points in the released state to 0.85 points after adaptation. The change was −1.91 points (within-class interval −2.23 to −1.62, whole-class −2.35 to −1.50). Each state first used its own 10,800 lowest-confidence vertical images. A sensitivity analysis kept exactly the same released request set across states.

```sh
OPENBLAS_NUM_THREADS=1 python extensions/multiseed/replay.py --workers 4 --output runs/multiseed
```

`inputs/scores.npz` holds the predicted class and maximum softmax probability on the labeled policy calibration set and the evaluation set. For example, `test_variant1_msp[3]` contains the seed 3 transfer vertical scores in `test_ids` order. Axis 0 runs from 0 (released) through 5. Reference vertical and horizontal require policy calibration scores only. The evaluation order and source labels are shared across states. [Raw adaptation and regeneration](RAW_REGENERATION.md)

`results/by_seed.json` gives counts, conditional discordance rates, accuracies and yields by order, fit origin, rule and request fraction. `policy_sweep.csv` provides the same operating points in long form. Six compressed source tables retain the first-scan request, terminal prediction and changes in correctness. Paired bootstrap draws for both resampling designs are stored in the NumPy arrays; metric names and axis order are given in `paired.json` (draw × released/mean adapted/adapted minus released × metric). The intervals resample the calibration and evaluation sources and condition on the five trained states. Training variability is summarized separately across seeds. Set `--workers 1` to limit concurrent memory use.
