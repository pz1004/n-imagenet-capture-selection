# Choosing the follow-up trajectory

An unadapted classifier first sees a vertical scan. When its confidence falls below the cutoff calibrated for 30% requests on the reference trajectories, it receives a horizontal or square follow-up. Both candidate paths use exactly the same requested source IDs. All comparisons use up to 30,000 events per capture and the frozen 9,000 calibration and 36,000 test photographs across 900 classes. This retrospective analysis uses test sources examined in earlier experiments.

For Event Histogram on the transfer trajectories, 11,076 images received a second scan. Accepting the follow-up prediction outright produced 43.18% accuracy for horizontal and 46.53% for square. With path-specific fitted selectors the accuracies were 43.84% and 46.69%. The paired square-minus-horizontal difference was 2.85 percentage points (95% within-class interval 2.58–3.11, whole-class interval 2.39–3.26). A single square scan reached 46.84% when that initial path was available.

## Replay

From the repository root, use Python with NumPy installed. The verifier checks the distributed decisions. The replay fits the selectors and recalculates all 2,000-draw paired bootstrap intervals and the calibration-refit sensitivity. It then writes the source-level decisions, counts, fits, and interval tables into the named output directory.

```sh
python extensions/path/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python extensions/path/replay.py --output replay-path
python extensions/path/verify.py --results replay-path
```

- `inputs/square_confidence.csv.gz` gives the highest softmax probability for each checkpoint on all 45,000 full-cohort square scans. Square top classes come from `data/original_sanity_predictions.csv`.
- `inputs/square_fusion.csv.gz` gives the top class after averaging the vertical and square softmax vectors on each test source. It contains one column per first trajectory and checkpoint. The corresponding linear-path fusion classes come from `data/trajectory_baseline_decisions.csv.gz`.
- `results/{reference,transfer}_{histogram,timestamp}.json` contains single-scan accuracy, thresholds, outcome counts, targeted and matched-random accuracy, paired path and rule contrasts, and intervals conditional on the fitted cutoffs and selectors. The accompanying `*_decisions.csv.gz` files give the request, predicted classes, hypothetical class if a follow-up is requested, final selected class and change in correctness. `all_contrasts.csv` combines the 276 estimates and their intervals. `transfer_histogram_refit_sensitivity.json` also resamples and refits the two local selectors while holding the request cutoff fixed.

To rebuild the two compact inputs from logits, use the repository's `regeneration/infer.py` to produce the full-checkpoint `original`, `vertical` and `variant1` inference folders for `events30k`. Square needs both splits. Each first vertical trajectory needs the test split. Pass the directories containing `predictions.csv`, `histogram.npy` and `timestamp.npy`:

```sh
python extensions/path/build_inputs.py --square logits/original --reference-vertical logits/vertical --transfer-vertical logits/variant1 --output regenerated-square
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python extensions/path/replay.py --inputs regenerated-square --output replay-from-logits
```

The input builder joins by source ID and checks the manifest labels, predictions, and confidence scores. It calculates softmax in float64 before averaging equal-weight probability vectors.
