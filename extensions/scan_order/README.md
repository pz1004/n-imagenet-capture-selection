# Scan order and expected gain

Calibration selected square first and optional vertical second among four orders at a 30% request budget. This policy reached 48.49% on 36,000 previously examined photographs, compared with 46.63% in the reverse order. The paired advantage was 1.87 percentage points (within-class 95% interval 1.53 to 2.21, whole-class 1.31 to 2.49). Of the 673 additional correct decisions, 664 occurred on the 20,288 photographs that neither order rescanned. The terminal selections agreed when every photograph received both scans.

From the repository root, use Python 3.11 or later and NumPy.

```sh
OPENBLAS_NUM_THREADS=1 python extensions/scan_order/replay.py --output runs/scan_order
```

`inputs/scores.npz` stores source IDs and labels in calibration and test hash order. The trajectory codes are V (transfer vertical, variant 1), H (transfer horizontal, variant 2), and S (original square). Fields for each split and trajectory give the predicted class, full-precision maximum softmax probability and duration in milliseconds for the hash-placed 30,000-event crop. For example, `test_S_class` and `test_S_duration_ms` belong to the photographs in `test_ids`.

The selector uses the difference between the two confidence scores. At request time, low first-scan confidence and predicted signed gain are alternative allocation criteria. For predicted gain, fivefold out-of-fold terminal decisions supply calibration targets for a ridge regression using confidence and crop duration. The primary learned-gate comparison was vertical to horizontal. The analysis refits the terminal rule and learned gate and reranks requests in 2,000 within-class and 2,000 whole-class bootstrap draws. The fixed-quota gate ranks all first scans together. A separate fixed-cutoff analysis records the realized request rate.

`results/orders.json` contains the calibration-selected order, settings, accuracies, random-request expectations, request counts and intervals. Each test CSV records both path predictions, the terminal class, first-scan acquisition scores, and whether each rule requested the second scan. `results/requested_discordance.csv` gives subgroup counts among the 10,800 requests for each order. The [acquisition verification script](../../analysis/verify_acquisition.py) reconstructs those counts from the decisions.

The [raw-data regeneration instructions](../../regeneration/capture_windows.md) cover the released checkpoints, event archives and conversion of raw inference runs into a fresh score table.
