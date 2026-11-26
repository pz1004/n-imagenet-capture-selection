# Variant 5 square follow-up

The two square trajectories ran counterclockwise at 5 Hz. The original square had a 3 mm diagonal and matched training. Variant 5 had a 6 mm diagonal (release mode 3). The first scan was reference vertical (variant 3); the horizontal follow-up was variant 4. The 6 mm linear amplitude and square diagonal are different geometric measures.

The first-scan confidence rule requested a second recording on 10,800 photographs per classifier. Directly accepting the 6 mm square instead of horizontal gave 38.70% versus 36.86% accuracy with Event Histogram, a paired difference of 1.83 percentage points (95% within-class interval 1.60 to 2.06, whole-class 1.49 to 2.17). Timestamp Image gave a difference of 1.61 points. For Histogram, the original square reached 42.10%, 3.40 points above variant 5. These figures describe policies evaluated on the 36,000 source photographs held out from calibration.

```sh
OPENBLAS_NUM_THREADS=1 python extensions/variant5/replay.py --output runs/variant5
```

The [protocol](PROTOCOL.md) and [calibration rules](calibration_rules.json) are the original recorded files. The source photographs and the previously published benchmark accuracies were known. The paired contrast and common request policy were fixed before the new trajectory was evaluated on the test recordings. Both representations passed the original-condition and calibration preprocessing checks. The [run chronology](chronology.json) binds the calibration rules and the two original inference receipts to their SHA-256 digests. The calibration freeze preceded test inference by 1.37 seconds. This is a record of the original local sequence and subsequent public deposit.

In `inputs/scores.npz`, arrays such as `test_variant5_histogram_class` and `test_variant5_histogram_msp` align with `test_ids` and `test_labels`. The other arm names are `vertical`, `horizontal` and `original`. Analogous arrays cover the 9,000 calibration photographs and the Timestamp classifier. The replay refits all three selectors per representation and checks the frozen cutoffs and test request-ID digests. It reproduces 2,000 paired replicates per resampling design. The result JSONs give counts, percentage estimates and both intervals for direct acceptance and fitted selection, at the quota and the fixed cutoff. The compressed test CSVs give source-level classes and requests. The raw inference receipts record checksums of the original full logit arrays. [Raw-data regeneration](../../regeneration/capture_windows.md)
