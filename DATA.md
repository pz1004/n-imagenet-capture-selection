# Data guide

The study tables contain predictions and measurements for physical event-camera scans of images from the ImageNet validation set. The event recordings and classifier weights come from Kim et al.'s N-ImageNet release. The whole-image gradient measurements use the ILSVRC 2012 source photographs. The original inputs are available through their respective providers. [N-ImageNet project](https://82magnolia.github.io/n_imagenet/) · [ImageNet downloads](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)

## Matching images and scans

Join records on `source_id`, which combines the class synset and the ImageNet validation filename stem. The ID identifies the photograph across trajectories. Match the cohort, model and observation window before comparing predictions. In the original-scan table, use both `model_set` and `source_id` because the full checkpoints also evaluated all Mini sources.

| Cohort in the paired tables | Classes | Calibration images | Test images | Outputs per classifier |
|---|---:|---:|---:|---:|
| `mini` | 100 | 1,000 | 4,000 | 100 |
| `full` | 900 other classes | 9,000 | 36,000 | 1,000 |

Every class contributes 10 calibration and 40 test images. Within class, the split orders sources by the first 64 bits of SHA-256 of `split/` followed by the source ID. The first 10 enter calibration. Both checkpoints, windows and scan directions use the same split within a cohort. The test sources were held out from the study's cutoff calibration and selector fitting.

## Per-image measurements

These files are in `data/`. In filenames, `events30k` denotes a contiguous crop of up to 30,000 events and `time20` denotes the first 20 ms, excluding the endpoint. The classifier names are `histogram` and `timestamp`. The suffixes `_v` and `_h` denote the vertical and horizontal scans.

| Files | Contents |
|---|---|
| `source_index.csv`, `class_map.csv` | Assignments for 50,000 unique images and the mapping from synsets to zero-based class labels. In the class map, `excluded_mini` identifies the classes used in the pilot. |
| `*_pairs.csv` (8) | One row per source for each cohort, crop and model. The two directions each have a predicted class, correctness indicator, maximum softmax probability (`msp`) and difference between the two largest logits (`margin`). |
| `*_windows.csv` (4) | Full-recording and crop event counts, first and last included event times in microseconds, and zero-duration flags. Subtract the first time from the last and divide by 1,000 to obtain the crop duration in milliseconds. |
| `original_sanity_predictions.csv` | Original square-scan predictions from both classifiers for 50,000 full-model and 5,000 Mini-model sources. |
| `whole_source_features.csv`, `registration_landmarks.csv` | Whole-photograph dimensions and directional gradient energies. The orientation index is `(energy_y − energy_x) / (energy_y + energy_x + 1e−12)`. The registration file records 240 landmark attempts per direction with errors in native sensor pixels. |
| `orientation_*_discordant.csv` (4) | Held-out sources with exactly one correct scan. Rows give the observed direction, orientation bin, predictors, predicted probabilities of vertical-only correctness, and log losses from the baseline and augmented models. |
| `fusion.csv` | Classes selected by higher confidence and by averaging class probabilities for 36,000 held-out sources, using Event Histogram and up to 30,000 events. |
| `input_releases.csv` | Filenames, upstream download locations, byte counts and SHA-256 hashes for the six event archives and four checkpoints. |

## Estimates and uncertainty

The files in `tables/` use fractions for CSV accuracy, ranking and coverage rates. Log losses are in nats per discordant image. JSON fields ending in `_pp` are percentage points. Blank cells mark inapplicable or unestimated quantities.

| Files | What to read |
|---|---|
| `original_sanity.csv`, `crop_summary.csv` | Original-scan accuracy alongside the published full benchmark or Mini release logs, and distributions of event counts and observed crop durations. |
| `joint_correctness.csv`, `ranking.csv`, `selection.csv` | Joint outcomes, directional ranking and selection accuracy. Ranking compares confidence only where one scan is correct, gives ties half credit and averages the two directional rates equally. Selection uses every test source, breaks ties in favor of vertical and counts rescues and losses against always choosing vertical. |
| `transfer.csv` | Cutoffs targeting 70% calibration coverage, then coverage, error among accepted images and error transitions in both transfer directions. Scores equal to the source cutoff are accepted. Numerator and denominator columns identify each rate's population. |
| `registration.csv`, `orientation_activity.csv` | Registration outcomes by direction and the calibration-set association of orientation with the log vertical-to-horizontal event-count ratio. |
| `orientation.csv`, `orientation_coefficients.csv`, `orientation_assignment.csv`, `orientation_quintiles.csv`, `orientation_bootstrap_draws.csv` | Held-out model comparisons, fitted coefficients, within-class reassignment control, results by orientation quintile, and 2,000 paired bootstrap draws for each crop and model. Positive loss reduction favors adding orientation. |
| `capture_policies.json`, `fusion.json`, `paired_intervals.json` | Fitted switching thresholds, square-scan and probability-averaging comparisons, paired contrasts for cutoff transfer and selection, whole-class intervals and a conditional permutation test for ranking. |

In `selection.csv`, each policy row has its own estimate, confidence limits and correct or net count, while every row for the same crop and model repeats the higher-MSP selector's `rescues` and `harms` relative to always choosing vertical and its `ties_to_vertical` count.

## File integrity

`data_manifest.csv` lists the relative path, size in bytes and SHA-256 digest of each of the 39 quantitative files. On Linux, run the following in the repository root to check every digest.

```sh
awk -F, 'NR > 1 {print $3 "  " $1}' data_manifest.csv | sha256sum --check
```
