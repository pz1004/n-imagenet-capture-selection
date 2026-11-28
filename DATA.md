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
| `input_releases.csv` | Filenames, upstream download locations, byte counts and SHA-256 hashes for the eight event archives and four checkpoints. |

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

## Acquisition-transfer files

Paper variants 1 and 2 are the new vertical (8.33 Hz, 4.5 mm) and horizontal (5 Hz, 3 mm) trajectories. Variants 3 and 4 formed the original 5 Hz, 6 mm linear pair. The horizontal same-axis comparison takes variant 2 first and variant 4 second.

| Files | Contents |
|---|---|
| `data/variant{1,2}_{events30k,time20}.csv.gz` | Each of the four files contains both checkpoints and both splits for one acquisition and observation window. Join by source ID. Each record gives the true class, split, crop metadata, predicted class, raw maximum softmax probability and logit margin for each checkpoint. |
| `data/trajectory_baseline_decisions.csv.gz` | The 36,000 test sources for each combination of pair, window and checkpoint. Each row contains both raw and temperature-scaled probabilities for the predicted class, the two predicted classes, the raw and scaled selections, and the class predicted by probability averaging. |
| `tables/trajectory_forecast.json`, `trajectory_forecast_draws.npz`, `trajectory_forecast_receipt.json` | Calibration forecasts, paired bootstrap replicates, and the timestamped receipt recorded before evaluating the held-out conditions. The SHA-256 values for the forecast file and draws in the receipt verify the corresponding files here. |
| `tables/trajectory_gains.json`, `trajectory_baselines.json` | Paired policy gains, directional counts, forecast errors, score fusion, calibration diagnostics and the fixed-axis amplitude comparison. The matched differences in policy effects between acquisition pairs appear in `trajectory_effect_changes.json`. In the gain table, forecast error equals forecast minus observed gain. |
| `tables/trajectory_rescans_{old,new}.csv` | Costs and accuracy at 0%, 10%, 30%, 50% and 100% nominal second-scan requests. `old_frozen` transports a numerical first-scan cutoff. `target_local` estimates it on local calibration sources and refits it for each calibration resample. `within` and `class` limits are paired 95% percentile intervals. The random-request expected accuracy uses the same realized mean scan count and frozen terminal decision. |

The shared tables suffice for recomputing the two forecasts, held-out decision counts, policy contrasts, calibration bins, and acquisition-budget curves. Multiclass temperature refitting, negative log likelihood and probability-vector averaging require 1,000 logits per scan. The regeneration scripts calculate these from the authors’ event archives and checkpoints.

## Transfer breakdown, label budgets and timing records

`tables/trajectory_attribution.json` contains changes in the mix of directional successes, the two conditional success rates, and the corresponding midpoint-weighted contributions to the change in discordant accuracy. The same file reports additional correct classifications per 100 second scans, their paired intervals and the fraction of the always-two gain recovered. `tables/label_budget.csv` gives the annotation budget and error summaries for each checkpoint at 30,000 events. The central 95% range across sampled calibration subsets describes variation among choices from the available pool. For the class-balanced design, the separate containment fraction counts subsets whose entire two-sample 95% error interval fits inside one percentage point of zero.

The provenance directory contains byte-preserved originals from the transfer experiment. The following table maps paths recorded in those receipts to the released copies. Historical paths are relative to the trajectory-transfer analysis unless they begin with an explicit study directory.

| Original artifact | Released location |
|---|---|
| `FROZEN_PROTOCOL.md`, `protocol_receipt.json`, `test_release.json` | Same filenames in `provenance/` |
| `acquisition/input_checks.json`, `acquisition/new_parity.json`, `acquisition/inference_verification.json` | Same filenames in `provenance/` |
| `results/gain/sealed_forecast_verification.json` | `provenance/sealed_forecast_verification.json` |
| `outputs/{split}/{variant}/run.json` | `provenance/{variant}_{split}.json` |
| `forecast_receipt.json`, `results/gain/forecast.json`, `results/gain/forecast_draws.npz` | `tables/trajectory_forecast_receipt.json`, `tables/trajectory_forecast.json`, `tables/trajectory_forecast_draws.npz` |
| `outputs/{split}/{variant}/{window}/predictions.csv` | Retain the header and matching split rows in `data/{variant}_{window}.csv.gz` |
| `manuscript-mva/verification/revised_policy.json`, `full-confirmation/checks/class_map.csv` | `tables/capture_policies.json`, `data/class_map.csv` |

The protocol receipt also contains checksums of earlier working files. The public code implements the replay using the released measurements. Each run record carries the original float32 logit-array checksums. Those arrays can be regenerated with the upstream inputs as described in the README. The 9,000 calibration and 36,000 test prediction rows for each variant and window reproduce the individual CSV hashes in the run records.

## Adaptation, follow-up paths and first-scan event rate

All three extensions join records using the existing `source_id` and class label. Keep the trajectory pair, representation and model state with each result. In the tables, a gain over vertical is the change in accuracy across **all 36,000 test sources**. A yield per 100 extra scans uses the smaller number of requested sources as its denominator.

| Directory | Measurements and comparisons |
|---|---|
| [Adaptation](extensions/adaptation/README.md) | Source-keyed predictions and confidence for the recalibration and calibration-only Ev-TTA arms, fitted request and switching rules, and comparisons to the released model fitted with the same policy-calibration allowance. Includes exact-quota comparisons and the directional breakdown on pairs with one correct scan. |
| [Follow-up path](extensions/path/README.md) | Square-scan scores and derived fusion classes supplement the original linear records. A fixed vertical request mask permits a paired comparison of horizontal and square follow-ups under each terminal rule. |
| [First-scan prefix](extensions/prefix/README.md) | The vertical event count in the first 20 ms and the vertical prediction score supply the request features. Results include fitted rules, per-source decisions at a fixed quota and at a transferred cutoff, and switching thresholds fitted on matching 20-ms reference windows. |

The adaptation comparison conditions on frozen classifier states and policy fits. An exact quota fixes the number of requests but can select different source IDs across classifier states. The path comparison holds both the number and identity of requested sources fixed across alternative paths. The released derived measurements support CPU evaluation. To regenerate full score vectors from individual captures, obtain the event recordings and classifier checkpoints from the upstream providers.

## Acquisition comparisons and five adaptation runs

Each v1.2.0 extension supplies a compressed NumPy score table and source-level decisions. Load a table with `np.load(path, allow_pickle=False)` and inspect its `files` attribute. Source IDs use the same synset/filename-stem convention as `data/source_index.csv`. The calibration and test arrays each follow the frozen class and within-class hash order. Each replay checks that order and its labels against the public source manifest.

| Extension | Inputs | Output unit |
|---|---|---|
| [Scan order](extensions/scan_order/README.md) | First and second classes, maximum softmax probability and hash-window duration on original square and transfer vertical/horizontal | Source and order, with confidence and learned-gain requests |
| [Variant 5](extensions/variant5/README.md) | Two released classifiers on reference vertical, reference horizontal and both square diagonals | Source, model, request rule and follow-up path |
| [Continuation](extensions/continuation/README.md) | Prefix and subsequent-block classes, scores, completeness, event counts, observed spans and averaged-probability classes | Source, request and action, including incomplete requests |
| [Five runs](extensions/multiseed/README.md) | Released and five adapted classifiers on the same 4,500 policy calibration and 36,000 evaluation sources; reference linear paths supply calibration rows | Source, seed, scan order and selector fit |

The included score tables and average-probability classes suffice for CPU policy refitting and both uncertainty analyses. The original event archives and pretrained checkpoints are distributed by the [N-ImageNet authors](https://82magnolia.github.io/n_imagenet/). Follow the [released-model regeneration instructions](regeneration/capture_windows.md) and [five-seed adaptation instructions](extensions/multiseed/RAW_REGENERATION.md) to generate logits from those upstream inputs.

## Adapted continuation, DiST and fusion

The source IDs in all three extensions join to `data/source_index.csv`. The adapted comparison uses 4,500 labeled policy-calibration photographs disjoint from the 4,500 sources reserved for adaptation. DiST also reports the 9,000-source policy fit. The fusion rules use the 9,000 transfer calibration pairs and retain all 1,000 classifier outputs.

| Directory | Released measurements and inference inputs |
|---|---|
| [Adapted continuation](extensions/adapted_continuation/README.md) | Per-source chronological predictions, confidence scores and completion information for the released and five adapted states, with the resulting requests, decisions and bootstrap draws. The regeneration instructions describe how to recompute the chronological scores from the original recordings. |
| [DiST](extensions/dist/README.md) | Per-source measurements from the original square, reference vertical and horizontal, and transfer vertical and horizontal recordings. Results cover the original-condition check, ranking, gain and chronological follow-up comparisons. The provenance record identifies the upstream classifier and input implementation. |
| [Fusion](extensions/fusion/README.md) | Per-source test decisions, request indicators, fitted-rule results and uncertainty estimates. The [logit archive](https://github.com/pz1004/n-imagenet-capture-selection/releases/download/v1.3.0/fusion-transfer-logits.zip) contains the two transfer trajectories for both calibration and test. These arrays are sufficient to refit the product and stacking models. |

The [v1.3.0 results ZIP](https://github.com/pz1004/n-imagenet-capture-selection/releases/download/v1.3.0/capture-selection-results-v1.3.0.zip) collects the three extensions’ numerical summaries. Per-source files and the replay programs accompany the tagged repository. The [N-ImageNet project](https://82magnolia.github.io/n_imagenet/) distributes the raw event recordings and pretrained classifiers.

## File integrity

`data_manifest.csv` lists the relative path, size in bytes and SHA-256 digest of each released data, table and provenance file, including the extension measurements and results. On Linux, run the following in the repository root to check every digest.

```sh
awk -F, 'NR > 1 {print $3 "  " $1}' data_manifest.csv | sha256sum --check
```
