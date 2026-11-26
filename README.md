# N-ImageNet capture selection

When does a second physical scan improve recognition, and can the first scan identify the photographs worth scanning again? This repository contains paired predictions and analysis code for vertical, horizontal and square N-ImageNet recordings of the same photographs.

The Mini N-ImageNet pilot used 100 classes. The confirmation used the remaining 900 classes, with 9,000 images for cutoff calibration and selector fitting and 36,000 for evaluation. With Event Histogram and up to 30,000 events per capture, confidence ranked the correct scan higher in 75.61% of vertical-only successes and 68.33% of horizontal-only successes. The equally weighted mean was 71.97%. Across all 36,000 images, selecting the more confident scan corrected 2,015 errors and lost 1,875 correct predictions relative to always using the vertical scan. Accuracy rose from 37.08% to 37.47%.

## Adaptation and follow-up path comparisons in v1.1.0

These analyses reuse the 36,000 previously evaluated test photographs. Each asks what an optional second capture adds after a prescribed vertical pass. The cost measure is the number of captures in offline replay.

| Analysis | Comparison on the transfer trajectories with Event Histogram |
|---|---|
| [Calibration-only adaptation](extensions/adaptation/README.md) | Directed horizontal requests on 30.10% of sources raised accuracy from 43.94% for adapted vertical alone to 46.00%. At exactly 10,800 requests per classifier state, the gain over each state's vertical baseline was 1.32 percentage points larger after Ev-TTA adaptation (paired 95% within-class interval 1.07–1.60). |
| [Choice of follow-up trajectory](extensions/path/README.md) | The same 11,076 sources received a horizontal or square follow-up with the unadapted classifier. Path-specific fitted selectors reached 43.84% and 46.69%, respectively (square minus horizontal 2.85 points, 95% interval 2.58–3.11). A single square capture reached 46.84% when the initial path was freely selectable. |
| [First 20 ms of the vertical scan](extensions/prefix/README.md) | At 10,800 requests, adding the vertical event rate to confidence increased net corrections from 434 to 516, or 0.23 points across all sources (95% interval 0.09–0.35). Transferring the numerical cutoff for the combined score requested only 11.41% of sources. |

Each linked directory gives the CPU replay commands, source-level measurements or decisions, and results for both Event Histogram and Timestamp Image. The adaptation arm used disjoint sets of 4,500 calibration photographs for unlabeled model updates and for fitting the request and switching rules. The follow-up trajectory comparison used the released, unadapted classifiers. Instructions for regenerating adapted scores from the upstream recordings and checkpoints are included with the adaptation analysis.

[Download the aggregate results for v1.1.0](https://github.com/pz1004/n-imagenet-capture-selection/releases/download/v1.1.0/capture-selection-results-v1.1.0.zip)

## Reproduce the comparisons

From the repository root, use Python 3.11 or later in a virtual environment. The analysis of the included tables requires NumPy and runs on a CPU.

```sh
python -m pip install -r requirements.txt
python analysis/verify.py --bootstrap
```

The verification command checks the file hashes, recomputes counts and rates from saved predictions, and checks the primary bootstrap intervals. To recompute individual analyses, run the following from the same directory.

```sh
python analysis/ranking.py events30k histogram msp
python analysis/paired_intervals.py --output intervals.json
python analysis/capture_policies.py --output policies.json
python analysis/orientation.py events30k histogram --output orientation.json
```

The ranking analysis reports each directional win rate, their equal-weight mean, and how a cutoff calibrated to 70% coverage transfers between directions. The paired intervals compare outcomes on the same images and also resample whole classes. The capture-policy analysis compares the raw confidence rule with an exploratory fitted threshold and a single square scan. The orientation analysis tests whether whole-photograph gradient energy improves prediction after accounting for confidence, event count and crop duration.

## Transfer to two further trajectories

The transfer test uses paper variants 1 and 2, with variant 1 as the first capture. The new acquisition settings cover the same 9,000 calibration and 36,000 test photographs as the original linear pair. The following commands recompute the two calibration forecasts, compare the held-out policies, reconstruct the accuracy-versus-acquisition curves, and check temperature-scaled choices and probability averaging.

```sh
python analysis/verify_chronology.py
python analysis/trajectory_transfer.py --bootstrap --output runs/trajectory_gain.json
python analysis/trajectory_sequential.py --pair old --output runs/sequential_old.csv
python analysis/trajectory_sequential.py --pair new --output runs/sequential_new.csv
python analysis/trajectory_baselines.py
```

On the new pair, the frozen Histogram selector gained 1.07 percentage points over the first scan. The forecast carried over from the original pair was +2.06 points, while the direct calibration forecast was +1.18. With a second scan requested for 30.77% of images, first-scan confidence gave 43.85% accuracy compared with 43.46% for random requests at the same rate (paired difference +0.39 points, within-class 95% interval +0.27 to +0.51).

The directed requests yielded 2.32 net additional correct classifications per 100 extra scans, compared with 1.07 for random requests at the same rate. A single training-matched square scan reached 46.84%. These comparisons count acquisitions during offline replay.

## Gain transfer and the number of target labels

The transfer breakdown measures how much the change in accuracy on discordant pairs comes from the mix of directional successes and from the fraction of each subgroup the rule selects correctly. The labeling analysis estimates how accurately the gain can be measured using fewer labeled source photographs. Its second comparison restricts labeling to photographs where the frozen selector changes the predicted class. Both approaches use recordings from both trajectories on the calibration pool.

From a clean clone, recompute the released retrospective tables with the following commands. The output directories are created by the scripts. With the original NumPy environment, the regenerated tables match the released files byte for byte. Cross-platform comparisons can use numerical tolerance for floating-point output.

```sh
python analysis/trajectory_attribution.py --output runs/trajectory_attribution.json
python analysis/label_budget.py --output runs/label_budget.csv
cmp runs/trajectory_attribution.json tables/trajectory_attribution.json
cmp runs/label_budget.csv tables/label_budget.csv
```

The label-budget estimates use NumPy. To reproduce the plot, install Matplotlib and add `--figure runs/label_budget.pdf` to the label-budget command.

The [frozen trajectory-transfer protocol](provenance/FROZEN_PROTOCOL.md) and [test release record](provenance/test_release.json) document the authors' recorded sequence of calibration, forecasting and test inference. The chronology check verifies the hashes and ordering across these files, including the eight prediction CSVs reconstructed from the combined data. The [data guide](DATA.md) maps the original file names to the public artifacts. The later repository release archives these records.

## Data and inputs

The `data/` directory holds the source assignments, paired scores and predictions, crop measurements, image-orientation features and derived comparison labels. The `tables/` directory contains the estimates and uncertainty intervals. File names, join keys, class labels, units and integrity checks are documented in the [data guide](DATA.md).

The upstream authors distribute the event recordings and trained checkpoints. Archive names, release identifiers and SHA-256 hashes are in [the input inventory](data/input_releases.csv). Access to the original photographs is managed by ImageNet. [N-ImageNet project](https://82magnolia.github.io/n_imagenet/) · [ImageNet downloads](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)

## Regenerate measurements from the original inputs

The optional scripts in `regeneration/` work with locally obtained inputs. Install their dependencies with `python -m pip install -r regeneration/requirements.txt`. For example, to recreate the full vertical Histogram and Timestamp Image predictions, give the script the vertical validation archive, the two full-model checkpoints and a new output directory.

```sh
python regeneration/infer.py full vertical events30k \
  --archive /path/to/extracted_val_mode_6.zip \
  --histogram "/path/to/Event Histogram/best.tar" \
  --timestamp "/path/to/Timestamp Image/best.tar" \
  --output runs/vertical
```

For the matching horizontal run, select `horizontal` and supply the mode 7 archive. Each run writes a per-image row index and both logit arrays. The original inference used batches of 32. A different batch size can slightly change floating-point confidence scores. With both runs complete, use `python regeneration/fusion.py runs/vertical runs/horizontal runs/fusion` to recompute the probability-averaging comparison.

To regenerate the trajectory-transfer logits, use `full variant1` (mode 1) or `full variant2` (mode 5) with the corresponding archive and specify `--split calibration` or `--split test`. For example, give the four result folders for one window to the temperature and fusion script.

```sh
python regeneration/trajectory_baselines.py --model histogram \
  --first-cal runs/variant1-cal --first-test runs/variant1-test \
  --second-cal runs/variant2-cal --second-test runs/variant2-test \
  --output runs/new_events30k_histogram_baselines.csv
```

To regenerate the whole-photograph orientation measurements, run `python regeneration/rgb_features.py --images /path/to/validation-jpegs --output runs/orientation-features.csv`. Both the inference and RGB commands accept `--source-id` to process one photograph. The supplied tables support the analyses above without downloading any raw inputs.

To repeat the display-registration check, run `python regeneration/registration.py --images /path/to/validation-jpegs --vertical /path/to/extracted_val_mode_6.zip --horizontal /path/to/extracted_val_mode_7.zip --output runs/registration`. The script fits the display geometry on 60 calibration photographs and evaluates it on another 60. The study used the whole-photograph measure because the fitted geometry missed the prespecified accuracy criterion.
