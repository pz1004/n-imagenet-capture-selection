# N-ImageNet capture selection

How often does a classifier assign higher confidence to the correct scan of an image, and how many additional images does it classify correctly by choosing between scans? This repository contains paired predictions and analysis code for vertical and horizontal N-ImageNet recordings of the same source photographs.

The Mini N-ImageNet pilot used 100 classes. The confirmation used the remaining 900 classes, with 9,000 images for cutoff calibration and selector fitting and 36,000 for evaluation. With Event Histogram and up to 30,000 events per capture, confidence ranked the correct scan higher in 75.61% of vertical-only successes and 68.33% of horizontal-only successes. The equally weighted mean was 71.97%. Across all 36,000 images, selecting the more confident scan corrected 2,015 errors and lost 1,875 correct predictions relative to always using the vertical scan. Accuracy rose from 37.08% to 37.47%.

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

To regenerate the whole-photograph orientation measurements, run `python regeneration/rgb_features.py --images /path/to/validation-jpegs --output runs/orientation-features.csv`. Both the inference and RGB commands accept `--source-id` to process one photograph. The supplied tables support the analyses above without downloading any raw inputs.

To repeat the display-registration check, run `python regeneration/registration.py --images /path/to/validation-jpegs --vertical /path/to/extracted_val_mode_6.zip --horizontal /path/to/extracted_val_mode_7.zip --output runs/registration`. The script fits the display geometry on 60 calibration photographs and evaluates it on another 60. The study used the whole-photograph measure because the fitted geometry missed the prespecified accuracy criterion.
