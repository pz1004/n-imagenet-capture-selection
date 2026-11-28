# Discounted Sorted Timestamp Image

The full N-ImageNet DiST classifier recognized 24,217 of the 50,000 original square recordings (48.434%) with a deterministic crop of up to 30,000 events. Kim et al. reported 48.43% using the upstream evaluation procedure. The analysis used the 1,000-way ResNet-34 checkpoint. [N-ImageNet paper, Table 4](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_N-ImageNet_Towards_Robust_Fine-Grained_Object_Recognition_With_Event_Cameras_ICCV_2021_paper.pdf) · [Official DiST checkpoint and configuration](https://drive.google.com/drive/folders/14mdWMeqYSvBGrvYvsydKbGJURuMzn2SW)

On the matched reference paths, confidence ranked the correct scan first for 6,336 of 8,428 vertical-only successes and 1,908 of 2,789 horizontal-only successes. The equal-weight mean was 71.79% (within-class 95% interval 70.83–72.77, whole-class interval 70.45–73.05). Choosing the higher-confidence prediction corrected 1,908 vertical errors and lost 2,092 correct vertical predictions, reducing accuracy by 0.51 percentage points across 36,000 evaluation photographs. [Ranking results](results/ranking.json)

In chronological replay, a cutoff fitted on 4,500 labeled policy sources requested another observation for 11,070 of those 36,000 photographs. Continuing the vertical stream reached 46.16% accuracy, switching to a horizontal recording reached 45.08%, and switching to the original square reached 47.52%. The paired horizontal-minus-continuation difference was −1.09 points (within-class interval −1.31 to −0.79, whole-class interval −1.37 to −0.72). Fitting on all 9,000 calibration sources gave a difference of −0.94 points. Each action used the same first observation and request decision within a fit. Incomplete requests retained the first prediction. These are retrospective comparisons on recordings previously evaluated in the study. [4,500-source fit](results/chrono_4500.json) · [9,000-source fit](results/chrono_9000.json)

## Reproduce the results

Run the following commands from the repository root with Python 3.11 or later, NumPy and SciPy. The released tables contain the measurements needed for CPU replay. The verifier checks the original-condition count, reference ranking and all 216,000 held-out arm decisions with independent terminal fits. The replay refits each policy in 2,000 within-class and 2,000 whole-class draws for each calibration size.

```sh
python -m pip install -r requirements.txt
python -m pip install scipy
OPENBLAS_NUM_THREADS=1 python extensions/dist/verify.py
OPENBLAS_NUM_THREADS=1 python extensions/dist/replay.py --output runs/dist
python extensions/dist/verify.py --replayed runs/dist
```

For point estimates and decision tables alone, add `--draws 0` to the replay command and `--points-only` to the final verification. The ranking and acquisition analyses can also run separately using `--analysis ranking` or `--analysis chrono`.

## Files

| Location | Contents |
|---|---|
| `inputs/{original,vertical,horizontal,variant1,variant2}.csv.gz` | One row per source per recorded trajectory. The original square contains 50,000 rows, including the 5,000 excluded Mini photographs. Each other trajectory has 45,000 rows. |
| `results/ranking_decisions.csv.gz` | Both top classes and maximum softmax probabilities (MSP), source label and raw selector prediction on each of the 36,000 reference pairs. |
| `results/chrono_decisions_{4500,9000}.csv.gz` | First prediction and MSP, cutoff and quota request flags, candidate completion and terminal class. Apply the request flag to the terminal class and retain the first class otherwise. |
| `results/*.json` | Point estimates, counts and fitted coefficients, or lower and upper bootstrap limits. Files ending in `_within` resample sources within class. Files ending in `_class` resample entire classes jointly across the calibration and test sets. |
| `provenance.json` | Code revision and input hashes, recorded per-array logit hashes, original-condition gate, software and hardware, and all 21 calibration checks against the upstream input tensors and logits. |

The trajectory rows use the original archive order. Join them by `source_id`. The `hash` columns describe the deterministic, source-hash-placed window, `first` the first chronological block, and `second` the following block. Start indices are inclusive and stop indices exclusive. A complete block contains 30,000 events. Event timestamps are microseconds. A missing block has start equal to stop and timestamps of −1. Its prediction is ignored in the decision rule. The `class` columns are zero-based indices and `msp` columns are probabilities. [Source assignments](../../data/source_index.csv) · [Class mapping](../../data/class_map.csv)

## Regenerate logits or repeat the upstream parity check

Download the official full DiST `best.tar`, its `config.ini`, and the desired N-ImageNet event archive. The raw regeneration command verifies the file hashes, reconstructs the two polarity channels, saves float32 logits in source-row order and writes their top classes and MSP to `sources.csv`. It requires PyTorch, torchvision and a C++17 compiler on Linux. The study used a batch size of 32. Device and batch size can affect the last digits of the scores. [Recording archive inventory](../../data/input_releases.csv) · [N-ImageNet project](https://82magnolia.github.io/n_imagenet/)

```sh
python -m pip install -r regeneration/requirements.txt
python extensions/dist/regenerate.py original \
  --archive /path/to/extracted_val.zip --weights /path/to/DiST/best.tar \
  --config /path/to/DiST/config.ini --output runs/dist-origin
```

Use `vertical` and `horizontal` for reference modes 6 and 7. Use `variant1` and `variant2` for the chronological transfer modes 1 and 5. The original contains both the hash and first windows. The transfer vertical contains both chronological blocks. To run a single photograph, append `--source-id` and its synset/filename stem.

The optional command below runs the authors’ loader and model on the same calibration crops as our encoder and classifier. It also needs a clean upstream checkout at commit `6815335a35b5bdf65623b3e6b1acbe3fe74b2e33`, the official ImageNet devkit `mapping.txt`, and the upstream dependency `torch_scatter` for the installed PyTorch version. Repeat it with the corresponding archive for each condition to cover all 21 probes. [Pinned upstream implementation](https://github.com/82magnolia/n_imagenet/tree/6815335a35b5bdf65623b3e6b1acbe3fe74b2e33)

```sh
python extensions/dist/parity.py original --upstream /path/to/n_imagenet \
  --archive /path/to/extracted_val.zip --weights /path/to/DiST/best.tar \
  --config /path/to/DiST/config.ini --mapping /path/to/mapping.txt
```
