# Combining two scan predictions

On the transfer vertical and horizontal scans, stacking the two probability vectors improved accuracy by 1.02 percentage points over choosing one top prediction when every source received both scans. A common 30% request gate reduced that advantage to 0.63 points. The analysis used 9,000 calibration and 36,000 test photographs with all 1,000 possible output classes. [Paired contrasts, error counts and intervals](RESULTS.md) · [Analysis protocol](PROTOCOL.md)

## Check the published results

Run the following from the repository root with Python 3.11 or later and NumPy. It verifies the source split and request mask, the counts for each policy and the published intervals from 4,000 bootstrap draws.

```sh
python3 extensions/fusion/check_results.py
```

In `results/per_source.csv.gz`, `vertical` and `horizontal` are the individual predictions. `local_selector`, `arithmetic_average`, `product` and `stack` are the classes that each rule returns when given both scans. To score the 30% workload, substitute `vertical` for all rows where `request_30` is zero. The exact quota ranks every first-scan confidence in the cohort.

## Refit from the full scores

The release asset contains four float32 NumPy arrays from the released Event Histogram classifier, one array per split and trajectory. Each row holds 1,000 logits. The leading dimension is 9,000 for calibration or 36,000 for testing. For each array in `{split}/{variant}/histogram.npy`, the row order matches `data/{variant}_events30k.csv.gz` after filtering for that split. Variant 1 is vertical and variant 2 is horizontal. The replay checks each array against its original inference receipt, joins source IDs and labels, and independently checks top classes and confidences. [Download the transfer logits](https://github.com/pz1004/n-imagenet-capture-selection/releases/download/v1.3.0/fusion-transfer-logits.zip)

Use Linux with Python 3.11 or later, NumPy, SciPy and a C++ compiler with OpenMP SIMD support. These commands run from the repository root. Keep about 3 GB of memory available for refitting and verification.

```sh
mkdir -p runs/fusion
curl -fL --retry 3 -o runs/fusion/logits.zip https://github.com/pz1004/n-imagenet-capture-selection/releases/download/v1.3.0/fusion-transfer-logits.zip
unzip -q runs/fusion/logits.zip -d runs/fusion/logits
g++ -O3 -march=native -ffast-math -fopenmp-simd -shared -fPIC extensions/fusion/likelihood.cpp -o extensions/fusion/likelihood.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extensions/fusion/replay.py --logits runs/fusion/logits --output runs/fusion/results --workers 6
python3 extensions/fusion/budget_sensitivity.py --results runs/fusion/results
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 extensions/fusion/verify.py --logits runs/fusion/logits --results runs/fusion/results --receipt runs/fusion/verification.json
```

For just the full-cohort fits, both gate settings and the per-source table, add `--point-only` to the replay command. The full replay recalibrates each path's temperature, the three stacking coefficients and the local selector in every bootstrap draw. It also reranks the requests. Interrupted bootstrap runs resume from saved rows in the output directory. The independent verifier evaluates every held-out prediction across all 1,000 classes, checks all stored intervals and repeats three full-score bootstrap comparisons for each sampling design.

Each row of `results/draws_within.npy` or `results/draws_classes.npy` stores the three stacking coefficients, the two inverse temperatures, maximum absolute gradient, Newton iteration count, then ten counts of correct decisions. The first five counts are for full acquisition and the last five are for 30% requests. Within each group the methods are vertical alone, local selection, probability average, product and stacking. `results/budget_sensitivity.json` compares accuracy at 30% and full acquisition and counts the corrections and losses on the additional 25,200 photographs.
