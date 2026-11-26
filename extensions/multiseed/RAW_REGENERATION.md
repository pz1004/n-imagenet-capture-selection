# Regenerating the five Histogram adaptation runs

The [compact scores](inputs/scores.npz) are sufficient to reproduce the policy comparisons on a CPU. This procedure starts from the upstream event recordings and full 1,000-class Histogram checkpoint. It fits and freezes a separate classifier for every requested seed and trajectory, then saves all 1,000 logits for each scored photograph.

The [input manifest](../../data/input_releases.csv) gives the download locations, filenames, sizes and SHA-256 hashes. Set `ARCHIVES` to the directory containing the five full validation ZIPs and `HISTOGRAM` to the downloaded full Event Histogram checkpoint. Run from the repository root, with CUDA-enabled PyTorch and torchvision matching your system. The [regeneration dependencies](../../regeneration/requirements.txt) specify the versions used. The adaptation needs only the pinned Ev-TTA loss file from the upstream checkout.

```sh
python -m pip install -r regeneration/requirements.txt
mkdir -p runs
git clone https://github.com/82magnolia/ev_tta.git runs/ev_tta
git -C runs/ev_tta checkout 130fbd35132a25c609f3d32f3e32a1cc426078e2

for pair in vertical:extracted_val_mode_6.zip horizontal:extracted_val_mode_7.zip \
            variant1:extracted_val_mode_1.zip variant2:extracted_val_mode_5.zip \
            original:extracted_val.zip; do
    trajectory=${pair%:*}
    archive=${pair#*:}
    python extensions/multiseed/regenerate_histogram.py "$trajectory" \
        --archive "$ARCHIVES/$archive" --histogram "$HISTOGRAM" \
        --ev-tta runs/ev_tta --output "runs/multiseed/$trajectory"
done
```

The default fits all five seeds. For a single run, add `--seeds 2` and give it a new output directory. An optional `--cache-directory` stages one compressed archive on fast local storage for the duration of that trajectory. Allow up to 20 GB of additional disk space for the copy. The program keeps event-channel indices for 4,500 adaptation recordings in host memory and trains on one GPU. Full regeneration produces approximately 2.6 GB of float32 logit arrays, in addition to the original data.

The [frozen source manifest](../../data/source_index.csv) specifies the split. Each model uses five unlabeled calibration photographs per class for adaptation and the other five per class for policy fitting. Seed 1 uses the original hashed batch order and three auxiliary crop draws. Seeds 2–5 prefix each of those hashes with `seed-2/` through `seed-5/`. The 30,000-event anchor uses the same `crop/` hash for every seed and trajectory. The batch size is 64 with 71 updates per fit. Only batch-normalization scales, offsets and running statistics change. No evaluation photograph contributes to adaptation. The script verifies the archive, checkpoint, source/synset inventory and 32 released-model predictions before training.

Each output directory contains `seedN_bn.pt` (adapted batch-normalization state), `logits_seedN.npy` (one float32 logit per class), `predictions.csv.gz` (source ID, split, label and each seed's top class and maximum softmax probability) and `receipt.json` (training-order and output checksums, batch losses, software versions and verification errors). The CSV gives the row order of every logit array for that trajectory. The two reference trajectories need 4,500 policy predictions each to fit the transported selector. Transfer vertical, transfer horizontal and the original square each need 4,500 policy predictions and 36,000 test predictions.

For each regenerated row, the script joins the source ID to the matching row of `policy_ids` or `test_ids` in the compact release. It verifies the label and predicted class exactly, and the float64 softmax maximum to absolute tolerance `1e-5`. The arrays `<split>_<trajectory>_class` and `<split>_<trajectory>_msp` store the released classifier at row 0 and adaptation seeds 1–5 at rows 1–5. Different CUDA or convolution implementations can produce numerical differences. A failed comparison stops the run and reports the mismatch.

For the release check, regenerating transfer vertical (`variant1`) with seed 2 reproduced all 40,500 original top classes and maximum softmax probabilities exactly. An independent comparison against the archived float32 logits found zero differences among 40,500,000 values. All 180 batch-normalization state tensors and all 71 recorded batch losses also matched.
