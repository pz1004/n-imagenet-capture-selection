# Calibration-only adaptation and second-scan requests

This experiment compares two adapted classifiers with their released counterparts on the same 36,000 held-out photographs. Each source has paired linear scans in a reference setting (modes 6 and 7) and a transfer setting (modes 1 and 5). A first-scan confidence cutoff determines who gets a second scan. A separately fitted rule chooses which class prediction to return when both scans are available.

For Event Histogram on the transfer pair, the Ev-TTA calibration variant raised accuracy from 43.94% with the first scan alone to 46.00% with 10,837 second scans. Its gain was 2.06 percentage points (within-class 95% interval 1.84 to 2.28). When both the released and adapted classifiers received exactly 10,800 second scans, adaptation increased the gain over each classifier's own first scan by 1.32 points. The two policies may request different photographs at the same quota.

## Reproduce the results from the released predictions

Run from the repository root with Python 3.11 or later.

```sh
python -m pip install -r requirements.txt
python extensions/adaptation/replay.py --bootstrap --verify --output runs/adaptation
```

The output directory must be new. Omit `--bootstrap` for a faster check of the policy fits, event counts, individual decisions and point estimates. The complete command also verifies 2,000 paired bootstrap replicates per resampling design through their reported intervals.

Of the ten calibration photographs per class in the frozen split, the first five in within-class hash order supply unlabeled model updates (4,500 sources). The other five supply labels to fit the first-scan cutoff and terminal switching rule on the reference pair (4,500 sources). Each trajectory has its own adapted model. Both pairs use the same reference-fitted cutoff and switching rule for a given classifier and state. In this experiment, the released classifier also uses the smaller, 4,500-source policy fit. The 36,000 test photographs enter neither fit. [Source assignments](../../data/source_index.csv)

| Files | Contents |
|---|---|
| `data/{vertical,horizontal,variant1,variant2}.csv.gz` | Source ID, split, label, predicted class and maximum softmax probability (`msp`) for Event Histogram and Timestamp Image under both adaptations. Each file has 4,500 policy and 36,000 test sources. The repository's [data guide](../../DATA.md) identifies the original released-classifier scores. |
| `data/*_decisions.csv.gz` | One row per held-out source for each pair and representation. The `all_rescan_class` field gives the output if everyone gets a second scan. `request` and `selected_class` describe the cutoff policy. Fields with the `quota_` prefix use exactly 10,800 requests. |
| `tables/reference_fits.json` | Standardization parameters, intercept and slope for each switching rule, the equivalent confidence-difference threshold, and the first-scan request cutoff. |
| `tables/calibration_budget.json` | Event counts on the 4,500 update sources for each trajectory. Distinct counts measure the union of event indices in the anchor and three auxiliary crops, counting overlap once. |
| `tables/{reference,transfer}_{histogram,timestamp}.json` | Counts, estimates and 95% intervals for the confidence cutoff (`gated`), the all-rescan decision among pairs with exactly one correct scan (`discordance`), and the exact request count (`quota`). |

In these files, `raw` denotes a released classifier, `bn` denotes batch-normalization moment recalibration, and `evtta` denotes the calibration-only Ev-TTA variant. All three use fitted switching rules. Accuracy and request rates are percentages. Gain is the percentage-point change from the same classifier's first-scan accuracy. The random-request comparator gives expected accuracy at each policy's request rate. Among discordant pairs, `v_share_pct` is the first scan's share of the correct predictions and `selector_success_pct` is the fraction correctly selected. The membership of that subgroup can change with adaptation.

The `within` intervals resample photographs within each class. The `class` intervals resample whole classes. Contrasts use identical draws across classifier states and condition on the fitted models and policies. The offline quota ranks sources by first-scan confidence, including bootstrap multiplicities, to request exactly 10,800 scans in each draw.

## Regenerate the adapted predictions and logits

GPU regeneration uses the four full N-ImageNet archives, both full 1,000-output checkpoints, and the Ev-TTA loss implementation. Filenames, download locations and input checksums are in the [input manifest](../../data/input_releases.csv). The commands below assume that `ARCHIVES` contains the archive files and `CHECKPOINTS` contains the two checkpoint directories.

```sh
python -m pip install -r regeneration/requirements.txt
git clone https://github.com/82magnolia/ev_tta.git runs/ev_tta
git -C runs/ev_tta checkout 130fbd35132a25c609f3d32f3e32a1cc426078e2
python extensions/adaptation/check_objective.py --ev-tta runs/ev_tta

for item in vertical/6 horizontal/7 variant1/1 variant2/5; do
    condition=${item%/*}
    mode=${item#*/}
    python extensions/adaptation/regenerate.py "$condition" \
        --archive "$ARCHIVES/extracted_val_mode_${mode}.zip" \
        --histogram "$CHECKPOINTS/Event Histogram/best.tar" \
        --timestamp "$CHECKPOINTS/Timestamp Image/best.tar" \
        --ev-tta runs/ev_tta --output "runs/adaptation-inference/$condition"
done

python extensions/adaptation/replay.py --scores runs/adaptation-inference \
    --bootstrap --output runs/regenerated-analysis
```

The objective check compares the scalar loss and its gradient against the pinned upstream trainer on a fixed mixed-consistency batch. Each condition starts by checking archive and checkpoint hashes, unique source IDs, and the released top class and confidence for 32 calibration recordings per classifier. Use `--check-only` to stop after those checks. The optional `--cache-directory` stages a compressed archive on fast local storage and needs up to 19.4 GB of free space. GPU inference writes four float32 logit arrays of shape (40,500, 1,000), the adapted BN parameters and moments, training diagnostics, and the compact per-source predictions. Allow about 650 MB of output space per trajectory. Row order in the predictions file matches the first axis of each logit array. The original recording archive, prediction CSV and four logit files for each trajectory have SHA-256 digests in [the inference fingerprints](tables/inference_hashes.json).

For this release, we regenerated the transfer vertical trajectory from its recording archive. The prediction CSV and all four logit arrays matched the original study files byte for byte. The CPU policy replay checks both trajectory pairs and both representations.

Moment recalibration uses one anchor crop of up to 30,000 events per calibration recording. The Ev-TTA variant uses that anchor and three deterministic temporal crops, with overlap allowed. It updates BN scale, offset and running moments in one calibration pass. Inference freezes those parameters and uses the original single anchor. The experiment evaluates offline calibration followed by fixed-model inference. GPU kernels and library versions can affect floating-point results when regenerating the adapted model. The released per-source scores support CPU replay independently of the GPU environment. [Upstream Ev-TTA](https://github.com/82magnolia/ev_tta)
