# Regenerate the recordings used for the acquisition comparisons

The release includes the compact input arrays required for the CPU analyses. The following steps regenerate those arrays from the published event recordings and full checkpoints. Run from the repository root with the packages in `regeneration/requirements.txt`. The two capture classifiers trained on the original 3-mm square. Variant 5 uses a different square trajectory with a 6-mm diagonal.

Obtain the indicated validation ZIPs and full Event Histogram and Timestamp Image checkpoints from the upstream authors. Each inference command verifies the input sizes and SHA-256 digests against the [input inventory](../data/input_releases.csv). Original RGB photographs are unnecessary for these comparisons. [N-ImageNet recordings](https://huggingface.co/datasets/82magnolia/N-ImageNet/tree/a7c6726b1a2e0f1b4ff9c8f4726c57cda0269146/validation) · [N-ImageNet models and documentation](https://82magnolia.github.io/n_imagenet/)

Set the three paths to your local copies. On a CUDA device the study used batches of 32 and float32 inference. Use `--device cpu` if required. Small floating-point differences in other environments can affect a decision near a tie or cutoff.

```sh
ARCHIVES=/path/to/validation
HIST=/path/to/Event-Histogram/best.tar
STAMP=/path/to/Timestamp-Image/best.tar
```

## Variant 5 and scan order

The main capture window contains up to 30,000 consecutive events. The source ID fixes its fractional starting position in the recording. The capture window and the 20-ms sensitivity both remain available from `regeneration/infer.py`. The reported variant 5 comparison uses only the 30,000-event window. For the original square, the inference file also contains Mini sources for compatibility with the original sanity check. The input builder selects only the 900-class frozen cohort.

| Condition | Validation archive | Analysis |
|---|---|---|
| `original` | `extracted_val.zip` | Both |
| `vertical` | `extracted_val_mode_6.zip` | Variant 5 |
| `horizontal` | `extracted_val_mode_7.zip` | Variant 5 |
| `variant5` | `extracted_val_mode_3.zip` | Variant 5 |
| `variant1` | `extracted_val_mode_1.zip` | Scan order |
| `variant2` | `extracted_val_mode_5.zip` | Scan order |

This shell function evaluates both representations for the calibration and test sources separately. Each destination must be new. Each run writes the per-source measurements and the two 1,000-column logit arrays in recording order.

```sh
score_capture() {
    for split in calibration test; do
        python regeneration/infer.py full "$1" events30k \
            --archive "$ARCHIVES/$2" --histogram "$HIST" --timestamp "$STAMP" \
            --split "$split" --output "runs/hashed/$1/$split" || return 1
    done
}
score_capture original extracted_val.zip
score_capture vertical extracted_val_mode_6.zip
score_capture horizontal extracted_val_mode_7.zip
score_capture variant5 extracted_val_mode_3.zip
score_capture variant1 extracted_val_mode_1.zip
score_capture variant2 extracted_val_mode_5.zip
python regeneration/acquisition_inputs.py variant5 --runs runs/hashed --output runs/variant5.npz
python regeneration/acquisition_inputs.py scan_order --runs runs/hashed --output runs/scan_order.npz
```

The optional `--source-id synset/filename-stem` on an inference command scores one photograph and records its row index. Building the policy inputs requires the complete 9,000 calibration and 36,000 test sources. Two variant 5 recordings had a single out-of-order final tick after the selected window. The encoder verifies the recorded exception and uses the original record length to place the crop.

## Chronological continuation

The first vertical observation uses events 1–30,000. The second uses events 30,001–60,000. The pooled input counts the events in both blocks. Horizontal and square use only their first blocks. These commands generate all 45,000 calibration and test rows per trajectory, with row IDs, counts, elapsed-time boundaries, predicted classes and maximum softmax scores in `sources.csv`. Separate float32 arrays preserve the 1,000 class logits from each view.

```sh
python regeneration/continuation.py variant1 --archive "$ARCHIVES/extracted_val_mode_1.zip" \
    --histogram "$HIST" --output runs/chronological/variant1
python regeneration/continuation.py variant2 --archive "$ARCHIVES/extracted_val_mode_5.zip" \
    --histogram "$HIST" --output runs/chronological/variant2
python regeneration/continuation.py original --archive "$ARCHIVES/extracted_val.zip" \
    --histogram "$HIST" --output runs/chronological/original
python regeneration/acquisition_inputs.py continuation --runs runs/chronological --output runs/continuation.npz
```

The input builder joins all trajectories by source ID and checks labels. It computes probability-averaging predictions from the corresponding logit vectors. Short or empty second blocks have a stored model output. The replay uses the first prediction on incomplete requests and charges the observed part of the requested window, including the gap between successive vertical blocks.

## Replay the policies

The generated files have the same array names and split order as `extensions/<analysis>/inputs/scores.npz`. Refit and evaluate the calibration policies and bootstrap intervals with the regenerated inputs.

```sh
python extensions/variant5/replay.py --inputs runs/variant5.npz --output runs/variant5-results
python extensions/scan_order/replay.py --inputs runs/scan_order.npz --output runs/scan-order-results
python extensions/continuation/replay.py --inputs runs/continuation.npz --output runs/continuation-results
```
