# Chronological continuation after trajectory-specific adaptation

Given the same first vertical events and the same request decision, is a separate horizontal recording more useful than the next vertical block? Each state used 4,500 labeled photographs for policy calibration and the same 36,000 test photographs for evaluation. The five adapted states had previously been fitted on a disjoint set of 4,500 unlabeled photographs. The two blocks from the vertical recording share a classifier. Horizontal and square use their own trajectory-specific classifier. An incomplete second observation counts as a request and retains the first prediction.

| Classifier | Requests | Vertical alone (%) | Continue (%) | Horizontal (%) | Square (%) | Horizontal minus continuation (points) |
|---|---:|---:|---:|---:|---:|---:|
| Released | 10,905 | 41.733 | 44.006 | 42.783 | 45.444 | −1.222 |
| Adaptation seed 1 | 10,747 | 43.064 | 44.892 | 45.319 | 46.006 | +0.428 |
| Adaptation seed 2 | 10,435 | 42.903 | 44.750 | 45.044 | 45.825 | +0.294 |
| Adaptation seed 3 | 10,618 | 43.339 | 45.131 | 45.383 | 46.217 | +0.253 |
| Adaptation seed 4 | 10,709 | 42.939 | 44.831 | 45.150 | 45.892 | +0.319 |
| Adaptation seed 5 | 10,669 | 42.758 | 44.650 | 44.992 | 45.611 | +0.342 |
| Five-run mean | 10,635.6 | 43.001 | 44.851 | 45.178 | 45.910 | +0.327 |

For the released model, the paired 95% interval for horizontal minus continuation was −1.49 to −0.98 points under within-class resampling and −1.54 to −0.91 under whole-class resampling. For the five-run mean, these intervals were +0.10 to +0.54 and +0.06 to +0.57. The paired change was +1.549 points, with intervals of +1.25 to +1.84 and +1.19 to +1.89. On average, each classifier state's candidate follow-up ordering also agreed with its standalone ordering. The switch uses a separately recorded prefix, so these experiments measure recognition and request counts rather than mechanical switching costs.

## Reproduce the comparisons

Run from the repository root. Replay needs NumPy. The independent verification also uses SciPy. Each bootstrap design refits the numerical cutoffs and terminal selectors while preserving source pairing across all states and arms. The intervals condition on the five fitted classifiers.

```sh
OPENBLAS_NUM_THREADS=1 python extensions/adapted_continuation/replay.py --output runs/adapted_continuation
python extensions/adapted_continuation/order.py --output runs/adapted_continuation
python -m pip install scipy
OPENBLAS_NUM_THREADS=1 python extensions/adapted_continuation/verify.py --results runs/adapted_continuation
```

The default uses four workers and 2,000 draws per design. The independent check without `--results` checks the distributed files. It reconstructs the request masks and terminal decisions, refits the reported point policies and six bootstrap replicates with SciPy BFGS, and reconstructs all interval endpoints. The separate order command uses the earlier [hash-crop adaptation decisions](../multiseed/README.md).

- [Inputs](inputs/scores.npz) contain all class predictions, float64 maximum softmax probabilities and block eligibility needed for the analysis. For each prefix (`policy_` and `test_`), `ids` and `labels` identify the sources in frozen order. `class` and `msp` have axes state × view × source. State 0 is the released model. States 1–5 are the adaptation seeds. The views are first vertical, second vertical, first horizontal, and first square. `eligible` has axes view × source.
- [Decisions](results/decisions.csv.gz) contain the source ID, label, predictions, confidence, completion flags, request flags and policy decisions for each evaluation photograph and state. A terminal class is the choice when the follow-up is requested. A policy class incorporates the state's actual request mask.
- [Seed results](results/by_seed.json) give the fitted gates, coefficients, workload, attainable and realized rescues, and losses. [Paired results](results/paired.json) give the full-cohort, common-mask, complete-window and fixed-quota contrasts. Their metric and state names index the final two axes of the two distributed bootstrap arrays (`bootstrap_within.npy`, `bootstrap_class.npy`).
- [Protocol](PROTOCOL.md) and [candidate-order diagnostic](CANDIDATE_ORDER.md) document the retrospective analysis and the timing of the exploratory comparisons. [Input provenance](input_provenance.json) records the original archive, trained-state and full logit hashes, device and inference batches. [Verification](results/verification.json) records the original independent checks.

## Regenerate logits from the recordings

The compact file contains the quantities used for every decision. To regenerate 1,000-class logits from the official recordings, first generate the five frozen states for transfer vertical, transfer horizontal and the original square using the [multiseed instructions](../multiseed/RAW_REGENERATION.md). Use the same official full Event Histogram checkpoint and event ZIPs. Then run the following from the repository root. The regeneration dependencies include PyTorch and torchvision.

```sh
for pair in variant1:extracted_val_mode_1.zip variant2:extracted_val_mode_5.zip original:extracted_val.zip; do
    trajectory=${pair%:*}
    archive=${pair#*:}
    python extensions/adapted_continuation/regenerate.py "$trajectory" \
        --archive "$ARCHIVES/$archive" --histogram "$HISTOGRAM" \
        --states "runs/multiseed/$trajectory" --output "runs/chronological_logits/$trajectory"
done
```

Each output directory holds a source-row index, one float32 logit array per view and state, and a checksum receipt. `--source-id` can select one calibration or evaluation photograph. `--seeds` can restrict the adapted states. The script checks the official archive and checkpoint, trajectory state hashes, source IDs, labels, flag for a complete block, and predictions against the paired inputs. Numerical agreement on confidence is tested at an absolute tolerance of 0.00001 for cross-device regeneration.
