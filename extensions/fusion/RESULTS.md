# Combining vertical and horizontal scores on the transfer pair

The class-shared stack reached 45.25% on 36,000 test photographs with both 30,000-event windows available. The locally fitted selector reached 44.23%. At the exact 30% request quota, the corresponding cohort accuracies were 44.47% and 43.84%. Product fusion and stacking used all 1,000 logits from the released Event Histogram classifier and the same 9,000 separate calibration sources. The local selector was fitted to the discordant pairs in that calibration set.

| Method | Both scans for all (%) | Exactly 10,800 requests (%) | Full / gated corrections | Full / gated losses |
|---|---:|---:|---:|---:|
| Vertical alone | 43.133 | 43.133 | 0 / 0 | 0 / 0 |
| Local selector | 44.231 | 43.842 | 1,126 / 585 | 731 / 330 |
| Arithmetic probability average | 43.981 | 44.036 | 2,161 / 1,035 | 1,856 / 710 |
| Temperature-scaled product | 44.050 | 44.425 | 2,551 / 1,132 | 2,221 / 667 |
| Regularized stack | 45.253 | 44.472 | 1,757 / 921 | 994 / 439 |

Corrections and losses use vertical alone as the baseline. At 30% the gate selected the lowest vertical confidences. Unrequested images retained their vertical prediction. The first-only baseline used one scan throughout.

## Paired accuracy contrasts

| Request fraction | Comparison | Difference (points) | Within-class 95% interval | Whole-class 95% interval |
|---:|---|---:|---:|---:|
| 100% | Temperature-scaled product minus vertical alone | +0.917 | +0.56 to +1.31 | +0.31 to +1.58 |
| 100% | Temperature-scaled product minus local selector | -0.181 | -0.51 to +0.18 | -0.63 to +0.27 |
| 100% | Temperature-scaled product minus arithmetic probability average | +0.069 | -0.19 to +0.33 | -0.26 to +0.38 |
| 100% | Regularized stack minus vertical alone | +2.119 | +1.86 to +2.43 | +1.69 to +2.64 |
| 100% | Regularized stack minus local selector | +1.022 | +0.78 to +1.30 | +0.73 to +1.38 |
| 100% | Regularized stack minus arithmetic probability average | +1.272 | +1.04 to +1.54 | +0.99 to +1.59 |
| 100% | Regularized stack minus temperature-scaled product | +1.203 | +0.96 to +1.49 | +0.92 to +1.54 |
| 30% | Temperature-scaled product minus vertical alone | +1.292 | +1.07 to +1.52 | +0.99 to +1.59 |
| 30% | Temperature-scaled product minus local selector | +0.583 | +0.37 to +0.75 | +0.34 to +0.78 |
| 30% | Temperature-scaled product minus arithmetic probability average | +0.389 | +0.24 to +0.53 | +0.21 to +0.54 |
| 30% | Regularized stack minus vertical alone | +1.339 | +1.14 to +1.54 | +1.09 to +1.61 |
| 30% | Regularized stack minus local selector | +0.631 | +0.44 to +0.79 | +0.42 to +0.81 |
| 30% | Regularized stack minus arithmetic probability average | +0.436 | +0.27 to +0.60 | +0.26 to +0.61 |
| 30% | Regularized stack minus temperature-scaled product | +0.047 | -0.09 to +0.17 | -0.09 to +0.19 |

Both bootstrap designs used 2,000 shared source draws. Temperatures, stack weights and the local selector were refitted, and the budget was reranked in each draw. The original and bootstrap fits attained maximum absolute penalized score gradients below 1e-8.

## What happened on individual photographs?

| Policy | Requests | Images with both individual predictions wrong rescued on requests | Missed either-correct images on requests |
|---|---:|---:|---:|
| Local selector | 36,000 | 0 | 2,558 |
| Arithmetic probability average | 36,000 | 109 | 2,757 |
| Temperature-scaled product | 36,000 | 682 | 3,305 |
| Regularized stack | 36,000 | 435 | 2,625 |
| Local selector | 10,800 | 0 | 857 |
| Arithmetic probability average | 10,800 | 66 | 853 |
| Temperature-scaled product | 10,800 | 351 | 998 |
| Regularized stack | 10,800 | 274 | 904 |

At full acquisition, 18,481 images had a correct vertical or horizontal top prediction. The stack identified 435 correct third classes among 17,519 both-wrong images. It missed 2,625 of the 18,481 either-correct cases, 67 more than the selector. The net difference was 368 additional correct predictions. Numerically this reduces the selector’s 7.11-point gap from the 51.34% either-correct reference by 1.02 points, or about 14%. That reference bounds policies restricted to the two original top predictions. The stack and the product can choose any class.

Among 10,800 requests, 2,608 had a correct individual prediction and 8,192 had two wrong predictions. The stack recovered 274 third classes and missed 904 either-correct cases, 47 more misses than the selector. The stack and product differed by 17 correct cohort decisions at this workload.

## Applying each rule to the other 25,200 sources

This descriptive comparison was added after inspecting the fusion accuracies. On the 25,200 photographs outside the low-confidence request group, the 30% policy keeps the vertical prediction. The full-acquisition policy applies the same fusion or selection rule to these sources as well. Positive differences favor requesting only 30%. All accuracy differences use the full cohort denominator.

| Terminal rule | Additional rescues / losses from acquiring horizontal on the other 25,200 | Accuracy at 30% minus 100% (points) | Within-class 95% interval | Whole-class 95% interval |
|---|---:|---:|---:|---:|
| Local selector | 541 / 401 | -0.389 | -0.54 to -0.19 | -0.60 to -0.15 |
| Arithmetic probability average | 1,126 / 1,146 | +0.056 | -0.19 to +0.31 | -0.34 to +0.44 |
| Temperature-scaled product | 1,419 / 1,554 | +0.375 | +0.07 to +0.67 | -0.09 to +0.81 |
| Regularized stack | 836 / 555 | -0.781 | -1.00 to -0.60 | -1.10 to -0.51 |

Using product fusion on the additional sources cost 135 correct predictions in this cohort. Its whole-class interval includes zero. Applying the stack to the same additional sources gained 281 correct predictions. These contrasts resample sources and refit the same policies jointly across workloads.

## Fits and checks

The vertical and horizontal temperatures were 1.341683 and 1.579591. The class-shared stack used the natural log probabilities from each scan and their classwise minimum, with respective weights 0.643383, 0.367819, -0.195719. The objective was mean multiclass log loss plus 0.005 times the squared coefficient norm.

The four input logit files matched their transfer inference receipts. An independent implementation checked source pairing, class order, calibration optimality, every full-class test prediction, the two historical baselines and the common request mask. It recovered all intervals from the paired counts and replayed three bootstrap fits and all 36,000 predictions for each design. The tests reused previously scored source images, so these comparisons are retrospective.

Commands and output definitions are in [README.md](README.md).
