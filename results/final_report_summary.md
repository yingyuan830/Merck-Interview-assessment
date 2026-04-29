### Final Evaluation Summary

| Dataset | IoU | Dice | mAP@0.5 |
|:--|:--:|:--:|:--:|
| Train | 0.9401 | 0.9685 | 0.7597 |
| Validation | 0.9390 | 0.9673 | 0.7686 |
| Test | 0.9283 | 0.9705 | 0.7727 |

Observations:
- The model generalizes well from train to validation (ΔDice < 0.02).
- Slight drop on test mAP suggests minor over-segmentation or staining variation.
- Visualization (see `dataset_comparison.png`) confirms consistent morphology learning.
