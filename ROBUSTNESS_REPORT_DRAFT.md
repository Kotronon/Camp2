# NLST Lung Nodule Detection Robustness Draft

## Setup

- Model: MONAI Model-Zoo `lung_nodule_ct_detection` RetinaNet bundle.
- Data: first 30 NLST volumes from `NLST/imagesTr`.
- Corruptions: 10 targeted lung-region occlusions, 10 full-slice dropouts, 10 low-dose noise cases.
- Rationale: lung-region occlusion is a targeted ablation/stress test; full-slice dropout and low-dose noise are closer to acquisition or reconstruction failures.
- Metrics: prediction stability, not Dice. Boxes are matched between clean and corrupted inference using score threshold `0.3` and IoU threshold `0.1`.

## Main Results

From `nlst_detection_outputs/original_vs_corrupted_aggregates.csv` after rerunning corrupted inference:

- Clean detections: 62
- Corrupted detections: 52
- Matched detections: 45
- Disappeared detections: 17
- New detections: 7
- Overall disappearance rate: 0.274
- New detection rate: 0.135
- Mean matched IoU: 0.833
- Mean score delta: -0.017

Per corruption type:

- `lung_region_occlusion`: disappearance rate 0.385, new detection rate 0.111, mean score delta -0.023.
- `full_slice_dropout`: disappearance rate 0.353, new detection rate 0.154, mean score delta +0.024.
- `low_dose_noise`: disappearance rate 0.053, new detection rate 0.143, mean score delta -0.041.

## Mitigation

The mitigation is a post-processing/data-quality step:

- Mark cases as low-quality when `removed_fraction >= 0.5`.
- For region-based corruptions, remove predictions whose box center falls inside the documented corrupted z-region.
- For low-dose noise, do not region-filter; only add quality flags because the corruption is global.
- Keep quality flags in the mitigated prediction JSON.

From `nlst_detection_outputs/original_vs_mitigated_aggregates.csv`:

- Mitigated corrupted detections: 50
- New detections after mitigation: 5
- New detection rate drops from 0.135 to 0.100.
- Lung-region occlusion new detections drop from 2 to 0.
- Disappearance rate remains 0.274 because post-processing cannot recover detections lost by the model.

Interpretation: the mitigation does not make the detector robust by itself, but it makes corrupted inputs explicit and removes region-based false positives. Global noise remains a quality issue rather than a local region-filtering problem.

## Visual Evidence

Use the PNGs in `nlst_detection_outputs/visual_evidence/`:

- `stable_NLST_0011_0000.png`
- `disappeared_NLST_0003_0000.png`
- `new_detection_NLST_0004_0000.png`
- `score_drop_NLST_0004_0000.png`

## Presentation Outline

1. Task and constraint: one-week robustness analysis, no retraining.
2. Pipeline: clean inference, corrupted inference, prediction matching.
3. Corruption design: targeted ablation vs. acquisition-style corruptions with same filenames.
4. Metrics: detections, disappeared/new, IoU, score delta.
5. Results table: overall and per corruption type.
6. Visual failure examples.
7. Mitigation: quality flag plus corrupted-region filtering.
8. Takeaway: detector robustness depends on corruption type; post-processing can flag unreliable inputs and reduce misleading outputs but cannot recover detections lost by the model.
