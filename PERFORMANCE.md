# Pretrained checkpoints and held-out performance

We provide four pretrained checkpoints: separate UKB and ADNI models for the four-index task and Width. All were trained from random initialization on CPU.

The small architecture uses patch_size=512, embed_dim=64, depth=4 (five Transformer blocks), and four attention heads. Each model was trained for 50 epochs; the checkpoint was selected by mean validation Pearson r. Test results were not used for model selection or hyperparameter tuning.

UKB labels are the available-reader means from the six-reader table; ADNI labels are the two-reader means from the R3 table. Seed=42; 70/15/15% split in original annotation row order. No images or subject-level labels, predictions, identifiers or paths are included in this release.

| Cohort | Train | Validation | Test |
|---|---:|---:|---:|
| UKB | 175 | 37 | 38 |
| ADNI | 140 | 30 | 30 |

## Independent test set

| Cohort | Target | Pearson r (95% CI) | MAE | RMSE | R² | Mean-baseline MAE |
|---|---|---:|---:|---:|---:|---:|
| UKB | EI | 0.787 (0.675, 0.869) | 0.0153 | 0.0182 | 0.526 | 0.0217 |
| UKB | z-EI | 0.699 (0.542, 0.817) | 0.0155 | 0.0190 | 0.450 | 0.0215 |
| UKB | BVR_AC | 0.713 (0.522, 0.850) | 0.1195 | 0.1505 | 0.489 | 0.1735 |
| UKB | BVR_PC | 0.865 (0.780, 0.929) | 0.0957 | 0.1238 | 0.735 | 0.1889 |
| UKB | Width | 0.466 (0.277, 0.721) | 0.9324 | 1.4884 | 0.131 | 1.2009 |
| ADNI | EI | 0.762 (0.563, 0.885) | 0.0167 | 0.0198 | 0.356 | 0.0200 |
| ADNI | z-EI | 0.828 (0.737, 0.907) | 0.0185 | 0.0251 | 0.646 | 0.0322 |
| ADNI | BVR_AC | 0.892 (0.813, 0.953) | 0.1001 | 0.1302 | 0.771 | 0.2098 |
| ADNI | BVR_PC | 0.853 (0.758, 0.920) | 0.1176 | 0.1487 | 0.682 | 0.2245 |
| ADNI | Width | 0.796 (0.622, 0.910) | 0.6693 | 0.9152 | 0.628 | 1.1822 |

MAE and RMSE use the annotation scale without unit conversion. The baseline predicts the training-set mean. Pearson intervals use 2,000 bootstrap resamples of the held-out subjects (seed=20260918). Each cohort has its own model and internal held-out set; these results are not cross-cohort external validation.

## Model size and selected epoch

| Cohort | Model | Parameters | Original parameters | Reduction | Best epoch |
|---|---|---:|---:|---:|---:|
| UKB | masked | 282,905 | 3,794,761 | 92.5% | 35 |
| UKB | width | 282,518 | 3,793,990 | 92.6% | 23 |
| ADNI | masked | 282,905 | 3,794,761 | 92.5% | 29 |
| ADNI | width | 282,518 | 3,793,990 | 92.6% | 20 |

Public checkpoint predictions were verified in a fresh CPU process against the private evaluation outputs; maximum difference was zero for both cohorts. Private checkpoints remain on the training server with the split audit metadata. Public bundles omit identifiers and paths and support `predict`; `evaluate` requires private bundles for subject-overlap checks.

## Usage

```bash
python pipeline.py predict --manifest subjects.csv --checkpoints pretrained/ukb --output ukb_predictions.csv --device cpu
python pipeline.py predict --manifest subjects.csv --checkpoints pretrained/adni --output adni_predictions.csv --device cpu
```

Use a manifest with `eid,image`. Inputs must have the same registered grid as the supplied masks. Bundles include the exact masks, voxel ordering, label normalization, model architecture and weights.
