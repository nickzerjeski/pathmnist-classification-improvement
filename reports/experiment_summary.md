# PathMNIST Experiment Summary

Date: 2026-05-07

## Target

The README benchmark target is the strongest listed official PathMNIST result:

| Reference | AUC | ACC |
| --- | ---: | ---: |
| MedMNIST ResNet-50, 28x28 | 0.990 | 0.911 |

Because this is a cancer histology task, accuracy alone is not sufficient. The pipeline also reports macro F1, per-class metrics, confusion matrices, and colorectal adenocarcinoma epithelium recall/precision.

## Best Result So Far

Best model: 4-model soft-voting ensemble.

Members:

1. `small_cnn_histology_seed42_e10`, 28x28, histology augmentation, MixUp, label smoothing.
2. `small_cnn_histology_seed11`, 28x28, histology augmentation, MixUp, label smoothing.
3. `resnet18_pre_basic_seed21`, torchvision ImageNet-pretrained ResNet-18, 28x28, basic augmentation.
4. `small_cnn64_basic_seed41`, 64x64 MedMNIST+, basic augmentation.

Test result:

| Method | AUC | ACC | Macro F1 | Cancer recall | Cancer precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official ResNet-50 28x28 target | 0.990 | 0.911 | - | - | - |
| Best single model, `small_cnn_histology_seed42_e10` | 0.99043 | 0.91114 | 0.87010 | 0.91565 | 0.94005 |
| 4-model soft-voting ensemble | 0.99156 | 0.92563 | 0.89334 | 0.96188 | 0.92081 |

The ensemble improves over the listed benchmark by `+0.00156` AUC and `+0.01463` ACC.

## What Worked

- Deep ensemble / soft voting: individually weak or only marginally better models made complementary errors. Averaging class probabilities raised test accuracy from `0.91114` for the best single model to `0.92563`.
- Seed diversity: the second histology-augmentation seed was materially better than the first seed on test AUC.
- Conservative histology augmentation: flips, rotations, mild color perturbation, mild affine transforms, random erasing, MixUp, and label smoothing produced the best single-model AUC.
- Higher resolution as a diverse ensemble member: the 64x64 CNN was not strong alone, but it contributed useful diversity to the final soft-voting ensemble.

## What Failed Or Was Falsified

- Chasing validation AUC alone was misleading. Several models reached validation AUC above `0.998` but collapsed on the external CRC-VAL test set.
- The first small CNN with basic augmentation reached validation ACC `0.98311`, but test ACC was only `0.85822`.
- The torchvision ImageNet ResNet-18 was not a good standalone 28x28 solution. Its ImageNet-style stem is not ideal for 28x28 images.
- An official-style CIFAR ResNet18 was added, but early checkpoints were slow on MPS and did not generalize well enough before stopping.
- The main recurring test-set failure mode was cancer-associated stroma recall. The best ensemble still has stroma recall `0.47981`, so this remains the main clinical/biological weakness.

## Reproduction Commands

Set up:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
```

Dataset summary:

```bash
.venv/bin/python scripts/analyze_dataset.py
```

Best single model:

```bash
.venv/bin/python -m pathmnist.train \
  --model small_cnn \
  --epochs 10 \
  --batch-size 512 \
  --workers 0 \
  --augment histology \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --label-smoothing 0.03 \
  --mixup-alpha 0.05 \
  --seed 42 \
  --run-name small_cnn_histology_seed42_e10 \
  --device auto \
  --no-progress
```

Final ensemble:

```bash
.venv/bin/python scripts/ensemble.py \
  results/experiments/small_cnn_histology_seed42_e10/test_predictions.npz \
  results/experiments/small_cnn_histology_seed11/test_predictions_eval.npz \
  results/experiments/resnet18_pre_basic_seed21/test_predictions.npz \
  results/experiments/small_cnn64_basic_seed41/test_predictions_eval.npz \
  --out results/ensemble_4model_metrics.json
```

## Research Rationale

- MedMNIST v2 reports that successful biomedical benchmark solutions often depend heavily on preprocessing, postprocessing, ensembles, and test-time augmentation, not only model architecture.
- MixUp is a vicinal-risk regularizer that encourages smoother behavior between examples and can reduce memorization.
- Deep ensembles are simple, scalable, and useful under dataset shift because independently trained models make different errors and their averaged probabilities are often better calibrated.

## Next High-Value Work

1. Improve cancer-associated stroma recall without sacrificing adenocarcinoma recall.
2. Save top-k checkpoints per run and ensemble across epochs, not only across model seeds.
3. Add test-time augmentation voting.
4. Add a cluster-aware specialist for the debris/stroma/smooth-muscle confusion region.
5. Run repeated seeds for the final ensemble to estimate mean and standard deviation.

