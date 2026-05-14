# Experiment Design

This design follows the project goal in `README.md`: improve cancer-class
F2 for PathMNIST while always reporting accuracy, precision, recall,
specificity, ROC-AUC, F1, and F2. The cancer-positive class is label `8`,
colorectal adenocarcinoma epithelium.

## Completion Rules

No step is considered complete until the relevant external target is reached
on the official MedMNIST split. The test split must not be used for model
selection, threshold selection, or hyperparameter tuning.

| Step | Completion gate |
| --- | --- |
| Baseline | Match or exceed the MedMNIST v2 PathMNIST ResNet-50 28x28 benchmark: AUC `0.990`, ACC `0.911`. |
| Experiment 1 | Match or exceed the original Kather external 9-class accuracy target: ACC `0.943`. |
| Experiment 2 | Match or exceed the ViT study PathMNIST target: ACC `0.9462`, cancer recall `0.95`, cancer precision `0.97`, cancer F1 `0.96`. |
| Experiment 3 | Match or exceed the best replicated cancer recall while reporting the specificity and F2 trade-off. |

## Baseline: Official MedMNIST ResNet-50

**Purpose:** Recreate the official benchmark used as the project baseline.

**Reference:** MedMNIST v2 reports PathMNIST ResNet-50 at 28x28 with AUC
`0.990` and ACC `0.911`.

**Pipeline:**

- Dataset: official PathMNIST split, 28x28 RGB.
- Model: CIFAR-style ResNet-50 trained from scratch, matching the official
  small-image ResNet implementation.
- Preprocessing: official normalization to `[-1, 1]`.
- Loss: multiclass cross-entropy.
- Optimizer and schedule: Adam, learning rate `0.001`, batch size `128`,
  100 epochs, LR decayed by `0.1` after epochs `50` and `75`.
- Outputs: saved checkpoint in `models/baseline`, metrics and predictions in
  `results/baseline`.

**Primary command:**

```bash
python -m pathmnist.train \
  --model cifar_resnet50 \
  --image-size 28 \
  --augment none \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.001 \
  --optimizer adam \
  --scheduler multistep \
  --milestones 50 75 \
  --gamma 0.1 \
  --norm official \
  --run-name baseline \
  --results-dir results \
  --models-dir models \
  --target-auc 0.990 \
  --target-acc 0.911 \
  --device auto
```

## Experiment 1: Kather-Style Transfer Learning

**Purpose:** Replicate the colorectal histology source-study style before
testing recall-oriented modifications.

**Reference:** Kather et al. compared ImageNet-pretrained CNNs and reported
external 9-class accuracy around `94.3%` using VGG19-style transfer learning,
SGD with momentum, learning rate `3e-4`, batch size `360`, and random
horizontal/vertical flips.

**Pipeline:**

- Dataset: PathMNIST+ 224x224.
- Model: ImageNet-pretrained `vgg19_bn`.
- Augmentation: horizontal and vertical flips, 90-degree rotations.
- Loss: multiclass cross-entropy.
- Optimizer: SGD with momentum `0.9`, learning rate `3e-4`.
- Gate: test ACC at least `0.943`.
- Outputs: `models/experiment1`, `results/experiment1`.

**Primary command:**

```bash
python -m pathmnist.train \
  --model vgg19_bn \
  --pretrained \
  --image-size 224 \
  --augment basic \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.0003 \
  --optimizer sgd \
  --momentum 0.9 \
  --norm pathmnist \
  --run-name experiment1 \
  --results-dir results \
  --models-dir models \
  --target-acc 0.943 \
  --device auto
```

## Experiment 2: Halder-Style ViT Fine-Tuning

**Purpose:** Replicate the strongest paper result in `data/research.md` that
reports cancer-class recall directly.

**Reference:** Halder et al. fine-tuned ViT-Base-Patch16-224 and reported
PathMNIST ACC `94.62%`, cancer precision `0.97`, cancer recall `0.95`, and
cancer F1 `0.96`.

**Pipeline:**

- Dataset: PathMNIST+ 224x224.
- Model: `vit_base_patch16_224` from `timm`, ImageNet-pretrained.
- Optimizer: AdamW.
- Learning rate: `5e-5`.
- Batch size: `32`.
- Epochs: start with `2` to match the paper; extend only if validation shows
  the paper target is not reproduced.
- Gate: test ACC at least `0.9462`, cancer recall at least `0.95`, cancer
  precision at least `0.97`, cancer F1 at least `0.96`.
- Outputs: `models/experiment2`, `results/experiment2`.

**Primary command:**

```bash
python -m pathmnist.train \
  --model vit_base_patch16_224 \
  --pretrained \
  --image-size 224 \
  --source-size 28 \
  --augment none \
  --epochs 2 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --log-every-batches 200 \
  --lr 0.00005 \
  --weight-decay 0.01 \
  --optimizer adamw \
  --norm pathmnist \
  --run-name experiment2 \
  --results-dir results \
  --models-dir models \
  --target-acc 0.9462 \
  --target-cancer-precision 0.97 \
  --target-cancer-recall 0.95 \
  --target-cancer-f1 0.96 \
  --device auto
```

## Experiment 3: Recall-Oriented Cancer Head Proxy

**Purpose:** Move toward the project objective: improve cancer-class F2 by
raising cancer recall while measuring specificity loss.

**Reference:** `data/research.md` identifies cost-sensitive learning,
hard-negative focus, calibration, and threshold tuning as the best-supported
recall-oriented strategy. The direct high-sensitivity result is from a
binarized PathMNIST setting, so it is not used as a native 9-class numeric
gate.

**Pipeline:**

- Dataset: official PathMNIST split, 28x28 first; optionally rerun at 224x224
  after reproducing Experiments 1 and 2.
- Model: strongest replicated baseline architecture.
- Loss: multiclass cross-entropy with cancer-class weighting.
- Augmentation: histology-safe flips, rotations, mild color jitter, affine
  transforms, and random erasing.
- Decision rule: report standard argmax metrics first. Threshold tuning for
  cancer-vs-rest can be added only using validation predictions.
- Gate: test cancer recall must match or exceed the best replicated paper
  recall, and the report must include cancer specificity and cancer F2.
- Outputs: `models/experiment3`, `results/experiment3`.

**Primary command:**

```bash
python -m pathmnist.train \
  --model resnet50 \
  --image-size 28 \
  --augment histology \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001 \
  --optimizer adamw \
  --class-weights cancer \
  --label-smoothing 0.03 \
  --mixup-alpha 0.05 \
  --norm pathmnist \
  --run-name experiment3 \
  --results-dir results \
  --models-dir models \
  --device auto
```

## Reporting Template

For every completed baseline or experiment, report:

- Overall ACC, macro precision, macro recall, macro specificity, macro AUC,
  macro F1, macro F2.
- Cancer precision, recall, specificity, F1, and F2.
- Full confusion matrix.
- Cancer false-negative destinations, especially lymphocytes, normal colon
  mucosa, debris, and cancer-associated stroma.
- Whether the paper gate passed.
