# Improving PathMNIST Classification

This project studies how to improve image-classification performance
on **PathMNIST**, a MedMNIST histopathology benchmark. The task is to
classify colorectal cancer tissue image patches into 9 tissue classes.
The starting point is the official MedMNIST benchmark; the project will
reproduce a baseline and then test result-improvement strategies such
as stronger augmentation, transfer learning, ensembles, and cluster-aware
specialist models.

The main question is:

> How can the cancer-class F2 score on PathMNIST be improved using a
combination of different ML approaches?

We try to maximize the F2-score because it has more emphasis on the recall.
Simply maximizing the recall would lead to classifiying every image as
cancer. We have an emphasis on the recall simply because it is more
important to try and catch as many cancer images as possible. You'd
rather want more false positives when it comes to cancer than false
negatives.

For final results the following metrics should always be available:
* accuracy
* precision
* recall
* specificity
* ROC-AUC
* F1
* F2

## Dataset

**Dataset:** PathMNIST from [MedMNIST v2](https://medmnist.com/v2)  
**Domain:** colorectal cancer histology  
**Task:** multi-class image classification  
**Input:** RGB tissue patches, available as 28x28, 64x64, 128x128, and 224x224 images through MedMNIST+  
**Classes:** 9 tissue categories  
**License:** CC BY 4.0  
**Clinical note:** MedMNIST states that the dataset is not intended for clinical use.

PathMNIST is based on NCT-CRC-HE-100K and CRC-VAL-HE-7K. The official split uses NCT-CRC-HE-100K for training and validation and CRC-VAL-HE-7K as the test set.

| Split | Samples |
| --- | ---: |
| Train | 89,996 |
| Validation | 10,004 |
| Test | 7,180 |

Classes:

| Label | Tissue type |
| ---: | --- |
| 0 | adipose |
| 1 | background |
| 2 | debris |
| 3 | lymphocytes |
| 4 | mucus |
| 5 | smooth muscle |
| 6 | normal colon mucosa |
| 7 | cancer-associated stroma |
| 8 | colorectal adenocarcinoma epithelium |

## Questions

### 1. Is there also a baseline for the other resolutions than 28x28 from the original dataset or is it the only benchmark they provided?

> **Baseline availability:** The official MedMNIST v2 benchmark provides PathMNIST results for 28x28 and 224x224 inputs, at least for ResNet-18 and ResNet-50. For PathMNIST, the official table reports ResNet-18 (28) with AUC `0.983` and ACC `0.907`, ResNet-50 (28) with AUC `0.990` and ACC `0.911`, ResNet-18 (224) with AUC `0.989` and ACC `0.909`, and ResNet-50 (224) with AUC `0.989` and ACC `0.892`.
>
> **Other resolutions:** The original lightweight MedMNIST task is centered on 28x28 images. MedMNIST+ additionally provides 64x64, 128x128, and 224x224 versions, but the most commonly cited official benchmark table is for 28x28 and 224x224. If we use 64x64 or 128x128, we should treat those as our own experimental settings unless we cite a specific MedMNIST+ paper/table for those exact resolutions.

### 2. Experiments 1, 2 use 224x224 images while the baseline and experiment 3 use 28x28 images so they aren't really comparable, are they? So to make them all comparable, shouldn't all three approaches be used with 28x28 images to make them comparable? After that we could just select the model with the best performance and then train them on the 224x224 images and still have the best model, right?

> **Are 224x224 and 28x28 runs directly comparable?** Not perfectly. They use the same train/validation/test split and the same labels, but the input representation is different. A 224x224 model has more spatial detail if it is trained on true 224x224 PathMNIST+ images, and it also has a different compute budget. Therefore, a 224x224 result should not be compared as a pure algorithmic improvement over a 28x28 result without stating that the resolution changed.
>
> **Should all approaches first be tested at 28x28?** Yes, for a fair first comparison. The cleanest experimental design is to run all candidate approaches on the same 28x28 PathMNIST input first. That isolates the modeling approach from the effect of image resolution.
>
> **Should the best 28x28 model then be retrained at 224x224?** Yes, that is a reasonable second stage. First compare models at 28x28, select the strongest approach by the target metric, then rerun the strongest candidates at 224x224 to test whether more resolution improves performance. One caveat is that some architectures, especially ImageNet-pretrained ViTs, are naturally designed for 224x224 inputs. If we resize 28x28 images to 224x224, that does not add new information; it only changes the input size. To test the value of higher resolution, we should use true MedMNIST+ 224x224 images.

### 3. Are the approaches in the experiments really a combination of different ML classifiers as it is the task for this research question? For example experiment 1 and 2 only use a vision transformer without combining different ML classifiers am I right? So they should for example more or less function as how high state of the art architectures can go. And can I achieve similar results with simpler ML techniques by combining different ML classifiers could be another research question, right?

> **Are the current experiments combinations of different classifiers?** No, not all of them. Experiment 1 is a single transfer-learning CNN-style approach, and Experiment 2 is a single ViT-style architecture. They are not combinations of multiple classifiers.
>
> **Is Experiment 2 only a Vision Transformer?** Yes. Experiment 2 is best interpreted as a high-capacity architecture benchmark: it tests how far a modern pretrained ViT-style model can go on PathMNIST, not whether a combination of classifiers improves the result.
>
> **What role should Experiments 1 and 2 play?** They should function as reference points for strong individual architectures. They answer: "How good is a strong transfer-learning CNN or transformer by itself?"
>
> **Could a separate research question ask whether simpler combined classifiers can match those results?** Yes. A good refined question would be: "Can an ensemble or hybrid of simpler classifiers match or exceed stronger single architectures on cancer-class F2 while remaining more efficient or interpretable?" That is closer to the stated project goal of combining different ML approaches.

### 4. Can you explain what the approach of experiment 3 is? Are here multiple ML classifiers combined to achieve a better result?

> **What is Experiment 3 trying to do?** Experiment 3 is recall-oriented. The goal is not just to maximize overall accuracy, but to improve the colorectal adenocarcinoma epithelium class by penalizing missed cancer cases more strongly. The current implementation uses class weighting, histology-safe augmentation, label smoothing, and MixUp to push the model toward better cancer recall and F2.
>
> **Are multiple ML classifiers currently combined in Experiment 3?** Not yet. The current Experiment 3 is still primarily a single classifier with a recall-oriented training objective. It is a useful step toward the project goal, but by itself it is not a true combination of different classifiers.
>
> **How should Experiment 3 become a true combination approach?** It should be extended into a hybrid system, for example: a 9-class tissue classifier plus a separate cancer-vs-rest classifier, or an ensemble combining a CNN, a transfer-learning model, and a specialist cancer detector. The final decision could combine their probabilities by soft voting, stacking, threshold tuning, or a specialist override for likely cancer false negatives.
>
> **Why is this useful for F2?** F2 gives more weight to recall than precision. A combined system can keep the 9-class classifier for general tissue recognition while adding a cancer-focused classifier or threshold rule that specifically reduces false negatives for label `8`.

### 5. Ok yeah I guess that is a good idea. So I guess we take the results from the baseline as our baseline and try to improve upon that by building a ML pipeline that combines different classifiers and techniques like supervised, unsupervised learning and RL? Do experiment 1,2,3 do help for that? I guess only experiment 3 since its trying to maximize recall am I right?

> **Should we use the trained baseline as our baseline?** Yes. The practical baseline for this project should be the model we trained and evaluated on the official PathMNIST split, because it gives all required project metrics: accuracy, precision, recall, specificity, ROC-AUC, F1, and F2. The official MedMNIST baseline is still useful as a reference, but it only reports AUC and accuracy.
>
> **Should the improvement stage combine different classifiers and techniques?** Yes. That is better aligned with the research question than exactly reproducing individual papers. A strong pipeline could combine supervised learning, unsupervised structure, and decision optimization. For example: train several supervised classifiers, use unsupervised clustering or embeddings to find hard tissue neighborhoods, add a cancer-vs-rest specialist, and combine outputs with ensembling, stacking, threshold tuning, or selective review.
>
> **Should reinforcement learning be included?** Maybe, but only if it has a clear role. Standard RL is not necessary for PathMNIST classification itself. A reasonable RL-style component would be decision-policy optimization, for example learning when to accept a prediction, when to flag cancer, and when to defer to manual review. If time is limited, threshold optimization and calibration are simpler and more defensible than adding RL.
>
> **Do Experiments 1, 2, and 3 help with this combined-pipeline goal?** Experiment 3 helps the most because it is already focused on cancer recall and F2. Experiments 1 and 2 are still useful, but mainly as reference single-model baselines: they show how strong transfer-learning CNNs or ViTs can be by themselves. They do not directly answer the "combination of classifiers" research question unless their predictions are later included in an ensemble or stacking pipeline.
>
> **Is it correct that only Experiment 3 directly targets recall?** Yes. Experiment 3 is the only one of the current experiments explicitly designed around improving cancer recall and F2. However, the best final project direction should not be "Experiment 3 only"; it should be an expanded version of Experiment 3 that combines multiple classifiers and techniques into one recall-oriented pipeline.

## Notes
Interpretability of the models should also be included in the paper

## Approach 1: Cancer specialist override

The selected multi-classifier approach combines:

* the trained 9-class `28x28` baseline classifier
* a separate binary cancer-vs-rest specialist
* a validation-tuned specialist override threshold

The baseline test cancer F2 is `0.9204304581770748`. Approach 1 improves
test cancer F2 to `0.9458159666559794`.

Reproduce approach 1 after the baseline artifacts exist:

```bash
PYTHONPATH=src python scripts/approach_01_cancer_vs_rest.py \
  --epochs 10 \
  --batch-size 512 \
  --lr 0.0008 \
  --weight-decay 0.0001 \
  --label-smoothing 0.02 \
  --mixup-alpha 0.05 \
  --seed 1042 \
  --workers 0 \
  --device auto \
  --data-root data
```

The main report notebook is
`notebooks/approach_01_cancer_vs_rest_override.ipynb`. It explains the
classifier combination, validation threshold tuning, final metrics, and
artifact paths needed to reconstruct the result.

The separate true-224 transfer-learning result in
`results/bounded_224_resnet18_unweighted_threshold/metrics.json` reached
test cancer F2 `0.9831053901850362`, but it is a single classifier plus
thresholding, not the selected multi-classifier approach.

## Approach C/D/E/F/G: Epithelium-safe feature-based stroma expert correction

This pipeline targets the cancer-associated stroma failure mode in the
`baseline_224` model. Instead of training full CNN specialists, it freezes the
trained 9-class baseline CNN and uses the penultimate feature vector, baseline
probabilities, confidence, stroma-vs-competitor margins, and entropy as input to
three lightweight binary experts:

* debris (`2`) vs stroma (`7`)
* smooth muscle (`5`) vs stroma (`7`)
* colorectal adenocarcinoma epithelium (`8`) vs stroma (`7`)

Extract frozen baseline features:

```bash
PYTHONPATH=src python scripts/feature_stroma_experts.py extract-features \
  --checkpoint models/baseline_224/best.pt \
  --batch-size 512 \
  --workers 0 \
  --device auto
```

This writes `features_train.npz`, `features_val.npz`, and `features_test.npz`
under `results/stroma_feature_experts/`. Each file contains the sample index,
true label, baseline prediction, baseline probability vector, baseline
confidence, and frozen feature vector.

Train and compare the expert families:

```bash
PYTHONPATH=src python scripts/feature_stroma_experts.py train-sklearn
PYTHONPATH=src python scripts/feature_stroma_experts.py train-mlp \
  --epochs 50 \
  --patience 8 \
  --mlp-variant small
PYTHONPATH=src python scripts/feature_stroma_experts.py evaluate
```

The script trains class-balanced logistic regression, calibrated linear SVM,
calibrated RBF SVM, linear discriminant, and small weighted-cross-entropy MLP
experts on the same filtered feature arrays. At inference time the baseline
confidence gate is fixed at `0.90`: if the baseline predicts debris, smooth
muscle, or epithelium below that confidence, the matching expert is consulted.
The expert has its own validation-tuned stroma score threshold before a label is
changed to stroma.

Selection is epithelium-safe. Validation chooses thresholds and systems only if
colorectal adenocarcinoma epithelium recall and F2 remain at least the
`baseline_224` validation values. Among safe systems, the selected model
maximizes validation stroma F2. If the full three-expert system is unsafe, the
pipeline automatically tries debris plus smooth-muscle experts, then single
experts, then baseline-only.

Run the full pipeline end to end:

```bash
PYTHONPATH=src python scripts/feature_stroma_experts.py run \
  --checkpoint models/baseline_224/best.pt \
  --batch-size 512 \
  --mlp-batch-size 1024 \
  --epochs 50 \
  --patience 8 \
  --workers 0 \
  --device auto
```

Final artifacts are written to `results/stroma_feature_experts/` and
`models/stroma_feature_experts/`, including trained sklearn and MLP experts,
validation threshold search results, selected approach, final thresholds,
baseline vs corrected metrics, threshold-only ablation, per-approach validation
and test rows, correction accounting, confusion matrices, and a Markdown report.

Leakage controls:

* experts are trained only on filtered training split samples
* validation selects MLP checkpoints, expert thresholds, fallback set, and the
  better approach
* validation selection enforces non-decreasing epithelium recall and F2
* test data is not used for training, threshold selection, hyperparameter tuning,
  or approach selection
* final test evaluation uses only the validation-selected approach and frozen
  thresholds

Smoke-test the epithelium guardrail and fallback logic:

```bash
PYTHONPATH=src python scripts/feature_stroma_experts.py smoke-test
```
