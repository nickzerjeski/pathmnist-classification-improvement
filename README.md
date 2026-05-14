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
