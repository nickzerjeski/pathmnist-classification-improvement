# Improving PathMNIST Classification

Machine Learning Lab project proposal for **Format 4: improving the results of a machine learning algorithm**.

## Project Idea

This project studies how to improve image-classification performance on **PathMNIST**, a MedMNIST histopathology benchmark. The task is to classify colorectal cancer tissue image patches into 9 tissue classes. The starting point is the official MedMNIST benchmark; the project will reproduce a baseline and then test result-improvement strategies such as stronger augmentation, transfer learning, ensembles, and cluster-aware specialist models.

The main question is:

> Can a principled combination of model training strategies improve PathMNIST test AUC and accuracy over the official MedMNIST baselines?

## Course Format

The project follows **Format 4** from the Machine Learning Lab guidelines. The focus is on the **impact of combining learning models and training strategies on the final results**, especially effectiveness:

- improve accuracy and AUC over a reproducible baseline;
- compare alternative improvement strategies under the same data split;
- explain which strategy helps, where it fails, and why.

Efficiency will be tracked secondarily through training time, inference time, and parameter count when relevant.

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

## Official Benchmark Targets

MedMNIST reports AUC and accuracy for several PathMNIST baselines:

| Method | Input size | AUC | ACC |
| --- | ---: | ---: | ---: |
| ResNet-18 | 28x28 | 0.983 | 0.907 |
| ResNet-18 | 224x224 | 0.989 | 0.909 |
| ResNet-50 | 28x28 | 0.990 | 0.911 |
| ResNet-50 | 224x224 | 0.989 | 0.892 |
| auto-sklearn | 28x28 | 0.934 | 0.716 |
| AutoKeras | 28x28 | 0.959 | 0.834 |
| Google AutoML Vision | 28x28 | 0.944 | 0.728 |

The strongest official reference point for accuracy is **ResNet-50 at 28x28 with 0.911 ACC**, while the strongest AUC in the listed table is also **ResNet-50 at 28x28 with 0.990 AUC**.

## Planned Approach

1. **Reproducible baseline**
   - Train ResNet-18 and/or ResNet-50 on the official PathMNIST split.
   - Match the MedMNIST evaluation protocol with macro AUC and accuracy.
   - Use fixed random seeds, logged hyperparameters, and saved predictions.

2. **Effectiveness improvements**
   - Apply histology-safe data augmentation: flips, rotations, color jitter, stain-like perturbations, MixUp/CutMix, and random erasing.
   - Compare image sizes: 28x28 versus higher-resolution MedMNIST+ variants where compute allows.
   - Use transfer learning from ImageNet-pretrained CNNs and compare with training from scratch.
   - Train multiple independently seeded models and combine them through soft-voting ensembles.
   - Explore cluster-aware training: cluster embeddings or image features, then train specialist classifiers or use cluster membership as an additional signal.

3. **Analysis**
   - Compare every intervention against the same baseline.
   - Report mean and standard deviation over repeated runs when feasible.
   - Inspect per-class performance and confusion matrices to identify which tissue types benefit most.
   - Track the cost of improvements using training time and inference time.

## Evaluation

Primary metrics:

- **AUC:** official MedMNIST metric for ranking quality.
- **Accuracy:** official MedMNIST metric for classification correctness.

Secondary diagnostics:

- macro F1-score;
- per-class precision and recall;
- confusion matrix;
- training and inference time;
- parameter count and model size.

The main success criterion is an improvement over the reproduced baseline and, if possible, over the official MedMNIST PathMNIST benchmark values.

## Expected Outcome

The final project will contain:

- a clear description of PathMNIST and the 9-class classification task;
- a reproducible baseline implementation;
- several Format-4 improvement strategies based on combining models or training paradigms;
- experiments validating the effect of each strategy;
- a 5-10 page report and a 15-20 minute presentation.

## Repository Structure

```text
.
├── README.md
├── Machine Learning Lab Project Guidelines.pdf
├── Project09-Ayaziov  Gavenda-Fake News Detection.pdf
├── Project11-MLProjectIdea_MehmetMertTezcan.pdf
└── preview/
    └── main.tex
```

Planned implementation folders:

```text
src/          training and evaluation code
configs/      experiment configurations
notebooks/    exploration and result analysis
results/      metrics, plots, and saved predictions
reports/      final report material
```

## References

- MedMNIST project page: <https://medmnist.com/v2>
- MedMNIST GitHub repository: <https://github.com/MedMNIST/MedMNIST>
- Yang, J. et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." *Scientific Data*, 2023.
- Yang, J., Shi, R., and Ni, B. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis." *ISBI*, 2021.
