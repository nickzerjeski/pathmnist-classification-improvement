# Improving PathMNIST Classification

This project studies how to improve image-classification performance on
**PathMNIST**, a [MedMNIST v2](https://medmnist.com/v2) histopathology
benchmark. The task is to classify colorectal cancer tissue image patches into 9
tissue classes. The starting point is the official MedMNIST benchmark; the
project will reproduce a baseline and then test result-improvement strategies
such as stronger augmentation, transfer learning, ensembles, and cluster-aware
specialist models.

The main question is:

> How can we predict cancer tissue more reliably?

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

Size:

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
