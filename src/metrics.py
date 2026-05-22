from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class MetricResult:
    auc: float
    acc: float
    precision_macro: float
    recall_macro: float
    specificity_macro: float
    f1_macro: float
    f2_macro: float
    cancer_precision: float
    cancer_recall: float
    cancer_specificity: float
    cancer_f1: float
    cancer_f2: float


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    values = []
    for cls in range(n_classes):
        negative = y_true != cls
        tn = int(np.sum(negative & (y_pred != cls)))
        fp = int(np.sum(negative & (y_pred == cls)))
        values.append(tn / (tn + fp) if tn + fp else 0.0)
    return np.asarray(values, dtype=float)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, cancer_label: int = 8) -> MetricResult:
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]
    target_label = cancer_label if cancer_label < n_classes else n_classes - 1
    labels = list(range(n_classes))
    y_bin = label_binarize(y_true, classes=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specificity = specificity_per_class(y_true, y_pred, n_classes)
    return MetricResult(
        auc=float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")),
        acc=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        specificity_macro=float(np.mean(specificity)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f2_macro=float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)),
        cancer_precision=float(precision[target_label]),
        cancer_recall=float(recall[target_label]),
        cancer_specificity=float(specificity[target_label]),
        cancer_f1=float(f1[target_label]),
        cancer_f2=float(fbeta_score(y_true == target_label, y_pred == target_label, beta=2, zero_division=0)),
    )


def report_dict(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_prob.argmax(axis=1)
    result = compute_metrics(y_true, y_prob).__dict__
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    result["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return result
