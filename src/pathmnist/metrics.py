from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class MetricResult:
    auc: float
    acc: float
    macro_f1: float
    cancer_recall: float
    cancer_precision: float


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, cancer_label: int = 8) -> MetricResult:
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_prob.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=list(range(y_prob.shape[1])))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(y_prob.shape[1])), zero_division=0
    )
    return MetricResult(
        auc=float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")),
        acc=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        cancer_recall=float(recall[cancer_label]),
        cancer_precision=float(precision[cancer_label]),
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

