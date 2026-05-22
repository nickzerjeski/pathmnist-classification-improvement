from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, precision_score, recall_score


STROMA_LABEL = 7
EPITHELIUM_LABEL = 8
N_CLASSES = 9
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth_muscle",
    "normal_colon_mucosa",
    "cancer_associated_stroma",
    "colorectal_adenocarcinoma_epithelium",
]


def load_probability_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path)
    y_true = arr["y_true"].reshape(-1).astype(int)
    y_prob = arr["y_prob"].astype(float)
    return y_true, y_prob.argmax(axis=1), y_prob


def load_official_resnet50(path: Path, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(path, header=None).to_numpy(dtype=float)
    y_prob = raw[:, 1:]
    if len(y_prob) != len(y_true):
        raise ValueError(f"{path} has {len(y_prob)} rows, expected {len(y_true)}")
    return y_prob.argmax(axis=1), y_prob


def load_hard_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["y_true"].reshape(-1).astype(int), arr["y_pred"].reshape(-1).astype(int)


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    return float(recall_score(y_true == label, y_pred == label, zero_division=0))


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    return float(precision_score(y_true == label, y_pred == label, zero_division=0))


def binary_f2(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    return float(fbeta_score(y_true == label, y_pred == label, beta=2, zero_division=0))


def metric_row(system: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    merged_true = np.isin(y_true, [STROMA_LABEL, EPITHELIUM_LABEL])
    merged_pred = np.isin(y_pred, [STROMA_LABEL, EPITHELIUM_LABEL])
    return {
        "system": system,
        "acc": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_stroma": binary_precision(y_true, y_pred, STROMA_LABEL),
        "recall_stroma": binary_recall(y_true, y_pred, STROMA_LABEL),
        "f2_stroma": binary_f2(y_true, y_pred, STROMA_LABEL),
        "precision_epithelium": binary_precision(y_true, y_pred, EPITHELIUM_LABEL),
        "recall_epithelium": binary_recall(y_true, y_pred, EPITHELIUM_LABEL),
        "f2_epithelium": binary_f2(y_true, y_pred, EPITHELIUM_LABEL),
        "recall_stroma_epithelium_vs_rest": float(recall_score(merged_true, merged_pred, zero_division=0)),
        "f2_stroma_epithelium_vs_rest": float(fbeta_score(merged_true, merged_pred, beta=2, zero_division=0)),
    }


def write_prediction_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample_index", "y_true", "y_pred"]
        if y_prob is not None:
            header.extend(f"prob_{idx}_{name}" for idx, name in enumerate(CLASS_NAMES))
        writer.writerow(header)
        for idx, (true, pred) in enumerate(zip(y_true, y_pred, strict=True)):
            row = [idx, int(true), int(pred)]
            if y_prob is not None:
                row.extend(float(value) for value in y_prob[idx])
            writer.writerow(row)


def write_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--official-resnet50-csv",
        type=Path,
        default=Path("artifacts/medmnist_predictions/predictions/pathmnist_test_[AUC]0.989_[ACC]0.924@resnet50_28_3.csv"),
    )
    args = parser.parse_args()

    basemodel_dir = args.results_dir / "basemodel"
    cancer_expert_dir = args.results_dir / "cancer_expert"
    resnet50_dir = args.results_dir / "baseline_resnet50"

    y_true, basemodel_pred, basemodel_prob = load_probability_npz(basemodel_dir / "test_predictions.npz")
    resnet50_pred, resnet50_prob = load_official_resnet50(args.official_resnet50_csv, y_true)
    expert_true, cancer_expert_pred = load_hard_npz(cancer_expert_dir / "pair_test_targeted_predictions.npz")
    if not np.array_equal(y_true, expert_true):
        raise ValueError("Cancer-Expert predictions do not align with Basemodel test labels")

    systems = [
        ("Baseline ResNet50", resnet50_dir, resnet50_pred, resnet50_prob),
        ("Basemodel", basemodel_dir, basemodel_pred, basemodel_prob),
        ("Cancer-Expert", cancer_expert_dir, cancer_expert_pred, None),
    ]

    rows = []
    for name, out_dir, y_pred, y_prob in systems:
        rows.append(metric_row(name, y_true, y_pred))
        write_prediction_csv(out_dir / "test_predictions.csv", y_true, y_pred, y_prob)
        write_confusion_matrix(out_dir / "confusion_matrix.csv", y_true, y_pred)
        (out_dir / "metrics.json").write_text(json.dumps(rows[-1], indent=2))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_dir / "final_model_comparison.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
