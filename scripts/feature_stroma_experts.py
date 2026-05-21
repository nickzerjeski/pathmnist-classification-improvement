from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathmnist-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pathmnist-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pathmnist.data import build_dataset, dataset_meta
from pathmnist.metrics import report_dict, softmax_np
from pathmnist.models import CifarResNet, SmallCNN, build_model
from pathmnist.train import pick_device, set_seed


STROMA_LABEL = 7
BASELINE_CONFIDENCE_GATE = 0.90
EXPERTS = {
    "debris": {"label": 2, "display": "debris"},
    "smooth_muscle": {"label": 5, "display": "smooth muscle"},
    "epithelium": {"label": 8, "display": "colorectal adenocarcinoma epithelium"},
}
SKLEARN_APPROACHES = ("logreg", "linear_svm", "rbf_svm", "lda")


@dataclass(frozen=True)
class FeatureSplit:
    sample_index: np.ndarray
    y_true: np.ndarray
    baseline_pred: np.ndarray
    baseline_prob: np.ndarray
    baseline_confidence: np.ndarray
    features: np.ndarray


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, variant: str, dropout: float) -> None:
        super().__init__()
        if variant == "large":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_features(path: Path) -> FeatureSplit:
    arr = np.load(path)
    return FeatureSplit(
        sample_index=arr["sample_index"].astype(int),
        y_true=arr["y_true"].reshape(-1).astype(int),
        baseline_pred=arr["baseline_pred"].reshape(-1).astype(int),
        baseline_prob=arr["baseline_prob"].astype(np.float32),
        baseline_confidence=arr["baseline_confidence"].reshape(-1).astype(np.float32),
        features=arr["features"].astype(np.float32),
    )


def save_features(path: Path, split: FeatureSplit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_index=split.sample_index,
        y_true=split.y_true,
        baseline_pred=split.baseline_pred,
        baseline_prob=split.baseline_prob,
        baseline_confidence=split.baseline_confidence,
        features=split.features,
    )


def augmented_features(split: FeatureSplit) -> np.ndarray:
    stroma_prob = split.baseline_prob[:, [STROMA_LABEL]]
    margins = []
    for spec in EXPERTS.values():
        margins.append(split.baseline_prob[:, [spec["label"]]] - stroma_prob)
    entropy = -np.sum(split.baseline_prob * np.log(np.clip(split.baseline_prob, 1e-8, 1.0)), axis=1, keepdims=True)
    return np.concatenate(
        [
            split.features,
            split.baseline_prob,
            split.baseline_confidence.reshape(-1, 1),
            *margins,
            entropy.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def feature_fn_for_model(model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(model, SmallCNN):
        return lambda x: torch.flatten(model.net(x), 1)
    if isinstance(model, CifarResNet):
        def cifar_features(x: torch.Tensor) -> torch.Tensor:
            out = torch.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = model.avgpool(out)
            return torch.flatten(out, 1)

        return cifar_features
    if hasattr(model, "forward_features"):
        def timm_features(x: torch.Tensor) -> torch.Tensor:
            out = model.forward_features(x)
            if out.ndim > 2:
                out = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(out, 1), 1)
            return out

        return timm_features
    if hasattr(model, "fc") and hasattr(model, "avgpool"):
        children = list(model.children())[:-1]
        backbone = nn.Sequential(*children)
        return lambda x: torch.flatten(backbone(x), 1)
    raise ValueError(f"Unsupported feature extraction model type: {type(model).__name__}")


def logits_from_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SmallCNN):
        return model.head(features)
    if isinstance(model, CifarResNet):
        return model.fc(features)
    if hasattr(model, "fc"):
        return model.fc(features)
    if hasattr(model, "head"):
        return model.head(features)
    raise ValueError(f"Unsupported classifier head for model type: {type(model).__name__}")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    meta = dataset_meta()
    model = build_model(config["model"], meta.n_classes, config.get("pretrained", False)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


@torch.inference_mode()
def extract_split_features(
    model: nn.Module,
    split: str,
    config: dict,
    batch_size: int,
    workers: int,
    device: torch.device,
    progress: bool,
) -> FeatureSplit:
    dataset = build_dataset(
        split,
        image_size=config["image_size"],
        source_size=config.get("source_size"),
        augment="none",
        root=config.get("data_root", "data"),
        norm=config.get("norm", "pathmnist"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    feature_fn = feature_fn_for_model(model)
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    features: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    offset = 0
    for x, y in tqdm(loader, desc=f"extract {split}", leave=False, disable=not progress):
        x = x.to(device)
        feature_tensor = feature_fn(x)
        logits = logits_from_features(model, feature_tensor).cpu().numpy()
        batch_features = feature_tensor.cpu().numpy()
        n = len(y)
        labels.append(y.numpy().reshape(-1))
        probs.append(softmax_np(logits))
        features.append(batch_features.astype(np.float32))
        indices.append(np.arange(offset, offset + n, dtype=np.int64))
        offset += n
    y_true = np.concatenate(labels).astype(int)
    baseline_prob = np.concatenate(probs).astype(np.float32)
    baseline_pred = baseline_prob.argmax(axis=1).astype(int)
    baseline_confidence = baseline_prob.max(axis=1).astype(np.float32)
    return FeatureSplit(
        sample_index=np.concatenate(indices),
        y_true=y_true,
        baseline_pred=baseline_pred,
        baseline_prob=baseline_prob,
        baseline_confidence=baseline_confidence,
        features=np.concatenate(features).astype(np.float32),
    )


def extract_features(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    model, config = load_checkpoint_model(args.checkpoint, device)
    out_dir = args.results_dir / args.run_name
    effective_config = dict(config)
    if args.data_root is not None:
        effective_config["data_root"] = args.data_root
    for split in ("train", "val", "test"):
        out = out_dir / f"features_{split}.npz"
        if out.exists() and not args.force:
            continue
        features = extract_split_features(
            model,
            split,
            effective_config,
            args.batch_size,
            args.workers,
            device,
            not args.no_progress,
        )
        save_features(out, features)
    write_json(
        out_dir / "feature_extraction_config.json",
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_config": effective_config,
            "feature_definition": "penultimate representation before final 9-class classification layer",
        },
    )


def filtered_features(split: FeatureSplit, competitor_label: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (split.y_true == competitor_label) | (split.y_true == STROMA_LABEL)
    return augmented_features(split)[mask], (split.y_true[mask] == STROMA_LABEL).astype(np.int64)


def binary_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
        "support_stroma": int(np.sum(y_true == 1)),
        "support_competitor": int(np.sum(y_true == 0)),
    }


def stroma_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return class_metric_dict(y_true, y_pred, STROMA_LABEL)


def class_metric_dict(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> dict:
    return {
        "precision": float(precision_score(y_true == label, y_pred == label, zero_division=0)),
        "recall": float(recall_score(y_true == label, y_pred == label, zero_division=0)),
        "f1": float(f1_score(y_true == label, y_pred == label, zero_division=0)),
        "f2": float(fbeta_score(y_true == label, y_pred == label, beta=2, zero_division=0)),
    }


def epithelium_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return class_metric_dict(y_true, y_pred, EXPERTS["epithelium"]["label"])


def calibrated_classifier(estimator: object) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, cv=3)


def sklearn_classifier(approach: str, seed: int) -> object:
    if approach == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed)
    if approach == "linear_svm":
        return calibrated_classifier(LinearSVC(class_weight="balanced", max_iter=10000, random_state=seed))
    if approach == "rbf_svm":
        return calibrated_classifier(SVC(kernel="rbf", class_weight="balanced", gamma="scale", C=1.0, random_state=seed))
    if approach == "lda":
        return LinearDiscriminantAnalysis()
    raise ValueError(f"Unsupported sklearn approach: {approach}")


def train_sklearn(args: argparse.Namespace, approach: str) -> None:
    result_dir = args.results_dir / args.run_name
    model_dir = args.models_dir / args.run_name / approach
    train = load_features(result_dir / "features_train.npz")
    val = load_features(result_dir / "features_val.npz")
    metrics = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in EXPERTS.items():
        x_train, y_train = filtered_features(train, spec["label"])
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", sklearn_classifier(approach, args.seed)),
            ]
        )
        clf.fit(x_train, y_train)
        with (model_dir / f"{name}.pkl").open("wb") as handle:
            pickle.dump(clf, handle)
        pair_metrics = {}
        x_eval, y_eval = filtered_features(val, spec["label"])
        pair_metrics["val"] = binary_metric_dict(y_eval, clf.predict(x_eval))
        metrics[name] = pair_metrics
    write_json(result_dir / f"{approach}_binary_metrics.json", metrics)


def train_logreg(args: argparse.Namespace) -> None:
    train_sklearn(args, "logreg")


def class_weights_for_binary(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def predict_mlp(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        logits = model(batch_x.to(device)).cpu().numpy()
        preds.append(logits.argmax(axis=1))
    return np.concatenate(preds).astype(int)


@torch.inference_mode()
def score_mlp(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    for (batch_x,) in loader:
        logits = model(batch_x.to(device))
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def train_one_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    out_path: Path,
) -> tuple[FeatureMLP, list[dict]]:
    model = FeatureMLP(input_dim, args.mlp_variant, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_for_binary(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.mlp_batch_size,
        shuffle=True,
    )
    history = []
    best_key = (-1.0, -1.0)
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)
        val_pred = predict_mlp(model, x_val, device, args.mlp_batch_size)
        val_metrics = binary_metric_dict(y_val, val_pred)
        row = {"epoch": epoch, "train_loss": total / max(seen, 1), **val_metrics}
        history.append(row)
        key = (val_metrics["f2"], val_metrics["precision"])
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "variant": args.mlp_variant,
            "dropout": args.dropout,
            "history": history,
        },
        out_path,
    )
    return model, history


def train_mlp(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = pick_device(args.device)
    result_dir = args.results_dir / args.run_name
    model_dir = args.models_dir / args.run_name / "mlp"
    train = load_features(result_dir / "features_train.npz")
    val = load_features(result_dir / "features_val.npz")
    metrics = {}
    histories = {}
    input_dim = augmented_features(train).shape[1]
    for name, spec in EXPERTS.items():
        x_train, y_train = filtered_features(train, spec["label"])
        x_val, y_val = filtered_features(val, spec["label"])
        model, history = train_one_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim,
            args,
            device,
            model_dir / name / "best.pt",
        )
        histories[name] = history
        pair_metrics = {}
        x_eval, y_eval = filtered_features(val, spec["label"])
        pair_metrics["val"] = binary_metric_dict(y_eval, predict_mlp(model, x_eval, device, args.mlp_batch_size))
        metrics[name] = pair_metrics
    write_json(result_dir / "mlp_binary_metrics.json", metrics)
    write_json(result_dir / "mlp_history.json", histories)


def load_logreg_experts(args: argparse.Namespace) -> dict[str, object]:
    return load_sklearn_experts(args, "logreg")


def load_sklearn_experts(args: argparse.Namespace, approach: str) -> dict[str, object]:
    model_dir = args.models_dir / args.run_name / approach
    experts = {}
    for name in EXPERTS:
        with (model_dir / f"{name}.pkl").open("rb") as handle:
            experts[name] = pickle.load(handle)
    return experts


def load_mlp_experts(args: argparse.Namespace, input_dim: int, device: torch.device) -> dict[str, FeatureMLP]:
    model_dir = args.models_dir / args.run_name / "mlp"
    experts = {}
    for name in EXPERTS:
        checkpoint = torch.load(model_dir / name / "best.pt", map_location=device)
        model = FeatureMLP(input_dim, checkpoint["variant"], checkpoint["dropout"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        experts[name] = model
    return experts


def expert_predictions(
    approach: str,
    split: FeatureSplit,
    experts: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    preds = {}
    x = augmented_features(split)
    for name, expert in experts.items():
        if approach in SKLEARN_APPROACHES:
            preds[name] = expert.predict(x).astype(int)
        else:
            preds[name] = predict_mlp(expert, x, device, batch_size)
    return preds


def expert_scores(
    approach: str,
    split: FeatureSplit,
    experts: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    scores = {}
    x = augmented_features(split)
    for name, expert in experts.items():
        if approach in SKLEARN_APPROACHES:
            if hasattr(expert, "predict_proba"):
                scores[name] = expert.predict_proba(x)[:, 1].astype(np.float32)
            elif hasattr(expert, "decision_function"):
                decision = expert.decision_function(x)
                scores[name] = (1.0 / (1.0 + np.exp(-decision))).astype(np.float32)
            else:
                scores[name] = expert.predict(x).astype(np.float32)
        else:
            scores[name] = score_mlp(expert, x, device, batch_size)
    return scores


def corrected_predictions(
    split: FeatureSplit,
    expert_score: dict[str, np.ndarray],
    expert_thresholds: dict[str, float],
    threshold_only: bool,
    enabled_experts: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    corrected = split.baseline_pred.copy()
    changed_by_expert = {name: 0 for name in EXPERTS}
    enabled = set(EXPERTS if enabled_experts is None else enabled_experts)
    for name, spec in EXPERTS.items():
        if name not in enabled:
            continue
        mask = (split.baseline_pred == spec["label"]) & (split.baseline_confidence < BASELINE_CONFIDENCE_GATE)
        if threshold_only:
            change_mask = mask
        else:
            change_mask = mask & (expert_score[name] >= expert_thresholds[name])
        corrected[change_mask] = STROMA_LABEL
        changed_by_expert[name] = int(np.sum(change_mask))
    return corrected, changed_by_expert


def prediction_probabilities_from_labels(base_prob: np.ndarray, corrected_pred: np.ndarray) -> np.ndarray:
    out = base_prob.copy()
    original_pred = base_prob.argmax(axis=1)
    changed = corrected_pred != original_pred
    if np.any(changed):
        rows = np.where(changed)[0]
        out[rows, corrected_pred[rows]] = out[rows].max(axis=1) + 1e-6
        out = out / out.sum(axis=1, keepdims=True)
    return out


def correction_accounting(y_true: np.ndarray, baseline_pred: np.ndarray, corrected_pred: np.ndarray, changed_by: dict[str, int]) -> dict:
    baseline_fn = (y_true == STROMA_LABEL) & (baseline_pred != STROMA_LABEL)
    corrected_fn = (y_true == STROMA_LABEL) & (corrected_pred != STROMA_LABEL)
    recovered = baseline_fn & (corrected_pred == STROMA_LABEL)
    new_fp = (y_true != STROMA_LABEL) & (baseline_pred != STROMA_LABEL) & (corrected_pred == STROMA_LABEL)
    epithelium_label = EXPERTS["epithelium"]["label"]
    true_epithelium_to_stroma = (y_true == epithelium_label) & (baseline_pred == epithelium_label) & (corrected_pred == STROMA_LABEL)
    return {
        "changed_predictions_per_expert": changed_by,
        "total_changed_predictions": int(np.sum(corrected_pred != baseline_pred)),
        "recovered_stroma_false_negatives": int(np.sum(recovered)),
        "newly_introduced_stroma_false_positives": int(np.sum(new_fp)),
        "true_epithelium_changed_to_stroma": int(np.sum(true_epithelium_to_stroma)),
        "remaining_stroma_false_negatives": int(np.sum(corrected_fn)),
    }


def threshold_candidates() -> list[float]:
    return [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]


def guardrail_passes(epithelium: dict, baseline_epithelium: dict) -> bool:
    return epithelium["recall"] >= baseline_epithelium["recall"] and epithelium["f2"] >= baseline_epithelium["f2"]


def tune_thresholds(
    split: FeatureSplit,
    expert_score: dict[str, np.ndarray],
    enabled_experts: tuple[str, ...],
    baseline_epithelium: dict,
) -> tuple[dict[str, float], list[dict], dict | None]:
    rows = []
    best_row = None
    names = list(enabled_experts)
    for values in itertools.product(threshold_candidates(), repeat=len(names)):
        thresholds = {name: 1.0 for name in EXPERTS}
        thresholds.update(dict(zip(names, values, strict=True)))
        corrected, changed_by = corrected_predictions(
            split,
            expert_score,
            thresholds,
            threshold_only=False,
            enabled_experts=enabled_experts,
        )
        stroma = stroma_metric_dict(split.y_true, corrected)
        epithelium = epithelium_metric_dict(split.y_true, corrected)
        accounting = correction_accounting(split.y_true, split.baseline_pred, corrected, changed_by)
        changed = int(np.sum(corrected != split.baseline_pred))
        row = {
            "thresholds": thresholds,
            "enabled_experts": list(enabled_experts),
            "stroma": stroma,
            "epithelium": epithelium,
            "guardrail_passed": guardrail_passes(epithelium, baseline_epithelium),
            "true_epithelium_changed_to_stroma": accounting["true_epithelium_changed_to_stroma"],
            "changed_predictions": changed,
            "changed_predictions_per_expert": changed_by,
        }
        rows.append(row)
        if not row["guardrail_passed"]:
            continue
        key = (
            row["stroma"]["f2"],
            -row["true_epithelium_changed_to_stroma"],
            row["stroma"]["precision"],
            -row["changed_predictions"],
        )
        best_key = None
        if best_row is not None:
            best_key = (
                best_row["stroma"]["f2"],
                -best_row["true_epithelium_changed_to_stroma"],
                best_row["stroma"]["precision"],
                -best_row["changed_predictions"],
            )
        if best_row is None or key > best_key:
            best_row = row
    return (best_row["thresholds"] if best_row else {name: 1.0 for name in EXPERTS}), rows, best_row


def save_confusion_artifacts(out_prefix: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with (out_prefix.with_suffix(".csv")).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *row.tolist()])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".svg"))
    plt.close(fig)


def load_available_approaches(args: argparse.Namespace, input_dim: int, device: torch.device) -> dict[str, dict[str, object]]:
    approaches = {}
    for approach in SKLEARN_APPROACHES:
        model_dir = args.models_dir / args.run_name / approach
        if all((model_dir / f"{name}.pkl").exists() for name in EXPERTS):
            approaches[approach] = load_sklearn_experts(args, approach)
    mlp_dir = args.models_dir / args.run_name / "mlp"
    if all((mlp_dir / name / "best.pt").exists() for name in EXPERTS):
        approaches["mlp"] = load_mlp_experts(args, input_dim, device)
    if not approaches:
        raise FileNotFoundError("No trained expert models were found. Run train commands before evaluate.")
    return approaches


def system_record(
    split_name: str,
    system_name: str,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    y_pred: np.ndarray,
    changed_by: dict[str, int],
    thresholds: dict[str, float] | None = None,
    enabled_experts: tuple[str, ...] | list[str] | None = None,
) -> dict:
    return {
        "split": split_name,
        "system": system_name,
        "thresholds": thresholds or {},
        "enabled_experts": list(enabled_experts or []),
        "stroma": stroma_metric_dict(y_true, y_pred),
        "epithelium": epithelium_metric_dict(y_true, y_pred),
        "accounting": correction_accounting(y_true, baseline_pred, y_pred, changed_by),
    }


def fallback_groups() -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    return [
        ("full_three_experts", (tuple(EXPERTS),)),
        ("disable_epithelium", (("debris", "smooth_muscle"),)),
        ("single_expert", tuple((name,) for name in EXPERTS)),
    ]


def select_validation_system(candidates: list[dict], baseline_record: dict) -> dict:
    for group_name, _ in fallback_groups():
        safe = [row for row in candidates if row["fallback_group"] == group_name and row["guardrail_passed"]]
        if safe:
            return sorted(
                safe,
                key=lambda row: (
                    row["stroma"]["f2"],
                    -row["accounting"]["true_epithelium_changed_to_stroma"],
                    row["stroma"]["precision"],
                    -row["accounting"]["total_changed_predictions"],
                ),
                reverse=True,
            )[0]
    return {
        **baseline_record,
        "approach": "baseline_only",
        "fallback_group": "baseline_only",
        "thresholds": {name: 1.0 for name in EXPERTS},
        "enabled_experts": [],
        "guardrail_passed": True,
    }


def evaluate_with_system(
    split_name: str,
    split: FeatureSplit,
    approach: str,
    experts: dict[str, object] | None,
    thresholds: dict[str, float],
    enabled_experts: tuple[str, ...] | list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict, np.ndarray]:
    if approach == "baseline_only" or experts is None:
        changed_by = {name: 0 for name in EXPERTS}
        pred = split.baseline_pred.copy()
    else:
        scores = expert_scores(approach, split, experts, device, batch_size)
        pred, changed_by = corrected_predictions(
            split,
            scores,
            thresholds,
            threshold_only=False,
            enabled_experts=enabled_experts,
        )
    record = system_record(
        split_name,
        f"{approach}_expert_system" if approach != "baseline_only" else "baseline_only",
        split.y_true,
        split.baseline_pred,
        pred,
        changed_by,
        thresholds,
        enabled_experts,
    )
    return record, pred


def evaluate_system(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    result_dir = args.results_dir / args.run_name
    val = load_features(result_dir / "features_val.npz")
    input_dim = augmented_features(val).shape[1]
    meta = dataset_meta()

    approaches = load_available_approaches(args, input_dim, device)
    threshold_search = {}
    val_baseline_record = system_record(
        "validation",
        "baseline_only",
        val.y_true,
        val.baseline_pred,
        val.baseline_pred,
        {name: 0 for name in EXPERTS},
    )
    baseline_epithelium = val_baseline_record["epithelium"]
    candidates = []
    for approach, experts in approaches.items():
        scores = expert_scores(approach, val, experts, device, args.mlp_batch_size)
        threshold_search[approach] = {}
        for group_name, enabled_sets in fallback_groups():
            for enabled in enabled_sets:
                thresholds, rows, best = tune_thresholds(val, scores, enabled, baseline_epithelium)
                threshold_search[approach][",".join(enabled)] = rows
                if best is None:
                    best = sorted(
                        rows,
                        key=lambda row: (
                            row["stroma"]["f2"],
                            -row["true_epithelium_changed_to_stroma"],
                            row["stroma"]["precision"],
                            -row["changed_predictions"],
                        ),
                        reverse=True,
                    )[0]
                    thresholds = best["thresholds"]
                pred, changed_by = corrected_predictions(
                    val,
                    scores,
                    thresholds,
                    threshold_only=False,
                    enabled_experts=enabled,
                )
                record = system_record(
                    "validation",
                    f"{approach}_expert_system",
                    val.y_true,
                    val.baseline_pred,
                    pred,
                    changed_by,
                    thresholds,
                    enabled,
                )
                candidates.append(
                    {
                        **record,
                        "approach": approach,
                        "fallback_group": group_name,
                        "guardrail_passed": guardrail_passes(record["epithelium"], baseline_epithelium),
                    }
                )

    selected_system = select_validation_system(candidates, val_baseline_record)
    best_approach = selected_system["approach"]
    final_thresholds = selected_system["thresholds"]
    final_enabled = tuple(selected_system["enabled_experts"])

    test = load_features(result_dir / "features_test.npz")
    val_system_rows = [val_baseline_record]
    test_system_rows = [
        system_record("test", "baseline_only", test.y_true, test.baseline_pred, test.baseline_pred, {name: 0 for name in EXPERTS})
    ]
    val_threshold_pred, val_threshold_changed = corrected_predictions(
        val,
        {name: np.ones(len(val.y_true), dtype=np.float32) for name in EXPERTS},
        {name: 0.0 for name in EXPERTS},
        threshold_only=True,
        enabled_experts=tuple(EXPERTS),
    )
    val_system_rows.append(
        system_record("validation", "threshold_only", val.y_true, val.baseline_pred, val_threshold_pred, val_threshold_changed)
    )
    test_threshold_pred, test_threshold_changed = corrected_predictions(
        test,
        {name: np.ones(len(test.y_true), dtype=np.float32) for name in EXPERTS},
        {name: 0.0 for name in EXPERTS},
        threshold_only=True,
        enabled_experts=tuple(EXPERTS),
    )
    test_system_rows.append(
        system_record("test", "threshold_only", test.y_true, test.baseline_pred, test_threshold_pred, test_threshold_changed)
    )

    approach_test_scores = {}
    approach_test_preds = {}
    approach_val_preds = {}
    for approach, experts in approaches.items():
        best_for_approach = sorted(
            [row for row in candidates if row["approach"] == approach],
            key=lambda row: (
                row["guardrail_passed"],
                row["stroma"]["f2"],
                -row["accounting"]["true_epithelium_changed_to_stroma"],
                row["stroma"]["precision"],
            ),
            reverse=True,
        )[0]
        val_record, val_pred = evaluate_with_system(
            "validation",
            val,
            approach,
            experts,
            best_for_approach["thresholds"],
            best_for_approach["enabled_experts"],
            device,
            args.mlp_batch_size,
        )
        test_record, test_pred = evaluate_with_system(
            "test",
            test,
            approach,
            experts,
            best_for_approach["thresholds"],
            best_for_approach["enabled_experts"],
            device,
            args.mlp_batch_size,
        )
        val_system_rows.append(val_record)
        test_system_rows.append(test_record)
        approach_val_preds[approach] = val_pred
        approach_test_preds[approach] = test_pred
        approach_test_scores[approach] = expert_scores(approach, test, experts, device, args.mlp_batch_size)

    if best_approach == "baseline_only":
        val_corrected_pred = val.baseline_pred.copy()
        test_corrected_pred = test.baseline_pred.copy()
        final_val_record = val_baseline_record | {"system": "final_selected_epithelium_safe_system"}
        final_test_record = test_system_rows[0] | {"system": "final_selected_epithelium_safe_system"}
    else:
        final_val_record, val_corrected_pred = evaluate_with_system(
            "validation",
            val,
            best_approach,
            approaches[best_approach],
            final_thresholds,
            final_enabled,
            device,
            args.mlp_batch_size,
        )
        final_test_record, test_corrected_pred = evaluate_with_system(
            "test",
            test,
            best_approach,
            approaches[best_approach],
            final_thresholds,
            final_enabled,
            device,
            args.mlp_batch_size,
        )
        final_val_record["system"] = "final_selected_epithelium_safe_system"
        final_test_record["system"] = "final_selected_epithelium_safe_system"
    val_system_rows.append(final_val_record)
    test_system_rows.append(final_test_record)

    selected_binary_test_metrics = {}
    if best_approach != "baseline_only":
        for name, spec in EXPERTS.items():
            _, y_eval = filtered_features(test, spec["label"])
            mask = (test.y_true == spec["label"]) | (test.y_true == STROMA_LABEL)
            pred = (approach_test_scores[best_approach][name][mask] >= final_thresholds[name]).astype(int)
            selected_binary_test_metrics[name] = binary_metric_dict(y_eval, pred)

    validation_epithelium_preserved = guardrail_passes(final_val_record["epithelium"], val_baseline_record["epithelium"])
    test_epithelium_preserved = guardrail_passes(final_test_record["epithelium"], test_system_rows[0]["epithelium"])
    metrics = {
        "selected_approach": best_approach,
        "selected_fallback_group": selected_system["fallback_group"],
        "baseline_confidence_gate": BASELINE_CONFIDENCE_GATE,
        "final_thresholds": final_thresholds,
        "enabled_experts": list(final_enabled),
        "validation_epithelium_guardrail_passed": validation_epithelium_preserved,
        "test_epithelium_preserved_after_locked_selection": test_epithelium_preserved,
        "validation": {
            "systems": val_system_rows,
            "baseline_only_stroma": val_baseline_record["stroma"],
            "baseline_only_epithelium": val_baseline_record["epithelium"],
            "selected_expert_correction_stroma": final_val_record["stroma"],
            "selected_expert_correction_epithelium": final_val_record["epithelium"],
            "selected_accounting": final_val_record["accounting"],
        },
        "test": {
            "systems": test_system_rows,
            "baseline_stroma": test_system_rows[0]["stroma"],
            "baseline_epithelium": test_system_rows[0]["epithelium"],
            "threshold_only_stroma": test_system_rows[1]["stroma"],
            "corrected_stroma": final_test_record["stroma"],
            "corrected_epithelium": final_test_record["epithelium"],
            "selected_binary_expert_metrics": selected_binary_test_metrics,
            "threshold_only_accounting": test_system_rows[1]["accounting"],
            "corrected_accounting": final_test_record["accounting"],
            "net_stroma_f2_change": float(final_test_record["stroma"]["f2"] - test_system_rows[0]["stroma"]["f2"]),
            "net_epithelium_recall_change": float(final_test_record["epithelium"]["recall"] - test_system_rows[0]["epithelium"]["recall"]),
            "net_epithelium_f2_change": float(final_test_record["epithelium"]["f2"] - test_system_rows[0]["epithelium"]["f2"]),
        },
    }

    write_json(result_dir / "threshold_search_results.json", threshold_search)
    write_json(result_dir / "selected_approach.json", selected_system)
    write_json(result_dir / "thresholds.json", {"approach": best_approach, "thresholds": final_thresholds, "enabled_experts": list(final_enabled)})
    write_json(result_dir / "metrics.json", metrics)
    write_json(result_dir / "baseline_multiclass_metrics.json", report_dict(test.y_true, test.baseline_prob, meta.class_names))
    corrected_prob = prediction_probabilities_from_labels(test.baseline_prob, test_corrected_pred)
    write_json(result_dir / "corrected_multiclass_metrics.json", report_dict(test.y_true, corrected_prob, meta.class_names))

    np.savez_compressed(result_dir / "val_baseline_only_predictions.npz", y_true=val.y_true, y_pred=val.baseline_pred)
    np.savez_compressed(result_dir / "val_threshold_only_predictions.npz", y_true=val.y_true, y_pred=val_threshold_pred)
    np.savez_compressed(result_dir / "val_feature_expert_correction_predictions.npz", y_true=val.y_true, y_pred=val_corrected_pred)
    np.savez_compressed(result_dir / "test_baseline_only_predictions.npz", y_true=test.y_true, y_pred=test.baseline_pred)
    np.savez_compressed(result_dir / "test_threshold_only_predictions.npz", y_true=test.y_true, y_pred=test_threshold_pred)
    np.savez_compressed(result_dir / "test_feature_expert_correction_predictions.npz", y_true=test.y_true, y_pred=test_corrected_pred)

    save_confusion_artifacts(result_dir / "confusion_matrix_baseline", test.y_true, test.baseline_pred, meta.class_names)
    save_confusion_artifacts(result_dir / "confusion_matrix_threshold_only", test.y_true, test_threshold_pred, meta.class_names)
    save_confusion_artifacts(result_dir / "confusion_matrix_feature_expert_correction", test.y_true, test_corrected_pred, meta.class_names)
    save_confusion_artifacts(result_dir / "confusion_matrix_validation_baseline", val.y_true, val.baseline_pred, meta.class_names)
    save_confusion_artifacts(result_dir / "confusion_matrix_validation_threshold_only", val.y_true, val_threshold_pred, meta.class_names)
    save_confusion_artifacts(
        result_dir / "confusion_matrix_validation_feature_expert_correction",
        val.y_true,
        val_corrected_pred,
        meta.class_names,
    )
    for split_name, split_obj, rows in (("validation", val, val_system_rows), ("test", test, test_system_rows)):
        for row in rows:
            if row["system"] in {"baseline_only", "threshold_only", "final_selected_epithelium_safe_system"}:
                continue
            pred = approach_val_preds[row["system"].replace("_expert_system", "")] if split_name == "validation" else approach_test_preds[row["system"].replace("_expert_system", "")]
            save_confusion_artifacts(
                result_dir / f"confusion_matrix_{split_name}_{row['system']}",
                split_obj.y_true,
                pred,
                meta.class_names,
            )
    save_summary_table(result_dir / "stroma_metrics_summary.csv", metrics)
    write_report(result_dir, metrics)


def save_summary_table(path: Path, metrics: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "system",
                "stroma_precision",
                "stroma_recall",
                "stroma_f1",
                "stroma_f2",
                "epithelium_precision",
                "epithelium_recall",
                "epithelium_f1",
                "epithelium_f2",
                "changed_predictions",
                "recovered_stroma_false_negatives",
                "newly_introduced_stroma_false_positives",
                "true_epithelium_changed_to_stroma",
            ]
        )
        for split_key in ("validation", "test"):
            for row in metrics[split_key]["systems"]:
                stroma = row["stroma"]
                epithelium = row["epithelium"]
                accounting = row["accounting"]
                writer.writerow(
                    [
                        split_key,
                        row["system"],
                        stroma["precision"],
                        stroma["recall"],
                        stroma["f1"],
                        stroma["f2"],
                        epithelium["precision"],
                        epithelium["recall"],
                        epithelium["f1"],
                        epithelium["f2"],
                        accounting["total_changed_predictions"],
                        accounting["recovered_stroma_false_negatives"],
                        accounting["newly_introduced_stroma_false_positives"],
                        accounting["true_epithelium_changed_to_stroma"],
                    ]
                )


def write_report(result_dir: Path, metrics: dict) -> None:
    test = metrics["test"]
    validation = metrics["validation"]
    lines = [
        "# Epithelium-Safe Feature-Based Stroma Expert Correction Report",
        "",
        "## Selected System",
        "",
        f"- Selected approach: `{metrics['selected_approach']}`",
        f"- Selected fallback group: `{metrics['selected_fallback_group']}`",
        f"- Enabled experts: `{metrics['enabled_experts']}`",
        f"- Baseline confidence gate: `{metrics['baseline_confidence_gate']}`",
        f"- Thresholds: `{metrics['final_thresholds']}`",
        f"- Validation epithelium guardrail passed: `{metrics['validation_epithelium_guardrail_passed']}`",
        f"- Locked test epithelium preserved: `{metrics['test_epithelium_preserved_after_locked_selection']}`",
        "",
        "## Validation Metrics",
        "",
        "| System | Stroma P | Stroma R | Stroma F2 | Epithelium R | Epithelium F2 | True epithelium changed to stroma |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in validation["systems"]:
        stroma = row["stroma"]
        epithelium = row["epithelium"]
        acct = row["accounting"]
        lines.append(
            f"| {row['system']} | {stroma['precision']:.6f} | {stroma['recall']:.6f} | {stroma['f2']:.6f} | "
            f"{epithelium['recall']:.6f} | {epithelium['f2']:.6f} | {acct['true_epithelium_changed_to_stroma']} |"
        )
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| System | Stroma P | Stroma R | Stroma F2 | Epithelium R | Epithelium F2 | True epithelium changed to stroma |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test["systems"]:
        stroma = row["stroma"]
        epithelium = row["epithelium"]
        acct = row["accounting"]
        lines.append(
            f"| {row['system']} | {stroma['precision']:.6f} | {stroma['recall']:.6f} | {stroma['f2']:.6f} | "
            f"{epithelium['recall']:.6f} | {epithelium['f2']:.6f} | {acct['true_epithelium_changed_to_stroma']} |"
        )
    acct = test["corrected_accounting"]
    lines.extend(
        [
            "",
            "## Correction Accounting",
            "",
            f"- Changed predictions per expert: `{acct['changed_predictions_per_expert']}`",
            f"- Recovered stroma false negatives: `{acct['recovered_stroma_false_negatives']}`",
            f"- Newly introduced stroma false positives: `{acct['newly_introduced_stroma_false_positives']}`",
            f"- True epithelium changed to stroma: `{acct['true_epithelium_changed_to_stroma']}`",
            f"- Net stroma F2 change: `{test['net_stroma_f2_change']:.6f}`",
            f"- Net epithelium recall change: `{test['net_epithelium_recall_change']:.6f}`",
            f"- Net epithelium F2 change: `{test['net_epithelium_f2_change']:.6f}`",
            "",
            "## Leakage Controls",
            "",
            "- Experts are trained only on filtered training split features.",
            "- The baseline confidence gate is fixed at 0.90 before validation threshold search.",
            "- Expert thresholds and approach selection are based only on validation metrics.",
            "- Validation selection requires epithelium recall and F2 to remain at least baseline.",
            "- Test metrics are computed once with the selected validation approach and frozen thresholds.",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n")


def smoke_test_guardrail() -> None:
    y_true = np.array([STROMA_LABEL, EXPERTS["epithelium"]["label"], STROMA_LABEL, EXPERTS["epithelium"]["label"]])
    baseline_pred = np.array([EXPERTS["debris"]["label"], EXPERTS["epithelium"]["label"], EXPERTS["smooth_muscle"]["label"], EXPERTS["epithelium"]["label"]])
    baseline_prob = np.full((4, 9), 0.01, dtype=np.float32)
    baseline_prob[np.arange(4), baseline_pred] = 0.80
    baseline_prob = baseline_prob / baseline_prob.sum(axis=1, keepdims=True)
    split = FeatureSplit(
        sample_index=np.arange(4),
        y_true=y_true,
        baseline_pred=baseline_pred,
        baseline_prob=baseline_prob,
        baseline_confidence=np.full(4, 0.50, dtype=np.float32),
        features=np.zeros((4, 3), dtype=np.float32),
    )
    scores = {
        "debris": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "smooth_muscle": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "epithelium": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    baseline_record = system_record(
        "validation",
        "baseline_only",
        split.y_true,
        split.baseline_pred,
        split.baseline_pred,
        {name: 0 for name in EXPERTS},
    )
    candidates = []
    for group_name, enabled_sets in fallback_groups():
        for enabled in enabled_sets:
            thresholds, rows, best = tune_thresholds(split, scores, enabled, baseline_record["epithelium"])
            if best is None:
                best = sorted(rows, key=lambda row: row["stroma"]["f2"], reverse=True)[0]
                thresholds = best["thresholds"]
            pred, changed_by = corrected_predictions(split, scores, thresholds, threshold_only=False, enabled_experts=enabled)
            record = system_record(
                "validation",
                "synthetic",
                split.y_true,
                split.baseline_pred,
                pred,
                changed_by,
                thresholds,
                enabled,
            )
            candidates.append(
                {
                    **record,
                    "approach": "synthetic",
                    "fallback_group": group_name,
                    "guardrail_passed": guardrail_passes(record["epithelium"], baseline_record["epithelium"]),
                }
            )
    selected = select_validation_system(candidates, baseline_record)
    assert selected["fallback_group"] == "disable_epithelium", selected
    assert selected["enabled_experts"] == ["debris", "smooth_muscle"], selected
    assert selected["epithelium"]["recall"] >= baseline_record["epithelium"]["recall"], selected
    print("smoke_test_guardrail passed")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    extract_features(args)
    for approach in SKLEARN_APPROACHES:
        train_sklearn(args, approach)
    train_mlp(args)
    evaluate_system(args)
    write_json(
        args.results_dir / args.run_name / "run_config.json",
        {
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "results_dir": str(args.results_dir),
            "models_dir": str(args.models_dir),
            "seconds": time.time() - started,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-based binary stroma experts for PathMNIST correction.")
    parser.add_argument(
        "phase",
        nargs="?",
        choices=[
            "run",
            "extract-features",
            "train-logreg",
            "train-linear-svm",
            "train-rbf-svm",
            "train-lda",
            "train-sklearn",
            "train-mlp",
            "evaluate",
            "smoke-test",
        ],
        default="run",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/baseline_224/best.pt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--run-name", default="stroma_feature_experts")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--mlp-batch-size", type=int, default=1024)
    parser.add_argument("--mlp-variant", choices=["small", "large"], default="small")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "run":
        run(args)
    elif args.phase == "extract-features":
        extract_features(args)
    elif args.phase == "train-logreg":
        train_logreg(args)
    elif args.phase == "train-linear-svm":
        train_sklearn(args, "linear_svm")
    elif args.phase == "train-rbf-svm":
        train_sklearn(args, "rbf_svm")
    elif args.phase == "train-lda":
        train_sklearn(args, "lda")
    elif args.phase == "train-sklearn":
        for approach in SKLEARN_APPROACHES:
            train_sklearn(args, approach)
    elif args.phase == "train-mlp":
        train_mlp(args)
    elif args.phase == "evaluate":
        evaluate_system(args)
    elif args.phase == "smoke-test":
        smoke_test_guardrail()


if __name__ == "__main__":
    main()
