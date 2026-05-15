from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from pathmnist.data import build_dataset, dataset_meta
from pathmnist.metrics import compute_metrics, report_dict, softmax_np
from pathmnist.models import build_model, count_parameters
from pathmnist.train import class_weight_tensor, mixup, pick_device, save_predictions, set_seed


CANCER_LABEL = 8
BASELINE_RUN = "baseline"
SPECIALIST_RUN = "approach_01_cancer_vs_rest_specialist"
OVERRIDE_RUN = "approach_01_cancer_vs_rest_override"


class BinaryCancerDataset(Dataset):
    def __init__(self, split: str, image_size: int, source_size: int | None, augment: str, root: str) -> None:
        self.base = build_dataset(
            split,
            image_size=image_size,
            source_size=source_size,
            augment=augment,
            root=root,
            norm="pathmnist",
        )
        self.labels = np.asarray(self.base.labels).reshape(-1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        target = int(np.asarray(y).reshape(-1)[0] == CANCER_LABEL)
        return x, target


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def binary_loader(
    split: str,
    batch_size: int,
    image_size: int,
    source_size: int | None,
    augment: str,
    workers: int,
    root: str,
) -> DataLoader:
    dataset = BinaryCancerDataset(split, image_size, source_size, augment, root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train_binary_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)
        x, ya, yb, lam = mixup(x, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * x.size(0)
        seen += x.size(0)
    return total / max(seen, 1)


@torch.inference_mode()
def predict_binary(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    probs = []
    for x, y in loader:
        labels.append(y.numpy().reshape(-1))
        probs.append(softmax_np(model(x.to(device)).cpu().numpy()))
    return np.concatenate(labels), np.concatenate(probs)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def override_cancer(y_prob: np.ndarray, cancer_score: np.ndarray, threshold: float) -> np.ndarray:
    out = y_prob.copy()
    mask = cancer_score >= threshold
    if np.any(mask):
        out[mask, CANCER_LABEL] = np.maximum(out[mask].max(axis=1) + 1e-6, out[mask, CANCER_LABEL])
    return normalize_probs(out)


def tune_threshold(
    y_true: np.ndarray,
    base_prob: np.ndarray,
    cancer_score: np.ndarray,
    min_recall: float,
) -> tuple[float, dict]:
    best_threshold = 0.5
    best = {"cancer_f2": -1.0, "cancer_recall": 0.0}
    for threshold in np.linspace(0.05, 0.95, 181):
        tuned = override_cancer(base_prob, cancer_score, float(threshold))
        metric = compute_metrics(y_true, tuned)
        candidate = {"cancer_f2": metric.cancer_f2, "cancer_recall": metric.cancer_recall}
        recall_ok = candidate["cancer_recall"] >= min_recall
        if recall_ok and candidate["cancer_f2"] > best["cancer_f2"]:
            best_threshold = float(threshold)
            best = candidate
    return best_threshold, best


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    return arr["y_true"].reshape(-1).astype(int), arr["y_prob"].astype(float)


def train_specialist(args: argparse.Namespace, device: torch.device) -> None:
    model_dir = args.models_dir / SPECIALIST_RUN
    result_dir = args.results_dir / SPECIALIST_RUN
    if not args.force and (result_dir / "specialist_val_predictions.npz").exists():
        return

    set_seed(args.seed)
    model = build_model("small_cnn", 2, False).to(device)
    train_loader = binary_loader("train", args.batch_size, args.image_size, args.source_size, "histology", args.workers, args.data_root)
    val_loader = binary_loader("val", args.batch_size, args.image_size, args.source_size, "none", args.workers, args.data_root)
    test_loader = binary_loader("test", args.batch_size, args.image_size, args.source_size, "none", args.workers, args.data_root)

    train_labels = train_loader.dataset.labels.reshape(-1)
    weights = class_weight_tensor("balanced", (train_labels == CANCER_LABEL).astype(int), 2, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        "run_name": SPECIALIST_RUN,
        "type": "cancer_vs_rest_specialist",
        "model": "small_cnn",
        "epochs": args.epochs,
        "image_size": args.image_size,
        "source_size": args.source_size,
        "augment": "histology",
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "seed": args.seed,
        "device": str(device),
        "parameters": count_parameters(model),
    }
    write_json(result_dir / "config.json", config)
    write_json(model_dir / "config.json", config)

    history = []
    best_score = -1.0
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_binary_epoch(model, train_loader, criterion, optimizer, device, args.mixup_alpha)
        scheduler.step()
        y_val, p_val = predict_binary(model, val_loader, device)
        pred_val = p_val[:, 1] >= 0.5
        binary_f2 = fbeta_score(y_val, pred_val, beta=2, zero_division=0)
        binary_recall = recall_score(y_val, pred_val, zero_division=0)
        binary_auc = roc_auc_score(y_val, p_val[:, 1])
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "binary_auc": float(binary_auc),
            "binary_recall": float(binary_recall),
            "binary_f2": float(binary_f2),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = binary_f2 + 0.1 * binary_auc
        if score > best_score:
            best_score = float(score)
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, model_dir / "best.pt")
    write_json(result_dir / "history.json", history)

    checkpoint = torch.load(model_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    specialist_metrics = {"best_epoch": checkpoint["epoch"], "seconds": time.time() - started}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y_bin, p_bin = predict_binary(model, loader, device)
        np.savez_compressed(result_dir / f"specialist_{split}_predictions.npz", y_true=y_bin, y_prob=p_bin)
        pred_bin = p_bin[:, 1] >= 0.5
        specialist_metrics[split] = {
            "binary_auc": float(roc_auc_score(y_bin, p_bin[:, 1])),
            "binary_recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
            "binary_f2": float(fbeta_score(y_bin, pred_bin, beta=2, zero_division=0)),
        }
    write_json(result_dir / "specialist_metrics.json", specialist_metrics)


def write_override(args: argparse.Namespace) -> None:
    meta = dataset_meta()
    baseline_dir = args.results_dir / BASELINE_RUN
    specialist_dir = args.results_dir / SPECIALIST_RUN
    result_dir = args.results_dir / OVERRIDE_RUN
    model_dir = args.models_dir / OVERRIDE_RUN

    y_val, p_val_base = load_npz(baseline_dir / "val_predictions.npz")
    y_test, p_test_base = load_npz(baseline_dir / "test_predictions.npz")
    _, p_val_specialist = load_npz(specialist_dir / "specialist_val_predictions.npz")
    _, p_test_specialist = load_npz(specialist_dir / "specialist_test_predictions.npz")

    baseline_val = compute_metrics(y_val, p_val_base)
    baseline_test = compute_metrics(y_test, p_test_base)
    threshold, threshold_selection = tune_threshold(
        y_val,
        p_val_base,
        p_val_specialist[:, 1],
        min_recall=baseline_val.cancer_recall,
    )
    p_val = override_cancer(p_val_base, p_val_specialist[:, 1], threshold)
    p_test = override_cancer(p_test_base, p_test_specialist[:, 1], threshold)

    config = {
        "approach": 1,
        "source_models": [BASELINE_RUN, SPECIALIST_RUN],
        "specialist_threshold": threshold,
        "threshold_selection": threshold_selection,
        "min_validation_recall": baseline_val.cancer_recall,
        "baseline_cancer_f2": baseline_test.cancer_f2,
        "target_cancer_f2": baseline_test.cancer_f2 + 0.01,
        "validation_only_selection": True,
    }
    metrics = {
        "val": report_dict(y_val, p_val, meta.class_names),
        "test": report_dict(y_test, p_test, meta.class_names),
        "delta_vs_baseline_cancer_f2": float(compute_metrics(y_test, p_test).cancer_f2 - baseline_test.cancer_f2),
    }
    write_json(result_dir / "config.json", config)
    write_json(model_dir / "config.json", config)
    save_predictions(result_dir / "val_predictions.npz", y_val, p_val)
    save_predictions(result_dir / "test_predictions.npz", y_test, p_test)
    write_json(result_dir / "metrics.json", metrics)
    print(json.dumps({"threshold": threshold, "test_cancer_f2": metrics["test"]["cancer_f2"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate approach 1: baseline + cancer-vs-rest specialist override.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--source-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    train_specialist(args, device)
    write_override(args)


if __name__ == "__main__":
    main()
