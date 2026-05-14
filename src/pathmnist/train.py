from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from pathmnist.data import build_loader, dataset_meta
from pathmnist.metrics import compute_metrics, report_dict, softmax_np
from pathmnist.models import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def class_weight_tensor(mode: str, labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels.reshape(-1).astype(int), minlength=n_classes).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    if mode == "cancer":
        weights[:] = 1.0
        weights[8] = 2.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    mixup_alpha: float,
    progress: bool,
    max_batches: int | None,
    grad_accum_steps: int,
    log_every_batches: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="train", leave=False, disable=not progress), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x = x.to(device)
        y = y.squeeze().long().to(device)
        x, ya, yb, lam = mixup(x, y, mixup_alpha)
        logits = model(x)
        raw_loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss = raw_loss / grad_accum_steps
        loss.backward()
        if batch_idx % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach().cpu()) * x.size(0)
        seen += x.size(0)
        if log_every_batches and batch_idx % log_every_batches == 0:
            print(json.dumps({"batch": batch_idx, "seen": seen, "train_loss_running": total_loss / seen}), flush=True)
    if seen and batch_idx % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(seen, 1)


@torch.inference_mode()
def predict(model: nn.Module, loader, device: torch.device, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out = []
    y_out = []
    for x, y in tqdm(loader, desc="predict", leave=False, disable=not progress):
        logits_out.append(model(x.to(device)).cpu().numpy())
        y_out.append(y.numpy().reshape(-1))
    logits = np.concatenate(logits_out)
    labels = np.concatenate(y_out)
    return labels, softmax_np(logits)


def save_predictions(path: Path, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=y_true, y_prob=y_prob)


def build_optimizer(args, model: nn.Module):
    if args.optimizer == "sgd":
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    if args.scheduler == "multistep":
        return MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    return CosineAnnealingLR(optimizer, T_max=args.epochs)


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name in ("head", "fc", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--source-size", type=int, default=None)
    parser.add_argument("--augment", choices=["none", "basic", "strong", "histology"], default="none")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["cosine", "multistep"], default="cosine")
    parser.add_argument("--milestones", type=int, nargs="*", default=[50, 75])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--class-weights", choices=["none", "balanced", "cancer"], default="none")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--norm", choices=["pathmnist", "official"], default="official")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-auc", type=float, default=None)
    parser.add_argument("--target-acc", type=float, default=None)
    parser.add_argument("--target-cancer-precision", type=float, default=None)
    parser.add_argument("--target-cancer-recall", type=float, default=None)
    parser.add_argument("--target-cancer-f1", type=float, default=None)
    parser.add_argument("--target-cancer-f2", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--log-every-batches", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    meta = dataset_meta()
    device = pick_device(args.device)
    run_name = args.run_name or (
        f"{args.model}_{args.image_size}px_{args.augment}_s{args.seed}_{'pre' if args.pretrained else 'scratch'}"
    )
    result_dir = Path(args.results_dir) / run_name
    model_dir = Path(args.models_dir) / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "train": build_loader(
            "train", args.batch_size, args.image_size, args.source_size, args.augment, args.workers, args.data_root, args.norm
        ),
        "val": build_loader(
            "val", args.batch_size, args.image_size, args.source_size, "none", args.workers, args.data_root, args.norm
        ),
        "test": build_loader(
            "test", args.batch_size, args.image_size, args.source_size, "none", args.workers, args.data_root, args.norm
        ),
    }
    model = build_model(args.model, meta.n_classes, args.pretrained).to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
    train_labels = loaders["train"].dataset.labels.reshape(-1)
    weights = class_weight_tensor(args.class_weights, train_labels, meta.n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    config = vars(args) | {
        "device_resolved": str(device),
        "parameters": count_parameters(model),
        "model_dir": str(model_dir),
        "result_dir": str(result_dir),
    }
    (result_dir / "config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))

    history = []
    best_score = -1.0
    best_epoch = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            args.mixup_alpha,
            not args.no_progress,
            args.max_train_batches,
            args.grad_accum_steps,
            args.log_every_batches,
        )
        scheduler.step()
        y_val, p_val = predict(model, loaders["val"], device, not args.no_progress)
        val = compute_metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": loss, **val.__dict__, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        score = val.auc + val.acc
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, model_dir / "best.pt")
            save_predictions(result_dir / "val_predictions.npz", y_val, p_val)
        (result_dir / "history.json").write_text(json.dumps(history, indent=2))

    checkpoint = torch.load(model_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    results = {"best_epoch": best_epoch, "seconds": time.time() - started}
    for split in ("val", "test"):
        y, p = predict(model, loaders[split], device, not args.no_progress)
        save_predictions(result_dir / f"{split}_predictions.npz", y, p)
        results[split] = report_dict(y, p, meta.class_names)
    test = results["test"]
    target_checks = {
        "auc": (args.target_auc, test["auc"]),
        "acc": (args.target_acc, test["acc"]),
        "cancer_precision": (args.target_cancer_precision, test["cancer_precision"]),
        "cancer_recall": (args.target_cancer_recall, test["cancer_recall"]),
        "cancer_f1": (args.target_cancer_f1, test["cancer_f1"]),
        "cancer_f2": (args.target_cancer_f2, test["cancer_f2"]),
    }
    results["target"] = {
        key: {"target": target, "actual": actual, "passed": target is None or actual >= target}
        for key, (target, actual) in target_checks.items()
        if target is not None
    }
    results["target"]["passed"] = all(item["passed"] for item in results["target"].values() if isinstance(item, dict))
    (result_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))
    if args.target_auc is not None or args.target_acc is not None:
        print(json.dumps(results["target"], indent=2))


if __name__ == "__main__":
    main()
