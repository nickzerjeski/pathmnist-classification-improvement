from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float, progress: bool) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    for x, y in tqdm(loader, desc="train", leave=False, disable=not progress):
        x = x.to(device)
        y = y.squeeze().long().to(device)
        x, ya, yb, lam = mixup(x, y, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch = x.size(0)
        total_loss += float(loss.detach().cpu()) * batch
        seen += batch
    return total_loss / seen


@torch.inference_mode()
def predict(model, loader, device, progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out = []
    y_out = []
    for x, y in tqdm(loader, desc="predict", leave=False, disable=not progress):
        logits = model(x.to(device)).cpu().numpy()
        logits_out.append(logits)
        y_out.append(y.numpy().reshape(-1))
    logits = np.concatenate(logits_out)
    labels = np.concatenate(y_out)
    return labels, softmax_np(logits)


def save_predictions(path: Path, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=y_true, y_prob=y_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--augment", choices=["none", "basic", "strong", "histology"], default="strong")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--norm", choices=["pathmnist", "official"], default="pathmnist")
    parser.add_argument("--out-dir", default="results/experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    meta = dataset_meta()
    device = pick_device(args.device)
    run_name = args.run_name or (
        f"{args.model}_s{args.seed}_{args.image_size}px_{args.augment}_"
        f"{'pre' if args.pretrained else 'scratch'}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "train": build_loader(
            "train", args.batch_size, args.image_size, args.augment, args.workers, args.data_root, args.norm
        ),
        "val": build_loader("val", args.batch_size, args.image_size, "none", args.workers, args.data_root, args.norm),
        "test": build_loader("test", args.batch_size, args.image_size, "none", args.workers, args.data_root, args.norm),
    }
    model = build_model(args.model, meta.n_classes, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_auc = -1.0
    best_epoch = 0
    started = time.time()
    config = vars(args) | {"device_resolved": str(device), "parameters": count_parameters(model)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, args.mixup_alpha, not args.no_progress
        )
        scheduler.step()
        y_val, p_val = predict(model, loaders["val"], device, not args.no_progress)
        val = compute_metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": loss, **val.__dict__, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val.auc > best_auc:
            best_auc = val.auc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "best.pt")
            save_predictions(out_dir / "val_predictions.npz", y_val, p_val)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    results = {"best_epoch": best_epoch, "seconds": time.time() - started}
    for split in ("val", "test"):
        y, p = predict(model, loaders[split], device, not args.no_progress)
        save_predictions(out_dir / f"{split}_predictions.npz", y, p)
        results[split] = report_dict(y, p, meta.class_names)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
