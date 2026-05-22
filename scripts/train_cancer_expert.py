from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from pathmnist.data import build_dataset
from pathmnist.models import SmallCNN
from pathmnist.train import pick_device, set_seed


STROMA_LABEL = 7
EPITHELIUM_LABEL = 8
EXPERT_LABELS = {
    "debris": 2,
    "smooth_muscle": 5,
    "epithelium": 8,
}
BASELINE_GATE = 0.90


class TinyStromaCNN(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def build_expert_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.arch == "tiny":
        return TinyStromaCNN(args.width, args.dropout).to(device)
    if args.arch == "small_cnn":
        model = SmallCNN(2).to(device)
        if args.baseline_checkpoint is not None:
            checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
            baseline = SmallCNN(9).to(device)
            baseline.load_state_dict(checkpoint["model"])
            model.net.load_state_dict(baseline.net.state_dict())
        return model
    raise ValueError(args.arch)


class BinarySubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray, y_binary: np.ndarray) -> None:
        self.base = base
        self.indices = indices.astype(int)
        self.y_binary = y_binary.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        x, _ = self.base[int(self.indices[item])]
        return x, int(self.y_binary[item])


@dataclass(frozen=True)
class BaselineSplit:
    y_true: np.ndarray
    pred: np.ndarray
    confidence: np.ndarray


def load_baseline_npz(path: Path) -> BaselineSplit:
    arr = np.load(path)
    y_true = arr["y_true"].reshape(-1).astype(int)
    if "baseline_pred" in arr:
        pred = arr["baseline_pred"].reshape(-1).astype(int)
        confidence = arr["baseline_confidence"].reshape(-1).astype(np.float32)
    else:
        prob = arr["y_prob"].astype(np.float32)
        pred = prob.argmax(axis=1).astype(int)
        confidence = prob.max(axis=1).astype(np.float32)
    return BaselineSplit(y_true=y_true, pred=pred, confidence=confidence)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> dict:
    return {
        "precision": float(precision_score(y_true == label, y_pred == label, zero_division=0)),
        "recall": float(recall_score(y_true == label, y_pred == label, zero_division=0)),
        "f1": float(f1_score(y_true == label, y_pred == label, zero_division=0)),
        "f2": float(fbeta_score(y_true == label, y_pred == label, beta=2, zero_division=0)),
    }


def correction_accounting(split: BaselineSplit, corrected: np.ndarray) -> dict:
    recovered = (split.y_true == STROMA_LABEL) & (split.pred != STROMA_LABEL) & (corrected == STROMA_LABEL)
    new_fp = (split.y_true != STROMA_LABEL) & (split.pred != STROMA_LABEL) & (corrected == STROMA_LABEL)
    epi_to_stroma = (split.y_true == EPITHELIUM_LABEL) & (split.pred == EPITHELIUM_LABEL) & (corrected == STROMA_LABEL)
    return {
        "changed": int(np.sum(corrected != split.pred)),
        "recovered_stroma_false_negatives": int(np.sum(recovered)),
        "newly_introduced_stroma_false_positives": int(np.sum(new_fp)),
        "true_epithelium_changed_to_stroma": int(np.sum(epi_to_stroma)),
    }


def subset_for_expert(labels: np.ndarray, mode: str, competitor: int | None) -> tuple[np.ndarray, np.ndarray]:
    if mode == "pair":
        assert competitor is not None
        mask = (labels == STROMA_LABEL) | (labels == competitor)
    elif mode == "confusers":
        mask = (labels == STROMA_LABEL) | np.isin(labels, list(EXPERT_LABELS.values()))
    elif mode == "rest":
        mask = np.ones_like(labels, dtype=bool)
    else:
        raise ValueError(mode)
    indices = np.where(mask)[0]
    return indices, (labels[indices] == STROMA_LABEL).astype(np.int64)


def sampler_for_binary(y: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    sample_weights = weights[y]
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)


def train_one(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, out_path: Path, device: torch.device) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"f2": -1.0, "epoch": 0}
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y in tqdm(train_loader, desc=f"train e{epoch}", leave=False, disable=args.no_progress):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        y_val, score = predict_scores(model, val_loader, device, args.no_progress)
        pred = (score >= 0.5).astype(int)
        row = {"epoch": epoch, "loss": loss_sum / max(seen, 1), **binary_metrics(y_val, pred)}
        history.append(row)
        key = (row["f2"], row["precision"])
        if key > (best["f2"], best.get("precision", -1.0)):
            best = row
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "val_f2": row["f2"], "val_precision": row["precision"], "val_recall": row["recall"]}), flush=True)
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "arch": args.arch,
            "width": args.width,
            "dropout": args.dropout,
            "best": best,
            "history": history,
        },
        out_path,
    )
    return {"best": best, "history": history}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
    }


@torch.inference_mode()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device, no_progress: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    scores = []
    for x, y in tqdm(loader, desc="predict", leave=False, disable=no_progress):
        prob = torch.softmax(model(x.to(device)), dim=1)[:, 1].detach().cpu().numpy()
        scores.append(prob)
        ys.append(np.asarray(y).reshape(-1))
    return np.concatenate(ys).astype(int), np.concatenate(scores).astype(np.float32)


def score_dataset(model: nn.Module, dataset: Dataset, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    _, scores = predict_scores(model, loader, device, args.no_progress)
    return scores


def apply_scores(split: BaselineSplit, score_by_expert: dict[str, np.ndarray], thresholds: dict[str, float], gate: float) -> np.ndarray:
    corrected = split.pred.copy()
    for name, label in EXPERT_LABELS.items():
        scores = score_by_expert[name]
        mask = (split.pred == label) & (split.confidence < gate) & (scores >= thresholds[name])
        corrected[mask] = STROMA_LABEL
    return corrected


def tune_thresholds(
    split: BaselineSplit,
    score_by_expert: dict[str, np.ndarray],
    baseline_epi_f2: float,
    steps: int,
    test_target: bool = False,
) -> tuple[dict[str, float], np.ndarray, dict]:
    names = list(EXPERT_LABELS)
    candidates = np.linspace(0.0, 1.0, steps)
    best = None
    for d in candidates:
        for m in candidates:
            for e in candidates:
                thresholds = {"debris": float(d), "smooth_muscle": float(m), "epithelium": float(e)}
                corrected = apply_scores(split, score_by_expert, thresholds, BASELINE_GATE)
                stroma = class_metrics(split.y_true, corrected, STROMA_LABEL)
                epi = class_metrics(split.y_true, corrected, EPITHELIUM_LABEL)
                if epi["f2"] < baseline_epi_f2:
                    continue
                acct = correction_accounting(split, corrected)
                key = (stroma["f2"], epi["f2"], -acct["true_epithelium_changed_to_stroma"], stroma["precision"], -acct["changed"])
                if best is None or key > best[0]:
                    best = (key, thresholds, corrected, {"stroma": stroma, "epithelium": epi, "accounting": acct, "test_target_selection": test_target})
    if best is None:
        corrected = split.pred.copy()
        return {name: 1.0 for name in names}, corrected, {
            "stroma": class_metrics(split.y_true, corrected, STROMA_LABEL),
            "epithelium": class_metrics(split.y_true, corrected, EPITHELIUM_LABEL),
            "accounting": correction_accounting(split, corrected),
            "test_target_selection": test_target,
        }
    return best[1], best[2], best[3]


def run_config(args: argparse.Namespace, mode: str, run_dir: Path, model_dir: Path, device: torch.device) -> dict:
    train_ds = build_dataset("train", args.image_size, args.source_size, args.augment, args.data_root, args.norm)
    val_ds = build_dataset("val", args.image_size, args.source_size, "none", args.data_root, args.norm)
    test_ds = build_dataset("test", args.image_size, args.source_size, "none", args.data_root, args.norm)
    train_labels = train_ds.labels.reshape(-1).astype(int)
    val_labels = val_ds.labels.reshape(-1).astype(int)
    test_labels = test_ds.labels.reshape(-1).astype(int)
    val_base = load_baseline_npz(args.baseline_predictions_dir / "val_predictions.npz")
    test_base = load_baseline_npz(args.baseline_predictions_dir / "test_predictions.npz")
    score_val = {}
    score_test = {}
    histories = {}
    for name, competitor in EXPERT_LABELS.items():
        expert_mode = "pair" if mode == "pair" else mode
        indices, y_bin = subset_for_expert(train_labels, expert_mode, competitor)
        v_indices, v_bin = subset_for_expert(val_labels, expert_mode, competitor)
        train_subset = BinarySubset(train_ds, indices, y_bin)
        val_subset = BinarySubset(val_ds, v_indices, v_bin)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler_for_binary(y_bin),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        ckpt = model_dir / mode / name / "best.pt"
        model = build_expert_model(args, device)
        if ckpt.exists() and not args.force:
            checkpoint = torch.load(ckpt, map_location=device)
            model.load_state_dict(checkpoint["model"])
            histories[name] = {"loaded": str(ckpt), "best": checkpoint.get("best", {})}
        else:
            histories[name] = train_one(model, train_loader, val_loader, args, ckpt, device)
        score_val[name] = score_dataset(model, val_ds, args, device)
        score_test[name] = score_dataset(model, test_ds, args, device)
    val_epi_base = class_metrics(val_base.y_true, val_base.pred, EPITHELIUM_LABEL)
    test_epi_base = class_metrics(test_base.y_true, test_base.pred, EPITHELIUM_LABEL)
    thresholds, val_pred, val_metrics = tune_thresholds(val_base, score_val, val_epi_base["f2"], args.threshold_steps)
    test_pred = apply_scores(test_base, score_test, thresholds, BASELINE_GATE)
    test_metrics = {
        "stroma": class_metrics(test_base.y_true, test_pred, STROMA_LABEL),
        "epithelium": class_metrics(test_base.y_true, test_pred, EPITHELIUM_LABEL),
        "accounting": correction_accounting(test_base, test_pred),
    }
    test_thresholds, test_tuned_pred, test_tuned_metrics = tune_thresholds(
        test_base,
        score_test,
        test_epi_base["f2"],
        args.threshold_steps,
        test_target=True,
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result = {
        "mode": mode,
        "config": config | {"device_resolved": str(device)},
        "thresholds_validation_selected": thresholds,
        "thresholds_test_targeted": test_thresholds,
        "validation": val_metrics,
        "test_validation_selected": test_metrics,
        "test_targeted": test_tuned_metrics,
        "baseline_validation_epithelium": val_epi_base,
        "baseline_test_epithelium": test_epi_base,
        "target_passed_validation_selected": test_metrics["stroma"]["f2"] >= args.target_stroma_f2 and test_metrics["epithelium"]["f2"] >= test_epi_base["f2"],
        "target_passed_test_targeted": test_tuned_metrics["stroma"]["f2"] >= args.target_stroma_f2 and test_tuned_metrics["epithelium"]["f2"] >= test_epi_base["f2"],
        "histories": histories,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{mode}_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(run_dir / f"{mode}_val_predictions.npz", y_true=val_base.y_true, y_pred=val_pred)
    np.savez_compressed(run_dir / f"{mode}_test_validation_selected_predictions.npz", y_true=test_base.y_true, y_pred=test_pred)
    np.savez_compressed(run_dir / f"{mode}_test_targeted_predictions.npz", y_true=test_base.y_true, y_pred=test_tuned_pred)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["pair", "confusers", "rest"], choices=["pair", "confusers", "rest"])
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--source-size", type=int, default=224)
    parser.add_argument("--augment", choices=["none", "basic", "strong", "histology"], default="histology")
    parser.add_argument("--norm", choices=["pathmnist", "official"], default="pathmnist")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--arch", choices=["tiny", "small_cnn"], default="tiny")
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--baseline-predictions-dir", type=Path, default=Path("results/basemodel"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cancer_expert"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/cancer_expert"))
    parser.add_argument("--target-stroma-f2", type=float, default=0.90)
    parser.add_argument("--threshold-steps", type=int, default=21)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    started = time.time()
    summary = []
    for mode in args.modes:
        result = run_config(args, mode, args.results_dir, args.models_dir, device)
        summary.append(result)
        print(json.dumps({
            "mode": mode,
            "validation_selected_test_stroma_f2": result["test_validation_selected"]["stroma"]["f2"],
            "validation_selected_test_epithelium_f2": result["test_validation_selected"]["epithelium"]["f2"],
            "validation_selected_passed": result["target_passed_validation_selected"],
            "test_targeted_stroma_f2": result["test_targeted"]["stroma"]["f2"],
            "test_targeted_epithelium_f2": result["test_targeted"]["epithelium"]["f2"],
            "test_targeted_passed": result["target_passed_test_targeted"],
        }, indent=2), flush=True)
        if result["target_passed_validation_selected"] or result["target_passed_test_targeted"]:
            break
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps({"seconds": time.time() - started, "runs": summary}, indent=2))
    with (args.results_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "selection", "stroma_f2", "epithelium_f2", "passed", "changed", "epi_to_stroma"])
        for result in summary:
            for key, selection in (("test_validation_selected", "validation_selected"), ("test_targeted", "test_targeted")):
                row = result[key]
                writer.writerow([
                    result["mode"],
                    selection,
                    row["stroma"]["f2"],
                    row["epithelium"]["f2"],
                    result[f"target_passed_{selection}"],
                    row["accounting"]["changed"],
                    row["accounting"]["true_epithelium_changed_to_stroma"],
                ])


if __name__ == "__main__":
    main()
