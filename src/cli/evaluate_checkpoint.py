from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data import build_loader, dataset_meta
from metrics import report_dict, softmax_np
from models import build_model
from train import pick_device


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for x, y in loader:
        logits = model(x.to(device)).cpu().numpy()
        probs.append(softmax_np(logits))
        labels.append(y.numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    meta = dataset_meta()
    model = build_model(config["model"], meta.n_classes, config["pretrained"]).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = build_loader(
        args.split,
        args.batch_size,
        config["image_size"],
        config.get("source_size"),
        "none",
        args.workers,
        config.get("data_root", "src/dataset"),
        config.get("norm", "pathmnist"),
    )
    y_true, y_prob = predict(model, loader, device)
    report = report_dict(y_true, y_prob, meta.class_names)
    out_dir = Path(args.out_dir) if args.out_dir else Path(config.get("result_dir", checkpoint_path.parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.split}_metrics_eval.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / f"{args.split}_predictions_eval.npz", y_true=y_true, y_prob=y_prob)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
