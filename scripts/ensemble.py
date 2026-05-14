from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pathmnist.data import dataset_meta
from pathmnist.metrics import report_dict


def parse_weighted_file(item: str) -> tuple[float, str]:
    if "=" not in item:
        return 1.0, item
    weight, path = item.split("=", 1)
    return float(weight), path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+", help="Either path.npz or weight=path.npz")
    parser.add_argument("--out", default="results/ensemble_metrics.json")
    parser.add_argument("--pred-out", default=None)
    args = parser.parse_args()
    meta = dataset_meta()
    probs = []
    weights = []
    labels = None
    for item in args.prediction_files:
        weight, path = parse_weighted_file(item)
        arr = np.load(path)
        if labels is None:
            labels = arr["y_true"]
        elif not np.array_equal(labels, arr["y_true"]):
            raise ValueError(f"label mismatch in {path}")
        weights.append(weight)
        probs.append(arr["y_prob"])
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    y_prob = np.average(np.stack(probs), axis=0, weights=weights_arr)
    report = report_dict(labels, y_prob, meta.class_names)
    report["ensemble_weights"] = weights_arr.tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    if args.pred_out:
        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.pred_out, y_true=labels, y_prob=y_prob)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
