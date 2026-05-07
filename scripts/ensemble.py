from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pathmnist.data import dataset_meta
from pathmnist.metrics import report_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", nargs="+")
    parser.add_argument("--out", default="results/ensemble_metrics.json")
    args = parser.parse_args()
    meta = dataset_meta()
    probs = []
    labels = None
    for item in args.prediction_files:
        arr = np.load(item)
        if labels is None:
            labels = arr["y_true"]
        elif not np.array_equal(labels, arr["y_true"]):
            raise ValueError(f"label mismatch in {item}")
        probs.append(arr["y_prob"])
    y_prob = np.mean(probs, axis=0)
    report = report_dict(labels, y_prob, meta.class_names)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

