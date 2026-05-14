from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pathmnist.data import dataset_meta
from pathmnist.metrics import report_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    arr = np.load(args.predictions_npz)
    meta = dataset_meta()
    report = report_dict(arr["y_true"], arr["y_prob"], meta.class_names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
