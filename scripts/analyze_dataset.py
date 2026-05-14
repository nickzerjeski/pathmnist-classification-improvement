from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pathmnist.data import build_dataset, dataset_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/dataset_summary.json")
    args = parser.parse_args()
    meta = dataset_meta()
    summary = {"classes": meta.class_names, "splits": {}}
    for split in ["train", "val", "test"]:
        ds = build_dataset(split=split, image_size=28, augment="none", root="data")
        counts = Counter(int(y[0]) for y in ds.labels)
        summary["splits"][split] = {
            "n": len(ds),
            "counts": {meta.class_names[k]: counts[k] for k in range(meta.n_classes)},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
