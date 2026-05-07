from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pathmnist.data import build_dataset, dataset_meta


def main() -> None:
    meta = dataset_meta()
    summary = {"classes": meta.class_names, "splits": {}}
    for split in ["train", "val", "test"]:
        ds = build_dataset(split=split, image_size=28, augment="none", root="data")
        counts = Counter(int(y[0]) for y in ds.labels)
        summary["splits"][split] = {
            "n": len(ds),
            "counts": {meta.class_names[k]: counts[k] for k in range(meta.n_classes)},
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

