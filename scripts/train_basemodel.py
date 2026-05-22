from __future__ import annotations

import sys

from pathmnist.train import main


DEFAULT_ARGS = [
    "--model",
    "small_cnn",
    "--image-size",
    "224",
    "--source-size",
    "224",
    "--augment",
    "histology",
    "--epochs",
    "10",
    "--batch-size",
    "32",
    "--lr",
    "0.001",
    "--weight-decay",
    "0.0001",
    "--optimizer",
    "adamw",
    "--scheduler",
    "cosine",
    "--label-smoothing",
    "0.03",
    "--mixup-alpha",
    "0.05",
    "--class-weights",
    "none",
    "--seed",
    "42",
    "--workers",
    "0",
    "--norm",
    "pathmnist",
    "--run-name",
    "basemodel",
    "--target-cancer-f2",
    "0.98",
    "--no-progress",
]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_ARGS)
    main()
