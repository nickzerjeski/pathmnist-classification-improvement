from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json")
    parser.add_argument("--target-auc", type=float, default=None)
    parser.add_argument("--target-acc", type=float, default=None)
    parser.add_argument("--target-cancer-precision", type=float, default=None)
    parser.add_argument("--target-cancer-recall", type=float, default=None)
    parser.add_argument("--target-cancer-f1", type=float, default=None)
    parser.add_argument("--target-cancer-f2", type=float, default=None)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text())
    test = metrics.get("test", metrics)
    checks = {
        "auc": args.target_auc,
        "acc": args.target_acc,
        "cancer_precision": args.target_cancer_precision,
        "cancer_recall": args.target_cancer_recall,
        "cancer_f1": args.target_cancer_f1,
        "cancer_f2": args.target_cancer_f2,
    }
    report = {}
    for key, target in checks.items():
        if target is None:
            continue
        actual = float(test[key])
        report[key] = {"target": target, "actual": actual, "passed": actual >= target}
    report["passed"] = all(item["passed"] for item in report.values())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
