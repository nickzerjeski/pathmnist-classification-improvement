from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathmnist.data import build_dataset, dataset_meta
from pathmnist.metrics import report_dict


def flatten_images(split: str, root: str) -> tuple[np.ndarray, np.ndarray]:
    ds = build_dataset(split=split, image_size=28, augment="none", root=root)
    x = ds.imgs.astype("float32").reshape(len(ds), -1) / 255.0
    y = ds.labels.reshape(-1).astype(int)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised cluster features + supervised linear classifier.")
    parser.add_argument("--clusters", type=int, default=36)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/cluster_features_metrics.json")
    args = parser.parse_args()

    meta = dataset_meta()
    x_train, y_train = flatten_images("train", args.data_root)
    x_val, y_val = flatten_images("val", args.data_root)
    x_test, y_test = flatten_images("test", args.data_root)

    kmeans = MiniBatchKMeans(n_clusters=args.clusters, batch_size=4096, random_state=7, n_init="auto")
    train_cluster = kmeans.fit_predict(x_train)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    c_train = encoder.fit_transform(train_cluster.reshape(-1, 1))
    c_val = encoder.transform(kmeans.predict(x_val).reshape(-1, 1))
    c_test = encoder.transform(kmeans.predict(x_test).reshape(-1, 1))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=2.0, multi_class="ovr", n_jobs=-1),
    )
    clf.fit(np.hstack([x_train, c_train]), y_train)
    report = {
        "val_acc": float(accuracy_score(y_val, clf.predict(np.hstack([x_val, c_val])))),
        "test": report_dict(y_test, clf.predict_proba(np.hstack([x_test, c_test])), meta.class_names),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

