from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathmnist.data import PATHMNIST_MEAN, PATHMNIST_STD, build_dataset, dataset_meta
from pathmnist.models import build_model


LAYER_SPECS = [
    ("Convolutional layer 1", 2),
    ("Convolutional layer 2", 9),
    ("Convolutional layer 3", 16),
]

SHORT_CLASS_NAMES = {
    "colorectal adenocarcinoma epithelium": "adenocarcinoma epithelium",
    "cancer-associated stroma": "cancer-associated stroma",
}


def load_baseline224(checkpoint_path: Path) -> torch.nn.Module:
    meta = dataset_meta()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = build_model(config.get("model", "small_cnn"), meta.n_classes, config.get("pretrained", False))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(PATHMNIST_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(PATHMNIST_STD, dtype=tensor.dtype)[:, None, None]
    image = tensor.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def capture_activations(model: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for label, layer_index in LAYER_SPECS:
        layer = model.net[layer_index]
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, name=label: activations.__setitem__(name, output.detach().cpu())
            )
        )
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    for handle in handles:
        handle.remove()
    return activations, probabilities


def select_top_channels(activation: torch.Tensor, top_k: int) -> list[int]:
    maps = activation.squeeze(0)
    scores = maps.abs().mean(dim=(1, 2))
    return torch.topk(scores, k=min(top_k, maps.shape[0])).indices.tolist()


def plot_feature_maps(
    *,
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    target_class: int,
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    meta = dataset_meta()
    x, y = dataset[sample_index]
    y_int = int(np.asarray(y).reshape(-1)[0])
    x_batch = x.unsqueeze(0)
    image = denormalize(x)
    activations, probabilities = capture_activations(model, x_batch)
    pred_class = int(np.argmax(probabilities))

    fig = plt.figure(figsize=(13.8, 7.6), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=top_k + 2,
        width_ratios=[1.15, 1.05, *([1.0] * top_k)],
        wspace=0.10,
        hspace=0.18,
    )

    input_ax = fig.add_subplot(grid[:, 0])
    input_ax.imshow(image)
    input_ax.axis("off")
    input_ax.set_title("Input patch", fontsize=8.5, pad=6)

    for row, (layer_label, _layer_index) in enumerate(LAYER_SPECS):
        activation = activations[layer_label].squeeze(0)
        channel_indices = select_top_channels(activation.unsqueeze(0), top_k)
        label_ax = fig.add_subplot(grid[row, 1])
        label_ax.axis("off")
        label_ax.text(0.5, 0.5, layer_label, ha="center", va="center", fontsize=9, clip_on=False)
        for col, channel in enumerate(channel_indices, start=2):
            ax = fig.add_subplot(grid[row, col])
            feature_map = activation[channel].numpy()
            ax.imshow(feature_map, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"ch {channel}", fontsize=8, pad=3)

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def default_stroma_index(prediction_path: Path, target_class: int) -> int:
    predictions = np.load(prediction_path)
    y_true = predictions["y_true"]
    y_prob = predictions["y_prob"]
    y_pred = y_prob.argmax(axis=1)
    candidates = np.where((y_true == target_class) & (y_pred == target_class))[0]
    if len(candidates) == 0:
        candidates = np.where(y_true == target_class)[0]
    return int(candidates[np.argmax(y_prob[candidates, target_class])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline224 layer activation map figures.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/baseline_224/best.pt")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "results/baseline_224/test_predictions.npz")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--cancer-index", type=int, default=5754)
    parser.add_argument("--stroma-index", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cancer-out", type=Path, default=PROJECT_ROOT / "report/figures/baseline224_cancer_feature_maps.svg")
    parser.add_argument("--stroma-out", type=Path, default=PROJECT_ROOT / "report/figures/baseline224_stroma_feature_maps.svg")
    args = parser.parse_args()

    cancer_class = 8
    stroma_class = 7
    stroma_index = args.stroma_index
    if stroma_index is None:
        stroma_index = default_stroma_index(args.predictions, stroma_class)

    model = load_baseline224(args.checkpoint)
    dataset = build_dataset(
        split="test",
        image_size=224,
        source_size=224,
        augment="none",
        root=str(args.data_root),
        norm="pathmnist",
    )

    args.cancer_out.parent.mkdir(parents=True, exist_ok=True)
    args.stroma_out.parent.mkdir(parents=True, exist_ok=True)

    plot_feature_maps(
        model=model,
        dataset=dataset,
        sample_index=args.cancer_index,
        target_class=cancer_class,
        output_path=args.cancer_out,
        title="Baseline 224 feature maps for colorectal adenocarcinoma epithelium",
        top_k=args.top_k,
    )
    plot_feature_maps(
        model=model,
        dataset=dataset,
        sample_index=stroma_index,
        target_class=stroma_class,
        output_path=args.stroma_out,
        title="Baseline 224 feature maps for cancer-associated stroma",
        top_k=args.top_k,
    )
    print(f"cancer_index={args.cancer_index}")
    print(f"stroma_index={stroma_index}")


if __name__ == "__main__":
    main()
