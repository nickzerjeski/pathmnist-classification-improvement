from __future__ import annotations

import argparse
import base64
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from PIL import ImageFilter
from torch import nn

from pathmnist.data import PATHMNIST_MEAN, PATHMNIST_STD, dataset_meta
from pathmnist.models import build_model


CANCER_CLASS = 8


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(PATHMNIST_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(PATHMNIST_STD, device=x.device).view(3, 1, 1)
    image = (x * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).detach().cpu().numpy()


def load_test_sample(path: str | Path, sample_index: int, device: torch.device) -> tuple[torch.Tensor, int]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("test_images.npy") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
            else:
                raise ValueError(f"Unsupported NPY version: {version}")
            if fortran_order:
                raise ValueError("Fortran-order arrays are not supported")
            sample_bytes = int(np.prod(shape[1:]) * dtype.itemsize)
            remaining = sample_index * sample_bytes
            while remaining:
                chunk = fh.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IndexError(f"Sample index {sample_index} is outside test_images")
                remaining -= len(chunk)
            image = np.frombuffer(fh.read(sample_bytes), dtype=dtype).reshape(shape[1:]).copy()

        labels = np.load(archive.open("test_labels.npy"))
        label = int(np.asarray(labels[sample_index]).reshape(-1)[0])

    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(PATHMNIST_MEAN).view(3, 1, 1)
    std = torch.tensor(PATHMNIST_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device), label


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def blur_cam(cam: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return cam
    arr = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
    blurred = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=radius))
    return torch.from_numpy(np.asarray(blurred, dtype=np.float32) / 255.0).to(cam.device)


def tissue_mask(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = np.clip(image, 0, 1)
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    mask = np.clip((1.0 - brightness) * 1.4 + saturation * 1.8, 0, 1)
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def region_cam(cam: torch.Tensor, image: np.ndarray, radius: float = 8.0) -> torch.Tensor:
    smooth = blur_cam(cam, radius)
    masked = smooth * tissue_mask(image, cam.device).pow(0.7)
    return normalize_cam(masked)


def upsample_cam(cam: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]


class ActivationCapture:
    def __init__(self, layer: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        self.activations = output
        output.retain_grad()

        def save_grad(grad: torch.Tensor) -> None:
            self.gradients = grad

        output.register_hook(save_grad)

    def close(self) -> None:
        self.handle.remove()


def gradcam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_class: int, plus_plus: bool) -> torch.Tensor:
    capture = ActivationCapture(layer)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[:, target_class].sum().backward()
    activations = capture.activations.detach()[0]
    gradients = capture.gradients.detach()[0]
    capture.close()

    if plus_plus:
        grads2 = gradients.pow(2)
        grads3 = gradients.pow(3)
        denom = 2 * grads2 + (activations * grads3).sum(dim=(1, 2), keepdim=True)
        alpha = grads2 / (denom + 1e-8)
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
    else:
        weights = gradients.mean(dim=(1, 2))

    cam = F.relu((weights[:, None, None] * activations).sum(dim=0))
    return normalize_cam(cam)


def tta_gradcam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
    variants = [
        (x, lambda cam: cam),
        (torch.flip(x, dims=[3]), lambda cam: torch.flip(cam, dims=[1])),
        (torch.flip(x, dims=[2]), lambda cam: torch.flip(cam, dims=[0])),
        (torch.rot90(x, 1, dims=[2, 3]), lambda cam: torch.rot90(cam, -1, dims=[0, 1])),
        (torch.rot90(x, 2, dims=[2, 3]), lambda cam: torch.rot90(cam, -2, dims=[0, 1])),
        (torch.rot90(x, 3, dims=[2, 3]), lambda cam: torch.rot90(cam, -3, dims=[0, 1])),
    ]
    cams = [invert(gradcam(model, layer, variant, target_class, plus_plus=False)) for variant, invert in variants]
    return normalize_cam(torch.stack(cams).mean(dim=0))


@torch.inference_mode()
def ablation_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
    base_capture: dict[str, torch.Tensor] = {}
    base_handle = layer.register_forward_hook(lambda _m, _i, output: base_capture.setdefault("activation", output.detach()))
    base_logit = model(x)[:, target_class].item()
    base_handle.remove()
    activations = base_capture["activation"][0]

    if hasattr(model, "net") and hasattr(model, "head"):
        ablated_logits = []
        for start in range(0, activations.shape[0], 32):
            stop = min(start + 32, activations.shape[0])
            batch = activations.unsqueeze(0).repeat(stop - start, 1, 1, 1)
            for row, channel in enumerate(range(start, stop)):
                batch[row, channel] = 0
            tail = model.net[17:](batch)
            ablated_logits.append(model.head(tail)[:, target_class])
        ablated = torch.cat(ablated_logits)
        weights = (base_logit - ablated).clamp_min(0)
    else:
        weights_list = []
        for channel in range(activations.shape[0]):
            def ablate(_module, _inputs, output, channel=channel):
                replaced = output.clone()
                replaced[:, channel] = 0
                return replaced

            handle = layer.register_forward_hook(ablate)
            logit = model(x)[:, target_class].item()
            handle.remove()
            weights_list.append(max(base_logit - logit, 0.0))
        weights = torch.tensor(weights_list, device=x.device, dtype=activations.dtype)

    cam = F.relu((weights[:, None, None] * activations).sum(dim=0))
    return normalize_cam(cam)


def find_layer(model: nn.Module, layer_index: int | None = None) -> nn.Module:
    if hasattr(model, "net"):
        return model.net[16 if layer_index is None else layer_index]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    raise ValueError("Unsupported model architecture for automatic CAM layer selection")


def turbo_like(cam: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.0, 0.05, 0.17, 0.55],
            [0.0, 0.45, 0.95, 1.0],
            [0.0, 0.90, 0.50, 0.0],
            [1.0, 0.90, 0.05, 0.0],
            [0.85, 0.05, 0.02, 0.0],
        ],
        dtype=np.float32,
    )
    x = np.clip(cam, 0, 1) * (len(stops) - 1)
    lo = np.floor(x).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = x[..., None] - lo[..., None]
    return ((1 - frac) * stops[lo] + frac * stops[hi])


def overlay_image(image: np.ndarray, cam: torch.Tensor, alpha: float = 0.48) -> Image.Image:
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    cam_np = cam.detach().cpu().numpy()
    colors = turbo_like(cam_np)
    strength = np.clip((cam_np - 0.18) / 0.82, 0, 1)[..., None]
    mixed = base * (1 - alpha * strength) + (colors[..., :3] * 255) * (alpha * strength)
    return Image.fromarray(mixed.astype(np.uint8))


def heatmap_image(cam: torch.Tensor) -> Image.Image:
    colors = turbo_like(cam.detach().cpu().numpy())
    return Image.fromarray((colors[..., :3] * 255).astype(np.uint8))


def panel_with_title(image: Image.Image, title: str, width: int = 304, title_height: int = 34) -> Image.Image:
    resized = image.resize((width, width), Image.Resampling.BICUBIC)
    panel = Image.new("RGB", (width, width + title_height), "white")
    panel.paste(resized, (0, title_height))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    text_width = draw.textlength(title, font=font)
    draw.text(((width - text_width) / 2, 11), title, fill=(20, 20, 20), font=font)
    return panel


def save_svg_from_png(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>\n'
        f"</svg>\n"
    )


def render_figure(
    image: np.ndarray,
    cams: dict[str, torch.Tensor],
    target_name: str,
    probability: float,
    sample_index: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    panels = [panel_with_title(base, "Input image")]
    for name, cam in cams.items():
        if "heatmap" in name.lower():
            rendered = heatmap_image(cam)
        else:
            rendered = overlay_image(image, cam)
        panels.append(panel_with_title(rendered, name))

    caption = f"{target_name}; p={probability:.3f}; sample index {sample_index}"
    gap = 14
    caption_height = 30
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels) + caption_height
    canvas = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, caption_height))
        x_offset += panel.width + gap
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_width = draw.textlength(caption, font=font)
    draw.text(((width - text_width) / 2, 9), caption, fill=(20, 20, 20), font=font)

    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    canvas.save(png_path)
    if out_path.suffix.lower() == ".svg":
        save_svg_from_png(png_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/baseline_224/best.pt")
    parser.add_argument("--sample-index", type=int, default=5754)
    parser.add_argument("--out", default="report/figures/baseline224_cancer_gradcam.svg")
    parser.add_argument("--comparison-out", default="report/figures/baseline224_cancer_gradcam_variants.png")
    parser.add_argument("--data", default="data/pathmnist_224.npz")
    parser.add_argument("--layer-index", type=int, default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    meta = dataset_meta()
    model = build_model(config["model"], meta.n_classes, config["pretrained"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    layer = find_layer(model, args.layer_index)

    x, y = load_test_sample(args.data, args.sample_index, device)
    image = denormalize(x[0])

    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    probability = probs[CANCER_CLASS].item()
    target_name = meta.class_names[CANCER_CLASS]

    raw_cam = upsample_cam(gradcam(model, layer, x, CANCER_CLASS, plus_plus=False), image.shape[:2])
    plus_cam = upsample_cam(gradcam(model, layer, x, CANCER_CLASS, plus_plus=True), image.shape[:2])
    tta_cam = upsample_cam(tta_gradcam(model, layer, x, CANCER_CLASS), image.shape[:2])
    ablate_cam = None if args.skip_ablation else upsample_cam(ablation_cam(model, layer, x, CANCER_CLASS), image.shape[:2])

    cams = {
        "Logit Grad-CAM": blur_cam(raw_cam, 1.5),
        "TTA smoothed": blur_cam(tta_cam, 2.0),
        "Region overlay": region_cam(tta_cam, image),
        "Grad-CAM++": blur_cam(plus_cam, 1.5),
    }
    if ablate_cam is not None:
        cams["Ablation CAM"] = blur_cam(ablate_cam, 2.0)
    render_figure(image, cams, target_name, probability, args.sample_index, Path(args.comparison_out))

    selected = {
        "Grad-CAM heatmap": cams["Region overlay"],
        "Tissue-region overlay": cams["Region overlay"],
    }
    render_figure(image, selected, target_name, probability, args.sample_index, Path(args.out))
    print(
        json.dumps(
            {
                "sample_index": args.sample_index,
                "true_label": y,
                "target_class": CANCER_CLASS,
                "target_probability": probability,
                "comparison": args.comparison_out,
                "selected": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
