"""Render qualitative XAI heatmap panels for Gate 12 manuscript figures.

The script reads the frozen XAI sample, original dataset images, and completed XAI
attribution registries. It creates qualitative panels without training, inference,
or checkpoint modification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

ARCHITECTURES = ["yolov8n_cls", "resnet50", "swin_tiny", "vit_b_16"]
ARCHITECTURE_LABELS = {
    "yolov8n_cls": "YOLOv8n-cls",
    "resnet50": "ResNet50",
    "swin_tiny": "Swin-Tiny",
    "vit_b_16": "ViT-B/16",
}


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _load_dataset_root(project_root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["paths"]["dataset_root"])
    return _resolve(project_root, dataset_root).resolve()


def _load_heatmap(map_path: Path) -> np.ndarray:
    data = np.load(map_path)
    if "normalised_map" in data.files:
        heatmap = data["normalised_map"]
    elif "ranking_map" in data.files:
        heatmap = data["ranking_map"]
    else:
        raise ValueError(f"No supported heatmap array found in {map_path}: {data.files}")
    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum > minimum:
        heatmap = (heatmap - minimum) / (maximum - minimum)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def _load_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heat_rgb, 0.0, 1.0)


def _select_samples(sample: pd.DataFrame, images_per_class: int) -> pd.DataFrame:
    selected = []
    for _, group in sample.sort_values("selection_order").groupby("true_class", sort=True):
        selected.append(group.head(images_per_class))
    return pd.concat(selected, ignore_index=True)


def _registry_for(
    xai_root: Path,
    architecture: str,
    regime: str,
    seed: int,
    method: str,
) -> tuple[pd.DataFrame, Path]:
    if method == "integrated_gradients":
        root = xai_root / "integrated_gradients" / architecture / regime / f"seed_{seed}"
        registry_path = root / "integrated_gradients_registry.csv"
    elif method == "specific":
        root = xai_root / "specific" / architecture / regime / f"seed_{seed}"
        registry_path = root / "specific_xai_registry.csv"
    else:
        raise ValueError(f"Unsupported method: {method}")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing registry: {registry_path}")
    registry = pd.read_csv(registry_path)
    if method == "integrated_gradients" and "target_type" in registry.columns:
        registry = registry[registry["target_type"] == "predicted_class_logit"].copy()
    return registry, root


def _build_panel(
    sample_row: pd.Series,
    dataset_root: Path,
    registries: dict[str, tuple[pd.DataFrame, Path]],
    output_path: Path,
    image_size: int,
    alpha: float,
    method_label: str,
) -> dict[str, Any]:
    image_path = dataset_root / str(sample_row["relative_path"])
    image = _load_image(image_path, image_size)
    columns = 1 + len(registries)
    fig, axes = plt.subplots(1, columns, figsize=(3.0 * columns, 3.4))
    if columns == 1:
        axes = [axes]
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    rendered = []
    for ax, (architecture, (registry, root)) in zip(axes[1:], registries.items(), strict=False):
        rows = registry[registry["xai_sample_id"] == sample_row["xai_sample_id"]]
        if rows.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            continue
        row = rows.iloc[0]
        map_path = root / str(row["map_path"])
        heatmap = _load_heatmap(map_path)
        if heatmap.shape != image.shape[:2]:
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            heatmap = np.asarray(heatmap_img, dtype=np.float32) / 255.0
        ax.imshow(_overlay(image, heatmap, alpha=alpha))
        pred = str(row.get("predicted_class", ""))
        title = ARCHITECTURE_LABELS.get(architecture, architecture)
        ax.set_title(f"{title}\nPred: {pred}", fontsize=9)
        ax.axis("off")
        rendered.append(
            {
                "architecture": architecture,
                "map_path": str(map_path),
                "predicted_class": pred,
            }
        )
    true_class = str(sample_row["true_class"])
    fig.suptitle(f"{method_label} heatmaps | True class: {true_class}", y=1.02, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "xai_sample_id": str(sample_row["xai_sample_id"]),
        "true_class": true_class,
        "relative_path": str(sample_row["relative_path"]),
        "png": str(output_path.with_suffix(".png")),
        "pdf": str(output_path.with_suffix(".pdf")),
        "rendered_maps": rendered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.local.yaml"))
    parser.add_argument("--sample-manifest", type=Path, default=Path("results/xai/sample/xai_sample_manifest.csv"))
    parser.add_argument("--xai-root", type=Path, default=Path("results/gate11_xai"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regime", default="standard_ce")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--method", choices=["integrated_gradients", "specific"], default="integrated_gradients")
    parser.add_argument("--images-per-class", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = _resolve(project_root, args.config).resolve()
    sample_path = _resolve(project_root, args.sample_manifest).resolve()
    xai_root = _resolve(project_root, args.xai_root).resolve()
    output_dir = _resolve(project_root, args.output_dir).resolve()
    panels_dir = output_dir / "panels" / args.method
    dataset_root = _load_dataset_root(project_root, config_path)

    sample = pd.read_csv(sample_path)
    selected = _select_samples(sample, args.images_per_class)
    registries = {
        architecture: _registry_for(xai_root, architecture, args.regime, args.seed, args.method)
        for architecture in ARCHITECTURES
    }

    rows = []
    for _, sample_row in selected.iterrows():
        safe_class = str(sample_row["true_class"]).replace(" ", "_").replace("/", "_")
        stem = f"{safe_class}__order_{int(sample_row['selection_order']):02d}__{sample_row['xai_sample_id']}"
        rows.append(
            _build_panel(
                sample_row,
                dataset_root,
                registries,
                panels_dir / stem,
                args.image_size,
                args.alpha,
                "Integrated Gradients" if args.method == "integrated_gradients" else "Architecture-specific XAI",
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"xai_heatmap_panels_{args.method}.csv", index=False)
    metadata = {
        "schema_version": "1.0.0",
        "stage": "gate_12_xai_heatmap_rendering",
        "postprocessing_only": True,
        "method": args.method,
        "regime": args.regime,
        "seed": args.seed,
        "images_per_class": args.images_per_class,
        "panel_count": len(rows),
        "architectures": ARCHITECTURES,
        "dataset_root": str(dataset_root),
        "sample_manifest": str(sample_path),
        "xai_root": str(xai_root),
        "output_dir": str(output_dir),
    }
    (output_dir / f"xai_heatmap_manifest_{args.method}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
