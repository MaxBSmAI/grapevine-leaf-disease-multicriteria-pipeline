"""Render Gate 12 confusion matrices from confirmatory outputs.

Modes:
- normalised_mean: average the five seed matrices and row-normalise values.
- integer_sum: sum raw counts across the five seeds and display integer cells.
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

ARCHITECTURES = ["yolov8n_cls", "resnet50", "swin_tiny", "vit_b_16"]
REGIMES = ["standard_ce", "balanced_ce", "focal_loss"]
ARCHITECTURE_LABELS = {
    "yolov8n_cls": "YOLOv8n-cls",
    "resnet50": "ResNet50",
    "swin_tiny": "Swin-Tiny",
    "vit_b_16": "ViT-B/16",
}
REGIME_LABELS = {
    "standard_ce": "Standard CE",
    "balanced_ce": "Balanced CE",
    "focal_loss": "Focal Loss",
}


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    label_col = "true_label" if "true_label" in df.columns else df.columns[0]
    labels = [str(v) for v in df[label_col].tolist()]
    value_cols = [c for c in df.columns if c != label_col]
    return labels, df[value_cols].to_numpy(dtype=float)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def _plot_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_stem: Path,
    mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.3))
    vmax = 1.0 if mode == "normalised_mean" else float(np.nanmax(matrix))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = 0.55 if mode == "normalised_mean" else vmax * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            text = f"{value:.2f}" if mode == "normalised_mean" else f"{int(value):d}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirmatory-root", type=Path, default=Path("results/confirmatory"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=["test", "validation"], default="test")
    parser.add_argument(
        "--mode",
        choices=["normalised_mean", "integer_sum"],
        default="normalised_mean",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    confirmatory_root = _resolve(project_root, args.confirmatory_root).resolve()
    output_dir = _resolve(project_root, args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    rows: list[dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        for regime in REGIMES:
            seed_dirs = sorted((confirmatory_root / architecture / regime).glob("seed_*"))
            matrices: list[np.ndarray] = []
            labels: list[str] | None = None
            seed_values: list[int] = []
            for seed_dir in seed_dirs:
                matrix_path = seed_dir / f"confusion_matrix_{args.partition}.csv"
                if not matrix_path.is_file():
                    continue
                current_labels, matrix = _read_matrix(matrix_path)
                if labels is None:
                    labels = current_labels
                matrices.append(matrix)
                seed_values.append(int(seed_dir.name.replace("seed_", "")))
            if not matrices or labels is None:
                continue

            sum_counts = np.sum(matrices, axis=0).astype(int)
            mean_counts = np.mean(matrices, axis=0)
            normalised = _normalise_rows(mean_counts)
            if args.mode == "integer_sum":
                display_matrix = sum_counts
                title_suffix = "summed integer counts across seeds"
                figure_suffix = "integer_sum"
            else:
                display_matrix = normalised
                title_suffix = "row-normalised mean over seeds"
                figure_suffix = "normalised_mean"

            stem = figures_dir / f"confusion_matrix_{args.partition}_{figure_suffix}__{architecture}__{regime}"
            title = f"{ARCHITECTURE_LABELS[architecture]} / {REGIME_LABELS[regime]} ({args.partition}, {title_suffix})"
            _plot_matrix(display_matrix, labels, title, stem, args.mode)

            for i, true_label in enumerate(labels):
                for j, predicted_label in enumerate(labels):
                    rows.append(
                        {
                            "architecture": architecture,
                            "architecture_label": ARCHITECTURE_LABELS[architecture],
                            "regime": regime,
                            "regime_label": REGIME_LABELS[regime],
                            "partition": args.partition,
                            "mode": args.mode,
                            "seed_count": len(matrices),
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "sum_count": int(sum_counts[i, j]),
                            "mean_count": float(mean_counts[i, j]),
                            "row_normalised_value": float(normalised[i, j]),
                            "seeds": ";".join(str(s) for s in seed_values),
                        }
                    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    table_name = f"confusion_matrices_{args.partition}_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(tables_dir / table_name, index=False)
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "gate_12_confusion_matrices",
        "postprocessing_only": True,
        "partition": args.partition,
        "mode": args.mode,
        "confirmatory_root": str(confirmatory_root),
        "output_dir": str(output_dir),
        "matrix_count": len(list(figures_dir.glob("*.png"))),
        "figures": sorted(p.name for p in figures_dir.glob("*")),
        "tables": sorted(p.name for p in tables_dir.glob("*")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"confusion_matrices_{args.partition}_{args.mode}_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
