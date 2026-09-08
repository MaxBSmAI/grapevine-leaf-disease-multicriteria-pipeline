"""Render Gate 12 training-curve figures from confirmatory training histories.

This is a post-processing script. It reads completed training_history.csv files and
writes manuscript-oriented learning-curve figures without training or inference.
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
REGIME_COLORS = {
    "standard_ce": "#1f77b4",
    "balanced_ce": "#2ca02c",
    "focal_loss": "#d62728",
}


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _load_histories(confirmatory_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for history_path in sorted(confirmatory_root.rglob("training_history.csv")):
        rel = history_path.relative_to(confirmatory_root).parts
        if len(rel) < 4:
            continue
        architecture, regime, seed_part = rel[0], rel[1], rel[2]
        if not seed_part.startswith("seed_"):
            continue
        seed = int(seed_part.replace("seed_", ""))
        df = pd.read_csv(history_path)
        df["architecture"] = architecture
        df["regime"] = regime
        df["seed"] = seed
        df["history_path"] = str(history_path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No training_history.csv files found under {confirmatory_root}")
    return pd.concat(rows, ignore_index=True)


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["architecture", "regime", "epoch"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def _plot_metric_by_architecture(df: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    agg = _aggregate(df, metric)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes_flat = axes.ravel()
    for ax, architecture in zip(axes_flat, ARCHITECTURES, strict=True):
        for regime in REGIMES:
            sub = agg[(agg["architecture"] == architecture) & (agg["regime"] == regime)].sort_values("epoch")
            if sub.empty:
                continue
            x = sub["epoch"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            sd = sub["std"].to_numpy(dtype=float)
            color = REGIME_COLORS[regime]
            ax.plot(x, y, label=REGIME_LABELS[regime], color=color, linewidth=2)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0)
        ax.set_title(ARCHITECTURE_LABELS[architecture])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(ylabel + " during confirmatory training", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output.with_suffix(".png"), dpi=300)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_best_regime_per_architecture(df: pd.DataFrame, output: Path) -> None:
    final = (
        df.sort_values("epoch")
        .groupby(["architecture", "regime", "seed"], as_index=False)
        .tail(1)
    )
    regime_scores = (
        final.groupby(["architecture", "regime"], as_index=False)["validation_macro_f1"]
        .mean()
        .sort_values("validation_macro_f1", ascending=False)
    )
    best_rows = regime_scores.groupby("architecture", as_index=False).head(1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in best_rows.iterrows():
        architecture = row["architecture"]
        regime = row["regime"]
        sub = df[(df["architecture"] == architecture) & (df["regime"] == regime)]
        agg = _aggregate(sub, "validation_macro_f1").sort_values("epoch")
        x = agg["epoch"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        sd = agg["std"].to_numpy(dtype=float)
        label = f"{ARCHITECTURE_LABELS.get(architecture, architecture)} / {REGIME_LABELS.get(regime, regime)}"
        ax.plot(x, y, linewidth=2, label=label)
        ax.fill_between(x, y - sd, y + sd, alpha=0.12, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Best validation trajectory per architecture")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output.with_suffix(".png"), dpi=300)
    plt.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _write_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    final = (
        df.sort_values("epoch")
        .groupby(["architecture", "regime", "seed"], as_index=False)
        .tail(1)
    )
    summary = (
        final.groupby(["architecture", "regime"], as_index=False)
        .agg(
            seed_count=("seed", "count"),
            final_epoch_mean=("epoch", "mean"),
            validation_macro_f1_final_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_final_std=("validation_macro_f1", "std"),
            validation_loss_final_mean=("validation_loss", "mean"),
            train_loss_final_mean=("train_loss", "mean"),
            elapsed_seconds_total_mean=("elapsed_seconds", "sum"),
        )
    )
    summary["architecture_label"] = summary["architecture"].map(ARCHITECTURE_LABELS)
    summary["regime_label"] = summary["regime"].map(REGIME_LABELS)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "training_curve_final_epoch_summary.csv", index=False)

    per_epoch = df.groupby(["architecture", "regime", "epoch"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        validation_loss_std=("validation_loss", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
    )
    per_epoch.to_csv(output_dir / "training_curves_per_epoch_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirmatory-root", type=Path, default=Path("results/confirmatory"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    confirmatory_root = _resolve(project_root, args.confirmatory_root).resolve()
    output_dir = _resolve(project_root, args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    histories = _load_histories(confirmatory_root)
    _write_summary_tables(histories, tables_dir)
    _plot_metric_by_architecture(
        histories,
        "validation_macro_f1",
        "Validation Macro-F1",
        figures_dir / "figure_training_validation_macro_f1",
    )
    _plot_metric_by_architecture(
        histories,
        "validation_loss",
        "Validation loss",
        figures_dir / "figure_training_validation_loss",
    )
    _plot_metric_by_architecture(
        histories,
        "train_loss",
        "Training loss",
        figures_dir / "figure_training_train_loss",
    )
    _plot_best_regime_per_architecture(histories, figures_dir / "figure_training_best_trajectories")

    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "gate_12_training_curves",
        "postprocessing_only": True,
        "confirmatory_root": str(confirmatory_root),
        "output_dir": str(output_dir),
        "history_files": int(histories["history_path"].nunique()),
        "architectures": sorted(histories["architecture"].unique().tolist()),
        "regimes": sorted(histories["regime"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in histories["seed"].unique().tolist()),
        "figures": sorted(p.name for p in figures_dir.glob("*")),
        "tables": sorted(p.name for p in tables_dir.glob("*")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_curves_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
