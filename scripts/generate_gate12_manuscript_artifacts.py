"""Generate Gate 12 manuscript tables and figures from completed Gate 10/11 artefacts.

This script is intentionally post-processing only: it reads completed confirmatory,
profiling and XAI artefacts and writes publication-ready CSV/LaTeX/PNG/PDF files.
It does not train models, does not run inference, and does not modify checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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

CLASS_LABELS = {
    "Black Rot": "Black Rot",
    "ESCA": "ESCA",
    "Healthy": "Healthy",
    "Leaf Blight": "Leaf Blight",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    (output_dir / "gate12_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    latex = df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        column_format="l" * len(df.columns),
    )
    path.write_text(latex, encoding="utf-8")


def _prepare_predictive_table(aggregate_path: Path, output_dir: Path) -> pd.DataFrame:
    df = _read_csv(aggregate_path)
    test = df[df["evaluation_partition"] == "test"].copy()
    test["Architecture"] = test["model_name"].map(ARCHITECTURE_LABELS).fillna(test["model_name"])
    test["Training regime"] = test["regime"].map(REGIME_LABELS).fillna(test["regime"])
    table = pd.DataFrame(
        {
            "Architecture": test["Architecture"],
            "Training regime": test["Training regime"],
            "Seeds": test["seed_count"].astype(int),
            "Accuracy": test["accuracy_mean"].map(_format_float),
            "Macro-Precision": test["macro_precision_mean"].map(_format_float),
            "Macro-Recall": test["macro_recall_mean"].map(_format_float),
            "Macro-F1": test["macro_f1_mean"].map(_format_float),
        }
    ).sort_values(["Architecture", "Training regime"])
    table.to_csv(output_dir / "table_predictive_performance.csv", index=False)
    _save_latex_table(
        table,
        output_dir / "table_predictive_performance.tex",
        "Confirmatory predictive performance on the locked test set.",
        "tab:predictive-performance",
    )
    return test


def _prepare_xai_table(faithfulness_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(faithfulness_root.rglob("faithfulness_summary.csv")):
        df = _read_csv(summary_path)
        if df.empty:
            continue
        model = str(df.iloc[0]["model_name"])
        regime = str(df.iloc[0]["regime"])
        seed = int(df.iloc[0]["seed"])
        rows.append(
            {
                "Architecture": ARCHITECTURE_LABELS.get(model, model),
                "Training regime": REGIME_LABELS.get(regime, regime),
                "Seed": seed,
                "Images": int(len(df)),
                "Insertion AUC": _format_float(df["insertion_partial_auc"].mean()),
                "Deletion AUC": _format_float(df["deletion_partial_auc"].mean()),
                "Insertion AUC SD": _format_float(df["insertion_partial_auc"].std(ddof=1)),
                "Deletion AUC SD": _format_float(df["deletion_partial_auc"].std(ddof=1)),
            }
        )
    table = pd.DataFrame(rows).sort_values("Architecture")
    table.to_csv(output_dir / "table_xai_faithfulness.csv", index=False)
    _save_latex_table(
        table,
        output_dir / "table_xai_faithfulness.tex",
        "Perturbation-based XAI faithfulness on the frozen XAI sample.",
        "tab:xai-faithfulness",
    )
    return table


def _prepare_profiling_table(profiling_path: Path, output_dir: Path) -> pd.DataFrame:
    df = _read_csv(profiling_path)
    compact = df[df["precision"].isin(["fp32", "fp16"])].copy()
    compact["Architecture"] = compact["architecture"].map(ARCHITECTURE_LABELS).fillna(compact["architecture"])
    compact = compact.sort_values(["Architecture", "precision", "batch_size"])
    table = pd.DataFrame(
        {
            "Architecture": compact["Architecture"],
            "Precision": compact["precision"],
            "Batch size": compact["batch_size"].astype(int),
            "Params (M)": compact["parameter_count"].astype(float).div(1e6).map(lambda x: _format_float(x, 2)),
            "Checkpoint (MB)": compact["checkpoint_size_mb"].map(lambda x: _format_float(x, 2)),
            "Mean latency (ms)": compact["mean_latency_ms"].map(lambda x: _format_float(x, 2)),
        }
    )
    if "throughput_images_per_second" in compact.columns:
        table["Throughput (img/s)"] = compact["throughput_images_per_second"].map(
            lambda x: _format_float(x, 2)
        )
    table.to_csv(output_dir / "table_computational_feasibility.csv", index=False)
    _save_latex_table(
        table,
        output_dir / "table_computational_feasibility.tex",
        "Computational feasibility indicators for representative confirmatory checkpoints.",
        "tab:computational-feasibility",
    )
    return compact


def _plot_predictive_bars(test_df: pd.DataFrame, figures_dir: Path) -> None:
    pivot = test_df.pivot(index="model_name", columns="regime", values="macro_f1_mean")
    pivot = pivot.reindex(["yolov8n_cls", "resnet50", "swin_tiny", "vit_b_16"])
    pivot = pivot.rename(index=ARCHITECTURE_LABELS, columns=REGIME_LABELS)
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.78)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Confirmatory test Macro-F1 by architecture and training regime")
    ax.legend(title="Training regime", loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_predictive_macro_f1.png", dpi=300)
    plt.savefig(figures_dir / "figure_predictive_macro_f1.pdf")
    plt.close()


def _plot_tradeoff(test_df: pd.DataFrame, profiling_df: pd.DataFrame, figures_dir: Path) -> None:
    standard = test_df[test_df["regime"] == "standard_ce"].copy()
    fp32_b1 = profiling_df[(profiling_df["precision"] == "fp32") & (profiling_df["batch_size"] == 1)].copy()
    merged = standard.merge(fp32_b1, left_on="model_name", right_on="architecture", how="inner")
    if merged.empty:
        return
    sizes = np.sqrt(merged["checkpoint_size_mb"].astype(float)) * 60
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        merged["mean_latency_ms"].astype(float),
        merged["macro_f1_mean"].astype(float),
        s=sizes,
        alpha=0.72,
        edgecolor="black",
    )
    del scatter
    for _, row in merged.iterrows():
        ax.annotate(
            ARCHITECTURE_LABELS.get(row["model_name"], row["model_name"]),
            (float(row["mean_latency_ms"]), float(row["macro_f1_mean"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Mean latency, batch 1, fp32 (ms)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(max(0.0, merged["macro_f1_mean"].min() - 0.02), 1.005)
    ax.set_title("Performance--latency--size trade-off")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_tradeoff_macro_f1_latency_size.png", dpi=300)
    plt.savefig(figures_dir / "figure_tradeoff_macro_f1_latency_size.pdf")
    plt.close()


def _plot_xai_auc(xai_table: pd.DataFrame, figures_dir: Path) -> None:
    if xai_table.empty:
        return
    numeric = xai_table.copy()
    numeric["Insertion AUC"] = numeric["Insertion AUC"].astype(float)
    numeric["Deletion AUC"] = numeric["Deletion AUC"].astype(float)
    numeric = numeric.sort_values("Architecture")
    x = np.arange(len(numeric))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, numeric["Insertion AUC"], width, label="Insertion AUC")
    ax.bar(x + width / 2, numeric["Deletion AUC"], width, label="Deletion AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(numeric["Architecture"], rotation=20, ha="right")
    ax.set_ylabel("Partial AUC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Perturbation-based XAI faithfulness")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_xai_faithfulness_auc.png", dpi=300)
    plt.savefig(figures_dir / "figure_xai_faithfulness_auc.pdf")
    plt.close()


def _plot_best_confusion_matrix(confirmatory_root: Path, test_df: pd.DataFrame, figures_dir: Path) -> None:
    best = test_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
    model = str(best["model_name"])
    regime = str(best["regime"])
    seed_dirs = sorted((confirmatory_root / model / regime).glob("seed_*"))
    if not seed_dirs:
        return
    matrices: list[np.ndarray] = []
    labels: list[str] | None = None
    for seed_dir in seed_dirs:
        matrix_path = seed_dir / "confusion_matrix_test.csv"
        if not matrix_path.is_file():
            continue
        df = pd.read_csv(matrix_path)
        if labels is None:
            label_col = "true_label" if "true_label" in df.columns else df.columns[0]
            labels = [str(v) for v in df[label_col].tolist()]
        value_cols = [c for c in df.columns if c != ("true_label" if "true_label" in df.columns else df.columns[0])]
        matrices.append(df[value_cols].to_numpy(dtype=float))
    if not matrices or labels is None:
        return
    mean_matrix = np.mean(matrices, axis=0)
    row_sums = mean_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(mean_matrix, row_sums, out=np.zeros_like(mean_matrix), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([CLASS_LABELS.get(v, v) for v in labels], rotation=35, ha="right")
    ax.set_yticklabels([CLASS_LABELS.get(v, v) for v in labels])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Normalized confusion matrix: {ARCHITECTURE_LABELS.get(model, model)} / {REGIME_LABELS.get(regime, regime)}")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_best_model_confusion_matrix.png", dpi=300)
    plt.savefig(figures_dir / "figure_best_model_confusion_matrix.pdf")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predictive-aggregate",
        type=Path,
        default=Path("results/gate11_publication_confirmatory_only/aggregate_results.csv"),
    )
    parser.add_argument(
        "--profiling-summary",
        type=Path,
        default=Path("results/gate11_profiling/architecture_profiling_summary.csv"),
    )
    parser.add_argument(
        "--faithfulness-root",
        type=Path,
        default=Path("results/gate11_xai/faithfulness"),
    )
    parser.add_argument(
        "--confirmatory-root",
        type=Path,
        default=Path("results/confirmatory"),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def resolve(path: Path) -> Path:
        return (project_root / path).resolve() if not path.is_absolute() else path.resolve()

    predictive_path = resolve(args.predictive_aggregate)
    profiling_path = resolve(args.profiling_summary)
    faithfulness_root = resolve(args.faithfulness_root)
    confirmatory_root = resolve(args.confirmatory_root)

    test_df = _prepare_predictive_table(predictive_path, tables_dir)
    xai_table = _prepare_xai_table(faithfulness_root, tables_dir)
    profiling_df = _prepare_profiling_table(profiling_path, tables_dir)

    _plot_predictive_bars(test_df, figures_dir)
    _plot_xai_auc(xai_table, figures_dir)
    _plot_tradeoff(test_df, profiling_df, figures_dir)
    _plot_best_confusion_matrix(confirmatory_root, test_df, figures_dir)

    metadata = {
        "schema_version": "1.0.0",
        "stage": "gate_12_manuscript_artifacts",
        "postprocessing_only": True,
        "predictive_aggregate": str(predictive_path),
        "profiling_summary": str(profiling_path),
        "faithfulness_root": str(faithfulness_root),
        "confirmatory_root": str(confirmatory_root),
        "output_dir": str(output_dir),
        "tables": sorted(p.name for p in tables_dir.glob("*")),
        "figures": sorted(p.name for p in figures_dir.glob("*")),
    }
    _write_metadata(output_dir, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
