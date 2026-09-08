"""Generate Gate 12 multicriteria radar/spider chart for manuscript synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARCHITECTURE_LABELS = {
    "yolov8n_cls": "YOLOv8n-cls",
    "resnet50": "ResNet50",
    "swin_tiny": "Swin-Tiny",
    "vit_b_16": "ViT-B/16",
}
ORDER = ["yolov8n_cls", "resnet50", "swin_tiny", "vit_b_16"]
COLORS = {
    "yolov8n_cls": "#1f77b4",
    "resnet50": "#2ca02c",
    "swin_tiny": "#ff7f0e",
    "vit_b_16": "#d62728",
}
AXES = [
    "Predictive\nMacro-F1",
    "XAI\nInsertion",
    "XAI\n1-Deletion",
    "Inference\nSpeed",
    "Compactness\nParams",
]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _minmax_higher_is_better(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return pd.Series(np.ones(len(values)), index=series.index)
    return (values - lo) / (hi - lo)


def _minmax_lower_is_better(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        return pd.Series(np.ones(len(values)), index=series.index)
    return (hi - values) / (hi - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictive-table", type=Path, default=Path("results/gate12_manuscript_artifacts/tables/table_predictive_performance.csv"))
    parser.add_argument("--xai-table", type=Path, default=Path("results/gate12_manuscript_artifacts/tables/table_xai_faithfulness.csv"))
    parser.add_argument("--profiling-table", type=Path, default=Path("results/gate12_manuscript_artifacts/tables/table_computational_feasibility.csv"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = _resolve(root, args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    predictive = pd.read_csv(_resolve(root, args.predictive_table))
    xai = pd.read_csv(_resolve(root, args.xai_table))
    profiling = pd.read_csv(_resolve(root, args.profiling_table))

    # Use Standard CE because Gate 11 XAI/profiling were run on representative Standard CE checkpoints.
    pred_standard = predictive[predictive["Training regime"] == "Standard CE"].copy()
    pred_standard["architecture"] = pred_standard["Architecture"].map({v: k for k, v in ARCHITECTURE_LABELS.items()})
    xai["architecture"] = xai["Architecture"].map({v: k for k, v in ARCHITECTURE_LABELS.items()})
    prof = profiling[(profiling["Precision"] == "fp32") & (profiling["Batch size"].astype(int) == 1)].copy()
    prof["architecture"] = prof["Architecture"].map({v: k for k, v in ARCHITECTURE_LABELS.items()})

    merged = (
        pred_standard[["architecture", "Macro-F1"]]
        .merge(xai[["architecture", "Insertion AUC", "Deletion AUC"]], on="architecture")
        .merge(prof[["architecture", "Params (M)", "Mean latency (ms)"]], on="architecture")
    )
    for col in ["Macro-F1", "Insertion AUC", "Deletion AUC", "Params (M)", "Mean latency (ms)"]:
        merged[col] = merged[col].astype(float)

    merged["Predictive Macro-F1"] = _minmax_higher_is_better(merged["Macro-F1"])
    merged["XAI Insertion"] = _minmax_higher_is_better(merged["Insertion AUC"])
    merged["XAI 1-Deletion"] = _minmax_higher_is_better(1.0 - merged["Deletion AUC"])
    merged["Inference Speed"] = _minmax_lower_is_better(merged["Mean latency (ms)"])
    merged["Compactness Params"] = _minmax_lower_is_better(merged["Params (M)"])
    merged["multicriteria_mean"] = merged[
        ["Predictive Macro-F1", "XAI Insertion", "XAI 1-Deletion", "Inference Speed", "Compactness Params"]
    ].mean(axis=1)
    merged["Architecture"] = merged["architecture"].map(ARCHITECTURE_LABELS)
    merged = merged.set_index("architecture").loc[ORDER].reset_index()

    output_table = merged[[
        "Architecture", "Macro-F1", "Insertion AUC", "Deletion AUC", "Mean latency (ms)", "Params (M)",
        "Predictive Macro-F1", "XAI Insertion", "XAI 1-Deletion", "Inference Speed", "Compactness Params", "multicriteria_mean"
    ]]
    output_table.to_csv(tables_dir / "table_multicriteria_radar_values.csv", index=False)

    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for _, row in merged.iterrows():
        values = [
            row["Predictive Macro-F1"], row["XAI Insertion"], row["XAI 1-Deletion"],
            row["Inference Speed"], row["Compactness Params"],
        ]
        values += values[:1]
        arch = row["architecture"]
        ax.plot(angles, values, linewidth=2, label=ARCHITECTURE_LABELS[arch], color=COLORS[arch])
        ax.fill(angles, values, alpha=0.10, color=COLORS[arch])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Multicriteria radar analysis (higher is better)", pad=24, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.12), frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_multicriteria_radar.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "figure_multicriteria_radar.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema_version": "1.0.0",
        "stage": "gate_12_multicriteria_radar",
        "postprocessing_only": True,
        "normalisation": "min-max within compared Standard CE architectures; all axes transformed so higher is better",
        "axes": AXES,
        "output_dir": str(output_dir),
        "figures": ["figure_multicriteria_radar.png", "figure_multicriteria_radar.pdf"],
        "tables": ["table_multicriteria_radar_values.csv"],
    }
    (output_dir / "multicriteria_radar_manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
