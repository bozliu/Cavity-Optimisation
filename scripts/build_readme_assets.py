#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate README-ready benchmark tables and figure assets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_ml.config import load_config
from cavity_ml.io_utils import read_json, resolve_path
from cavity_ml.labels import class_to_radius, radius_to_class


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _load_arrays(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(cfg["_project_root"]).resolve()
    processed = resolve_path(cfg["paths"]["processed_csv"], root)
    splits = resolve_path(cfg["paths"]["splits_json"], root)

    df = pd.read_csv(processed)
    split_payload = read_json(splits)
    idx_test = np.asarray(split_payload["indices"]["test"], dtype=int)

    x_test = df[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]].to_numpy(dtype=float)[idx_test]
    y_r = df["cR"].to_numpy(dtype=float)[idx_test]
    y_h = df["cH"].to_numpy(dtype=float)[idx_test]
    return x_test, y_r, y_h


def _predict_selected(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    root = Path(cfg["_project_root"]).resolve()
    metadata_path = resolve_path(cfg["paths"]["best_metadata_json"], root)
    meta = read_json(metadata_path)
    x_test, y_r, y_h = _load_arrays(cfg)

    selected = meta["selected_models"]["artifact_paths"]
    if meta["selected_design"] == "multioutput":
        model = joblib.load(resolve_path(selected["joint_model"], root))
        pred = np.asarray(model.predict(x_test)).reshape(-1, 2)
        pred_r = class_to_radius(radius_to_class(pred[:, 0]))
        pred_h = pred[:, 1]
    else:
        model_r = joblib.load(resolve_path(selected["radius_model"], root))
        model_h = joblib.load(resolve_path(selected["height_model"], root))
        pred_r = class_to_radius(np.asarray(model_r.predict(x_test)).astype(int))
        pred_h = np.asarray(model_h.predict(x_test)).reshape(-1)

    return np.asarray(pred_r), np.asarray(pred_h)


def build_tables(cfg: dict) -> None:
    root = Path(cfg["_project_root"]).resolve()
    metrics = pd.read_csv(resolve_path(cfg["paths"]["metrics_csv"], root))

    out_csv = root / "results/benchmark/benchmark_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "method": str(row["model_name"]),
                "design": str(row["design"]),
                "task": str(row["task"]),
                "radius_accuracy": _fmt(None if pd.isna(row.get("test_radius_accuracy")) else float(row["test_radius_accuracy"])),
                "radius_within_1": _fmt(
                    None if pd.isna(row.get("test_radius_within_1_class")) else float(row["test_radius_within_1_class"])
                ),
                "height_r2": _fmt(None if pd.isna(row.get("test_height_r2")) else float(row["test_height_r2"])),
                "height_mae_mm": _fmt(None if pd.isna(row.get("test_height_mae")) else float(row["test_height_mae"])),
                "train_seconds": _fmt(float(row["train_seconds"])),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "design",
                "task",
                "radius_accuracy",
                "radius_within_1",
                "height_r2",
                "height_mae_mm",
                "train_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sota_csv = root / "results/benchmark/sota_comparison.csv"
    sota_rows = [
        {
            "citation_id": "[1]",
            "paper": "Wang and Ou, Applied Sciences, 2022",
            "task_summary": "DDPG-based cavity tuning in simulation and real setup",
            "reported_outcome": "Most test episodes tuned in <= 20 steps (paper claim)",
            "radius_accuracy": "N/A",
            "height_r2": "N/A",
            "url": "https://doi.org/10.3390/app122010498",
            "comparability_note": "Different objective/metric from geometry prediction",
        },
        {
            "citation_id": "[2]",
            "paper": "Nimara et al., CoRL/PMLR, 2023",
            "task_summary": "Model-based RL for cavity filter tuning",
            "reported_outcome": "4x and 10x lower sample complexity vs DDPG and DIRT",
            "radius_accuracy": "N/A",
            "height_r2": "N/A",
            "url": "https://proceedings.mlr.press/v229/nimara23a.html",
            "comparability_note": "RL tuning objective, not direct cR/cH regression",
        },
        {
            "citation_id": "[3]",
            "paper": "Wang et al., IEEE CBS, 2018",
            "task_summary": "Continuous RL with knowledge-inspired reward shaping",
            "reported_outcome": "Demonstrated autonomous cavity tuning (paper-level)",
            "radius_accuracy": "N/A",
            "height_r2": "N/A",
            "url": "https://doi.org/10.1109/CBS.2018.8612197",
            "comparability_note": "Different target and evaluation protocol",
        },
        {
            "citation_id": "[4]",
            "paper": "Wang et al., IEEE ROBIO, 2015",
            "task_summary": "RL approach to learning human tuning experience",
            "reported_outcome": "Early RL cavity tuning framework",
            "radius_accuracy": "N/A",
            "height_r2": "N/A",
            "url": "https://doi.org/10.1109/ROBIO.2015.7419091",
            "comparability_note": "Different target and evaluation protocol",
        },
        {
            "citation_id": "[5]",
            "paper": "Lindstahl and Lan, IEEE AIM, 2020",
            "task_summary": "Reinforcement learning with imitation for cavity filter tuning",
            "reported_outcome": "Imitation-enhanced RL for tuning policy learning",
            "radius_accuracy": "N/A",
            "height_r2": "N/A",
            "url": "https://doi.org/10.1109/AIM43001.2020.9158839",
            "comparability_note": "Policy-learning task, metrics not directly mappable",
        },
    ]
    with sota_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "citation_id",
                "paper",
                "task_summary",
                "reported_outcome",
                "radius_accuracy",
                "height_r2",
                "url",
                "comparability_note",
            ],
        )
        writer.writeheader()
        writer.writerows(sota_rows)


def build_figures(cfg: dict) -> None:
    root = Path(cfg["_project_root"]).resolve()
    fig_dir = root / "results/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(resolve_path(cfg["paths"]["metrics_csv"], root))
    radius_rows = metrics[metrics["task"] == "radius"].copy()
    height_rows = metrics[metrics["task"] == "height"].copy()

    plt.figure(figsize=(9, 4.5))
    plt.subplot(1, 2, 1)
    plt.bar(radius_rows["model_name"], radius_rows["test_radius_accuracy"], color="#2a9d8f")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test Radius Accuracy")
    plt.title("Radius Classification")

    plt.subplot(1, 2, 2)
    plt.bar(height_rows["model_name"], height_rows["test_height_r2"], color="#e76f51")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Test Height R2")
    plt.title("Height Regression")

    plt.tight_layout()
    plt.savefig(fig_dir / "model_compare.png", dpi=180)
    plt.close()

    x_test, y_r, y_h = _load_arrays(cfg)
    pred_r, pred_h = _predict_selected(cfg)

    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(y_h, pred_h, s=10, alpha=0.7, color="#264653")
    lo = min(y_h.min(), pred_h.min())
    hi = max(y_h.max(), pred_h.max())
    plt.plot([lo, hi], [lo, hi], "--", color="#e76f51", linewidth=1.4)
    plt.xlabel("True Height (mm)")
    plt.ylabel("Predicted Height (mm)")
    plt.title("Predicted vs True Height")
    plt.tight_layout()
    plt.savefig(fig_dir / "pred_vs_true.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.2, 4.2))
    plt.hist(pred_h - y_h, bins=28, color="#457b9d", alpha=0.85)
    plt.xlabel("Prediction Error (mm)")
    plt.ylabel("Count")
    plt.title("Height Error Distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "error_hist.png", dpi=180)
    plt.close()

    cm = confusion_matrix(radius_to_class(y_r), radius_to_class(pred_r))
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ConfusionMatrixDisplay(cm).plot(ax=ax, include_values=False, cmap="Blues", colorbar=False)
    ax.set_title("Radius Class Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_radius.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, project_root=ROOT)
    build_tables(cfg)
    build_figures(cfg)


if __name__ == "__main__":
    main()
