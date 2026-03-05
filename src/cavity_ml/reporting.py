"""Experiment reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .io_utils import read_json, resolve_path, to_rel_path


def _format_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df[columns].copy()
    for col in display.select_dtypes(include=["float64", "float32"]).columns:
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    return display.to_markdown(index=False)


def generate_report(config: dict[str, Any]) -> dict[str, str]:
    paths = config["paths"]
    project_root = Path(config["_project_root"]).resolve()
    metrics_csv = resolve_path(paths["metrics_csv"], project_root)
    metrics_json = resolve_path(paths["metrics_json"], project_root)
    metadata_json = resolve_path(paths["best_metadata_json"], project_root)

    if not metrics_csv.exists() or not metrics_json.exists() or not metadata_json.exists():
        raise FileNotFoundError("Missing metrics or metadata. Run training before evaluate.")

    metrics_df = pd.read_csv(metrics_csv)
    metadata = read_json(metadata_json)

    radius_df = metrics_df[metrics_df["task"] == "radius"].sort_values("val_radius_accuracy", ascending=False)
    height_df = metrics_df[metrics_df["task"] == "height"].sort_values("val_height_r2", ascending=False)
    joint_df = metrics_df[metrics_df["task"] == "joint"].sort_values(
        ["val_radius_accuracy", "val_height_r2"], ascending=False
    )

    selected = metadata["selected_test_metrics"]
    acceptance = metadata["acceptance"]

    lines = [
        "# Cavity Optimisation Experiment Report",
        "",
        "## Summary",
        f"- Selected design: `{metadata['selected_design']}`",
        f"- Model version: `{metadata['model_version']}`",
        f"- Radius accuracy (test): **{selected['radius_accuracy']:.4f}**",
        f"- Radius ±1 class (test): **{selected['radius_within_1_class']:.4f}**",
        f"- Height R² (test): **{selected['height_r2']:.4f}**",
        f"- Height MAE (test): **{selected['height_mae']:.4f} mm**",
        "",
        "## Acceptance Gates",
        f"- Radius accuracy threshold: {acceptance['radius_accuracy_threshold']:.3f} -> {acceptance['radius_accuracy_pass']}",
        f"- Height R² threshold: {acceptance['height_r2_threshold']:.3f} -> {acceptance['height_r2_pass']}",
        f"- Overall: **{acceptance['all_pass']}**",
        "",
        "## Radius Models (Validation Ranking)",
    ]

    if not radius_df.empty:
        lines.append(_format_table(radius_df, [
            "model_name",
            "family",
            "train_seconds",
            "val_radius_accuracy",
            "val_radius_within_1_class",
            "test_radius_accuracy",
        ]))
    else:
        lines.append("No radius rows found.")

    lines.extend(["", "## Height Models (Validation Ranking)"])
    if not height_df.empty:
        lines.append(_format_table(height_df, [
            "model_name",
            "family",
            "train_seconds",
            "val_height_r2",
            "val_height_mae",
            "test_height_r2",
        ]))
    else:
        lines.append("No height rows found.")

    lines.extend(["", "## Joint Models (Validation Ranking)"])
    if not joint_df.empty:
        lines.append(_format_table(joint_df, [
            "model_name",
            "family",
            "train_seconds",
            "val_radius_accuracy",
            "val_height_r2",
            "test_radius_accuracy",
            "test_height_r2",
        ]))
    else:
        lines.append("No joint rows found.")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Metrics CSV: `{to_rel_path(metrics_csv, project_root)}`",
            f"- Metrics JSON: `{to_rel_path(metrics_json, project_root)}`",
            f"- Best model metadata: `{to_rel_path(metadata_json, project_root)}`",
        ]
    )

    report_text = "\n".join(lines) + "\n"

    report_path = resolve_path(paths["report_md"], project_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "report_md": to_rel_path(report_path, project_root),
    }


def evaluate_from_config(config_path: str | Path | None = None) -> dict[str, str]:
    config = load_config(config_path=config_path)
    return generate_report(config)
