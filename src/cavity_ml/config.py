"""Configuration loading for experiments and serving."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "random_state": 44,
    "paths": {
        "raw_frequency_csv": "data/raw/run_3_frequencies.csv",
        "raw_eh_csv": "data/raw/run_3_e_h_fields.csv",
        "xlsx_path": "data/alternate/Mode1.xlsx",
        "processed_csv": "data/processed/cavity_dataset.csv",
        "splits_json": "artifacts/splits/split_indices.json",
        "metrics_json": "artifacts/metrics/metrics.json",
        "metrics_csv": "artifacts/metrics/metrics.csv",
        "report_md": "reports/experiment_report.md",
        "best_metadata_json": "artifacts/models/best_model_metadata.json",
    },
    "split": {
        "test_size": 0.2,
        "val_size": 0.1,
        "stratify_radius": True,
    },
    "acceptance": {
        "legacy_radius_accuracy": 0.863,
        "legacy_height_r2": 0.857,
    },
    "models": {
        "random_forest_n_estimators": 300,
        "extra_trees_n_estimators": 500,
        "multioutput_extra_trees_n_estimators": 700,
        "knn_neighbors": 8,
        "torch": {
            "batch_size": 256,
            "radius_epochs": 350,
            "height_epochs": 350,
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "radius_hidden": [512, 256, 128],
            "height_hidden": [512, 256, 128],
        },
    },
    "selection": {
        "runtime_penalty": 0.002,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    path_config = resolved.get("paths", {})
    for key, path_value in path_config.items():
        path_config[key] = str(Path(path_value).as_posix())
    resolved["_project_root"] = str(root.resolve())
    return resolved


def load_config(config_path: str | Path | None = None, project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = (root / override_path).resolve()
        override = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        cfg = _deep_merge(cfg, override)
    return _resolve_paths(cfg, root)
