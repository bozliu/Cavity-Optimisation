"""Training, evaluation, selection, and artifact persistence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .constants import DEFAULT_MODEL_VERSION, FEATURE_COLUMNS
from .data_pipeline import load_processed_dataset, prepare_data
from .io_utils import ensure_parent, read_json, resolve_path, to_rel_path, write_json
from .labels import class_to_radius, radius_to_class
from .models import (
    Candidate,
    build_height_candidates,
    build_multioutput_candidate,
    build_radius_candidates,
    evaluate_height_metrics,
    evaluate_radius_metrics,
)


def _fit_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Any:
    try:
        model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
    except (TypeError, ValueError):
        model.fit(x_train, y_train)
    return model


def _select_best_radius(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: (r["val_radius_accuracy"], r["val_radius_within_1_class"], -r["train_seconds"]))


def _select_best_height(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: (r["val_height_r2"], -r["val_height_mae"], -r["train_seconds"]))


def _safe_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _evaluate_two_task(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_radius_train: np.ndarray,
    y_radius_val: np.ndarray,
    y_radius_test: np.ndarray,
    y_height_train: np.ndarray,
    y_height_val: np.ndarray,
    y_height_test: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    num_classes = int(np.unique(np.concatenate([y_radius_train, y_radius_val, y_radius_test])).size)

    radius_results: list[dict[str, Any]] = []
    radius_builders: dict[str, Candidate] = {}
    for candidate in build_radius_candidates(config, num_classes=num_classes):
        model = candidate.builder()
        start = time.perf_counter()
        _fit_model(model, x_train, y_radius_train, x_val=x_val, y_val=y_radius_val)
        seconds = time.perf_counter() - start

        pred_val = np.asarray(model.predict(x_val)).astype(int)
        pred_test = np.asarray(model.predict(x_test)).astype(int)

        val_metrics = evaluate_radius_metrics(y_radius_val, pred_val)
        test_metrics = evaluate_radius_metrics(y_radius_test, pred_test)

        row = {
            "design": "two_task",
            "task": "radius",
            "model_name": candidate.name,
            "family": candidate.family,
            "train_seconds": float(seconds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        radius_results.append(row)
        radius_builders[candidate.name] = candidate

    height_results: list[dict[str, Any]] = []
    height_builders: dict[str, Candidate] = {}
    for candidate in build_height_candidates(config):
        model = candidate.builder()
        start = time.perf_counter()
        _fit_model(model, x_train, y_height_train, x_val=x_val, y_val=y_height_val)
        seconds = time.perf_counter() - start

        pred_val = np.asarray(model.predict(x_val)).astype(float)
        pred_test = np.asarray(model.predict(x_test)).astype(float)

        val_metrics = evaluate_height_metrics(y_height_val, pred_val)
        test_metrics = evaluate_height_metrics(y_height_test, pred_test)

        row = {
            "design": "two_task",
            "task": "height",
            "model_name": candidate.name,
            "family": candidate.family,
            "train_seconds": float(seconds),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        height_results.append(row)
        height_builders[candidate.name] = candidate

    best_radius = _select_best_radius(radius_results)
    best_height = _select_best_height(height_results)

    train_full_x = np.vstack([x_train, x_val])
    train_full_radius = np.concatenate([y_radius_train, y_radius_val])
    train_full_height = np.concatenate([y_height_train, y_height_val])

    best_radius_model = radius_builders[best_radius["model_name"]].builder()
    _fit_model(best_radius_model, train_full_x, train_full_radius, x_val=x_val, y_val=y_radius_val)

    best_height_model = height_builders[best_height["model_name"]].builder()
    _fit_model(best_height_model, train_full_x, train_full_height, x_val=x_val, y_val=y_height_val)

    final_radius_pred = np.asarray(best_radius_model.predict(x_test)).astype(int)
    final_height_pred = np.asarray(best_height_model.predict(x_test)).astype(float)

    final_radius_metrics = evaluate_radius_metrics(y_radius_test, final_radius_pred)
    final_height_metrics = evaluate_height_metrics(y_height_test, final_height_pred)

    penalty = float(config["selection"]["runtime_penalty"])
    selection_score = (
        _safe_float(best_radius["val_radius_accuracy"])
        + _safe_float(best_height["val_height_r2"])
        - penalty * (_safe_float(best_radius["train_seconds"]) + _safe_float(best_height["train_seconds"]))
    )

    design_summary = {
        "design": "two_task",
        "selected_radius_model": best_radius["model_name"],
        "selected_height_model": best_height["model_name"],
        "selection_score": float(selection_score),
        "test_radius_accuracy": final_radius_metrics["radius_accuracy"],
        "test_radius_within_1_class": final_radius_metrics["radius_within_1_class"],
        "test_height_r2": final_height_metrics["height_r2"],
        "test_height_mae": final_height_metrics["height_mae"],
        "selected_models": {
            "radius": best_radius_model,
            "height": best_height_model,
        },
    }

    return radius_results, height_results, design_summary


def _evaluate_multioutput(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_joint_train: np.ndarray,
    y_joint_val: np.ndarray,
    y_joint_test: np.ndarray,
    y_radius_val: np.ndarray,
    y_radius_test: np.ndarray,
    y_height_val: np.ndarray,
    y_height_test: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = build_multioutput_candidate(config)
    model = candidate.builder()

    start = time.perf_counter()
    model.fit(x_train, y_joint_train)
    seconds = time.perf_counter() - start

    pred_val = np.asarray(model.predict(x_val)).astype(float)
    pred_test = np.asarray(model.predict(x_test)).astype(float)

    pred_val_radius_class = radius_to_class(pred_val[:, 0])
    pred_test_radius_class = radius_to_class(pred_test[:, 0])

    val_radius_metrics = evaluate_radius_metrics(y_radius_val, pred_val_radius_class)
    test_radius_metrics = evaluate_radius_metrics(y_radius_test, pred_test_radius_class)

    val_height_metrics = evaluate_height_metrics(y_height_val, pred_val[:, 1])
    test_height_metrics = evaluate_height_metrics(y_height_test, pred_test[:, 1])

    row = {
        "design": "multioutput",
        "task": "joint",
        "model_name": candidate.name,
        "family": candidate.family,
        "train_seconds": float(seconds),
        "val_radius_accuracy": val_radius_metrics["radius_accuracy"],
        "val_radius_within_1_class": val_radius_metrics["radius_within_1_class"],
        "test_radius_accuracy": test_radius_metrics["radius_accuracy"],
        "test_radius_within_1_class": test_radius_metrics["radius_within_1_class"],
        "val_height_r2": val_height_metrics["height_r2"],
        "val_height_mae": val_height_metrics["height_mae"],
        "test_height_r2": test_height_metrics["height_r2"],
        "test_height_mae": test_height_metrics["height_mae"],
    }

    train_full_x = np.vstack([x_train, x_val])
    train_full_joint = np.vstack([y_joint_train, y_joint_val])
    final_model = candidate.builder()
    final_model.fit(train_full_x, train_full_joint)

    final_pred_test = np.asarray(final_model.predict(x_test)).astype(float)
    final_pred_radius_class = radius_to_class(final_pred_test[:, 0])
    final_radius_metrics = evaluate_radius_metrics(y_radius_test, final_pred_radius_class)
    final_height_metrics = evaluate_height_metrics(y_height_test, final_pred_test[:, 1])

    penalty = float(config["selection"]["runtime_penalty"])
    selection_score = row["val_radius_accuracy"] + row["val_height_r2"] - penalty * row["train_seconds"]

    summary = {
        "design": "multioutput",
        "selected_joint_model": candidate.name,
        "selection_score": float(selection_score),
        "test_radius_accuracy": final_radius_metrics["radius_accuracy"],
        "test_radius_within_1_class": final_radius_metrics["radius_within_1_class"],
        "test_height_r2": final_height_metrics["height_r2"],
        "test_height_mae": final_height_metrics["height_mae"],
        "selected_model": final_model,
    }

    return [row], summary


def _load_split_arrays(dataset: pd.DataFrame, split_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    idx_train = np.asarray(split_payload["indices"]["train"], dtype=int)
    idx_val = np.asarray(split_payload["indices"]["val"], dtype=int)
    idx_test = np.asarray(split_payload["indices"]["test"], dtype=int)

    x = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_radius_mm = dataset["cR"].to_numpy(dtype=float)
    y_height = dataset["cH"].to_numpy(dtype=float)
    y_radius_class = radius_to_class(y_radius_mm)
    y_joint = dataset[["cR", "cH"]].to_numpy(dtype=float)

    return {
        "x_train": x[idx_train],
        "x_val": x[idx_val],
        "x_test": x[idx_test],
        "y_radius_train": y_radius_class[idx_train],
        "y_radius_val": y_radius_class[idx_val],
        "y_radius_test": y_radius_class[idx_test],
        "y_height_train": y_height[idx_train],
        "y_height_val": y_height[idx_val],
        "y_height_test": y_height[idx_test],
        "y_joint_train": y_joint[idx_train],
        "y_joint_val": y_joint[idx_val],
        "y_joint_test": y_joint[idx_test],
    }


def _save_selected_models(selection: dict[str, Any], paths: dict[str, str], project_root: Path) -> dict[str, str]:
    model_dir = resolve_path(paths["best_metadata_json"], project_root).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    if selection["design"] == "two_task":
        radius_path = model_dir / "best_radius_model.joblib"
        height_path = model_dir / "best_height_model.joblib"
        joblib.dump(selection["selected_models"]["radius"], radius_path, compress=3)
        joblib.dump(selection["selected_models"]["height"], height_path, compress=3)
        artifact_paths = {
            "radius_model": to_rel_path(radius_path, project_root),
            "height_model": to_rel_path(height_path, project_root),
        }
    else:
        joint_path = model_dir / "best_joint_model.joblib"
        joblib.dump(selection["selected_model"], joint_path, compress=3)
        artifact_paths = {"joint_model": to_rel_path(joint_path, project_root)}

    return artifact_paths


def train_and_select(config: dict[str, Any], ensure_prepared: bool = True) -> dict[str, Any]:
    paths = config["paths"]
    project_root = Path(config["_project_root"]).resolve()
    processed_path = resolve_path(paths["processed_csv"], project_root)
    split_path = resolve_path(paths["splits_json"], project_root)

    if ensure_prepared and (not processed_path.exists() or not split_path.exists()):
        prepare_data(config)

    dataset = load_processed_dataset(processed_path)
    split_payload = read_json(split_path)
    arrays = _load_split_arrays(dataset, split_payload)

    radius_rows, height_rows, two_task_summary = _evaluate_two_task(
        x_train=arrays["x_train"],
        x_val=arrays["x_val"],
        x_test=arrays["x_test"],
        y_radius_train=arrays["y_radius_train"],
        y_radius_val=arrays["y_radius_val"],
        y_radius_test=arrays["y_radius_test"],
        y_height_train=arrays["y_height_train"],
        y_height_val=arrays["y_height_val"],
        y_height_test=arrays["y_height_test"],
        config=config,
    )

    multi_rows, multi_summary = _evaluate_multioutput(
        x_train=arrays["x_train"],
        x_val=arrays["x_val"],
        x_test=arrays["x_test"],
        y_joint_train=arrays["y_joint_train"],
        y_joint_val=arrays["y_joint_val"],
        y_joint_test=arrays["y_joint_test"],
        y_radius_val=arrays["y_radius_val"],
        y_radius_test=arrays["y_radius_test"],
        y_height_val=arrays["y_height_val"],
        y_height_test=arrays["y_height_test"],
        config=config,
    )

    selected_summary = two_task_summary if two_task_summary["selection_score"] >= multi_summary["selection_score"] else multi_summary

    artifact_paths = _save_selected_models(selected_summary, paths, project_root=project_root)

    acceptance_cfg = config["acceptance"]
    acceptance = {
        "radius_accuracy_threshold": float(acceptance_cfg["legacy_radius_accuracy"]),
        "height_r2_threshold": float(acceptance_cfg["legacy_height_r2"]),
        "radius_accuracy_pass": bool(selected_summary["test_radius_accuracy"] >= float(acceptance_cfg["legacy_radius_accuracy"])),
        "height_r2_pass": bool(selected_summary["test_height_r2"] >= float(acceptance_cfg["legacy_height_r2"])),
    }
    acceptance["all_pass"] = acceptance["radius_accuracy_pass"] and acceptance["height_r2_pass"]

    model_results = radius_rows + height_rows + multi_rows
    model_results_df = pd.DataFrame(model_results)

    metrics_csv_path = resolve_path(paths["metrics_csv"], project_root)
    ensure_parent(metrics_csv_path)
    model_results_df.to_csv(metrics_csv_path, index=False)

    metadata = {
        "_project_root": str(project_root),
        "model_version": DEFAULT_MODEL_VERSION,
        "selected_design": selected_summary["design"],
        "feature_columns": FEATURE_COLUMNS,
        "radius_transform": {
            "min_mm": 3.5,
            "step_mm": 0.5,
        },
        "selection": {
            "two_task": {k: v for k, v in two_task_summary.items() if not str(k).startswith("selected_")},
            "multioutput": {k: v for k, v in multi_summary.items() if not str(k).startswith("selected_")},
        },
        "selected_test_metrics": {
            "radius_accuracy": float(selected_summary["test_radius_accuracy"]),
            "radius_within_1_class": float(selected_summary["test_radius_within_1_class"]),
            "height_r2": float(selected_summary["test_height_r2"]),
            "height_mae": float(selected_summary["test_height_mae"]),
        },
        "selected_models": {
            "radius_model_name": selected_summary.get("selected_radius_model"),
            "height_model_name": selected_summary.get("selected_height_model"),
            "joint_model_name": selected_summary.get("selected_joint_model"),
            "artifact_paths": artifact_paths,
        },
        "acceptance": acceptance,
        "metrics_csv": to_rel_path(metrics_csv_path, project_root),
    }

    metadata_path = resolve_path(paths["best_metadata_json"], project_root)
    write_json(metadata_path, metadata)

    metrics_payload = {
        "model_results": model_results,
        "metadata": metadata,
    }
    metrics_json_path = resolve_path(paths["metrics_json"], project_root)
    write_json(metrics_json_path, metrics_payload)

    return {
        "metrics_json": to_rel_path(metrics_json_path, project_root),
        "metrics_csv": to_rel_path(metrics_csv_path, project_root),
        "best_metadata_json": to_rel_path(metadata_path, project_root),
        "selected_design": selected_summary["design"],
        "acceptance": acceptance,
    }


def train_from_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path=config_path)
    return train_and_select(config=config, ensure_prepared=True)


def predict_loaded_model(
    metadata: dict[str, Any],
    electric_abs_3d: float,
    magnetic_abs_3d: float,
    mode_1: float,
    project_root: str | Path | None = None,
) -> dict[str, float | str]:
    features = np.array([[electric_abs_3d, magnetic_abs_3d, mode_1]], dtype=float)
    selected = metadata["selected_models"]
    base_root = Path(project_root or Path.cwd()).resolve()

    if metadata["selected_design"] == "two_task":
        radius_model = joblib.load(resolve_path(selected["artifact_paths"]["radius_model"], base_root))
        height_model = joblib.load(resolve_path(selected["artifact_paths"]["height_model"], base_root))
        pred_radius_class = np.asarray(radius_model.predict(features)).astype(int)
        pred_radius_mm = class_to_radius(pred_radius_class)[0]
        pred_height_mm = float(np.asarray(height_model.predict(features)).reshape(-1)[0])
    else:
        joint_model = joblib.load(resolve_path(selected["artifact_paths"]["joint_model"], base_root))
        pred = np.asarray(joint_model.predict(features)).reshape(-1, 2)
        pred_radius_mm = float(class_to_radius(radius_to_class(pred[:, 0]))[0])
        pred_height_mm = float(pred[:, 1][0])

    return {
        "predicted_radius_mm": float(pred_radius_mm),
        "predicted_height_mm": float(pred_height_mm),
        "model_name": selected.get("joint_model_name") or f"{selected.get('radius_model_name')}+{selected.get('height_model_name')}",
        "model_version": metadata.get("model_version", DEFAULT_MODEL_VERSION),
    }
