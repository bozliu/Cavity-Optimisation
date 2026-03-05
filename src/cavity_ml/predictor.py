"""Prediction entrypoints using persisted model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .io_utils import read_json, resolve_path
from .trainer import predict_loaded_model


def load_metadata(metadata_path: str | Path | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    if metadata_path is not None:
        path = Path(metadata_path)
        if config_path is not None:
            cfg = load_config(config_path=config_path)
            project_root = Path(cfg["_project_root"]).resolve()
        path = resolve_path(path, project_root)
    else:
        cfg = load_config(config_path=config_path)
        project_root = Path(cfg["_project_root"]).resolve()
        path = resolve_path(cfg["paths"]["best_metadata_json"], project_root)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    payload = read_json(path)
    payload.setdefault("_project_root", str(project_root))
    return payload


def predict_from_artifacts(
    electric_abs_3d: float,
    magnetic_abs_3d: float,
    mode_1: float,
    metadata_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, float | str]:
    metadata = load_metadata(metadata_path=metadata_path, config_path=config_path)
    base_root = Path(metadata.get("_project_root", Path.cwd())).resolve()
    return predict_loaded_model(
        metadata=metadata,
        electric_abs_3d=float(electric_abs_3d),
        magnetic_abs_3d=float(magnetic_abs_3d),
        mode_1=float(mode_1),
        project_root=base_root,
    )
