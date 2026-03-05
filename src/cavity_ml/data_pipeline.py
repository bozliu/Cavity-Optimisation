"""Data ingestion and deterministic split generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMNS
from .io_utils import ensure_parent, resolve_path, to_rel_path, write_json
from .labels import radius_to_class

SCHEMA_COLUMNS = [ID_COLUMN, *TARGET_COLUMNS, *FEATURE_COLUMNS]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_raw_csv_dataset(freq_csv_path: str | Path, eh_csv_path: str | Path) -> pd.DataFrame:
    freq_path = Path(freq_csv_path)
    eh_path = Path(eh_csv_path)
    if not freq_path.exists() or not eh_path.exists():
        raise FileNotFoundError(
            "Raw CST CSV files are missing. Run `python scripts/reconstruct_data.py` first, "
            "or provide paths via config."
        )

    freq = pd.read_csv(freq_path, header=None).iloc[2:].reset_index(drop=True)
    eh = pd.read_csv(eh_path, header=None).iloc[2:].reset_index(drop=True)

    aligned_len = min(len(freq), len(eh))
    freq = freq.iloc[:aligned_len].reset_index(drop=True)
    eh = eh.iloc[:aligned_len].reset_index(drop=True)

    dataset = pd.DataFrame(
        {
            ID_COLUMN: _to_numeric(eh.iloc[:, 0]),
            "cH": _to_numeric(eh.iloc[:, 2]),
            "cR": _to_numeric(eh.iloc[:, 3]),
            "Electric_Abs_3D": _to_numeric(eh.iloc[:, 4]),
            "Magnetic_Abs_3D": _to_numeric(eh.iloc[:, 5]),
            "Mode 1": _to_numeric(freq.iloc[:, 4]),
        }
    )
    dataset = dataset.dropna().reset_index(drop=True)
    dataset[ID_COLUMN] = dataset[ID_COLUMN].astype(int)
    return dataset[SCHEMA_COLUMNS]


def load_xlsx_dataset(xlsx_path: str | Path) -> pd.DataFrame:
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"XLSX dataset not found: {path}")
    dataset = pd.read_excel(path)
    missing = [col for col in SCHEMA_COLUMNS if col not in dataset.columns]
    if missing:
        raise ValueError(f"XLSX file missing expected columns: {missing}")

    dataset = dataset[SCHEMA_COLUMNS].copy()
    for col in SCHEMA_COLUMNS:
        dataset[col] = _to_numeric(dataset[col])
    dataset = dataset.dropna().reset_index(drop=True)
    dataset[ID_COLUMN] = dataset[ID_COLUMN].astype(int)
    return dataset


def compare_datasets(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, Any]:
    min_len = min(len(df_a), len(df_b))
    a = df_a.iloc[:min_len][SCHEMA_COLUMNS].reset_index(drop=True)
    b = df_b.iloc[:min_len][SCHEMA_COLUMNS].reset_index(drop=True)

    numeric_diffs = {}
    for col in SCHEMA_COLUMNS:
        diff = (a[col] - b[col]).abs()
        numeric_diffs[col] = float(diff.max())

    equal_rounded = (a.round(6) == b.round(6)).all(axis=1)
    row_mismatch_count = int((~equal_rounded).sum())

    return {
        "rows_a": int(len(df_a)),
        "rows_b": int(len(df_b)),
        "compared_rows": int(min_len),
        "row_mismatch_count_round6": row_mismatch_count,
        "max_abs_diff_per_column": numeric_diffs,
    }


def make_deterministic_splits(
    dataset: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_radius: bool,
) -> dict[str, Any]:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1")

    indices = np.arange(len(dataset))
    radius_class = radius_to_class(dataset["cR"].to_numpy())

    strat_all = radius_class if stratify_radius else None
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_all,
    )

    val_ratio_within_train_val = val_size / (1.0 - test_size)
    strat_train_val = radius_class[train_val_idx] if stratify_radius else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train_val,
        random_state=random_state,
        stratify=strat_train_val,
    )

    return {
        "random_state": random_state,
        "test_size": test_size,
        "val_size": val_size,
        "stratify_radius": stratify_radius,
        "counts": {
            "total": int(len(indices)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "indices": {
            "train": sorted(int(i) for i in train_idx),
            "val": sorted(int(i) for i in val_idx),
            "test": sorted(int(i) for i in test_idx),
        },
    }


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    project_root = Path(config["_project_root"]).resolve()

    raw_df = load_raw_csv_dataset(
        resolve_path(paths["raw_frequency_csv"], project_root),
        resolve_path(paths["raw_eh_csv"], project_root),
    )
    xlsx_path = resolve_path(paths["xlsx_path"], project_root)
    if xlsx_path.exists():
        xlsx_df = load_xlsx_dataset(xlsx_path)
        comparison = compare_datasets(raw_df, xlsx_df)
    else:
        comparison = {
            "rows_a": int(len(raw_df)),
            "rows_b": 0,
            "compared_rows": 0,
            "row_mismatch_count_round6": 0,
            "max_abs_diff_per_column": {},
            "note": "xlsx source not present; raw CSV used as canonical input",
        }

    canonical_df = raw_df.copy()
    processed_csv = resolve_path(paths["processed_csv"], project_root)
    ensure_parent(processed_csv)
    canonical_df.to_csv(processed_csv, index=False)

    split_cfg = config["split"]
    splits_payload = make_deterministic_splits(
        dataset=canonical_df,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        random_state=int(config["random_state"]),
        stratify_radius=bool(split_cfg["stratify_radius"]),
    )

    splits_path = resolve_path(paths["splits_json"], project_root)
    write_json(splits_path, splits_payload)

    summary = {
        "processed_csv": to_rel_path(processed_csv, project_root),
        "splits_json": to_rel_path(splits_path, project_root),
        "schema_columns": SCHEMA_COLUMNS,
        "rows": int(len(canonical_df)),
        "comparison_raw_vs_xlsx": comparison,
    }

    summary_path = processed_csv.parent / "data_summary.json"
    write_json(summary_path, summary)
    return summary


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(Path(path))
    missing = [col for col in SCHEMA_COLUMNS if col not in dataset.columns]
    if missing:
        raise ValueError(f"Processed dataset missing columns: {missing}")
    return dataset[SCHEMA_COLUMNS].copy()


def validate_data_sources(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    project_root = Path(config["_project_root"]).resolve()
    freq_path = resolve_path(paths["raw_frequency_csv"], project_root)
    eh_path = resolve_path(paths["raw_eh_csv"], project_root)
    xlsx_path = resolve_path(paths["xlsx_path"], project_root)

    checks = {
        "raw_frequency_csv_exists": freq_path.exists(),
        "raw_eh_csv_exists": eh_path.exists(),
        "xlsx_exists": xlsx_path.exists(),
    }
    if not checks["raw_frequency_csv_exists"] or not checks["raw_eh_csv_exists"]:
        raise FileNotFoundError(
            "Required raw files missing. Run `python scripts/reconstruct_data.py` to regenerate public data sources."
        )

    raw_df = load_raw_csv_dataset(freq_path, eh_path)
    checks["raw_rows"] = int(len(raw_df))
    checks["raw_columns"] = list(raw_df.columns)
    checks["raw_has_null"] = bool(raw_df.isnull().any().any())

    if checks["xlsx_exists"]:
        xlsx_df = load_xlsx_dataset(xlsx_path)
        checks["xlsx_rows"] = int(len(xlsx_df))
        checks["raw_vs_xlsx"] = compare_datasets(raw_df, xlsx_df)
    else:
        checks["xlsx_rows"] = 0
        checks["raw_vs_xlsx"] = {"note": "xlsx missing"}

    if checks["raw_has_null"]:
        raise ValueError("Raw dataset validation failed: null values detected in canonical columns.")

    return checks
