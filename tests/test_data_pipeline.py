from __future__ import annotations

from pathlib import Path

import numpy as np

from cavity_ml.config import load_config
from cavity_ml.data_pipeline import (
    compare_datasets,
    load_raw_csv_dataset,
    load_xlsx_dataset,
    make_deterministic_splits,
    prepare_data,
)
from cavity_ml.labels import class_to_radius, radius_to_class
from cavity_ml.reconstruct import reconstruct_public_sources


def _prepare_public_sources(cfg: dict, root: Path, profile: str = "full") -> None:
    raw_freq = root / "data/raw/run_3_frequencies.csv"
    raw_eh = root / "data/raw/run_3_e_h_fields.csv"
    xlsx = root / "data/alternate/Mode1.xlsx"
    reconstruct_public_sources(raw_freq, raw_eh, xlsx, profile=profile, seed=44)
    cfg["paths"]["raw_frequency_csv"] = str(raw_freq.relative_to(root))
    cfg["paths"]["raw_eh_csv"] = str(raw_eh.relative_to(root))
    cfg["paths"]["xlsx_path"] = str(xlsx.relative_to(root))
    cfg["_project_root"] = str(root.resolve())


def test_csv_xlsx_equivalence(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    _prepare_public_sources(cfg, tmp_path, profile="full")
    raw_df = load_raw_csv_dataset(
        tmp_path / cfg["paths"]["raw_frequency_csv"],
        tmp_path / cfg["paths"]["raw_eh_csv"],
    )
    xlsx_df = load_xlsx_dataset(tmp_path / cfg["paths"]["xlsx_path"])

    comp = compare_datasets(raw_df, xlsx_df)
    assert comp["rows_a"] == comp["rows_b"]
    assert comp["row_mismatch_count_round6"] == 0


def test_radius_transform_roundtrip() -> None:
    radius_mm = np.array([3.5, 4.0, 10.5, 20.0], dtype=float)
    classes = radius_to_class(radius_mm)
    recovered = class_to_radius(classes)
    assert np.allclose(radius_mm, recovered)


def test_preprocessing_and_split_determinism(tmp_path: Path) -> None:
    cfg = load_config("configs/default.yaml")
    _prepare_public_sources(cfg, tmp_path, profile="full")
    cfg["paths"]["processed_csv"] = "data/processed/dataset.csv"
    cfg["paths"]["splits_json"] = "artifacts/splits/splits.json"
    cfg["_project_root"] = str(tmp_path.resolve())

    summary_a = prepare_data(cfg)
    summary_b = prepare_data(cfg)

    assert summary_a["rows"] == summary_b["rows"]
    assert summary_a["rows"] > 1000

    import pandas as pd

    dataset = pd.read_csv(tmp_path / cfg["paths"]["processed_csv"])
    split_a = make_deterministic_splits(dataset, test_size=0.2, val_size=0.1, random_state=44, stratify_radius=True)
    split_b = make_deterministic_splits(dataset, test_size=0.2, val_size=0.1, random_state=44, stratify_radius=True)

    assert split_a["indices"] == split_b["indices"]
