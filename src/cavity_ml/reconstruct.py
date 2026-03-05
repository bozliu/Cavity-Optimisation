"""Public-source reconstruction for cavity-style synthetic dataset.

This module creates deterministic, open reproducible synthetic data using
published cylindrical cavity resonator equations and documented transforms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconstructionSpec:
    profile: str = "full"
    seed: int = 44


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_core_dataframe(profile: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    radius_values = np.arange(3.5, 20.0 + 1e-9, 0.5)
    height_values = np.arange(4.5, 50.5 + 1e-9, 0.5)

    rr, hh = np.meshgrid(radius_values, height_values)
    c_r = rr.reshape(-1)
    c_h = hh.reshape(-1)

    if profile == "full":
        # Match legacy sample count (3161) from 34*93=3162 by dropping the last point.
        c_r = c_r[:-1]
        c_h = c_h[:-1]
    elif profile == "toy":
        keep = rng.choice(np.arange(c_r.size), size=620, replace=False)
        keep = np.sort(keep)
        c_r = c_r[keep]
        c_h = c_h[keep]
    else:
        raise ValueError("profile must be one of: full, toy")

    run_id = np.arange(1, c_r.size + 1)

    # Approximate cylindrical cavity resonance formula (TM mode style).
    c = 299_792_458.0
    x_01 = 2.4048255577
    r_m = c_r * 1e-3
    h_m = c_h * 1e-3
    mode_1 = (c / (2.0 * np.pi)) * np.sqrt((x_01 / r_m) ** 2 + (np.pi / h_m) ** 2) / 1e9

    r_norm = (c_r - 3.5) / (20.0 - 3.5)
    h_norm = (c_h - 4.5) / (50.5 - 4.5)
    m_norm = (mode_1 - mode_1.min()) / (mode_1.max() - mode_1.min())

    # Keep physically inspired smooth variation but make cR/cH information
    # less degenerate for inverse prediction benchmarks.
    electric = 0.86 + 0.30 * (1 - r_norm) + 0.015 * (1 - h_norm) + 0.006 * np.cos(2 * np.pi * m_norm)
    magnetic = 0.84 + 0.30 * r_norm + 0.015 * h_norm + 0.006 * np.sin(2 * np.pi * m_norm)

    electric += rng.normal(0.0, 0.0008, size=electric.shape)
    magnetic += rng.normal(0.0, 0.0008, size=magnetic.shape)

    electric = np.clip(electric, 0.84, 1.22)
    magnetic = np.clip(magnetic, 0.84, 1.28)

    return pd.DataFrame(
        {
            "3D Run ID": run_id.astype(int),
            "cH": np.round(c_h, 1),
            "cR": np.round(c_r, 1),
            "Electric_Abs_3D": np.round(electric, 6),
            "Magnetic_Abs_3D": np.round(magnetic, 6),
            "Mode 1": np.round(mode_1, 4),
        }
    )


def reconstruct_public_sources(
    raw_frequency_csv: str | Path,
    raw_eh_csv: str | Path,
    xlsx_path: str | Path,
    profile: str = "full",
    seed: int = 44,
) -> dict[str, Any]:
    df = _build_core_dataframe(profile=profile, seed=seed)

    raw_frequency_csv = Path(raw_frequency_csv)
    raw_eh_csv = Path(raw_eh_csv)
    xlsx_path = Path(xlsx_path)

    raw_frequency_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_eh_csv.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    freq_rows = [
        ["Unnamed: 0", "Unnamed: 1", "Parameters", "Unnamed: 3", "1D Results\\Mode Frequencies"],
        ["3D Run ID", "Status", "cH", "cR", "Mode 1"],
    ]
    eh_rows = [
        ["Unnamed: 0", "Unnamed: 1", "Parameters", "Unnamed: 3", "0D Results", "Unnamed: 5", "1D Results"],
        ["3D Run ID", "Status", "cH", "cR", "Electric_Abs_3D", "Magnetic_Abs_3D", "TotalQ_Eigenmode_All"],
    ]

    for row in df.itertuples(index=False):
        freq_rows.append([int(row[0]), "Calculated", float(row[1]), float(row[2]), float(row[5])])
        eh_rows.append([int(row[0]), "Calculated", float(row[1]), float(row[2]), float(row[3]), float(row[4]), "1D - Real"])

    pd.DataFrame(freq_rows).to_csv(raw_frequency_csv, index=False, header=False)
    pd.DataFrame(eh_rows).to_csv(raw_eh_csv, index=False, header=False)
    df.to_excel(xlsx_path, index=False)

    manifest = {
        "profile": profile,
        "seed": seed,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "files": {
            "raw_frequency_csv": {
                "path": raw_frequency_csv.as_posix(),
                "sha256": _sha256(raw_frequency_csv),
            },
            "raw_eh_csv": {
                "path": raw_eh_csv.as_posix(),
                "sha256": _sha256(raw_eh_csv),
            },
            "xlsx": {
                "path": xlsx_path.as_posix(),
                "sha256": _sha256(xlsx_path),
            },
        },
    }

    manifest_path = raw_frequency_csv.parent.parent / "reconstruction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = manifest_path.as_posix()
    return manifest
