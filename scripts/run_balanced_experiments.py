#!/usr/bin/env python3
# ruff: noqa: E402
"""Run prepare-data, train, and evaluate in sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_ml.cli import main
from cavity_ml.config import load_config
from cavity_ml.io_utils import resolve_path
from cavity_ml.reconstruct import reconstruct_public_sources


def run(config: str | None, reconstruct_if_missing: bool, profile: str) -> None:
    base_args = ["--config", config] if config else []
    if reconstruct_if_missing:
        cfg = load_config(config_path=config, project_root=ROOT)
        root = Path(cfg["_project_root"]).resolve()
        freq = resolve_path(cfg["paths"]["raw_frequency_csv"], root)
        eh = resolve_path(cfg["paths"]["raw_eh_csv"], root)
        xlsx = resolve_path(cfg["paths"]["xlsx_path"], root)
        if not freq.exists() or not eh.exists() or not xlsx.exists():
            reconstruct_public_sources(freq, eh, xlsx, profile=profile, seed=int(cfg["random_state"]))
    main(["prepare-data", *base_args])
    main(["train", *base_args])
    main(["evaluate", *base_args])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--reconstruct-if-missing", action="store_true")
    parser.add_argument("--profile", choices=["full", "toy"], default="full")
    args = parser.parse_args()
    run(args.config, reconstruct_if_missing=args.reconstruct_if_missing, profile=args.profile)
