#!/usr/bin/env python3
# ruff: noqa: E402
"""Reconstruct public-source dataset files used by this repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cavity_ml.config import load_config
from cavity_ml.io_utils import resolve_path, to_rel_path
from cavity_ml.reconstruct import reconstruct_public_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", choices=["full", "toy"], default="full")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, project_root=ROOT)
    project_root = Path(cfg["_project_root"]).resolve()

    manifest = reconstruct_public_sources(
        raw_frequency_csv=resolve_path(cfg["paths"]["raw_frequency_csv"], project_root),
        raw_eh_csv=resolve_path(cfg["paths"]["raw_eh_csv"], project_root),
        xlsx_path=resolve_path(cfg["paths"]["xlsx_path"], project_root),
        profile=args.profile,
        seed=args.seed,
    )

    # Print relative paths for public logs.
    for k in ["raw_frequency_csv", "raw_eh_csv", "xlsx"]:
        manifest["files"][k]["path"] = to_rel_path(manifest["files"][k]["path"], project_root)
    manifest["manifest_path"] = to_rel_path(manifest["manifest_path"], project_root)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
