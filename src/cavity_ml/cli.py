"""Command-line interface for cavity optimisation workflow."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config
from .data_pipeline import prepare_data, validate_data_sources
from .predictor import predict_from_artifacts
from .reporting import generate_report
from .trainer import train_and_select
from .web.app import create_app


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_prepare_data(args: argparse.Namespace) -> None:
    cfg = load_config(config_path=args.config)
    result = prepare_data(cfg)
    _print_json(result)


def run_validate_data(args: argparse.Namespace) -> None:
    cfg = load_config(config_path=args.config)
    result = validate_data_sources(cfg)
    _print_json(result)


def run_train(args: argparse.Namespace) -> None:
    cfg = load_config(config_path=args.config)
    result = train_and_select(cfg, ensure_prepared=not args.skip_prepare)
    _print_json(result)


def run_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(config_path=args.config)
    result = generate_report(cfg)
    _print_json(result)


def run_predict(args: argparse.Namespace) -> None:
    result = predict_from_artifacts(
        electric_abs_3d=float(args.electric_abs_3d),
        magnetic_abs_3d=float(args.magnetic_abs_3d),
        mode_1=float(args.mode_1),
        metadata_path=args.metadata,
        config_path=args.config,
    )
    _print_json(result)


def run_serve(args: argparse.Namespace) -> None:
    app = create_app(config_path=args.config, metadata_path=args.metadata)
    app.run(host=args.host, port=args.port, debug=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cavity optimisation reproducible ML workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare-data", help="Prepare canonical dataset and deterministic splits")
    p_prepare.add_argument("--config", default=None, help="Path to YAML config")
    p_prepare.set_defaults(func=run_prepare_data)

    p_validate = subparsers.add_parser("validate-data", help="Validate presence/schema of required data sources")
    p_validate.add_argument("--config", default=None, help="Path to YAML config")
    p_validate.set_defaults(func=run_validate_data)

    p_train = subparsers.add_parser("train", help="Train/evaluate all models and persist best artifacts")
    p_train.add_argument("--config", default=None, help="Path to YAML config")
    p_train.add_argument("--skip-prepare", action="store_true", help="Skip automatic prepare-data step")
    p_train.set_defaults(func=run_train)

    p_eval = subparsers.add_parser("evaluate", help="Generate markdown report from metrics artifacts")
    p_eval.add_argument("--config", default=None, help="Path to YAML config")
    p_eval.set_defaults(func=run_evaluate)

    p_predict = subparsers.add_parser("predict", help="Predict cR and cH from trained best model")
    p_predict.add_argument("--config", default=None, help="Path to YAML config")
    p_predict.add_argument("--metadata", default=None, help="Direct metadata JSON path")
    p_predict.add_argument("--electric-abs-3d", required=True, type=float)
    p_predict.add_argument("--magnetic-abs-3d", required=True, type=float)
    p_predict.add_argument("--mode-1", required=True, type=float)
    p_predict.set_defaults(func=run_predict)

    p_serve = subparsers.add_parser("serve", help="Run lightweight Flask app")
    p_serve.add_argument("--config", default=None, help="Path to YAML config")
    p_serve.add_argument("--metadata", default=None, help="Direct metadata JSON path")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.set_defaults(func=run_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def prepare_data_entry() -> None:
    main(["prepare-data", *sys.argv[1:]])


def train_entry() -> None:
    main(["train", *sys.argv[1:]])


def evaluate_entry() -> None:
    main(["evaluate", *sys.argv[1:]])


def validate_data_entry() -> None:
    main(["validate-data", *sys.argv[1:]])


def predict_entry() -> None:
    main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    main()
