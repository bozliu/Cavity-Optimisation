from __future__ import annotations

from pathlib import Path

from cavity_ml.config import load_config
from cavity_ml.data_pipeline import prepare_data
from cavity_ml.predictor import predict_from_artifacts
from cavity_ml.reconstruct import reconstruct_public_sources
from cavity_ml.reporting import generate_report
from cavity_ml.trainer import train_and_select
from cavity_ml.web.app import create_app


def _tmp_config(tmp_path: Path) -> dict:
    cfg = load_config("configs/lightweight_test.yaml")
    raw_freq = tmp_path / "data/raw/run_3_frequencies.csv"
    raw_eh = tmp_path / "data/raw/run_3_e_h_fields.csv"
    xlsx = tmp_path / "data/alternate/Mode1.xlsx"
    reconstruct_public_sources(raw_freq, raw_eh, xlsx, profile="toy", seed=44)

    cfg["paths"]["raw_frequency_csv"] = "data/raw/run_3_frequencies.csv"
    cfg["paths"]["raw_eh_csv"] = "data/raw/run_3_e_h_fields.csv"
    cfg["paths"]["xlsx_path"] = "data/alternate/Mode1.xlsx"
    cfg["paths"]["processed_csv"] = "data/processed/cavity_dataset.csv"
    cfg["paths"]["splits_json"] = "artifacts/splits/split_indices.json"
    cfg["paths"]["metrics_json"] = "artifacts/metrics/metrics.json"
    cfg["paths"]["metrics_csv"] = "artifacts/metrics/metrics.csv"
    cfg["paths"]["report_md"] = "reports/experiment_report.md"
    cfg["paths"]["best_metadata_json"] = "artifacts/models/best_model_metadata.json"
    cfg["_project_root"] = str(tmp_path.resolve())
    return cfg


def test_end_to_end_training_and_report(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)

    prepare_data(cfg)
    train_result = train_and_select(cfg, ensure_prepared=False)
    report_result = generate_report(cfg)

    assert (tmp_path / train_result["metrics_json"]).exists()
    assert (tmp_path / train_result["metrics_csv"]).exists()
    assert (tmp_path / train_result["best_metadata_json"]).exists()
    assert (tmp_path / report_result["report_md"]).exists()

    prediction = predict_from_artifacts(
        electric_abs_3d=1.01372,
        magnetic_abs_3d=1.04692,
        mode_1=32.7838,
        metadata_path=tmp_path / cfg["paths"]["best_metadata_json"],
        config_path=None,
    )

    assert "predicted_radius_mm" in prediction
    assert "predicted_height_mm" in prediction
    assert prediction["predicted_radius_mm"] >= 3.5


def test_flask_health_and_predict(tmp_path: Path) -> None:
    cfg = _tmp_config(tmp_path)
    prepare_data(cfg)
    train_and_select(cfg, ensure_prepared=False)

    app = create_app(metadata_path=tmp_path / cfg["paths"]["best_metadata_json"])
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    health_payload = health.get_json()
    assert health_payload["status"] == "ok"
    assert health_payload["model_loaded"] is True

    pred = client.post(
        "/predict",
        json={
            "electric_abs_3d": 1.01372,
            "magnetic_abs_3d": 1.04692,
            "mode_1": 32.7838,
        },
    )
    assert pred.status_code == 200
    pred_payload = pred.get_json()
    assert "predicted_radius_mm" in pred_payload
    assert "predicted_height_mm" in pred_payload
