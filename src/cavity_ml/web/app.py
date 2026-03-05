"""Flask application for local inference demo and JSON API."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template_string, request

from ..predictor import load_metadata, predict_from_artifacts

HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cavity Optimisation Predictor</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
    input { width: 100%; padding: 0.5rem; margin: 0.25rem 0 1rem 0; }
    button { padding: 0.6rem 1rem; }
    pre { background: #f4f4f4; padding: 1rem; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Cavity Predictor</h1>
  <p>Input features: <code>Electric_Abs_3D</code>, <code>Magnetic_Abs_3D</code>, <code>Mode 1</code></p>
  <label>electric_abs_3d</label>
  <input id="electric_abs_3d" type="number" step="any" value="1.01372" />
  <label>magnetic_abs_3d</label>
  <input id="magnetic_abs_3d" type="number" step="any" value="1.04692" />
  <label>mode_1</label>
  <input id="mode_1" type="number" step="any" value="32.7838" />
  <button onclick="runPredict()">Predict</button>
  <pre id="result">Awaiting input...</pre>

  <script>
    async function runPredict() {
      const payload = {
        electric_abs_3d: Number(document.getElementById('electric_abs_3d').value),
        magnetic_abs_3d: Number(document.getElementById('magnetic_abs_3d').value),
        mode_1: Number(document.getElementById('mode_1').value),
      };
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      document.getElementById('result').textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""


def create_app(config_path: str | None = None, metadata_path: str | None = None) -> Flask:
    app = Flask(__name__)

    metadata: dict[str, Any] | None = None
    metadata_error: str | None = None
    try:
        metadata = load_metadata(metadata_path=metadata_path, config_path=config_path)
    except Exception as exc:  # pragma: no cover
        metadata_error = str(exc)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "model_loaded": metadata is not None,
                "selected_design": metadata.get("selected_design") if metadata else None,
                "error": metadata_error,
            }
        )

    @app.post("/predict")
    def predict_endpoint() -> Any:
        nonlocal metadata
        if metadata is None:
            return jsonify({"error": f"Model metadata unavailable: {metadata_error}"}), 500

        payload = request.get_json(silent=True) or {}
        required = ["electric_abs_3d", "magnetic_abs_3d", "mode_1"]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        try:
            result = predict_from_artifacts(
                electric_abs_3d=float(payload["electric_abs_3d"]),
                magnetic_abs_3d=float(payload["magnetic_abs_3d"]),
                mode_1=float(payload["mode_1"]),
                metadata_path=metadata_path,
                config_path=config_path,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(result)

    return app
