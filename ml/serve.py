"""
SolarCast — XGBoost Prediction REST Server
==========================================
Exposes a single POST endpoint that the SolarCast browser app calls to
get XGBoost-based power predictions for a 72-hour weather window.

Requirements (install once):
    pip install -r ml/requirements.txt

Train the model first:
    python ml/train.py

Then start this server:
    python ml/serve.py

The browser app will automatically detect it at http://127.0.0.1:5050/predict
"""

import json
import math
import os
import sys

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
except ImportError:
    sys.exit("flask / flask-cors not found.  Run:  pip install -r ml/requirements.txt")

try:
    import xgboost as xgb
    import numpy as np
except ImportError:
    sys.exit("xgboost / numpy not found.  Run:  pip install -r ml/requirements.txt")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_model.json")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

if not os.path.exists(MODEL_PATH):
    sys.exit(
        f"Model file not found at {MODEL_PATH}.\n"
        "Train it first with:  python ml/train.py"
    )

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

meta = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)

print(f"Model loaded from {MODEL_PATH}")
if meta:
    print(f"  R² = {meta.get('r2', '?'):.4f}   MAE = {meta.get('mae', '?'):.4f}")

# ---------------------------------------------------------------------------
# Feature builder (must match ml/train.py)
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _to_rad(deg):
    return deg * math.pi / 180

def _solar_elevation(hour, day_of_year, latitude_deg):
    decl = 23.45 * math.sin(_to_rad(360 / 365 * (day_of_year - 81)))
    hour_angle = 15 * (hour - 12)
    lat_r = _to_rad(latitude_deg)
    decl_r = _to_rad(decl)
    ha_r = _to_rad(hour_angle)
    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    return math.degrees(math.asin(_clamp(sin_elev, -1, 1)))

def build_features(hour, day_of_year, latitude, irradiance, cloud_cover,
                   temperature, wind_speed, tilt, azimuth, efficiency, losses):
    solar_elev = _solar_elevation(hour, day_of_year, latitude)
    preferred = 180 if latitude >= 0 else 0
    az_diff = min(abs(azimuth - preferred), 360 - abs(azimuth - preferred))
    return np.array([[
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * day_of_year / 365),
        math.cos(2 * math.pi * day_of_year / 365),
        _clamp(solar_elev / 90, -1, 1),
        _clamp(irradiance / 1000, 0, 1.5),
        cloud_cover / 100,
        _clamp((temperature + 10) / 55, 0, 1),
        _clamp(wind_speed / 15, 0, 2),
        tilt / 45,
        az_diff / 180,
        _clamp((latitude + 60) / 120, 0, 1),
        _clamp(efficiency / 25, 0, 1.5),
        losses / 30,
    ]], dtype=np.float32)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "xgboost", "meta": meta})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
      "config": {
        "latitude": 28.6, "longitude": 77.2,
        "capacity": 5, "efficiency": 18,
        "tilt": 30, "azimuth": 180, "losses": 12
      },
      "hourly": [
        { "time": "2026-05-12T06:00", "irradiance": 420,
          "temperature": 32, "cloud_cover": 20, "wind_speed": 8 },
        ...
      ]
    }
    Returns:
    {
      "predictions": [0.82, 1.24, ...],   // predicted kW per hour
      "model": "xgboost",
      "meta": { "r2": ..., "mae": ... }
    }
    """
    try:
        body = request.get_json(force=True)
        cfg = body["config"]
        hourly = body["hourly"]

        lat = float(cfg["latitude"])
        tilt = float(cfg.get("tilt", 30))
        azimuth = float(cfg.get("azimuth", 180))
        capacity = float(cfg.get("capacity", 5))
        efficiency = float(cfg.get("efficiency", 18))
        losses = float(cfg.get("losses", 12))

        predictions = []
        for row in hourly:
            time_str = row.get("time", "2026-01-01T12:00")
            # Parse hour and day_of_year from ISO string
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(time_str)
                hour = dt.hour + dt.minute / 60
                day_of_year = dt.timetuple().tm_yday
            except Exception:
                hour = 12
                day_of_year = 180

            irradiance = float(row.get("irradiance", 0))
            temperature = float(row.get("temperature", 25))
            cloud_cover = float(row.get("cloud_cover", 0))
            wind_speed = float(row.get("wind_speed", 5))

            feats = build_features(hour, day_of_year, lat, irradiance,
                                   cloud_cover, temperature, wind_speed,
                                   tilt, azimuth, efficiency, losses)
            norm_power = float(model.predict(feats)[0])
            kw = max(0.0, min(capacity * 1.10, norm_power * capacity))
            predictions.append(round(kw, 3))

        return jsonify({"predictions": predictions, "model": "xgboost", "meta": meta})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\nXGBoost prediction server running -> http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=port, debug=False)
