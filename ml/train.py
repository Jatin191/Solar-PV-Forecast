"""
SolarCast — XGBoost PV Power Forecasting Model
================================================
Trains an XGBoost regressor on synthetic PV data that uses a more accurate
physics model than the one in app.js.  The trained model is saved to
ml/xgboost_model.json and can be served by ml/serve.py.

Run:
    pip install -r ml/requirements.txt
    python ml/train.py
"""

import json
import math
import random
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not found.  Run:  pip install -r ml/requirements.txt")

import joblib

RANDOM_SEED = 42
N_SAMPLES = 60_000
MODEL_PATH = "ml/xgboost_model.json"
SCALER_PATH = "ml/feature_scaler.pkl"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _to_rad(deg):
    return deg * math.pi / 180


def _day_length_factor(day_of_year, latitude_deg):
    """Approximate fraction of the day that has daylight."""
    decl = 23.45 * math.sin(_to_rad(360 / 365 * (day_of_year - 81)))
    decl_r = _to_rad(decl)
    lat_r = _to_rad(latitude_deg)
    cos_ha = -math.tan(lat_r) * math.tan(decl_r)
    cos_ha = _clamp(cos_ha, -1, 1)
    ha = math.degrees(math.acos(cos_ha))
    return 2 * ha / 360  # fraction of day with sun above horizon


def _solar_elevation(hour_decimal, day_of_year, latitude_deg, longitude_deg=0.0):
    """Approximate solar elevation angle (degrees) using simplified formula."""
    decl = 23.45 * math.sin(_to_rad(360 / 365 * (day_of_year - 81)))
    hour_angle = 15 * (hour_decimal - 12)  # degrees, solar noon = 0
    lat_r = _to_rad(latitude_deg)
    decl_r = _to_rad(decl)
    ha_r = _to_rad(hour_angle)
    sin_elev = (math.sin(lat_r) * math.sin(decl_r)
                + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    return math.degrees(math.asin(_clamp(sin_elev, -1, 1)))


def compute_power_accurate(irradiance_wm2, temperature_c, wind_speed_ms,
                            cloud_cover_pct, tilt_deg, azimuth_deg,
                            latitude_deg, hour, day_of_year, efficiency_pct,
                            losses_pct, capacity_kw):
    """
    More accurate PV power estimate (normalised to capacity).
    Returns power in kW.
    """
    if irradiance_wm2 <= 0:
        return 0.0

    # --- Temperature coefficient (IEC 61215 standard: -0.45 %/°C) ----------
    temp_factor = _clamp(1 - 0.0045 * max(0, temperature_c - 25), 0.70, 1.10)

    # --- Tilt factor using cosine of angle of incidence (simplified) --------
    solar_elev = _solar_elevation(hour, day_of_year, latitude_deg)
    if solar_elev <= 0:
        return 0.0
    optimum_tilt = _clamp(abs(latitude_deg) * 0.76, 10, 45)
    tilt_factor = _clamp(0.80 + 0.20 * math.cos(_to_rad(tilt_deg - optimum_tilt)), 0.68, 1.05)

    # --- Azimuth factor -----------------------------------------------------
    preferred = 180 if latitude_deg >= 0 else 0
    az_diff = min(abs(azimuth_deg - preferred), 360 - abs(azimuth_deg - preferred))
    az_factor = _clamp(0.84 + 0.16 * math.cos(_to_rad(az_diff)), 0.68, 1.0)

    # --- Cloud / diffuse irradiance model -----------------------------------
    # Direct-normal irradiance drops sharply; diffuse still ~15–20%
    cloud_frac = cloud_cover_pct / 100
    beam_factor = 1 - 0.85 * (cloud_frac ** 1.3)
    diffuse_fraction = 0.18 * cloud_frac
    effective_irr = irradiance_wm2 * (beam_factor + diffuse_fraction)
    effective_irr = _clamp(effective_irr, 0, irradiance_wm2)

    # --- Wind cooling (reduces cell temp, improves efficiency) ---------------
    wind_cooling = _clamp(1 + min(wind_speed_ms, 12) * 0.003, 1.0, 1.06)

    # --- Efficiency & losses ------------------------------------------------
    eff_factor = _clamp(efficiency_pct / 18, 0.50, 1.50)
    loss_factor = _clamp(1 - losses_pct / 100, 0.55, 1.0)

    raw = (capacity_kw * (effective_irr / 1000)
           * temp_factor * tilt_factor * az_factor
           * wind_cooling * eff_factor * loss_factor)

    return _clamp(raw, 0.0, capacity_kw * 1.10)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "solar_elevation_norm",
    "irradiance_norm",
    "cloud_cover_norm",
    "temp_norm",
    "wind_norm",
    "tilt_norm",
    "azimuth_diff_norm",
    "latitude_norm",
    "efficiency_norm",
    "losses_norm",
]


def build_features(hour, day_of_year, latitude_deg, irradiance_wm2,
                   cloud_cover_pct, temperature_c, wind_speed_ms,
                   tilt_deg, azimuth_deg, efficiency_pct, losses_pct):
    """Return a 1-D numpy array of engineered features."""
    solar_elev = _solar_elevation(hour, day_of_year, latitude_deg)
    preferred = 180 if latitude_deg >= 0 else 0
    az_diff = min(abs(azimuth_deg - preferred), 360 - abs(azimuth_deg - preferred))

    return np.array([
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * day_of_year / 365),
        math.cos(2 * math.pi * day_of_year / 365),
        _clamp(solar_elev / 90, -1, 1),
        _clamp(irradiance_wm2 / 1000, 0, 1.5),
        cloud_cover_pct / 100,
        _clamp((temperature_c + 10) / 55, 0, 1),
        _clamp(wind_speed_ms / 15, 0, 2),
        tilt_deg / 45,
        az_diff / 180,
        _clamp((latitude_deg + 60) / 120, 0, 1),
        _clamp(efficiency_pct / 25, 0, 1.5),
        losses_pct / 30,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_dataset(n=N_SAMPLES):
    print(f"Generating {n:,} synthetic training samples ...")
    X_list, y_list = [], []

    for _ in range(n):
        # Random system parameters
        lat = random.uniform(-55, 55)
        capacity = random.uniform(1, 50)
        efficiency = random.uniform(12, 24)
        losses = random.uniform(5, 25)
        tilt = random.uniform(0, 50)
        azimuth = random.uniform(0, 360)

        # Random time
        hour = random.uniform(0, 24)
        day = random.randint(1, 365)

        # Weather
        cloud = random.uniform(0, 100)
        temp = random.uniform(-5, 45)
        wind = random.uniform(0, 20)

        # Derived irradiance
        elev = _solar_elevation(hour, day, lat)
        if elev <= 2:
            irr = 0.0
        else:
            clear_sky = 900 * math.sin(_to_rad(elev))
            cloud_attn = 1 - 0.82 * (cloud / 100) ** 1.2
            irr = _clamp(clear_sky * cloud_attn * random.uniform(0.88, 1.08), 0, 1100)

        # Target: normalised power (0..1.08)
        power = compute_power_accurate(
            irr, temp, wind, cloud, tilt, azimuth,
            lat, hour, day, efficiency, losses, capacity
        )
        power_norm = _clamp(power / capacity, 0.0, 1.10)

        feats = build_features(hour, day, lat, irr, cloud, temp, wind, tilt, azimuth, efficiency, losses)
        X_list.append(feats)
        y_list.append(power_norm)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print("Training XGBoost model ...")

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.06,
        max_depth=7,
        min_child_weight=4,
        subsample=0.82,
        colsample_bytree=0.82,
        gamma=0.08,
        reg_alpha=0.04,
        reg_lambda=1.2,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=40,
        eval_metric="rmse",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Test-set metrics (normalised power 0–1) ===")
    print(f"  MAE  : {mae:.4f}  ({mae * 100:.2f}% of rated capacity)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")

    # Save
    model.save_model(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    # Save feature metadata for the server
    meta = {"feature_names": FEATURE_NAMES, "mae": float(mae), "r2": float(r2)}
    meta_path = "ml/model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved -> {meta_path}")

    return model


if __name__ == "__main__":
    train()
    print("\nDone! Start the prediction server with:  python ml/serve.py")
