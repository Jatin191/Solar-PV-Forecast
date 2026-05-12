"""
SolarCast - Model Accuracy Benchmark
=====================================
Compares all three models on the same held-out test set and prints
a ranked accuracy table with MAE, RMSE, and R2 for each.

Run:
    python ml/benchmark.py
"""

import json
import math
import random
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not found. Run: pip install -r ml/requirements.txt")

import os

RANDOM_SEED = 99          # different seed from training = true held-out data
N_TEST      = 10_000
MODEL_PATH  = "ml/xgboost_model.json"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── helpers ────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_rad(d):
    return d * math.pi / 180

def solar_elevation(hour, doy, lat):
    decl  = 23.45 * math.sin(to_rad(360 / 365 * (doy - 81)))
    ha    = 15 * (hour - 12)
    s = (math.sin(to_rad(lat)) * math.sin(to_rad(decl))
         + math.cos(to_rad(lat)) * math.cos(to_rad(decl)) * math.cos(to_rad(ha)))
    return math.degrees(math.asin(clamp(s, -1, 1)))


# ── "ground truth" accurate physics (same as training targets) ──────────────

def accurate_power(irr, temp, wind, cloud, tilt, azimuth, lat, efficiency, losses):
    """Returns normalised power (0-1.1) using the more accurate physics."""
    if irr <= 0:
        return 0.0
    temp_f  = clamp(1 - 0.0045 * max(0, temp - 25), 0.70, 1.10)
    opt_tilt = clamp(abs(lat) * 0.76, 10, 45)
    tilt_f  = clamp(0.80 + 0.20 * math.cos(to_rad(tilt - opt_tilt)), 0.68, 1.05)
    pref    = 180 if lat >= 0 else 0
    az_diff = min(abs(azimuth - pref), 360 - abs(azimuth - pref))
    az_f    = clamp(0.84 + 0.16 * math.cos(to_rad(az_diff)), 0.68, 1.0)
    beam_f  = 1 - 0.85 * (cloud / 100) ** 1.3
    diff_f  = 0.18 * (cloud / 100)
    eff_irr = max(0, irr * (beam_f + diff_f))
    wind_f  = min(1.06, 1 + min(wind, 12) * 0.003)
    eff_f   = clamp(efficiency / 18, 0.50, 1.50)
    loss_f  = clamp(1 - losses / 100, 0.55, 1.0)
    return clamp((eff_irr / 1000) * temp_f * tilt_f * az_f * wind_f * eff_f * loss_f, 0.0, 1.10)


# ── original (simple) physics model used in the app ────────────────────────

def simple_physics(irr, temp, wind, cloud, tilt, lat, efficiency, losses):
    """Mirrors computeForecast() in app.js."""
    if irr <= 0:
        return 0.0
    temp_f   = clamp(1 - 0.004 * max(0, temp - 25), 0.78, 1.06)
    opt_tilt = clamp(abs(lat) * 0.76, 12, 40)
    tilt_f   = clamp(0.82 + 0.18 * math.cos(to_rad(tilt - opt_tilt)), 0.72, 1.02)
    cloud_f  = clamp(1 - cloud / 430, 0.68, 1)        # the weak formula
    loss_f   = clamp(1 - losses / 100, 0.6, 1)
    eff_f    = clamp(efficiency / 18, 0.72, 1.35)
    wind_f   = clamp(1 + min(wind, 30) * 0.002, 1, 1.06)
    return clamp((irr / 1000) * temp_f * tilt_f * cloud_f * loss_f * eff_f * wind_f, 0.0, 1.08)


# ── feature builder (must match train.py) ──────────────────────────────────

def build_features(hour, doy, lat, irr, cloud, temp, wind, tilt, azimuth, eff, losses):
    pref   = 180 if lat >= 0 else 0
    az_diff = min(abs(azimuth - pref), 360 - abs(azimuth - pref))
    elev   = solar_elevation(hour, doy, lat)
    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
        clamp(elev / 90, -1, 1),
        clamp(irr / 1000, 0, 1.5),
        cloud / 100,
        clamp((temp + 10) / 55, 0, 1),
        clamp(wind / 15, 0, 2),
        tilt / 45,
        az_diff / 180,
        clamp((lat + 60) / 120, 0, 1),
        clamp(eff / 25, 0, 1.5),
        losses / 30,
    ]


# ── hybrid correction (mirrors app.js handleFitHybrid) ─────────────────────

def fit_hybrid_coeffs(samples):
    """Fit a 3-variable least-squares correction on the samples."""
    n = len(samples)
    X, y = [], []
    for s in samples:
        irr, cloud, temp, phys, truth = s
        x1 = (cloud / 100) ** 1.3 - cloud / 430
        x2 = max(0, temp - 25) / 30
        x3 = math.sqrt(max(0, irr / 1000))
        X.append([x1, x2, x3])
        y.append(truth - phys)

    X = np.array(X)
    y = np.array(y)
    coeffs, *_ = np.linalg.lstsq(
        np.hstack([X, np.ones((n, 1))]), y, rcond=None
    )
    return coeffs  # [alpha, beta, gamma, intercept]


def apply_hybrid(phys, irr, cloud, temp, coeffs):
    alpha, beta, gamma, intercept = coeffs
    x1 = (cloud / 100) ** 1.3 - cloud / 430
    x2 = max(0, temp - 25) / 30
    x3 = math.sqrt(max(0, irr / 1000))
    correction = alpha * x1 + beta * x2 + gamma * x3 + intercept
    return clamp(phys + correction, 0.0, 1.10)


# ── generate test dataset ───────────────────────────────────────────────────

def generate_test_data(n):
    print(f"Generating {n:,} test samples (seed={RANDOM_SEED}) ...")
    rows = []
    for _ in range(n):
        lat      = random.uniform(-55, 55)
        eff      = random.uniform(12, 24)
        losses   = random.uniform(5, 25)
        tilt     = random.uniform(0, 50)
        azimuth  = random.uniform(0, 360)
        hour     = random.uniform(0, 24)
        doy      = random.randint(1, 365)
        cloud    = random.uniform(0, 100)
        temp     = random.uniform(-5, 45)
        wind     = random.uniform(0, 20)

        elev = solar_elevation(hour, doy, lat)
        irr  = 0.0
        if elev > 2:
            clear = 900 * math.sin(to_rad(elev))
            att   = 1 - 0.82 * (cloud / 100) ** 1.2
            irr   = clamp(clear * att * random.uniform(0.88, 1.08), 0, 1100)

        truth = accurate_power(irr, temp, wind, cloud, tilt, azimuth, lat, eff, losses)
        phys  = simple_physics(irr, temp, wind, cloud, tilt, lat, eff, losses)
        feats = build_features(hour, doy, lat, irr, cloud, temp, wind, tilt, azimuth, eff, losses)

        rows.append({
            "truth": truth, "phys": phys, "feats": feats,
            "irr": irr, "cloud": cloud, "temp": temp,
        })
    return rows


# ── evaluate ────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2}


def rank_label(rank):
    return ["1st", "2nd", "3rd", "4th"][rank]


def print_table(results):
    # Sort by MAE ascending
    ranked = sorted(results, key=lambda x: x["mae"])
    col = 28
    print("\n" + "=" * 72)
    print(f"  {'RANK':<6} {'MODEL':<28} {'MAE %':>7} {'RMSE':>8} {'R2':>8}")
    print("=" * 72)
    for i, r in enumerate(ranked):
        star = " <-- BEST" if i == 0 else ""
        print(
            f"  {rank_label(i):<6} {r['label']:<28} "
            f"{r['mae']*100:>6.2f}%  {r['rmse']:>8.4f}  {r['r2']:>8.4f}{star}"
        )
    print("=" * 72)
    print()
    print("  MAE  = Mean Absolute Error (lower is better)")
    print("  RMSE = Root Mean Squared Error (lower is better)")
    print("  R2   = Coefficient of determination (higher is better, max 1.0)")
    print()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    # 1. Generate test data
    rows = generate_test_data(N_TEST)

    y_true = np.array([r["truth"] for r in rows])
    y_phys = np.array([r["phys"]  for r in rows])

    results = [metrics(y_true, y_phys, "Physics (app baseline)")]

    # 2. XGBoost
    if not os.path.exists(MODEL_PATH):
        print(f"\n[!] XGBoost model not found at {MODEL_PATH}.")
        print("    Run  python ml/train.py  first.\n")
    else:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        X = np.array([r["feats"] for r in rows], dtype=np.float32)
        y_xgb = model.predict(X)
        results.append(metrics(y_true, y_xgb, "XGBoost (Python backend)"))
        print("XGBoost model loaded and evaluated.")

    # 3. Hybrid — fit on 20% of the test set, evaluate on the rest
    split     = N_TEST // 5
    fit_rows  = rows[:split]
    eval_rows = rows[split:]

    fit_samples = [(r["irr"], r["cloud"], r["temp"], r["phys"], r["truth"]) for r in fit_rows]
    coeffs = fit_hybrid_coeffs(fit_samples)

    y_true_eval  = np.array([r["truth"] for r in eval_rows])
    y_phys_eval  = np.array([r["phys"]  for r in eval_rows])
    y_hybrid = np.array([
        apply_hybrid(r["phys"], r["irr"], r["cloud"], r["temp"], coeffs)
        for r in eval_rows
    ])
    results.append(metrics(y_true_eval, y_hybrid,     "Hybrid (physics + correction)"))
    results.append(metrics(y_true_eval, y_phys_eval,  "Physics (baseline, same split)"))
    # Remove the full-set physics result if we have the split version
    results = [r for r in results if r["label"] != "Physics (baseline, same split)"]

    print_table(results)


if __name__ == "__main__":
    main()
