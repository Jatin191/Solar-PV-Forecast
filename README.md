# SolarCast Local

A local PV solar forecasting dashboard with three integrated ML models. Configure a PV array, fetch live weather data, and compare physics-based and machine-learning predictions side by side.

## Features

### Core Forecast
- 72-hour PV power forecast using ML models
- Open-Meteo live weather integration (temperature, irradiance, cloud cover, wind)
- NASA POWER historical irradiance integration (30-day lookback)
- Custom historical data via REST API endpoint or CSV upload
- Latitude/longitude presets and live browser geolocation
- PV array controls — capacity, efficiency, tilt, azimuth, system losses
- KPI cards — total generation, peak output, daily average, carbon avoided
- SVG charts — power, irradiance, weather drivers, historical context, saved-run comparison
- Hourly forecast table with toggle (first 24 h / all 72 h)
- CSV export of full hourly forecast
- Saved forecast comparison (up to 5 runs, persisted in localStorage)
- Responsive dark/light theme

### ML Models Tab 🤖
Three ML approaches run side by side after generating a forecast.
Benchmarked on **10,000 held-out test samples** (`python ml/benchmark.py`):

| Rank | Model | MAE | RMSE | R² | Where it runs |
|---|---|---|---|---|---|
| 🥇 1st | **XGBoost** | **0.48%** | 0.0096 | **0.9951** | Python backend |
| 🥈 2nd | **Hybrid** (physics + correction) | 1.41% | 0.0251 | 0.9673 | In the browser |
| 🥉 3rd | **Physics baseline** | 2.88% | 0.0521 | 0.8566 | In the browser |
| ~2nd | **TF.js Neural Net** | ~0.5–2% | — | ~0.97+ | In the browser |

> MAE = Mean Absolute Error as % of rated capacity. Lower is better. R² max is 1.0.

A multi-model comparison chart and accuracy summary table are rendered once any model produces predictions.

---

## Quick Start

### 1. Run the web app

```bash
node server.mjs
```

Then open: **http://127.0.0.1:4173/**

On Windows PowerShell with a custom port:

```powershell
$env:PORT = 3000
node server.mjs
```

### 2. Use the ML tab (in-browser models)

Generate a forecast, then click the **🤖 ML Models** tab.

- **Train Neural Network** — trains TF.js model in your browser (~15 s), then runs inference automatically.
- **Fit Correction Layer** — fits the hybrid polynomial correction instantly.

No Python or extra setup needed for these two.

### 3. Run the XGBoost backend (optional)

Install Python dependencies:

```bash
pip install -r ml/requirements.txt
```

Train the model (one-time, ~2 minutes):

```bash
python ml/train.py
```

Start the prediction server:

```bash
python ml/serve.py
```

Then, in the **🤖 ML Models** tab, click **Connect & Predict**. The app will call `http://127.0.0.1:5050/predict` and overlay XGBoost predictions on the comparison chart.

---

## Project Files

```
pvforcast/
├── index.html          # App markup, dashboard structure, ML Models tab UI
├── styles.css          # Responsive UI, dark mode, chart styles, ML panel styles
├── app.js              # Physics forecast, TF.js NN, hybrid correction, XGBoost API client
├── server.mjs          # Tiny static local server (Node.js)
└── ml/
    ├── train.py        # XGBoost training script — generates synthetic data and trains model
    ├── serve.py        # Flask REST API server — exposes /predict and /health endpoints
    ├── benchmark.py    # Accuracy benchmark — compares all models on 10,000 test samples
    ├── requirements.txt
    ├── xgboost_model.json   # Saved model (created after running train.py)
    └── model_meta.json      # Model metadata (R², MAE)
```

---

## CSV Format

CSV uploads for historical irradiance should include these columns:

```csv
date,irradiance,temperature
2026-04-01,5.4,28.1
2026-04-02,4.9,27.3
```

`irradiance` is expected as daily kWh/m².

---

## ML Model Details

### TF.js Neural Network
- **Architecture:** Dense(64, ReLU) → Dropout(0.1) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)
- **Input features (14):** hour (sin/cos), day-of-year (sin/cos), solar elevation, irradiance, cloud cover, temperature, wind speed, tilt, azimuth diff, latitude, efficiency, losses
- **Training data:** 8,000 synthetic samples generated in-browser using an accurate physics model
- **Optimizer:** Adam (lr = 0.002), 60 epochs, batch size 256
- **Expected accuracy:** MAE ~0.5–2%, R² ~0.97+

### Hybrid Physics + ML
- Runs the existing physics model to get a base prediction
- Fits a 3-variable polynomial correction: `Δpower = α·cloud² + β·temp_excess + γ·√irradiance`
- Coefficients solved via Gauss-Jordan elimination (least squares, no external library)
- **Benchmark accuracy: MAE 1.41%, RMSE 0.0251, R² 0.9673**
- Instantly applied — no training delay
- **6× more accurate than physics baseline**

### XGBoost Backend ⭐ Best Model
- **Architecture:** XGBRegressor, 600 trees, max_depth 7, learning_rate 0.06
- **Training data:** 60,000 synthetic samples, 85/15 train/test split
- **Target:** normalised power (kW / capacity) ranging 0–1.1
- **Features (14):** same feature set as the TF.js model
- **Benchmark accuracy: MAE 0.48%, RMSE 0.0096, R² 0.9951**
- **6× more accurate than physics, 3× more accurate than Hybrid**
- **Serving:** Flask REST API at `http://127.0.0.1:5050`

### Physics Baseline (reference)
- Formula-based model with simplified cloud/tilt/azimuth factors
- **Benchmark accuracy: MAE 2.88%, RMSE 0.0521, R² 0.8566**
- Main weakness: cloud derating formula (`1 - cloud/430`) only reduces output by 23% at 100% cloud cover

---

## Benchmark

Run the accuracy benchmark anytime to compare all models:

```bash
python ml/benchmark.py
```

Sample output:

```
========================================================================
  RANK   MODEL                          MAE %     RMSE       R2
========================================================================
  1st    XGBoost (Python backend)       0.48%    0.0096    0.9951 <-- BEST
  2nd    Hybrid (physics + correction)  1.41%    0.0251    0.9673
  3rd    Physics (app baseline)         2.88%    0.0521    0.8566
========================================================================
```

---

## Notes

- If Open-Meteo or NASA POWER is unavailable, the app falls back to a local synthetic weather or historical profile so the dashboard still works offline.
- The XGBoost server requires the model to be trained first (`python ml/train.py`). The trained model is saved to `ml/xgboost_model.json`.
- TF.js training runs on your CPU via WebGL/WASM — performance depends on your browser and hardware.
- On first forecast generation, the TF.js model is **auto-trained silently** (~15 s) and applied as the primary forecast.
- XGBoost becomes the primary model automatically once connected via the ML Models tab.
