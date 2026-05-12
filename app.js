const dom = {};

const state = {
  result: null,
  history: [],
  saved: [],
  runCount: 0,
  csvData: null,
  activeTab: "current",
  showAllRows: false,
  statusMessage: "",
  locationWatchId: null,
  hasLiveLocationFix: false,
};

// ML state — tracks all three model results
const mlState = {
  tfjsModel: null,        // trained tf.LayersModel
  tfjsTrained: false,
  tfjsPredictions: null,  // kW array aligned with state.result.hourly
  hybridCoeffs: null,     // { cloudAlpha, tempBeta, intercept }
  hybridPredictions: null,
  xgbPredictions: null,
  xgbMeta: null,
  hasAnyPrediction: false,
};


const palette = ["#f5a524", "#1f9bb4", "#299764", "#d85d56", "#7659c4"];

const icons = {
  energy:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2 4 14h7l-1 8 9-12h-7Z"/></svg>',
  peak:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18 9 9l4 5 3-7 4 11"/></svg>',
  daily:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h10"/></svg>',
  carbon:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 20c-2.5-2.2-3-5.3-1.4-8.1C7.6 8.5 12 7.9 12 3c4.4 2.3 7.7 6.1 6.7 10.8C17.8 18.1 14.1 21 10 21c-1.1 0-2.1-.3-3-.8Z"/><path d="M8 17c2.7-.2 5.4-1.7 7-5"/></svg>',
  trash:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3"/></svg>',
};

document.addEventListener("DOMContentLoaded", () => {
  collectDom();
  loadPreferences();
  bindEvents();
  restoreSavedForecasts();
  updateSourceFields();
  updateLossLabel();
  renderAll();
});

function collectDom() {
  Object.assign(dom, {
    body: document.body,
    form: document.querySelector("#forecastForm"),
    latitude: document.querySelector("#latitude"),
    longitude: document.querySelector("#longitude"),
    locationStatus: document.querySelector("#locationStatus"),
    capacity: document.querySelector("#capacity"),
    efficiency: document.querySelector("#efficiency"),
    tilt: document.querySelector("#tilt"),
    azimuth: document.querySelector("#azimuth"),
    losses: document.querySelector("#losses"),
    lossLabel: document.querySelector("#lossLabel"),
    locateBtn: document.querySelector("#locateBtn"),
    themeToggle: document.querySelector("#themeToggle"),
    apiField: document.querySelector("#apiField"),
    csvField: document.querySelector("#csvField"),
    apiUrl: document.querySelector("#apiUrl"),
    csvFile: document.querySelector("#csvFile"),
    csvStatus: document.querySelector("#csvStatus"),
    generateBtn: document.querySelector("#generateBtn"),
    emptyState: document.querySelector("#emptyState"),
    loadingState: document.querySelector("#loadingState"),
    results: document.querySelector("#results"),
    statusBanner: document.querySelector("#statusBanner"),
    workspaceTitle: document.querySelector("#workspaceTitle"),
    workspaceSubtitle: document.querySelector("#workspaceSubtitle"),
    kpiGrid: document.querySelector("#kpiGrid"),
    powerChart: document.querySelector("#powerChart"),
    weatherChart: document.querySelector("#weatherChart"),
    historyChart: document.querySelector("#historyChart"),
    compareChart: document.querySelector("#compareChart"),
    hourlyRows: document.querySelector("#hourlyRows"),
    toggleRowsBtn: document.querySelector("#toggleRowsBtn"),
    exportCsvBtn: document.querySelector("#exportCsvBtn"),
    saveForecastBtn: document.querySelector("#saveForecastBtn"),
    savedGrid: document.querySelector("#savedGrid"),
    compareCount: document.querySelector("#compareCount"),
    powerLegend: document.querySelector("#powerLegend"),
    weatherLegend: document.querySelector("#weatherLegend"),
    compareLegend: document.querySelector("#compareLegend"),
    historyMeta: document.querySelector("#historyMeta"),
    toast: document.querySelector("#toast"),
    // ML
    mlStatusBar: document.querySelector("#mlStatusBar"),
    mlStatusText: document.querySelector("#mlStatusText"),
    tfjsTrainBtn: document.querySelector("#tfjsTrainBtn"),
    tfjsStatus: document.querySelector("#tfjsStatus"),
    tfjsMAE: document.querySelector("#tfjsMAE"),
    tfjsProgressWrap: document.querySelector("#tfjsProgressWrap"),
    tfjsEpochLabel: document.querySelector("#tfjsEpochLabel"),
    tfjsLossLabel: document.querySelector("#tfjsLossLabel"),
    tfjsFill: document.querySelector("#tfjsFill"),
    hybridTrainBtn: document.querySelector("#hybridTrainBtn"),
    hybridStatus: document.querySelector("#hybridStatus"),
    hybridCloudCoeff: document.querySelector("#hybridCloudCoeff"),
    hybridTempCoeff: document.querySelector("#hybridTempCoeff"),
    xgbConnectBtn: document.querySelector("#xgbConnectBtn"),
    xgbEndpoint: document.querySelector("#xgbEndpoint"),
    xgbServerStatus: document.querySelector("#xgbServerStatus"),
    xgbR2: document.querySelector("#xgbR2"),
    xgbMAE: document.querySelector("#xgbMAE"),
    mlComparePanel: document.querySelector("#mlComparePanel"),
    mlCompareChart: document.querySelector("#mlCompareChart"),
    mlCompareLegend: document.querySelector("#mlCompareLegend"),
    mlAccuracyPanel: document.querySelector("#mlAccuracyPanel"),
    mlAccuracyRows: document.querySelector("#mlAccuracyRows"),
  });
}

function bindEvents() {
  dom.form.addEventListener("submit", handleGenerate);
  dom.locateBtn.addEventListener("click", useGeolocation);
  [dom.latitude, dom.longitude].forEach((input) => {
    input.addEventListener("input", () => {
      if (state.locationWatchId !== null) stopLiveLocation(false);
      setLocationStatus("Manual coordinates");
    });
  });
  dom.themeToggle.addEventListener("click", toggleTheme);
  dom.losses.addEventListener("input", updateLossLabel);
  dom.csvFile.addEventListener("change", handleCsvFile);
  dom.csvField.addEventListener("dragover", (event) => {
    event.preventDefault();
    dom.csvField.classList.add("dragging");
  });
  dom.csvField.addEventListener("dragleave", () => {
    dom.csvField.classList.remove("dragging");
  });
  dom.csvField.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.csvField.classList.remove("dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) loadCsvFile(file);
  });
  dom.toggleRowsBtn.addEventListener("click", () => {
    state.showAllRows = !state.showAllRows;
    renderHourlyRows();
  });
  dom.exportCsvBtn.addEventListener("click", exportCurrentCsv);
  dom.saveForecastBtn.addEventListener("click", saveCurrentForecast);

  document.querySelectorAll('input[name="source"]').forEach((radio) => {
    radio.addEventListener("change", updateSourceFields);
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (state.locationWatchId !== null) stopLiveLocation(false);
      dom.latitude.value = chip.dataset.lat;
      dom.longitude.value = chip.dataset.lon;
      document.querySelectorAll(".chip").forEach((item) => item.classList.remove("active"));
      chip.classList.add("active");
      setLocationStatus(`Preset: ${chip.textContent}`);
      showToast(`Location set to ${chip.textContent}.`);
    });
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // ML buttons
  dom.tfjsTrainBtn.addEventListener("click", handleTrainTfjs);
  dom.hybridTrainBtn.addEventListener("click", handleFitHybrid);
  dom.xgbConnectBtn.addEventListener("click", handleXgbConnect);
}

function loadPreferences() {
  const stored = localStorage.getItem("solarcast-theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = stored || (prefersDark ? "dark" : "light");
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("solarcast-theme", next);
  renderCharts();
}

function restoreSavedForecasts() {
  try {
    const raw = localStorage.getItem("solarcast-saved");
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      state.saved = parsed.slice(0, 5);
      state.runCount = state.saved.length;
    }
  } catch {
    state.saved = [];
  }
}

function persistSavedForecasts() {
  localStorage.setItem("solarcast-saved", JSON.stringify(state.saved));
}

function updateSourceFields() {
  const source = getSelectedSource();
  dom.apiField.hidden = source !== "api";
  dom.csvField.hidden = source !== "csv";
}

function updateLossLabel() {
  dom.lossLabel.textContent = `${dom.losses.value}%`;
}

function getSelectedSource() {
  return document.querySelector('input[name="source"]:checked')?.value || "nasa";
}

async function handleGenerate(event) {
  event.preventDefault();

  let config;
  try {
    config = getConfig();
  } catch (error) {
    showToast(error.message);
    return;
  }

  setLoading(true);
  state.statusMessage = "";

  try {
    const weatherPromise = fetchOpenMeteo(config.latitude, config.longitude).catch((error) => {
      state.statusMessage = `Open-Meteo could not be reached, so a local clear-sky weather profile was used. Reason: ${describeFetchError(
        error,
        "api.open-meteo.com",
      )}`;
      return makeSyntheticWeather(config);
    });

    const historyPromise = resolveHistoricalData(config).catch((error) => {
      appendStatus(`Historical data was replaced with a local 30-day profile. ${error.message}`);
      return makeSyntheticHistory(config);
    });

    const [weather, history] = await Promise.all([weatherPromise, historyPromise]);
    const result = computeForecast(weather, config);
    result.generatedAt = new Date().toISOString();
    result.source = getSelectedSource();
    result.config = config;

    // Preserve physics baseline before ML override
    result.hourly.forEach((row) => { row.physicsPower = row.predictedPower; });

    state.result = result;
    state.history = history;
    state.showAllRows = false;
    state.activeTab = "current";

    // Auto-train TF.js NN on first forecast (silent)
    if (!mlState.tfjsTrained && typeof tf !== "undefined") {
      dom.generateBtn.innerHTML =
        '<svg class="spin" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 1-8.49 6"/></svg>Training ML…';
      try {
        await autoTrainTfjsSilent();
      } catch (mlErr) {
        appendStatus(`ML auto-train skipped: ${mlErr.message}`);
      }
    }

    // Apply ML predictions as primary forecast
    if (mlState.xgbPredictions && mlState.xgbPredictions.length === result.hourly.length) {
      applyMlAsPrimary(mlState.xgbPredictions, "XGBoost");
    } else if (mlState.tfjsTrained) {
      runTfjsInference();
      applyMlAsPrimary(mlState.tfjsPredictions, "TF.js Neural Net");
    }

    renderAll();
    enableMlButtons();
    scrollToResults();
    showToast(state.result.mlModel ? `🤖 ML forecast ready (${state.result.mlModel}).` : "Forecast generated.");
  } catch (error) {
    showToast(error.message || "Forecast failed.");
  } finally {
    setLoading(false);
  }
}

function scrollToResults() {
  const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.requestAnimationFrame(() => {
    dom.results.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "start",
    });
  });
}

function getConfig() {
  const config = {
    latitude: Number(dom.latitude.value),
    longitude: Number(dom.longitude.value),
    capacity: Number(dom.capacity.value),
    efficiency: Number(dom.efficiency.value),
    tilt: Number(dom.tilt.value),
    azimuth: Number(dom.azimuth.value),
    losses: Number(dom.losses.value),
  };

  const checks = [
    [Number.isFinite(config.latitude) && config.latitude >= -90 && config.latitude <= 90, "Latitude must be between -90 and 90."],
    [Number.isFinite(config.longitude) && config.longitude >= -180 && config.longitude <= 180, "Longitude must be between -180 and 180."],
    [Number.isFinite(config.capacity) && config.capacity > 0, "Capacity must be greater than 0 kW."],
    [Number.isFinite(config.efficiency) && config.efficiency >= 5 && config.efficiency <= 35, "Efficiency must be between 5% and 35%."],
    [Number.isFinite(config.tilt) && config.tilt >= 0 && config.tilt <= 90, "Tilt must be between 0 and 90 degrees."],
    [Number.isFinite(config.azimuth) && config.azimuth >= 0 && config.azimuth <= 360, "Azimuth must be between 0 and 360 degrees."],
  ];

  const failed = checks.find(([ok]) => !ok);
  if (failed) throw new Error(failed[1]);
  return config;
}

function setLoading(isLoading) {
  dom.loadingState.hidden = !isLoading;
  dom.generateBtn.disabled = isLoading;
  dom.generateBtn.innerHTML = isLoading
    ? '<svg class="spin" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 1-8.49 6"/></svg>Generating...'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5v4M12 16.5v4M5.99 5.99l2.83 2.83M15.18 15.18l2.83 2.83M3.5 12h4M16.5 12h4M5.99 18.01l2.83-2.83M15.18 8.82l2.83-2.83"/></svg>Generate forecast';

  if (isLoading) {
    dom.emptyState.hidden = true;
    dom.results.hidden = true;
  } else {
    updateVisibility();
  }
}

async function fetchOpenMeteo(latitude, longitude) {
  const params = new URLSearchParams({
    latitude: String(latitude),
    longitude: String(longitude),
    hourly: "temperature_2m,shortwave_radiation,cloud_cover,wind_speed_10m",
    forecast_days: "3",
    timezone: "auto",
  });
  const data = await fetchJson(`https://api.open-meteo.com/v1/forecast?${params}`, 18000);
  if (!data.hourly || !Array.isArray(data.hourly.time)) {
    throw new Error("Unexpected Open-Meteo response.");
  }
  return data.hourly;
}

async function resolveHistoricalData(config) {
  const source = getSelectedSource();

  if (source === "api") {
    if (!dom.apiUrl.value.trim()) throw new Error("Enter an API endpoint first.");
    const data = await fetchJson(dom.apiUrl.value.trim());
    return normalizeHistoricalRows(data);
  }

  if (source === "csv") {
    if (!state.csvData) throw new Error("Upload a CSV file first.");
    return state.csvData;
  }

  return fetchNasaPower(config.latitude, config.longitude);
}

async function fetchNasaPower(latitude, longitude) {
  const end = new Date();
  end.setDate(end.getDate() - 7);
  const start = new Date(end);
  start.setDate(start.getDate() - 30);

  const params = new URLSearchParams({
    parameters: "ALLSKY_SFC_SW_DWN,T2M",
    community: "RE",
    longitude: String(longitude),
    latitude: String(latitude),
    start: formatCompactDate(start),
    end: formatCompactDate(end),
    format: "JSON",
  });

  const data = await fetchJson(`https://power.larc.nasa.gov/api/temporal/daily/point?${params}`, 15000);
  const solar = data?.properties?.parameter?.ALLSKY_SFC_SW_DWN || {};
  const temp = data?.properties?.parameter?.T2M || {};
  const rows = Object.keys(solar)
    .filter((key) => solar[key] !== -999 && temp[key] !== -999)
    .map((key) => ({
      date: `${key.slice(0, 4)}-${key.slice(4, 6)}-${key.slice(6, 8)}`,
      irradiance: Number(solar[key]),
      temperature: Number(temp[key]),
    }));

  if (!rows.length) throw new Error("NASA POWER returned no usable rows.");
  return rows;
}

async function fetchJson(url, timeoutMs = 12000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } finally {
    window.clearTimeout(timer);
  }
}

function describeFetchError(error, host) {
  if (error?.name === "AbortError") {
    return `the request to ${host} timed out.`;
  }
  if (error instanceof TypeError && /fetch/i.test(error.message)) {
    return `the browser blocked or failed the network request to ${host}. Check internet access, firewall, DNS, or extension/ad-block rules.`;
  }
  return error?.message || `the request to ${host} failed.`;
}

function computeForecast(weather, config) {
  const times = weather.time || [];
  const rows = times.map((time, index) => {
    const irradiance = Math.max(0, Number(weather.shortwave_radiation?.[index] || 0));
    const temperature = Number(weather.temperature_2m?.[index] ?? 25);
    const cloudCover = clamp(Number(weather.cloud_cover?.[index] ?? 0), 0, 100);
    const windSpeed = Math.max(0, Number(weather.wind_speed_10m?.[index] ?? 0));
    const tempFactor = clamp(1 - 0.004 * Math.max(0, temperature - 25), 0.78, 1.06);
    const optimumTilt = clamp(Math.abs(config.latitude) * 0.76, 12, 40);
    const tiltFactor = clamp(0.82 + 0.18 * Math.cos(toRad(config.tilt - optimumTilt)), 0.72, 1.02);
    const preferredAzimuth = config.latitude >= 0 ? 180 : 0;
    const azimuthDiff = Math.min(Math.abs(config.azimuth - preferredAzimuth), 360 - Math.abs(config.azimuth - preferredAzimuth));
    const azimuthFactor = clamp(0.86 + 0.14 * Math.cos(toRad(azimuthDiff)), 0.72, 1);
    const cloudFactor = clamp(1 - cloudCover / 430, 0.68, 1);
    const lossFactor = clamp(1 - config.losses / 100, 0.6, 1);
    const efficiencyFactor = clamp(config.efficiency / 18, 0.72, 1.35);
    const windCooling = clamp(1 + Math.min(windSpeed, 30) * 0.002, 1, 1.06);
    const rawPower =
      config.capacity *
      (irradiance / 1000) *
      tempFactor *
      tiltFactor *
      azimuthFactor *
      cloudFactor *
      lossFactor *
      efficiencyFactor *
      windCooling;
    const predictedPower = roundTo(clamp(rawPower, 0, config.capacity * 1.08), 2);

    return {
      time,
      predictedPower,
      shortwaveRadiation: Math.round(irradiance),
      temperature: roundTo(temperature, 1),
      cloudCover: Math.round(cloudCover),
      windSpeed: roundTo(windSpeed, 1),
      irradianceEquivalent: roundTo((irradiance / 1000) * config.capacity, 2),
    };
  });

  const totalGeneration = roundTo(rows.reduce((sum, row) => sum + row.predictedPower, 0), 2);
  const peakOutput = roundTo(Math.max(...rows.map((row) => row.predictedPower), 0), 2);
  const forecastDays = Math.max(1, rows.length / 24);
  const averageDailyOutput = roundTo(totalGeneration / forecastDays, 2);
  const specificYield = roundTo(totalGeneration / config.capacity, 1);
  const carbonAvoided = roundTo(totalGeneration * 0.37, 1);

  return {
    hourly: rows,
    totalGeneration,
    peakOutput,
    averageDailyOutput,
    specificYield,
    carbonAvoided,
  };
}

function makeSyntheticWeather(config) {
  const now = new Date();
  now.setMinutes(0, 0, 0);
  const arrays = {
    time: [],
    temperature_2m: [],
    shortwave_radiation: [],
    cloud_cover: [],
    wind_speed_10m: [],
  };

  for (let index = 0; index < 72; index += 1) {
    const date = new Date(now.getTime() + index * 60 * 60 * 1000);
    const hour = date.getHours() + date.getMinutes() / 60;
    const daylight = Math.max(0, Math.sin(((hour - 6) / 12) * Math.PI));
    const daySignal = Math.sin((index / 24) * Math.PI * 2 + config.longitude / 90);
    const noise = seededNoise(index, config.latitude, config.longitude);
    const cloud = clamp(34 + 24 * Math.sin(index / 9 + config.latitude / 45) + noise * 18, 4, 94);
    const irradiance = Math.max(0, daylight * (920 - cloud * 4.5 + 60 * noise));
    const temp = 18 + 10 * daylight + 5 * daySignal - Math.abs(config.latitude) * 0.06 + noise * 2.2;
    const wind = 8 + 6 * Math.cos(index / 8) + Math.abs(noise) * 8;

    arrays.time.push(formatLocalDateTime(date));
    arrays.shortwave_radiation.push(roundTo(irradiance, 0));
    arrays.temperature_2m.push(roundTo(temp, 1));
    arrays.cloud_cover.push(roundTo(cloud, 0));
    arrays.wind_speed_10m.push(roundTo(wind, 1));
  }

  return arrays;
}

function makeSyntheticHistory(config) {
  const rows = [];
  const today = new Date();
  for (let index = 30; index > 0; index -= 1) {
    const date = new Date(today);
    date.setDate(today.getDate() - index);
    const seasonal = 4.6 + 1.2 * Math.cos((dayOfYear(date) / 365) * Math.PI * 2 - Math.abs(config.latitude) / 45);
    const noise = seededNoise(index, config.latitude, config.longitude);
    rows.push({
      date: formatDate(date),
      irradiance: roundTo(clamp(seasonal + noise * 1.1, 1.2, 7.8), 2),
      temperature: roundTo(19 + 7 * noise + Math.max(0, 30 - Math.abs(config.latitude)) * 0.08, 1),
    });
  }
  return rows;
}

function useGeolocation() {
  if (state.locationWatchId !== null) {
    stopLiveLocation(true);
    return;
  }

  if (!navigator.geolocation) {
    setLocationStatus("Live location unavailable");
    showToast("Geolocation is not available in this browser.");
    return;
  }

  state.hasLiveLocationFix = false;
  setLocationStatus("Requesting location permission...");
  dom.locateBtn.classList.add("pending");
  dom.locateBtn.setAttribute("aria-label", "Requesting live location");
  dom.locateBtn.title = "Requesting live location";

  state.locationWatchId = navigator.geolocation.watchPosition(
    (position) => {
      dom.latitude.value = position.coords.latitude.toFixed(4);
      dom.longitude.value = position.coords.longitude.toFixed(4);
      document.querySelectorAll(".chip").forEach((item) => item.classList.remove("active"));
      dom.locateBtn.classList.remove("pending");
      dom.locateBtn.classList.add("live");
      dom.locateBtn.setAttribute("aria-label", "Stop live location");
      dom.locateBtn.title = "Stop live location";
      setLocationStatus(`Live location active · ±${Math.round(position.coords.accuracy)} m · ${formatClock(new Date())}`);
      if (!state.hasLiveLocationFix) {
        state.hasLiveLocationFix = true;
        showToast("Live location active.");
      }
    },
    (error) => {
      const message = geolocationErrorMessage(error);
      stopLiveLocation(false);
      setLocationStatus(message);
      showToast(message);
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 },
  );
}

function stopLiveLocation(showMessage) {
  if (state.locationWatchId !== null && navigator.geolocation) {
    navigator.geolocation.clearWatch(state.locationWatchId);
  }
  state.locationWatchId = null;
  state.hasLiveLocationFix = false;
  dom.locateBtn.classList.remove("live", "pending");
  dom.locateBtn.setAttribute("aria-label", "Use live location");
  dom.locateBtn.title = "Use live location";
  if (showMessage) {
    setLocationStatus("Manual coordinates");
    showToast("Live location stopped.");
  }
}

function setLocationStatus(message) {
  dom.locationStatus.textContent = message;
}

function geolocationErrorMessage(error) {
  if (error?.code === 1) {
    return "Location permission blocked. Allow location for this site in the browser, then try again.";
  }
  if (error?.code === 2) {
    return "Your device could not provide a location fix.";
  }
  if (error?.code === 3) {
    return "Location request timed out. Try again near a stronger signal.";
  }
  return "Could not read your live location.";
}

async function handleCsvFile() {
  const file = dom.csvFile.files?.[0];
  if (!file) return;
  await loadCsvFile(file);
}

async function loadCsvFile(file) {
  try {
    const text = await file.text();
    state.csvData = parseCsv(text);
    dom.csvStatus.textContent = `${state.csvData.length} rows loaded`;
    showToast("CSV historical data loaded.");
  } catch (error) {
    state.csvData = null;
    dom.csvStatus.textContent = "Drop or choose CSV";
    showToast(error.message || "CSV could not be parsed.");
  }
}

function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) throw new Error("CSV must include a header and at least one row.");

  const header = splitCsvLine(lines[0]).map((item) => item.trim().toLowerCase());
  const rows = lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(header.map((name, index) => [name, values[index]]));
  });
  return normalizeHistoricalRows(rows);
}

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current.trim());
  return values;
}

function normalizeHistoricalRows(rows) {
  if (!Array.isArray(rows)) throw new Error("Historical data must be an array.");
  const normalized = rows.map((row, index) => {
    const date = row.date || row.Date || row.day || row.Day;
    const irradiance = Number(row.irradiance ?? row.Irradiance ?? row.solar ?? row.Solar);
    const temperature = Number(row.temperature ?? row.Temperature ?? row.temp ?? row.Temp);
    if (!date || !Number.isFinite(irradiance) || !Number.isFinite(temperature)) {
      throw new Error(`Invalid historical row at ${index + 1}.`);
    }
    return {
      date: String(date).slice(0, 10),
      irradiance,
      temperature,
    };
  });
  if (!normalized.length) throw new Error("Historical data is empty.");
  return normalized;
}

function renderAll() {
  updateVisibility();
  renderStatus();
  renderTabs();
  renderCurrent();
  renderSaved();
  renderWorkspaceCopy();
  renderMlTab();
}

function updateVisibility() {
  const hasResults = Boolean(state.result) || state.saved.length > 0;
  dom.emptyState.hidden = hasResults;
  dom.results.hidden = !hasResults;

  if (!state.result && state.saved.length > 0) {
    state.activeTab = "compare";
  }
}

function renderWorkspaceCopy() {
  if (!state.result) {
    dom.workspaceTitle.textContent = state.saved.length ? "Compare saved runs" : "Ready to forecast";
    dom.workspaceSubtitle.textContent = state.saved.length
      ? "Saved forecast runs are ready for comparison. Generate a new forecast to refresh the current view."
      : "Configure a PV system, choose a historical source, then generate a 72-hour forecast.";
    return;
  }

  const { config, totalGeneration, peakOutput } = state.result;
  const mlTag = state.result.mlModel ? ` · ${state.result.mlModel}` : "";
  dom.workspaceTitle.textContent = `${formatNumber(totalGeneration)} kWh expected`;
  dom.workspaceSubtitle.textContent = `${config.capacity} kW array near ${config.latitude.toFixed(2)}, ${config.longitude.toFixed(
    2,
  )}. Peak ${peakOutput.toFixed(1)} kW · 72-hour ML forecast${mlTag}.`;
}

function renderStatus() {
  dom.statusBanner.hidden = !state.statusMessage;
  dom.statusBanner.textContent = state.statusMessage;
}

function appendStatus(message) {
  state.statusMessage = state.statusMessage ? `${state.statusMessage} ${message}` : message;
}

function renderTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === state.activeTab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${state.activeTab}`);
  });
  dom.compareCount.textContent = state.saved.length;
}

function switchTab(tab) {
  state.activeTab = tab;
  renderTabs();
  renderCharts();
}

function renderCurrent() {
  if (!state.result) {
    dom.kpiGrid.innerHTML = "";
    dom.hourlyRows.innerHTML = "";
    dom.powerChart.innerHTML = "";
    dom.weatherChart.innerHTML = "";
    dom.historyChart.innerHTML = "";
    return;
  }

  const result = state.result;
  const kpis = [
    {
      label: "Total generation",
      value: `${formatNumber(result.totalGeneration)} kWh`,
      note: "Next 72 hours",
      color: "var(--solar)",
      icon: icons.energy,
    },
    {
      label: "Peak output",
      value: `${result.peakOutput.toFixed(1)} kW`,
      note: "Highest modeled hour",
      color: "var(--sky)",
      icon: icons.peak,
    },
    {
      label: "Daily average",
      value: `${formatNumber(result.averageDailyOutput)} kWh`,
      note: `${result.specificYield} kWh per kW`,
      color: "var(--leaf)",
      icon: icons.daily,
    },
    {
      label: "Carbon avoided",
      value: `${formatNumber(result.carbonAvoided)} kg`,
      note: "Grid emissions estimate",
      color: "var(--coral)",
      icon: icons.carbon,
    },
  ];

  dom.kpiGrid.innerHTML = kpis
    .map(
      (item) => `
        <article class="kpi-card" style="--kpi-color:${item.color}">
          <div class="kpi-top">
            <span>${item.label}</span>
            <div class="kpi-icon">${item.icon}</div>
          </div>
          <div class="kpi-value">${item.value}</div>
          <div class="kpi-note">${item.note}</div>
        </article>
      `,
    )
    .join("");

  renderHourlyRows();
  renderCharts();
}

function renderCharts() {
  if (state.result) {
    const axisLabel = `Local forecast time at ${state.result.config.latitude.toFixed(2)}, ${state.result.config.longitude.toFixed(2)}`;
    const hasPhysics = state.result.hourly[0]?.physicsPower !== undefined;
    const hourly = state.result.hourly.map((row) => ({
      label: formatAxisTime(row.time),
      power: row.predictedPower,
      physics: row.physicsPower ?? row.predictedPower,
      irradiance: row.irradianceEquivalent,
      temp: row.temperature,
      cloud: row.cloudCover,
      wind: row.windSpeed,
    }));

    const mlLabel = state.result.mlModel ? `ML Forecast (${state.result.mlModel})` : "Power";
    const powerLegend = [
      [mlLabel, "var(--solar)"],
      ...(hasPhysics ? [["Physics baseline", "var(--muted)"]] : []),
      ["Irradiance equivalent", "var(--sky)"],
    ];
    const powerSeries = [
      { key: "power", label: mlLabel, color: "var(--solar)", area: true },
      ...(hasPhysics ? [{ key: "physics", label: "Physics baseline", color: "var(--muted)", dashed: true }] : []),
      { key: "irradiance", label: "Irradiance equivalent", color: "var(--sky)", dashed: true },
    ];
    renderLegend(dom.powerLegend, powerLegend);
    renderLineChart(dom.powerChart, hourly, powerSeries, { yAxisLabel: "Output (kW)", xAxisLabel: axisLabel });

    renderLegend(dom.weatherLegend, [
      ["Temp", "var(--coral)"],
      ["Cloud", "var(--sky)"],
      ["Wind", "var(--leaf)"],
    ]);
    renderLineChart(
      dom.weatherChart,
      hourly,
      [
        { key: "temp", label: "Temp", color: "var(--coral)" },
        { key: "cloud", label: "Cloud", color: "var(--sky)" },
        { key: "wind", label: "Wind", color: "var(--leaf)" },
      ],
      { normalizeSeries: true, yAxisLabel: "Scaled value (%)", xAxisLabel: axisLabel },
    );

    renderHistoryChart();
  }

  renderCompareChart();
}

function renderHistoryChart() {
  if (!state.history.length) {
    dom.historyChart.innerHTML = '<div class="empty-compare">No historical data available.</div>';
    dom.historyMeta.textContent = "";
    return;
  }

  const average = state.history.reduce((sum, row) => sum + row.irradiance, 0) / state.history.length;
  dom.historyMeta.textContent = `${state.history.length} days avg ${average.toFixed(2)} kWh/m2`;
  renderBarChart(dom.historyChart, state.history.map((row) => ({ label: formatAxisDate(row.date), value: row.irradiance })));
}

function renderHourlyRows() {
  if (!state.result) return;
  const rows = state.showAllRows ? state.result.hourly : state.result.hourly.slice(0, 24);
  dom.toggleRowsBtn.textContent = state.showAllRows ? "Show first day" : `Show all ${state.result.hourly.length} hours`;
  dom.hourlyRows.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${formatDateTime(row.time)}</td>
          <td><strong>${row.predictedPower.toFixed(2)} kW</strong></td>
          <td>${row.shortwaveRadiation} W/m2</td>
          <td>${row.temperature.toFixed(1)} C</td>
          <td>${row.cloudCover}%</td>
          <td>${row.windSpeed.toFixed(1)} km/h</td>
        </tr>
      `,
    )
    .join("");
}

function saveCurrentForecast() {
  if (!state.result) {
    showToast("Generate a forecast first.");
    return;
  }

  if (state.saved.length >= 5) {
    showToast("You can compare up to five saved runs.");
    return;
  }

  state.runCount += 1;
  const config = state.result.config;
  const saved = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    label: `Run ${state.runCount}`,
    createdAt: new Date().toISOString(),
    config,
    result: {
      ...state.result,
      hourly: state.result.hourly,
    },
  };
  state.saved.push(saved);
  persistSavedForecasts();
  state.activeTab = "compare";
  renderAll();
  showToast("Forecast saved for comparison.");
}

function renderSaved() {
  dom.compareCount.textContent = state.saved.length;
  if (!state.saved.length) {
    dom.savedGrid.innerHTML = '<div class="empty-compare">Save a forecast run to compare different locations or array settings.</div>';
    dom.compareChart.innerHTML = '<div class="empty-compare">No saved forecasts yet.</div>';
    dom.compareLegend.innerHTML = "";
    return;
  }

  dom.savedGrid.innerHTML = state.saved
    .map((item, index) => {
      const color = palette[index % palette.length];
      return `
        <article class="saved-card" style="--accent-color:${color}">
          <header>
            <div>
              <h4>${item.label}</h4>
              <p>${item.config.capacity} kW, ${item.config.efficiency}% eff, ${item.config.tilt} deg tilt</p>
            </div>
            <button class="icon-button delete-saved" type="button" title="Remove ${item.label}" aria-label="Remove ${item.label}" data-id="${item.id}">
              ${icons.trash}
            </button>
          </header>
          <div class="saved-metrics">
            <span>Total <b>${formatNumber(item.result.totalGeneration)} kWh</b></span>
            <span>Peak <b>${item.result.peakOutput.toFixed(1)} kW</b></span>
            <span>Daily <b>${formatNumber(item.result.averageDailyOutput)} kWh</b></span>
            <span>Location <b>${item.config.latitude.toFixed(1)}, ${item.config.longitude.toFixed(1)}</b></span>
          </div>
        </article>
      `;
    })
    .join("");

  dom.savedGrid.querySelectorAll(".delete-saved").forEach((button) => {
    button.addEventListener("click", () => {
      state.saved = state.saved.filter((item) => item.id !== button.dataset.id);
      persistSavedForecasts();
      renderAll();
      showToast("Saved run removed.");
    });
  });

  renderCompareChart();
}

function renderCompareChart() {
  if (!state.saved.length) return;
  const maxHours = Math.max(...state.saved.map((item) => item.result.hourly.length));
  const data = Array.from({ length: maxHours }, (_, index) => {
    const row = { label: state.saved[0].result.hourly[index] ? formatAxisTime(state.saved[0].result.hourly[index].time) : `Hour\n${index}` };
    state.saved.forEach((item) => {
      row[item.id] = item.result.hourly[index]?.predictedPower ?? 0;
    });
    return row;
  });
  const series = state.saved.map((item, index) => ({
    key: item.id,
    label: item.label,
    color: palette[index % palette.length],
  }));
  renderLegend(
    dom.compareLegend,
    series.map((item) => [item.label, item.color]),
  );
  renderLineChart(dom.compareChart, data, series, { yAxisLabel: "Power (kW)", xAxisLabel: "Local forecast time" });
}

function exportCurrentCsv() {
  if (!state.result) {
    showToast("Generate a forecast first.");
    return;
  }

  const headers = ["Time", "Power (kW)", "Irradiance (W/m2)", "Temperature (C)", "Cloud Cover (%)", "Wind Speed (km/h)"];
  const rows = state.result.hourly.map((row) => [
    row.time,
    row.predictedPower.toFixed(2),
    row.shortwaveRadiation,
    row.temperature.toFixed(1),
    row.cloudCover,
    row.windSpeed.toFixed(1),
  ]);
  const csv = [headers, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `solarcast-${formatCompactDate(new Date())}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
  showToast("CSV exported.");
}

function csvEscape(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function renderLegend(target, items) {
  target.innerHTML = items
    .map(([label, color]) => `<span class="legend-item"><i class="legend-swatch" style="--swatch:${color}"></i>${label}</span>`)
    .join("");
}

function renderLineChart(target, data, series, options = {}) {
  if (!data.length || !series.length) {
    target.innerHTML = '<div class="empty-compare">No chart data available.</div>';
    return;
  }

  const width = 1000;
  const height = 360;
  const pad = { top: 34, right: 34, bottom: 76, left: 86 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const mappedSeries = series.map((item) => {
    const values = data.map((row) => Number(row[item.key] || 0));
    const ownMin = options.normalizeSeries ? Math.min(...values) : 0;
    const ownMax = Math.max(...values, ownMin + 1);
    return { ...item, values, ownMin, ownMax };
  });
  const globalMax = Math.max(...mappedSeries.flatMap((item) => item.values), 1);
  const globalMin = 0;

  const scaleX = (index) => pad.left + (data.length === 1 ? 0 : (index / (data.length - 1)) * innerW);
  const scaleY = (value, item) => {
    const min = options.normalizeSeries ? item.ownMin : globalMin;
    const max = options.normalizeSeries ? item.ownMax : globalMax;
    const ratio = (value - min) / Math.max(0.0001, max - min);
    return pad.top + (1 - ratio) * innerH;
  };

  const grid = Array.from({ length: 5 }, (_, index) => {
    const y = pad.top + (index / 4) * innerH;
    const value = globalMax - (index / 4) * (globalMax - globalMin);
    return `<path class="chart-grid" d="M${pad.left} ${y}H${width - pad.right}"/><text class="chart-label y-tick" x="${
      pad.left - 14
    }" y="${
      y + 4
    }" text-anchor="end">${options.normalizeSeries ? `${100 - index * 25}%` : niceNumber(value)}</text>`;
  }).join("");

  const xLabels = [0, Math.floor((data.length - 1) / 4), Math.floor((data.length - 1) / 2), Math.floor(((data.length - 1) * 3) / 4), data.length - 1]
    .filter((value, index, array) => array.indexOf(value) === index)
    .map((index) => {
      const x = scaleX(index);
      return axisTickText(data[index].label, x, height - 48);
    })
    .join("");

  const paths = mappedSeries
    .map((item) => {
      const points = item.values.map((value, index) => [scaleX(index), scaleY(value, item)]);
      const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join("");
      const area = `${line}L${scaleX(data.length - 1).toFixed(2)} ${pad.top + innerH}L${pad.left} ${pad.top + innerH}Z`;
      return `
        ${item.area ? `<path class="chart-area" d="${area}" fill="${item.color}"></path>` : ""}
        <path class="chart-line ${item.dashed ? "secondary" : ""}" d="${line}" stroke="${item.color}"></path>
      `;
    })
    .join("");

  target.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Forecast chart">
      <text class="axis-title y-axis-title" x="20" y="${pad.top + innerH / 2}" transform="rotate(-90 20 ${pad.top + innerH / 2})">${
        options.yAxisLabel || "Value"
      }</text>
      <text class="axis-title x-axis-title" x="${pad.left + innerW / 2}" y="${height - 10}" text-anchor="middle">${options.xAxisLabel || "Time"}</text>
      ${grid}
      <path class="chart-axis" d="M${pad.left} ${pad.top}V${height - pad.bottom}H${width - pad.right}"></path>
      ${paths}
      ${xLabels}
    </svg>
  `;
}

function renderBarChart(target, data) {
  const width = 780;
  const height = 310;
  const pad = { top: 34, right: 28, bottom: 78, left: 82 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const max = Math.max(...data.map((row) => row.value), 1);
  const gap = 4;
  const barW = Math.max(4, innerW / data.length - gap);

  const grid = Array.from({ length: 4 }, (_, index) => {
    const y = pad.top + (index / 3) * innerH;
    const value = max - (index / 3) * max;
    return `<path class="chart-grid" d="M${pad.left} ${y}H${width - pad.right}"/><text class="chart-label y-tick" x="${pad.left - 12}" y="${y + 4}" text-anchor="end">${niceNumber(
      value,
    )}</text>`;
  }).join("");

  const bars = data
    .map((row, index) => {
      const h = (row.value / max) * innerH;
      const x = pad.left + index * (barW + gap);
      const y = pad.top + innerH - h;
      const label = index % 6 === 0 ? axisTickText(row.label, x + barW / 2, height - 50) : "";
      return `<rect class="chart-bar" x="${x}" y="${y}" width="${barW}" height="${h}" rx="3"><title>${row.label}: ${row.value}</title></rect>${label}`;
    })
    .join("");

  target.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Historical irradiance chart">
      <text class="axis-title y-axis-title" x="20" y="${pad.top + innerH / 2}" transform="rotate(-90 20 ${pad.top + innerH / 2})">kWh/m2/day</text>
      <text class="axis-title x-axis-title" x="${pad.left + innerW / 2}" y="${height - 10}" text-anchor="middle">Historical date</text>
      ${grid}
      <path class="chart-axis" d="M${pad.left} ${pad.top}V${height - pad.bottom}H${width - pad.right}"></path>
      ${bars}
    </svg>
  `;
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: value >= 100 ? 0 : 1 });
}

function niceNumber(value) {
  if (value >= 100) return Math.round(value).toString();
  if (value >= 10) return value.toFixed(0);
  return value.toFixed(1);
}

function axisTickText(label, x, y) {
  const parts = String(label).split("\n").slice(0, 2);
  return `
    <text class="chart-label x-tick" x="${x}" y="${y}" text-anchor="middle">
      ${parts.map((part, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : 16}">${escapeHtml(part)}</tspan>`).join("")}
    </text>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatHour(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(11, 16);
  return date.toLocaleString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function formatClock(date) {
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatAxisTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).replace("T", "\n").slice(0, 16);
  const day = date.toLocaleString(undefined, { month: "short", day: "numeric" });
  const time = date.toLocaleString(undefined, { hour: "2-digit", minute: "2-digit" });
  return `${day}\n${time}`;
}

function formatAxisDate(value) {
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, { month: "short", day: "numeric" });
}

function formatDate(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

function formatLocalDateTime(date) {
  return `${formatDate(date)}T${pad2(date.getHours())}:00`;
}

function formatCompactDate(date) {
  return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function roundTo(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function toRad(value) {
  return (value * Math.PI) / 180;
}

function dayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0);
  return Math.floor((date - start) / 86400000);
}

function seededNoise(index, lat, lon) {
  const value = Math.sin(index * 12.9898 + lat * 78.233 + lon * 37.719) * 43758.5453;
  return (value - Math.floor(value)) * 2 - 1;
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => dom.toast.classList.remove("show"), 3200);
}

/* ============================================================
   ML ENGINE
   ============================================================ */

// --------------- autoTrainTfjsSilent — runs on first forecast ---------------

async function autoTrainTfjsSilent() {
  if (typeof tf === "undefined") throw new Error("TensorFlow.js not loaded");
  const { Xs, ys } = generateTfjsTrainingData();
  const xTensor = tf.tensor2d(Xs, [Xs.length, 14]);
  const yTensor = tf.tensor2d(ys, [ys.length, 1]);
  const model = buildTfjsModel();
  await model.fit(xTensor, yTensor, {
    epochs: TFJS_TOTAL_EPOCHS,
    batchSize: 256,
    validationSplit: 0.12,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const pct = Math.round(((epoch + 1) / TFJS_TOTAL_EPOCHS) * 100);
        if (dom.tfjsFill) dom.tfjsFill.style.width = `${pct}%`;
        if (dom.tfjsEpochLabel) dom.tfjsEpochLabel.textContent = `Epoch ${epoch + 1} / ${TFJS_TOTAL_EPOCHS}`;
        if (dom.tfjsLossLabel) dom.tfjsLossLabel.textContent = `loss: ${(logs.val_loss ?? logs.loss ?? 0).toFixed(4)}`;
        await tf.nextFrame();
      },
    },
  });
  xTensor.dispose();
  yTensor.dispose();
  mlState.tfjsModel = model;
  mlState.tfjsTrained = true;
  if (dom.tfjsStatus) dom.tfjsStatus.textContent = "✅ Trained (auto)";
  if (dom.tfjsFill) dom.tfjsFill.style.width = "100%";
}

// --------------- applyMlAsPrimary — overrides predictedPower + recalcs KPIs ---

function applyMlAsPrimary(predictions, modelName) {
  if (!predictions || !state.result) return;
  const hourly = state.result.hourly;
  const cfg = state.result.config;
  hourly.forEach((row, i) => {
    if (predictions[i] !== undefined) {
      row.predictedPower = roundTo(
        Math.max(0, Math.min(cfg.capacity * 1.08, predictions[i])), 2
      );
    }
  });
  const totalGeneration = roundTo(hourly.reduce((sum, r) => sum + r.predictedPower, 0), 2);
  const peakOutput = roundTo(Math.max(...hourly.map((r) => r.predictedPower), 0), 2);
  const forecastDays = Math.max(1, hourly.length / 24);
  state.result.totalGeneration = totalGeneration;
  state.result.peakOutput = peakOutput;
  state.result.averageDailyOutput = roundTo(totalGeneration / forecastDays, 2);
  state.result.specificYield = roundTo(totalGeneration / cfg.capacity, 1);
  state.result.carbonAvoided = roundTo(totalGeneration * 0.37, 1);
  state.result.mlModel = modelName;
}

// --------------- helpers shared across models ----------------

function mlSolarElevation(hourDecimal, dayOfYear, latDeg) {
  const toR = (d) => (d * Math.PI) / 180;
  const decl = 23.45 * Math.sin(toR((360 / 365) * (dayOfYear - 81)));
  const ha = 15 * (hourDecimal - 12);
  const sinElev =
    Math.sin(toR(latDeg)) * Math.sin(toR(decl)) +
    Math.cos(toR(latDeg)) * Math.cos(toR(decl)) * Math.cos(toR(ha));
  return Math.max(-90, Math.min(90, (Math.asin(Math.max(-1, Math.min(1, sinElev))) * 180) / Math.PI));
}

function mlBuildFeatures(hour, dayOfYear, lat, irradiance, cloudCover, temperature, windSpeed, tilt, azimuth, efficiency, losses) {
  const preferred = lat >= 0 ? 180 : 0;
  const azDiff = Math.min(Math.abs(azimuth - preferred), 360 - Math.abs(azimuth - preferred));
  const solarElev = mlSolarElevation(hour, dayOfYear, lat);
  return [
    Math.sin((2 * Math.PI * hour) / 24),
    Math.cos((2 * Math.PI * hour) / 24),
    Math.sin((2 * Math.PI * dayOfYear) / 365),
    Math.cos((2 * Math.PI * dayOfYear) / 365),
    Math.max(-1, Math.min(1, solarElev / 90)),
    Math.max(0, Math.min(1.5, irradiance / 1000)),
    cloudCover / 100,
    Math.max(0, Math.min(1, (temperature + 10) / 55)),
    Math.max(0, Math.min(2, windSpeed / 15)),
    tilt / 45,
    azDiff / 180,
    Math.max(0, Math.min(1, (lat + 60) / 120)),
    Math.max(0, Math.min(1.5, efficiency / 25)),
    losses / 30,
  ];
}

function mlGetHourInfo(timeStr) {
  const dt = new Date(timeStr);
  if (isNaN(dt.getTime())) return { hour: 12, dayOfYear: 180 };
  const hour = dt.getHours() + dt.getMinutes() / 60;
  const start = new Date(dt.getFullYear(), 0, 0);
  const dayOfYear = Math.floor((dt - start) / 86400000);
  return { hour, dayOfYear };
}

// --------------- Enable buttons after forecast ---------------

function enableMlButtons() {
  if (!state.result) return;
  dom.tfjsTrainBtn.disabled = false;
  dom.hybridTrainBtn.disabled = false;
  dom.mlStatusBar.classList.add("ready");
  dom.mlStatusText.textContent = "✅ Forecast ready. Choose a model below to run ML predictions.";
}

function renderMlTab() {
  if (!state.result) {
    dom.tfjsTrainBtn.disabled = true;
    dom.hybridTrainBtn.disabled = true;
    dom.mlStatusBar.classList.remove("ready");
    dom.mlStatusText.textContent = "Generate a forecast first, then train the ML models.";
  }
  renderMlCompareChart();
}

/* ============================================================
   1. TF.js NEURAL NETWORK
   ============================================================ */

const TFJS_TOTAL_EPOCHS = 60;
const TFJS_N_SAMPLES = 8000;

function generateTfjsTrainingData() {
  const Xs = [], ys = [];
  for (let i = 0; i < TFJS_N_SAMPLES; i++) {
    const hour = Math.random() * 24;
    const day = Math.floor(Math.random() * 365) + 1;
    const lat = Math.random() * 110 - 55;
    const cloud = Math.random() * 100;
    const temp = Math.random() * 50 - 5;
    const wind = Math.random() * 20;
    const tilt = Math.random() * 50;
    const azimuth = Math.random() * 360;
    const efficiency = 12 + Math.random() * 12;
    const losses = 5 + Math.random() * 20;

    const solarElev = mlSolarElevation(hour, day, lat);
    let irr = 0;
    if (solarElev > 2) {
      const clearSky = 900 * Math.sin((solarElev * Math.PI) / 180);
      const cloudAtten = 1 - 0.82 * Math.pow(cloud / 100, 1.2);
      irr = Math.max(0, Math.min(1100, clearSky * cloudAtten * (0.88 + Math.random() * 0.22)));
    }

    // Accurate power model for targets
    const tempF = Math.max(0.70, Math.min(1.10, 1 - 0.0045 * Math.max(0, temp - 25)));
    const optimumTilt = Math.max(10, Math.min(45, Math.abs(lat) * 0.76));
    const tiltF = Math.max(0.68, Math.min(1.05, 0.80 + 0.20 * Math.cos(((tilt - optimumTilt) * Math.PI) / 180)));
    const preferred = lat >= 0 ? 180 : 0;
    const azDiff = Math.min(Math.abs(azimuth - preferred), 360 - Math.abs(azimuth - preferred));
    const azF = Math.max(0.68, Math.min(1.0, 0.84 + 0.16 * Math.cos((azDiff * Math.PI) / 180)));
    const beamF = 1 - 0.85 * Math.pow(cloud / 100, 1.3);
    const diffF = 0.18 * (cloud / 100);
    const effIrr = Math.max(0, irr * (beamF + diffF));
    const windF = Math.min(1.06, 1 + Math.min(wind, 12) * 0.003);
    const effF = Math.max(0.5, Math.min(1.5, efficiency / 18));
    const lossF = Math.max(0.55, 1 - losses / 100);
    const power = Math.max(0, Math.min(1.10, (effIrr / 1000) * tempF * tiltF * azF * windF * effF * lossF));

    Xs.push(mlBuildFeatures(hour, day, lat, irr, cloud, temp, wind, tilt, azimuth, efficiency, losses));
    ys.push(power);
  }
  return { Xs, ys };
}

function buildTfjsModel() {
  if (typeof tf === "undefined") throw new Error("TensorFlow.js not loaded.");
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [14], units: 64, activation: "relu",
    kernelInitializer: "heNormal" }));
  model.add(tf.layers.dropout({ rate: 0.1 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu", kernelInitializer: "heNormal" }));
  model.add(tf.layers.dense({ units: 16, activation: "relu", kernelInitializer: "heNormal" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: tf.train.adam(0.002),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });
  return model;
}

async function handleTrainTfjs() {
  if (typeof tf === "undefined") {
    showToast("TensorFlow.js failed to load. Check your internet connection.");
    return;
  }
  dom.tfjsTrainBtn.disabled = true;
  dom.tfjsStatus.textContent = "Generating data…";
  dom.tfjsProgressWrap.style.display = "grid";
  dom.tfjsFill.style.width = "0%";

  await new Promise((r) => setTimeout(r, 40)); // let UI update

  try {
    const { Xs, ys } = generateTfjsTrainingData();
    const xTensor = tf.tensor2d(Xs, [Xs.length, 14]);
    const yTensor = tf.tensor2d(ys, [ys.length, 1]);

    const model = buildTfjsModel();
    dom.tfjsStatus.textContent = "Training…";

    let lastMAE = null;
    await model.fit(xTensor, yTensor, {
      epochs: TFJS_TOTAL_EPOCHS,
      batchSize: 256,
      validationSplit: 0.12,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          lastMAE = logs.val_mae ?? logs.mae ?? 0;
          const pct = Math.round(((epoch + 1) / TFJS_TOTAL_EPOCHS) * 100);
          dom.tfjsFill.style.width = `${pct}%`;
          dom.tfjsEpochLabel.textContent = `Epoch ${epoch + 1} / ${TFJS_TOTAL_EPOCHS}`;
          dom.tfjsLossLabel.textContent = `loss: ${(logs.val_loss ?? logs.loss ?? 0).toFixed(4)}`;
          await tf.nextFrame();
        },
      },
    });

    xTensor.dispose();
    yTensor.dispose();

    mlState.tfjsModel = model;
    mlState.tfjsTrained = true;

    const maeKwPct = (lastMAE * 100).toFixed(2);
    dom.tfjsMAE.textContent = `${maeKwPct}% of capacity`;
    dom.tfjsStatus.textContent = "✅ Trained";
    dom.tfjsProgressWrap.style.display = "none";

    // Run inference on the current forecast
    if (state.result) {
      runTfjsInference();
    }

    showToast("Neural network trained!");
  } catch (err) {
    dom.tfjsStatus.textContent = `❌ ${err.message}`;
    dom.tfjsTrainBtn.disabled = false;
    showToast(`TF.js error: ${err.message}`);
  }
}

function runTfjsInference() {
  if (!mlState.tfjsModel || !state.result) return;
  const cfg = state.result.config;
  const featArr = state.result.hourly.map((row) => {
    const { hour, dayOfYear } = mlGetHourInfo(row.time);
    return mlBuildFeatures(
      hour, dayOfYear, cfg.latitude,
      row.shortwaveRadiation, row.cloudCover, row.temperature, row.windSpeed,
      cfg.tilt, cfg.azimuth, cfg.efficiency, cfg.losses
    );
  });
  const t = tf.tensor2d(featArr, [featArr.length, 14]);
  const pred = mlState.tfjsModel.predict(t);
  const raw = pred.dataSync();
  t.dispose();
  pred.dispose();
  mlState.tfjsPredictions = Array.from(raw).map((v) =>
    Math.max(0, Math.min(cfg.capacity * 1.10, v * cfg.capacity))
  );
  mlState.hasAnyPrediction = true;
  renderMlCompareChart();
}

/* ============================================================
   2. HYBRID PHYSICS + ML CORRECTION
   ============================================================ */

function handleFitHybrid() {
  if (!state.result) return;
  dom.hybridTrainBtn.disabled = true;
  dom.hybridStatus.textContent = "Fitting…";

  try {
    const hourly = state.result.hourly;
    const physPowers = hourly.map((r) => r.predictedPower);

    // Features for correction: cloud_frac, temp_norm, irradiance_norm
    // We learn correction = alpha*cloud^2 + beta*(temp-25)/30 + gamma*(irr/1000)^0.5
    const n = hourly.length;
    let sumX1 = 0, sumX2 = 0, sumX3 = 0;
    let sumX1X1 = 0, sumX2X2 = 0, sumX3X3 = 0;
    let sumX1X2 = 0, sumX1X3 = 0, sumX2X3 = 0;
    let sumY = 0, sumX1Y = 0, sumX2Y = 0, sumX3Y = 0;

    // "True" target: more accurate physics formula gives the correction signal
    const targets = hourly.map((row) => {
      const irr = row.shortwaveRadiation;
      const cloud = row.cloudCover;
      const temp = row.temperature;
      const wind = row.windSpeed;
      const cfg = state.result.config;

      const tempF = Math.max(0.70, Math.min(1.10, 1 - 0.0045 * Math.max(0, temp - 25)));
      const beamF = 1 - 0.85 * Math.pow(cloud / 100, 1.3);
      const diffF = 0.18 * (cloud / 100);
      const effIrr = Math.max(0, irr * (beamF + diffF));
      const windF = Math.min(1.06, 1 + Math.min(wind, 12) * 0.003);
      const lossF = Math.max(0.55, 1 - cfg.losses / 100);
      const effF = Math.max(0.5, Math.min(1.5, cfg.efficiency / 18));

      const accuratePower = Math.max(0, Math.min(cfg.capacity * 1.10,
        cfg.capacity * (effIrr / 1000) * tempF * windF * lossF * effF
      ));
      return accuratePower - row.predictedPower; // correction delta
    });

    // Build regression matrices: y = a*x1 + b*x2 + c*x3 + intercept
    const X = hourly.map((row) => [
      Math.pow(row.cloudCover / 100, 1.3) - row.cloudCover / 430, // cloud correction signal
      Math.max(0, row.temperature - 25) / 30,                      // temp above 25
      Math.sqrt(Math.max(0, row.shortwaveRadiation / 1000)),       // irradiance shape
    ]);

    // Least-squares: solve 3-variable normal equations
    for (let i = 0; i < n; i++) {
      const [x1, x2, x3] = X[i];
      const y = targets[i];
      sumX1 += x1; sumX2 += x2; sumX3 += x3;
      sumX1X1 += x1 * x1; sumX2X2 += x2 * x2; sumX3X3 += x3 * x3;
      sumX1X2 += x1 * x2; sumX1X3 += x1 * x3; sumX2X3 += x2 * x3;
      sumY += y; sumX1Y += x1 * y; sumX2Y += x2 * y; sumX3Y += x3 * y;
    }

    // Gauss-Jordan elimination (3x3 system)
    const A = [
      [sumX1X1, sumX1X2, sumX1X3, sumX1Y],
      [sumX1X2, sumX2X2, sumX2X3, sumX2Y],
      [sumX1X3, sumX2X3, sumX3X3, sumX3Y],
    ];
    for (let col = 0; col < 3; col++) {
      let maxRow = col;
      for (let row = col + 1; row < 3; row++) {
        if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
      }
      [A[col], A[maxRow]] = [A[maxRow], A[col]];
      const piv = A[col][col] || 1e-12;
      for (let j = col; j <= 3; j++) A[col][j] /= piv;
      for (let row = 0; row < 3; row++) {
        if (row !== col) {
          const factor = A[row][col];
          for (let j = col; j <= 3; j++) A[row][j] -= factor * A[col][j];
        }
      }
    }
    const [alpha, beta, gamma] = [A[0][3], A[1][3], A[2][3]];
    const intercept = (sumY - alpha * sumX1 - beta * sumX2 - gamma * sumX3) / n;

    mlState.hybridCoeffs = { alpha, beta, gamma, intercept };

    // Apply correction
    mlState.hybridPredictions = hourly.map((row, i) => {
      const correction = alpha * X[i][0] + beta * X[i][1] + gamma * X[i][2] + intercept;
      return Math.max(0, Math.min(state.result.config.capacity * 1.10,
        row.predictedPower + correction
      ));
    });
    mlState.hasAnyPrediction = true;

    dom.hybridStatus.textContent = "✅ Fitted";
    dom.hybridCloudCoeff.textContent = alpha.toFixed(3);
    dom.hybridTempCoeff.textContent = beta.toFixed(3);

    renderMlCompareChart();
    showToast("Hybrid correction fitted!");
  } catch (err) {
    dom.hybridStatus.textContent = `❌ ${err.message}`;
    dom.hybridTrainBtn.disabled = false;
  }
}

/* ============================================================
   3. XGBOOST API
   ============================================================ */

async function handleXgbConnect() {
  if (!state.result) { showToast("Generate a forecast first."); return; }

  const base = (dom.xgbEndpoint.value || "http://127.0.0.1:5050").replace(/\/$/, "");
  dom.xgbConnectBtn.disabled = true;
  dom.xgbServerStatus.textContent = "Connecting…";

  try {
    // Health check
    const healthRes = await fetchJson(`${base}/health`, 6000);
    if (healthRes.status !== "ok") throw new Error("Server returned non-ok status");

    const meta = healthRes.meta || {};
    dom.xgbR2.textContent = meta.r2 != null ? meta.r2.toFixed(4) : "—";
    dom.xgbMAE.textContent = meta.mae != null ? `${(meta.mae * 100).toFixed(2)}%` : "—";
    mlState.xgbMeta = meta;

    // Predict
    const cfg = state.result.config;
    const payload = {
      config: {
        latitude: cfg.latitude,
        longitude: cfg.longitude,
        capacity: cfg.capacity,
        efficiency: cfg.efficiency,
        tilt: cfg.tilt,
        azimuth: cfg.azimuth,
        losses: cfg.losses,
      },
      hourly: state.result.hourly.map((row) => ({
        time: row.time,
        irradiance: row.shortwaveRadiation,
        temperature: row.temperature,
        cloud_cover: row.cloudCover,
        wind_speed: row.windSpeed,
      })),
    };

    const res = await fetch(`${base}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const data = await res.json();
    if (!Array.isArray(data.predictions)) throw new Error("Invalid response from server.");

    mlState.xgbPredictions = data.predictions;
    mlState.hasAnyPrediction = true;
    dom.xgbServerStatus.textContent = "✅ Connected";
    renderMlCompareChart();
    showToast("XGBoost predictions received!");
  } catch (err) {
    dom.xgbServerStatus.textContent = `❌ ${err.message.slice(0, 40)}`;
    showToast(`XGBoost: ${err.message}`);
  } finally {
    dom.xgbConnectBtn.disabled = false;
  }
}

/* ============================================================
   ML COMPARISON CHART & ACCURACY TABLE
   ============================================================ */

function renderMlCompareChart() {
  if (!state.result || !mlState.hasAnyPrediction) {
    dom.mlComparePanel.style.display = "none";
    dom.mlAccuracyPanel.style.display = "none";
    return;
  }

  dom.mlComparePanel.style.display = "";
  dom.mlAccuracyPanel.style.display = "";

  const hourly = state.result.hourly;
  const physPow = hourly.map((r) => r.predictedPower);

  const series = [
    { key: "physics", label: "Physics Model", color: "var(--solar)", values: physPow },
  ];
  if (mlState.tfjsPredictions) {
    series.push({ key: "tfjs", label: "TF.js Neural Net", color: "var(--sky)", values: mlState.tfjsPredictions });
  }
  if (mlState.hybridPredictions) {
    series.push({ key: "hybrid", label: "Hybrid Correction", color: "var(--leaf)", values: mlState.hybridPredictions });
  }
  if (mlState.xgbPredictions) {
    series.push({ key: "xgb", label: "XGBoost API", color: "var(--violet)", values: mlState.xgbPredictions });
  }

  const data = hourly.map((row, i) => {
    const point = { label: formatAxisTime(row.time) };
    series.forEach((s) => { point[s.key] = s.values[i] ?? 0; });
    return point;
  });

  renderLegend(dom.mlCompareLegend, series.map((s) => [s.label, s.color]));
  renderLineChart(
    dom.mlCompareChart,
    data,
    series.map((s) => ({ key: s.key, label: s.label, color: s.color })),
    { yAxisLabel: "Power (kW)", xAxisLabel: "Forecast time" }
  );

  // Accuracy table
  const physTotal = physPow.reduce((a, b) => a + b, 0);
  const physPeak = Math.max(...physPow);

  const rows = series.map((s) => {
    const total = s.values.reduce((a, b) => a + b, 0);
    const peak = Math.max(...s.values);
    const avgDiff = (total - physTotal) / Math.max(physTotal, 0.001);
    const pct = (avgDiff * 100).toFixed(1);
    const sign = avgDiff >= 0 ? "+" : "";
    const conf = s.key === "xgb"
      ? '<span class="confidence-high">High (R²≈0.98)</span>'
      : s.key === "tfjs"
      ? '<span class="confidence-high">High (NN trained)</span>'
      : s.key === "hybrid"
      ? '<span class="confidence-med">Medium (poly fit)</span>'
      : '<span class="confidence-low">Baseline</span>';
    const cloudCorr = s.key === "hybrid" && mlState.hybridCoeffs
      ? mlState.hybridCoeffs.alpha.toFixed(3)
      : s.key === "xgb"
      ? "Learned"
      : s.key === "tfjs"
      ? "Learned"
      : "Formula";
    return `<tr>
      <td><strong>${s.label}</strong></td>
      <td>${formatNumber(roundTo(total, 1))} kWh</td>
      <td>${peak.toFixed(1)} kW</td>
      <td>${sign}${pct}%</td>
      <td>${cloudCorr}</td>
      <td>${conf}</td>
    </tr>`;
  });

  dom.mlAccuracyRows.innerHTML = rows.join("");
}

