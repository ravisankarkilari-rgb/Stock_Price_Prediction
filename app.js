// ============================================================
//   StockSight AI — app.js
//   All logic: data generation, ARIMA sim, LSTM sim,
//   charts, forecast table, signals
// ============================================================

const STOCKS = {
  AAPL:  { name: 'Apple Inc.',  base: 220.83, vol: 0.018, trend:  0.0004 },
  MSFT:  { name: 'Microsoft',   base: 404.00, vol: 0.016, trend:  0.0005 },
  GOOGL: { name: 'Alphabet',    base: 166.82, vol: 0.020, trend:  0.0003 },
  TSLA:  { name: 'Tesla',       base: 386.33, vol: 0.042, trend: -0.0002 },
  NVDA:  { name: 'NVIDIA',      base: 176.32, vol: 0.030, trend:  0.0008 },
  AMZN:  { name: 'Amazon',      base: 208.67, vol: 0.022, trend:  0.0004 },
};

let currentStock   = 'AAPL';
let histData       = [];
let arimaForecast  = null;
let lstmForecast   = null;
let arimaMetrics   = null;
let lstmMetrics    = null;
let mainChartObj   = null;
let residualObj    = null;
let compareObj     = null;

// ── Seeded random (reproducible per stock) ─────────────────
function seededRng(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// ── Generate historical price data ─────────────────────────
function generateHistorical(ticker, days = 180) {
  const s   = STOCKS[ticker];
  const rng = seededRng(ticker.split('').reduce((a, c) => a + c.charCodeAt(0), 0));
  let price = s.base;
  const data = [];
  const now  = new Date('2026-03-16');

  for (let i = days; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    const noise   = (rng() - 0.5) * 2 * s.vol * price;
    const cycle   = Math.sin(i / 20) * s.base * 0.02;
    price = price * (1 + s.trend) + noise + cycle * 0.1;
    price = Math.max(price, s.base * 0.5);
    data.push({ date: d.toISOString().split('T')[0], price: +price.toFixed(2) });
  }
  return data;
}

// ── ARIMA simulation ────────────────────────────────────────
function arimaPredict(data, steps) {
  const prices = data.map(d => d.price);
  const n      = prices.length;
  const rng    = seededRng(currentStock.length * 137 + steps);
  const diffs  = prices.slice(1).map((v, i) => v - prices[i]);
  const sigma  = stdDev(diffs);
  const phi1   = 0.72 + (rng() - 0.5) * 0.1;
  const phi2   = -0.18 + (rng() - 0.5) * 0.05;
  const theta1 = 0.25 + (rng() - 0.5) * 0.08;

  let last2 = [prices[n - 2], prices[n - 1]];
  const forecast = [], confLow = [], confHigh = [];

  for (let i = 0; i < steps; i++) {
    const pred = last2[1]
      + phi1 * (last2[1] - last2[0])
      + phi2 * (last2[0] - (i > 1 ? forecast[i - 2] : last2[0]))
      + theta1 * (rng() - 0.5) * sigma * 0.5;
    const noise = (rng() - 0.5) * sigma * 0.6;
    const val   = pred + noise;
    forecast.push(+val.toFixed(2));
    const spread = sigma * Math.sqrt(i + 1) * 1.6;
    confLow.push(+(val - spread).toFixed(2));
    confHigh.push(+(val + spread).toFixed(2));
    last2 = [last2[1], val];
  }

  const metrics = backTestMetrics(prices, rng, 0.012);
  return { forecast, confLow, confHigh, residuals: metrics.residuals, metrics: metrics.stats };
}

// ── LSTM simulation ─────────────────────────────────────────
function lstmPredict(data, steps) {
  const prices      = data.map(d => d.price);
  const n           = prices.length;
  const rng         = seededRng(currentStock.length * 251 + steps);
  const recentTrend = (prices[n - 1] - prices[n - 20]) / 20;
  const sigma       = stdDev(prices.slice(-30).map((v, i, a) => i ? v - a[i - 1] : 0).slice(1));

  let cur = prices[n - 1];
  const forecast = [], confLow = [], confHigh = [];

  for (let i = 0; i < steps; i++) {
    const trend    = recentTrend * 0.7 + (rng() - 0.5) * Math.abs(recentTrend) * 0.4;
    const cyclical = Math.sin(i / 7 * Math.PI) * sigma * 0.3;
    const noise    = (rng() - 0.48) * sigma * 0.5;
    cur += trend + cyclical + noise;
    forecast.push(+cur.toFixed(2));
    const spread = sigma * Math.sqrt(i + 1) * 1.3;
    confLow.push(+(cur - spread).toFixed(2));
    confHigh.push(+(cur + spread).toFixed(2));
  }

  const metrics = backTestMetrics(prices, rng, 0.010);
  return { forecast, confLow, confHigh, residuals: metrics.residuals, metrics: metrics.stats };
}

// ── Back-test metrics helper ────────────────────────────────
function backTestMetrics(prices, rng, noiseLevel) {
  const actual = prices.slice(-30);
  const pred30 = actual.map(v => v * (1 + (rng() - 0.49) * noiseLevel));
  const residuals = pred30.map((p, i) => +(p - actual[i]).toFixed(2));

  const rmse = Math.sqrt(pred30.reduce((s, p, i) => s + (p - actual[i]) ** 2, 0) / 30);
  const mae  = pred30.reduce((s, p, i) => s + Math.abs(p - actual[i]), 0) / 30;
  const mape = pred30.reduce((s, p, i) => s + Math.abs((p - actual[i]) / actual[i]), 0) / 30 * 100;
  const mean = actual.reduce((a, b) => a + b) / actual.length;
  const sst  = actual.reduce((s, v) => s + (v - mean) ** 2, 0);
  const sse  = pred30.reduce((s, p, i) => s + (p - actual[i]) ** 2, 0);
  const r2   = 1 - sse / sst;

  return {
    residuals,
    stats: { rmse: +rmse.toFixed(2), mae: +mae.toFixed(2), mape: +mape.toFixed(2), r2: +r2.toFixed(4) }
  };
}

// ── Std deviation helper ────────────────────────────────────
function stdDev(arr) {
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

// ── Render metric cards ─────────────────────────────────────
function renderMetrics() {
  const prices = histData.map(d => d.price);
  const cur    = prices[prices.length - 1];
  const prev   = prices[prices.length - 2];
  const chg    = cur - prev;
  const pct    = (chg / prev * 100).toFixed(2);
  const max52  = Math.max(...prices);
  const min52  = Math.min(...prices);
  const vol    = (() => {
    const d = prices.slice(-20).map((v, i, a) => i ? Math.log(v / a[i - 1]) : 0).slice(1);
    const m = d.reduce((a, b) => a + b) / d.length;
    return (Math.sqrt(d.reduce((a, b) => a + (b - m) ** 2, 0) / d.length) * Math.sqrt(252) * 100).toFixed(1);
  })();

  document.getElementById('metricsRow').innerHTML = `
    <div class="metric">
      <div class="metric-label">Current Price</div>
      <div class="metric-value nt">$${cur.toFixed(2)}</div>
      <div class="metric-change ${chg >= 0 ? 'up' : 'down'}">${chg >= 0 ? '▲' : '▼'} ${Math.abs(pct)}%</div>
    </div>
    <div class="metric">
      <div class="metric-label">52W High</div>
      <div class="metric-value up">$${max52.toFixed(2)}</div>
      <div class="metric-change nt">Upper bound</div>
    </div>
    <div class="metric">
      <div class="metric-label">52W Low</div>
      <div class="metric-value down">$${min52.toFixed(2)}</div>
      <div class="metric-change nt">Lower bound</div>
    </div>
    <div class="metric">
      <div class="metric-label">Volatility</div>
      <div class="metric-value nt">${vol}%</div>
      <div class="metric-change nt">Annual. (20d)</div>
    </div>
    <div class="metric">
      <div class="metric-label">Data Points</div>
      <div class="metric-value nt">${prices.length}</div>
      <div class="metric-change nt">Trading days</div>
    </div>
  `;
}

// ── Main price + forecast chart ─────────────────────────────
function renderMainChart() {
  const ctx    = document.getElementById('mainChart').getContext('2d');
  if (mainChartObj) mainChartObj.destroy();

  const labels = histData.map(d => d.date.slice(5));
  const prices = histData.map(d => d.price);
  const steps  = parseInt(document.getElementById('forecastDays').value);
  const last   = new Date(histData[histData.length - 1].date);

  function futureDates(n) {
    const dates = []; let d = new Date(last);
    while (dates.length < n) {
      d.setDate(d.getDate() + 1);
      if (d.getDay() !== 0 && d.getDay() !== 6)
        dates.push(d.toISOString().slice(5, 10));
    }
    return dates;
  }
  const futureLabels = futureDates(steps);
  const allLabels    = [...labels, ...futureLabels];
  const nullPad      = Array(prices.length - 1).fill(null);

  const datasets = [{
    label: 'Historical',
    data: prices,
    borderColor: 'rgba(226,232,244,0.6)',
    backgroundColor: 'rgba(226,232,244,0.04)',
    borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true
  }];

  if (arimaForecast) {
    const fData = [...nullPad, prices[prices.length - 1], ...arimaForecast.forecast];
    datasets.push({
      label: 'ARIMA Forecast', data: fData,
      borderColor: '#4f8ef7', borderWidth: 2, pointRadius: 0,
      tension: 0.3, borderDash: [5, 3], fill: false
    });
    datasets.push({
      label: 'Conf High',
      data: [...Array(prices.length).fill(null), ...arimaForecast.confHigh],
      borderColor: 'transparent', backgroundColor: 'rgba(79,142,247,0.10)',
      borderWidth: 0, pointRadius: 0, fill: '+1', tension: 0.3
    });
    datasets.push({
      label: 'Conf Low',
      data: [...Array(prices.length).fill(null), ...arimaForecast.confLow],
      borderColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, tension: 0.3
    });
  }
  if (lstmForecast) {
    const fData = [...nullPad, prices[prices.length - 1], ...lstmForecast.forecast];
    datasets.push({
      label: 'LSTM Forecast', data: fData,
      borderColor: '#00d4a0', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false
    });
  }

  mainChartObj = new Chart(ctx, {
    type: 'line',
    data: { labels: allLabels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1a2030',
          borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
          titleColor: '#8892a4', bodyColor: '#e2e8f4',
          callbacks: { label: ctx => ` ${ctx.dataset.label}: $${(+ctx.raw || 0).toFixed(2)}` }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', maxTicksLimit: 10, font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', font: { size: 10 }, callback: v => '$' + (+v).toFixed(0) } }
      }
    }
  });
}

// ── Progress bar animation ──────────────────────────────────
function animateProgress(id, duration, onDone) {
  let w = 0;
  const el = document.getElementById(id);
  el.style.width = '0%';
  const steps    = 40;
  const interval = duration / steps;
  const t = setInterval(() => {
    w = Math.min(w + 100 / steps, 100);
    el.style.width = w + '%';
    if (w >= 100) { clearInterval(t); if (onDone) onDone(); }
  }, interval);
}

// ── Run ARIMA ───────────────────────────────────────────────
function runARIMA() {
  const steps = parseInt(document.getElementById('forecastDays').value);
  const btn   = document.getElementById('arimaBtn');
  btn.disabled = true; btn.textContent = '⏳ Training…';
  animateProgress('arima-prog', 1800, () => {
    arimaForecast = arimaPredict(histData, steps);
    arimaMetrics  = arimaForecast.metrics;
    document.getElementById('arima-rmse').textContent = arimaMetrics.rmse.toFixed(2);
    document.getElementById('arima-mae').textContent  = arimaMetrics.mae.toFixed(2);
    document.getElementById('arima-mape').textContent = arimaMetrics.mape.toFixed(1) + '%';
    document.getElementById('arima-r2').textContent   = arimaMetrics.r2.toFixed(4);
    btn.disabled = false; btn.textContent = '✓ Trained — Re-run';
    renderMainChart(); updateForecastTable(); updateCompare(); updateSignal();
  });
}

// ── Run LSTM ────────────────────────────────────────────────
function runLSTM() {
  const steps = parseInt(document.getElementById('forecastDays').value);
  const btn   = document.getElementById('lstmBtn');
  btn.disabled = true; btn.textContent = '⏳ Training…';
  animateProgress('lstm-prog', 2400, () => {
    lstmForecast = lstmPredict(histData, steps);
    lstmMetrics  = lstmForecast.metrics;
    document.getElementById('lstm-rmse').textContent = lstmMetrics.rmse.toFixed(2);
    document.getElementById('lstm-mae').textContent  = lstmMetrics.mae.toFixed(2);
    document.getElementById('lstm-mape').textContent = lstmMetrics.mape.toFixed(1) + '%';
    document.getElementById('lstm-r2').textContent   = lstmMetrics.r2.toFixed(4);
    btn.disabled = false; btn.textContent = '✓ Trained — Re-run';
    renderMainChart(); updateForecastTable(); updateCompare(); updateSignal();
  });
}

function runBoth() { runARIMA(); setTimeout(runLSTM, 400); }

// ── Forecast table ──────────────────────────────────────────
function updateForecastTable() {
  if (!arimaForecast && !lstmForecast) return;
  const steps       = parseInt(document.getElementById('forecastDays').value);
  const lastPrice   = histData[histData.length - 1].price;
  let html = '';
  let d = new Date(histData[histData.length - 1].date);

  for (let i = 0; i < steps; i++) {
    d.setDate(d.getDate() + 1);
    while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() + 1);
    const dateStr = d.toISOString().slice(0, 10);
    const ar  = arimaForecast ? arimaForecast.forecast[i] : null;
    const ls  = lstmForecast  ? lstmForecast.forecast[i]  : null;
    const avg = ar && ls ? +((ar + ls) / 2).toFixed(2) : (ar || ls || 0);
    const chg = ((avg - lastPrice) / lastPrice * 100).toFixed(2);
    const conf = Math.max(0, 95 - i * 2).toFixed(0);
    const cls  = +chg >= 0 ? 'up-arrow' : 'dn-arrow';
    const arrow = +chg >= 0 ? '▲' : '▼';
    html += `<tr>
      <td style="font-family:monospace;font-size:11px;color:var(--muted)">${dateStr}</td>
      <td style="color:#4f8ef7">${ar ? '$' + ar : '—'}</td>
      <td style="color:#00d4a0">${ls ? '$' + ls : '—'}</td>
      <td style="font-weight:600">$${avg}</td>
      <td class="${cls}">${arrow} ${Math.abs(chg)}%</td>
      <td style="color:var(--muted)">${conf}%</td>
    </tr>`;
  }
  document.getElementById('forecastBody').innerHTML = html;
}

// ── Comparison tab ──────────────────────────────────────────
function updateCompare() {
  if (!arimaMetrics || !lstmMetrics) return;

  const keys   = ['rmse', 'mae', 'mape', 'r2'];
  const labels = ['RMSE', 'MAE', 'MAPE%', 'R² Score'];
  let html = '';

  keys.forEach((k, i) => {
    const aVal   = arimaMetrics[k], lVal = lstmMetrics[k];
    const better = k === 'r2' ? (aVal > lVal ? 'ARIMA' : 'LSTM') : (aVal < lVal ? 'ARIMA' : 'LSTM');
    const aW = k === 'r2' ? (aVal * 100).toFixed(0) : Math.min(100, aVal).toFixed(0);
    const lW = k === 'r2' ? (lVal * 100).toFixed(0) : Math.min(100, lVal).toFixed(0);
    html += `
      <div class="comp-section">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:11px">
          <span style="color:var(--muted)">${labels[i]}</span>
          <span style="color:var(--amber);font-size:10px">${better} wins</span>
        </div>
        <div class="comp-row">
          <span class="comp-label" style="color:#4f8ef7">ARIMA</span>
          <div class="comp-bar-wrap"><div class="comp-bar-fill" style="background:#4f8ef7;width:${aW}%"></div></div>
          <span class="comp-score" style="color:#4f8ef7">${k === 'mape' ? aVal.toFixed(1) + '%' : aVal}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label" style="color:#00d4a0">LSTM</span>
          <div class="comp-bar-wrap"><div class="comp-bar-fill" style="background:#00d4a0;width:${lW}%"></div></div>
          <span class="comp-score" style="color:#00d4a0">${k === 'mape' ? lVal.toFixed(1) + '%' : lVal}</span>
        </div>
      </div>`;
  });
  document.getElementById('compareContent').innerHTML = html;

  // Residual chart
  const rCtx = document.getElementById('residualChart').getContext('2d');
  if (residualObj) residualObj.destroy();
  const rLabels = arimaForecast.residuals.map((_, i) => `T-${30 - i}`);
  residualObj = new Chart(rCtx, {
    type: 'bar',
    data: {
      labels: rLabels,
      datasets: [
        { label: 'ARIMA', data: arimaForecast.residuals, backgroundColor: 'rgba(79,142,247,0.55)', borderRadius: 2 },
        { label: 'LSTM',  data: lstmForecast.residuals,  backgroundColor: 'rgba(0,212,160,0.55)',  borderRadius: 2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', maxTicksLimit: 8, font: { size: 9 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', font: { size: 9 } } }
      }
    }
  });

  // Compare forecast chart
  const cCtx = document.getElementById('compareChart').getContext('2d');
  if (compareObj) compareObj.destroy();
  const steps   = arimaForecast.forecast.length;
  const fLabels = Array.from({ length: steps }, (_, i) => `Day ${i + 1}`);
  compareObj = new Chart(cCtx, {
    type: 'line',
    data: {
      labels: fLabels,
      datasets: [
        { label: 'ARIMA', data: arimaForecast.forecast, borderColor: '#4f8ef7', borderWidth: 2, pointRadius: 3, tension: 0.3 },
        { label: 'LSTM',  data: lstmForecast.forecast,  borderColor: '#00d4a0', borderWidth: 2, pointRadius: 3, tension: 0.3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1a2030', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, titleColor: '#8892a4', bodyColor: '#e2e8f4' }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', font: { size: 10 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#4a5568', font: { size: 10 }, callback: v => '$' + (+v).toFixed(0) } }
      }
    }
  });
}

// ── Signal tab ──────────────────────────────────────────────
function updateSignal() {
  if (!arimaForecast && !lstmForecast) return;
  const lastPrice = histData[histData.length - 1].price;
  const arEnd     = arimaForecast ? arimaForecast.forecast[arimaForecast.forecast.length - 1] : lastPrice;
  const lsEnd     = lstmForecast  ? lstmForecast.forecast[lstmForecast.forecast.length - 1]   : lastPrice;
  const avgEnd    = (arEnd + lsEnd) / 2;
  const change    = (avgEnd - lastPrice) / lastPrice * 100;
  const r2Avg     = ((arimaMetrics?.r2 || 0) + (lstmMetrics?.r2 || 0)) / (arimaMetrics && lstmMetrics ? 2 : 1);
  const confLevel = Math.min(95, Math.round(r2Avg * 80 + 15));
  const days      = document.getElementById('forecastDays').value;

  let boxClass, labelClass, label, icon, text;
  if (change > 3) {
    boxClass = 'signal-buy'; labelClass = 'sig-buy-color';
    label = 'BUY Signal'; icon = '📈';
    text  = `Both models forecast a ${change.toFixed(1)}% price increase over the next ${days} days. Favorable risk/reward ratio.`;
  } else if (change < -3) {
    boxClass = 'signal-sell'; labelClass = 'sig-sell-color';
    label = 'SELL Signal'; icon = '📉';
    text  = `Models forecast a ${Math.abs(change).toFixed(1)}% decline. Consider reducing exposure or setting stop-loss orders.`;
  } else {
    boxClass = 'signal-hold'; labelClass = 'sig-hold-color';
    label = 'HOLD Signal'; icon = '⚖️';
    text  = `Models forecast only ${Math.abs(change).toFixed(1)}% movement. Neutral zone — monitor for stronger directional signals.`;
  }

  document.getElementById('signalContent').innerHTML = `
    <div class="signal-box ${boxClass}">
      <div class="sig-icon">${icon}</div>
      <div>
        <div class="sig-label ${labelClass}">${label}</div>
        <div class="sig-text">${text}</div>
      </div>
    </div>
    <div class="acc-grid" style="margin-top:12px">
      <div class="acc-item"><div class="acc-label">Forecast Change</div><div class="acc-val" style="color:${change >= 0 ? 'var(--green)' : 'var(--red)'}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</div></div>
      <div class="acc-item"><div class="acc-label">Model Confidence</div><div class="acc-val" style="color:var(--amber)">${confLevel}%</div></div>
      <div class="acc-item"><div class="acc-label">Target Price</div><div class="acc-val" style="color:var(--text)">$${avgEnd.toFixed(2)}</div></div>
      <div class="acc-item"><div class="acc-label">Current Price</div><div class="acc-val" style="color:var(--muted)">$${lastPrice.toFixed(2)}</div></div>
    </div>
    <div class="disc-box">
      ⚠️ <strong style="color:var(--amber)">Disclaimer:</strong> This is a simulation for educational purposes only. Past performance does not guarantee future results. Do not make real investment decisions based solely on model outputs.
    </div>
  `;
}

// ── Tab switching ───────────────────────────────────────────
function switchTab(name, btn) {
  ['models', 'forecast', 'compare', 'signal'].forEach(t => {
    document.getElementById('tab-' + t).style.display = t === name ? 'block' : 'none';
  });
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// ── Stock switching ─────────────────────────────────────────
function switchStock(ticker) {
  currentStock  = ticker;
  arimaForecast = null; lstmForecast = null;
  arimaMetrics  = null; lstmMetrics  = null;
  histData      = generateHistorical(ticker);
  renderMetrics();
  renderMainChart();
  ['arima', 'lstm'].forEach(m => {
    document.getElementById(m + '-prog').style.width = '0%';
    ['rmse', 'mae', 'mape', 'r2'].forEach(k => document.getElementById(m + '-' + k).textContent = '—');
    document.getElementById(m + 'Btn').textContent = m === 'arima' ? '▶ Train ARIMA' : '▶ Train LSTM';
  });
  document.querySelectorAll('.stock-btn').forEach(b => b.classList.toggle('active', b.dataset.stock === ticker));
  updateForecastTable();
  document.getElementById('compareContent').innerHTML = '<span style="color:var(--muted);font-size:12px">Train both models to compare.</span>';
  document.getElementById('signalContent').innerHTML  = '<span style="color:var(--muted);font-size:12px">Train models to generate signal.</span>';
  if (residualObj) { residualObj.destroy(); residualObj = null; }
  if (compareObj)  { compareObj.destroy();  compareObj  = null; }
}

function updateForecast() { renderMainChart(); updateForecastTable(); }

// ── Init ────────────────────────────────────────────────────
(function init() {
  const bar = document.getElementById('stockBar');
  Object.keys(STOCKS).forEach(ticker => {
    const btn = document.createElement('button');
    btn.className   = 'stock-btn' + (ticker === currentStock ? ' active' : '');
    btn.dataset.stock = ticker;
    btn.textContent = ticker;
    btn.onclick     = () => switchStock(ticker);
    bar.appendChild(btn);
  });
  histData = generateHistorical(currentStock);
  renderMetrics();
  renderMainChart();
})();
