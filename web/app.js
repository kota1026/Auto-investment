/* Auto-Investment dashboard frontend.
 *
 * Talks to the FastAPI server's JSON endpoints and renders three charts via
 * TradingView's lightweight-charts:
 *   - Price + EMA(12/26) with signal arrows
 *   - RSI(14)
 *   - Backtest equity curve
 *
 * Plus an AI evaluation panel that calls /api/evaluate on demand.
 */

const fmtUSD = (n) => `$${Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
const fmtPct = (n) => `${Number(n).toFixed(2)}%`;

const chartCommon = {
  layout: {
    background: { color: "#11161f" },
    textColor: "#94a3b8",
  },
  grid: {
    vertLines: { color: "#1f2937" },
    horzLines: { color: "#1f2937" },
  },
  crosshair: { mode: 1 },
  rightPriceScale: { borderColor: "#1f2937" },
  timeScale: { borderColor: "#1f2937", timeVisible: true, secondsVisible: false },
};

let priceChart, rsiChart, equityChart;
let candleSeries, emaFastSeries, emaSlowSeries, rsiSeries, equitySeries;

function initCharts() {
  priceChart = LightweightCharts.createChart(document.getElementById("price-chart"), {
    ...chartCommon,
    height: 460,
  });
  candleSeries = priceChart.addCandlestickSeries({
    upColor: "#22c55e",
    downColor: "#ef4444",
    borderUpColor: "#22c55e",
    borderDownColor: "#ef4444",
    wickUpColor: "#22c55e",
    wickDownColor: "#ef4444",
  });
  emaFastSeries = priceChart.addLineSeries({ color: "#22d3ee", lineWidth: 2, title: "EMA 12" });
  emaSlowSeries = priceChart.addLineSeries({ color: "#facc15", lineWidth: 2, title: "EMA 26" });

  rsiChart = LightweightCharts.createChart(document.getElementById("rsi-chart"), {
    ...chartCommon,
    height: 160,
  });
  rsiSeries = rsiChart.addLineSeries({ color: "#a78bfa", lineWidth: 2 });
  // RSI 30/70 reference lines
  rsiSeries.createPriceLine({ price: 70, color: "#ef4444", lineStyle: 2, axisLabelVisible: true, title: "70" });
  rsiSeries.createPriceLine({ price: 30, color: "#22c55e", lineStyle: 2, axisLabelVisible: true, title: "30" });
  rsiSeries.createPriceLine({ price: 50, color: "#94a3b8", lineStyle: 3, axisLabelVisible: true, title: "50" });

  equityChart = LightweightCharts.createChart(document.getElementById("equity-chart"), {
    ...chartCommon,
    height: 200,
  });
  equitySeries = equityChart.addAreaSeries({
    lineColor: "#22d3ee",
    topColor: "rgba(34, 211, 238, 0.4)",
    bottomColor: "rgba(34, 211, 238, 0.05)",
    lineWidth: 2,
  });

  window.addEventListener("resize", () => {
    priceChart.applyOptions({ width: document.getElementById("price-chart").clientWidth });
    rsiChart.applyOptions({ width: document.getElementById("rsi-chart").clientWidth });
    equityChart.applyOptions({ width: document.getElementById("equity-chart").clientWidth });
  });
}

async function loadCandles() {
  const r = await fetch("/api/candles?limit=400");
  if (!r.ok) throw new Error(`candles HTTP ${r.status}`);
  const data = await r.json();

  candleSeries.setData(data.candles);
  emaFastSeries.setData(data.ema_fast);
  emaSlowSeries.setData(data.ema_slow);
  rsiSeries.setData(data.rsi);
  candleSeries.setMarkers(data.markers);

  document.getElementById("symbol-label").textContent = `${data.symbol} · ${data.timeframe}`;
}

async function loadBacktest() {
  const r = await fetch("/api/backtest?limit=1000");
  if (!r.ok) throw new Error(`backtest HTTP ${r.status}`);
  const data = await r.json();

  equitySeries.setData(
    data.equity_curve.map((p) => ({
      time: Math.floor(new Date(p.timestamp).getTime() / 1000),
      value: p.equity,
    })),
  );

  const stats = document.getElementById("bt-stats");
  stats.innerHTML = `
    <dt>Trades</dt><dd>${data.n_trades}</dd>
    <dt>Win rate</dt><dd>${fmtPct(data.win_rate * 100)}</dd>
    <dt>Avg R:R</dt><dd>${data.avg_rr.toFixed(2)}</dd>
    <dt>Total return</dt><dd>${fmtPct(data.total_return_pct)}</dd>
    <dt>Max drawdown</dt><dd>${fmtPct(data.max_drawdown_pct)}</dd>
    <dt>Final equity</dt><dd>${fmtUSD(data.final_equity)}</dd>
  `;
}

async function runEvaluation() {
  const btn = document.getElementById("evaluate-btn");
  const box = document.getElementById("verdict-box");
  btn.disabled = true;
  btn.textContent = "Consulting Claude…";
  box.className = "";
  box.innerHTML = '<div class="empty">Running evaluation…</div>';

  try {
    const r = await fetch("/api/evaluate?limit=300", { method: "POST" });
    if (!r.ok) throw new Error(`evaluate HTTP ${r.status}`);
    const data = await r.json();
    renderVerdict(data);
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Evaluation";
  }
}

function renderVerdict(data) {
  const box = document.getElementById("verdict-box");
  if (data.status === "no_signal") {
    box.innerHTML = `
      <div class="empty">No active signal at the moment.</div>
      <div style="margin-top: 6px; font-size: 12px; color: #94a3b8">
        Last close: ${fmtUSD(data.last_close)}
      </div>`;
    return;
  }
  const v = data.verdict;
  const sig = data.signal;
  const fc = data.forecast;
  box.innerHTML = `
    <div>
      <span class="badge ${sig.side}">${sig.side.toUpperCase()} signal</span>
      <span style="margin-left: 8px; font-size: 12px; color: #94a3b8">@ ${fmtUSD(sig.price)}</span>
    </div>
    <div class="verdict-action ${v.action}">${v.action}</div>
    <div class="verdict-confidence">Confidence: ${fmtPct(v.confidence * 100)}</div>
    <div class="verdict-rationale">${escapeHtml(v.rationale)}</div>
    ${v.key_observations && v.key_observations.length
      ? `<ul class="verdict-obs">${v.key_observations.map((o) => `<li>${escapeHtml(o)}</li>`).join("")}</ul>`
      : ""}
    ${fc ? `<pre class="forecast">${escapeHtml(`${fc.backend} forecast (${fc.horizon} bars): ${fc.point[0].toFixed(2)} → ${fc.point[fc.point.length - 1].toFixed(2)}`)}</pre>` : ""}
    ${data.plan
      ? `<pre class="forecast">PLAN: ${data.plan.side} qty=${data.plan.qty.toFixed(6)} entry=${data.plan.entry.toFixed(2)} stop=${data.plan.stop.toFixed(2)} target=${data.plan.target.toFixed(2)} (R:R ${data.plan.rr.toFixed(2)})</pre>`
      : ""}
  `;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function init() {
  initCharts();
  try {
    await Promise.all([loadCandles(), loadBacktest()]);
    document.getElementById("status-label").textContent = "Connected";
  } catch (e) {
    document.getElementById("status-label").textContent = `Error: ${e.message}`;
  }
  document.getElementById("evaluate-btn").addEventListener("click", runEvaluation);
}

init();
