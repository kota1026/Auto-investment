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

async function refreshContext() {
  const btn = document.getElementById("context-btn");
  const box = document.getElementById("context-box");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const r = await fetch("/api/context?limit=300");
    if (!r.ok) throw new Error(`context HTTP ${r.status}`);
    const data = await r.json();
    renderContext(data);
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh Context";
  }
}

function renderContext(data) {
  const box = document.getElementById("context-box");
  const parts = [];

  if (data.forecast) {
    const fc = data.forecast;
    const first = fc.point[0];
    const last = fc.point[fc.point.length - 1];
    const ch = (((last - first) / first) * 100).toFixed(2);
    parts.push(`<div><strong>Forecast (${fc.backend}):</strong> ${first.toFixed(2)} → ${last.toFixed(2)} (${ch}%)</div>`);
  }

  if (data.news && data.news.summary) {
    parts.push(`<div style="margin-top: 8px;"><strong>News:</strong> ${escapeHtml(data.news.summary)}</div>`);
    if (data.news.items && data.news.items.length) {
      const items = data.news.items.slice(0, 3).map((i) =>
        `<li><a href="${escapeHtml(i.url)}" target="_blank" rel="noopener" style="color: #22d3ee;">${escapeHtml(i.title || i.source)}</a></li>`
      ).join("");
      parts.push(`<ul class="verdict-obs">${items}</ul>`);
    }
  } else {
    parts.push(`<div class="empty" style="margin-top: 6px;">News: not configured (set TAVILY_API_KEY)</div>`);
  }

  if (data.macro) {
    const lines = Object.entries(data.macro).map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join("");
    parts.push(`<div style="margin-top: 8px;"><strong>Macro (FRED):</strong></div><dl class="stats">${lines}</dl>`);
  } else {
    parts.push(`<div class="empty" style="margin-top: 6px;">Macro: not configured (set FRED_API_KEY)</div>`);
  }

  if (data.fundamentals) {
    parts.push(`<div style="margin-top: 8px;"><strong>Fundamentals (Novaquity):</strong> ${escapeHtml(JSON.stringify(data.fundamentals).slice(0, 200))}…</div>`);
  }

  box.className = "";
  box.innerHTML = parts.join("");
}

async function runOptimize() {
  const btn = document.getElementById("optimize-btn");
  const box = document.getElementById("optimize-box");
  btn.disabled = true;
  btn.textContent = "Optimizing…";
  box.className = "";
  box.innerHTML = '<div class="empty">Sweeping parameter space — this can take a moment.</div>';

  try {
    const r = await fetch("/api/optimize?method=random&n_samples=20&explain=true", { method: "POST" });
    if (!r.ok) throw new Error(`optimize HTTP ${r.status}`);
    const data = await r.json();
    renderOptimize(data);
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Optimization";
  }
}

function renderOptimize(data) {
  const box = document.getElementById("optimize-box");
  if (!data.best) {
    box.innerHTML = `<div class="empty">No viable parameter set found (${data.n_evaluated} evaluated).</div>`;
    return;
  }
  const top = data.top.slice(0, 5);
  const rows = top.map((c, i) => {
    const p = c.params;
    return `
      <tr>
        <td>${i + 1}</td>
        <td>${p.ema_fast}/${p.ema_slow}</td>
        <td>${p.rsi_threshold}</td>
        <td>${p.sl_atr_mult}/${p.tp_atr_mult}</td>
        <td>${c.sharpe.toFixed(2)}</td>
        <td>${fmtPct(c.total_return_pct)}</td>
      </tr>`;
  }).join("");

  box.innerHTML = `
    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 6px;">
      ${data.n_evaluated} configs evaluated · top 5 by Sharpe
    </div>
    <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
      <thead style="color: #94a3b8;">
        <tr><th>#</th><th>EMA</th><th>RSI</th><th>SL/TP</th><th>Sharpe</th><th>Return</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    ${data.claude_recommendation
      ? `<div style="margin-top: 10px; font-size: 12px; line-height: 1.5;"><strong style="color: #22d3ee;">Claude recommends:</strong> ${escapeHtml(data.claude_recommendation)}</div>`
      : ""}
  `;
}

async function runImprove() {
  const btn = document.getElementById("improve-btn");
  const box = document.getElementById("improve-box");
  const mode = document.getElementById("improve-mode").value;
  btn.disabled = true;
  btn.textContent = "Running…";
  box.className = "";
  box.innerHTML = `<div class="empty">Running improvement loop in mode ${mode} (this may take a moment — Claude is iterating)…</div>`;

  try {
    const r = await fetch(`/api/improve?mode=${mode}&max_iterations=5&limit=1000`, { method: "POST" });
    if (!r.ok) throw new Error(`improve HTTP ${r.status}`);
    const data = await r.json();
    renderImprove(data);
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Loop";
  }
}

function renderImprove(data) {
  const box = document.getElementById("improve-box");
  if (data.error) {
    box.innerHTML = `<div class="empty">${escapeHtml(data.error)}</div>`;
    return;
  }
  const before = data.initial_metrics;
  const after = data.final_metrics;
  const beforeSharpe = before.icir.toFixed(2);
  const afterSharpe = after.icir.toFixed(2);
  const historyHtml = data.history.map((h, i) => {
    const p = h.proposal;
    return `
      <div style="margin-top: 10px; padding: 8px; background: #0a0d12; border-left: 3px solid ${h.approved ? '#22c55e' : '#22d3ee'}; border-radius: 0 4px 4px 0;">
        <div style="font-size: 11px; color: #94a3b8; margin-bottom: 4px;">
          Iteration ${h.iteration} ${h.approved ? '· APPROVED' : ''} · trades=${h.metrics.n_trades} · winR=${(h.metrics.win_rate*100).toFixed(0)}% · ret=${h.metrics.total_return_pct.toFixed(1)}% · ICIR=${h.metrics.icir.toFixed(2)}
        </div>
        <div style="font-size: 12px; line-height: 1.4;">${escapeHtml(p.rationale || '')}</div>
        ${p.expected_change ? `<div style="font-size: 11px; color: #facc15; margin-top: 4px;">→ ${escapeHtml(p.expected_change)}</div>` : ""}
      </div>`;
  }).join("");

  box.innerHTML = `
    <div style="font-size: 12px; color: #94a3b8;">
      Mode <strong>${data.mode}</strong> · ${data.iterations_used} iterations · ${data.converged ? '<span style="color: #22c55e;">CONVERGED</span>' : '<span style="color: #facc15;">stopped</span>'}
    </div>
    <div style="font-size: 12px; margin-top: 6px;">
      <strong>Before:</strong> ret ${before.total_return_pct.toFixed(1)}% · DD ${before.max_drawdown_pct.toFixed(1)}% · ICIR ${beforeSharpe}<br>
      <strong>After:</strong>&nbsp; ret ${after.total_return_pct.toFixed(1)}% · DD ${after.max_drawdown_pct.toFixed(1)}% · ICIR ${afterSharpe}
    </div>
    ${historyHtml}
  `;
}

async function runContest() {
  const btn = document.getElementById("contest-btn");
  const box = document.getElementById("contest-box");
  const capital = document.getElementById("contest-capital").value || 30;
  const days = document.getElementById("contest-days").value || 30;
  const useAi = document.getElementById("contest-ai").checked;
  const useReal = document.getElementById("contest-real").checked;

  btn.disabled = true;
  btn.textContent = useAi ? "Claude trading..." : "Simulating...";
  box.className = "";
  box.innerHTML = `<div class="empty">Running ${days}-day contest with $${capital} starting capital...</div>`;

  try {
    const url = `/api/contest?starting_capital=${capital}&duration_days=${days}&use_ai=${useAi}&use_real_data=${useReal}&seed=42`;
    const r = await fetch(url, { method: "POST" });
    if (!r.ok) throw new Error(`contest HTTP ${r.status}`);
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    renderContest(data);
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Contest";
  }
}

function renderContest(data) {
  const box = document.getElementById("contest-box");
  const alphaPos = data.alpha_vs_benchmark >= 0;
  const equity = data.equity_curve;
  const minEq = Math.min(...equity.map((p) => p.equity));
  const maxEq = Math.max(...equity.map((p) => p.equity));
  // ASCII sparkline
  const spark = "▁▂▃▄▅▆▇█";
  const sparkLine = equity
    .filter((_, i) => i % Math.ceil(equity.length / 40) === 0)
    .map((p) => {
      const idx = Math.floor(((p.equity - minEq) / Math.max(maxEq - minEq, 0.001)) * (spark.length - 1));
      return spark[Math.max(0, Math.min(spark.length - 1, idx))];
    })
    .join("");

  const lastDecisions = data.decision_log.slice(-2).map((d) => {
    const decs = d.decisions
      .filter((x) => x.action !== "hold")
      .map((x) => `${x.symbol}:${x.action}`)
      .join(", ") || "all hold";
    return `<div style="font-size: 10px; color: #94a3b8; margin-top: 4px;">${d.timestamp.slice(0, 10)} · ${decs}</div>`;
  }).join("");

  box.innerHTML = `
    <div style="font-size: 12px; line-height: 1.6;">
      <div><strong>$${data.starting_capital}</strong> → <strong style="color: ${data.total_return_pct >= 0 ? '#22c55e' : '#ef4444'}">$${data.final_equity.toFixed(2)}</strong> (${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct.toFixed(2)}%)</div>
      <div>BTC HODL benchmark: ${data.benchmark_return_pct >= 0 ? '+' : ''}${data.benchmark_return_pct.toFixed(2)}%</div>
      <div>Alpha vs BTC: <strong style="color: ${alphaPos ? '#22c55e' : '#ef4444'}">${alphaPos ? '+' : ''}${data.alpha_vs_benchmark.toFixed(2)}%</strong></div>
    </div>
    <pre style="font-size: 14px; color: #22d3ee; margin: 8px 0; line-height: 1; letter-spacing: -1px;">${sparkLine}</pre>
    <div style="font-size: 11px; color: #94a3b8;">
      Sharpe ${data.sharpe.toFixed(2)} · Sortino ${data.sortino.toFixed(2)} · MaxDD ${data.max_drawdown_pct.toFixed(1)}%<br>
      ${data.n_decisions} decisions · ${data.n_trades} trades · ${data.n_liquidations} liquidations<br>
      Fees $${data.fees_paid.toFixed(2)} · Funding $${data.funding_paid.toFixed(2)} · ${data.used_ai ? 'Claude' : 'Heuristic'}
    </div>
    ${lastDecisions ? `<div style="margin-top: 6px;"><strong style="font-size: 11px; color: #94a3b8;">Recent decisions:</strong>${lastDecisions}</div>` : ""}
  `;
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
  document.getElementById("context-btn").addEventListener("click", refreshContext);
  document.getElementById("optimize-btn").addEventListener("click", runOptimize);
  document.getElementById("improve-btn").addEventListener("click", runImprove);
  document.getElementById("contest-btn").addEventListener("click", runContest);
}

init();
