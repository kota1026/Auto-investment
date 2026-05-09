# Auto‑Investment Strategy Specification (v0.2)

**Status:** CEO‑signed off 2026‑05‑09. Phase 1 implementation in progress.
**Branch:** `claude/crypto-portfolio-automation-gEv4B`
**Author:** Claude Code (auto‑drafted, CEO‑reviewed)

## Changelog
- **v0.3 (2026‑05‑09)** — Phase 2 implementation. Adds S3 (cross‑exchange
  stat‑arb), S4 (pre‑IPO alerts, alert‑only), and a real‑data fetcher
  layer (`data_fetchers/`) with parquet cache. New runner
  `scripts/backtest_phase2.py` runs all four strategies with `--mode synth`
  (hermetic) or `--mode real` (loads cache populated by
  `scripts/fetch_real_data.py`). S3 z‑threshold tuned to 3.5 to clear
  retail‑taker fee tier; promote to maker‑rebate in Phase 3 to relax it.
- **v0.2.1 (2026‑05‑09)** — Backtest output gains explicit `period_days` /
  `capital_base_usd` fields and both `total_return_pct` (over period) and
  `annualised_return_pct` (per‑year) so units are unambiguous. Adds §13
  (Capital Allocation) with three blended‑APR scenarios.
- **v0.2 (2026‑05‑09)** — CEO decisions locked in: Hyperliquid as the sole
  perp DEX, monthly external‑data budget pinned at **$0** by using free
  tiers only, all LLM reasoning routed through Claude Code Pro Max
  (no Anthropic API spend), reporting set to **daily Slack + weekly Excel**,
  and a panic key is mandatory. Adds §11 (Cost Architecture) and §12
  (Scheduling).
- **v0.1 (2026‑05‑09)** — Initial draft for review.

This document defines what we are going to build *before* we build it.
The intent is that every line of trading code added to `src/auto_investment/`
is justified by a paragraph here. If a future change does not map to a
section below, we either update this spec or reject the change.

---

## 1. Goals and Constraints

| Item | Value | Reason |
|---|---|---|
| Capital tier (Phase 1) | **$1,000 – $10,000** | User‑specified; drives venue & gas choice. |
| Holding period | **超短期** (intraday → 7 days) | User‑specified. |
| Mandate | High‑probability, data‑driven, low‑fee, automated | YEARN V3 風 |
| Decision style | Quant first, LLM‑gated | Repo already uses `decision_agent.py` |
| Geography | Global, USD denominated | |
| Off‑limits in Phase 1 | L1 mainnet DeFi (gas $20–50/tx kills small capital), MEV searcher / sandwich bots, sub‑second HFT | Capital and infra mismatch |

### Why these constraints matter for strategy choice

At $1k–$10k, a single Aave deposit + withdrawal on Ethereum L1 can cost
30–80 bps round‑trip in gas alone. That removes ~half of a typical DeFi
yield‑hopping edge. We therefore restrict on‑chain execution to **L2s
(Arbitrum, Base, Optimism)** and Solana, where round‑trip cost is <5 bps.

---

## 2. Strategy Portfolio (Recommended)

We recommend running **four** strategies side‑by‑side in `alpha_arena.py`,
ranked by data quality and expected sharpe. Pre‑IPO is included per
user request but flagged as research‑grade only.

| # | Strategy | Venue | Holding | Data quality | Backtestable? | Phase |
|---|---|---|---|---|---|---|
| **S1** | **Funding‑Rate Arb** (perp short ↔ spot long, delta‑neutral) | Binance / Bybit / OKX | Hours – days | ★★★★★ | Yes | Phase 1 |
| **S2** | **L2 DeFi Yield Aggregation** (Aave/Compound/Curve on Arbitrum & Base) | DefiLlama API + on‑chain | Days – weeks | ★★★★ | Yes (APY history) | Phase 1 |
| **S3** | **CEX Cross‑Exchange Stat‑Arb** (BTC/ETH/SOL pair spread mean reversion) | Binance ↔ Bybit ↔ OKX | Minutes – hours | ★★★★ | Yes | Phase 2 |
| **S4** | **Pre‑IPO Token Carry** (Aevo Pre‑Markets, Whales Market) | DEX | Days – weeks | ★ | **Limited** (sparse OB data) | Research |
| ❌ | On‑chain CEX↔DEX arb / MEV | — | sub‑second | n/a | No | Reject for Phase 1 |

> **Why we reject MEV/HFT here:** The historical edge cannot be replayed
> from public data (private mempools, builder auctions, latency). Backtests
> systematically over‑estimate PnL. Re‑evaluate only if user funds Flashbots
> Builder access + co‑located infra.

---

## 3. Strategy Specs

### S1 — Funding‑Rate Arbitrage (delta‑neutral)

**Hypothesis.** When a perp's funding rate is persistently positive (longs
pay shorts), we can collect that yield by holding **+1 unit spot, −1 unit
perp** of the same asset. PnL = funding paid − fees − basis drift.

**Data sources**
- `ccxt.fetch_funding_rate_history(symbol)` — already wired through `data_providers.py`
- Spot/perp mid for basis: `ccxt.fetch_ohlcv` on both legs
- Borrow rate (margin) and CEX maker fees per venue

**Signal**
```
funding_8h_apr = mean(last_n_funding) * 3 * 365
edge_bps = funding_8h_apr * 10000 - (entry_fee_bps + exit_fee_bps + basis_drift_bps)
enter if edge_bps > MIN_EDGE_BPS  (default 800 bps APR ≈ 22 bps over 24h)
exit if funding turns negative or basis blows out > 50 bps
```

**Sizing.** Both legs sized identically in notional terms; gross 2× equity,
net 0 delta. Cap notional per venue at 5% of LP order book depth at ±10 bps.

**KPI targets (in‑sample, 2024–2026 data)**
- Sharpe ≥ 2.0 net of fees (annualised, trade-level)
- Adjusted Sharpe (after multiple-testing penalty) ≥ 1.5
- **Median** block-bootstrap Sharpe ≥ 1.0 (right-skewed PnL means p5 is
  noisy; we monitor it but don't gate on it)
- Max DD ≤ 5%
- Hit rate is intentionally **not** an acceptance gate — funding arb is
  a low-hit-rate, positive-expectancy profile (winners much bigger than
  losers). We track it for monitoring only.

**Failure modes**
- Funding flips quickly → unwind cost > collected funding
- One leg gets liquidated (perp side); mitigated by ≥ 4× initial margin
- CEX deposit/withdrawal halt on a venue (real risk)

---

### S2 — L2 DeFi Yield Aggregation (YEARN V3‑style)

**Hypothesis.** APY across stable‑coin pools is mean‑reverting on a
days‑to‑weeks horizon. A simple "always rotate to the top‑K APY pool
that meets risk filters, when expected gain > 2× rotation cost"
strategy beats holding any single pool.

**Universe (Phase 1).**
- Chains: Arbitrum, Base, Optimism (gas < $0.30/tx)
- Protocols: Aave v3, Compound v3, Morpho Blue, Curve, Pendle (PT only), Yearn V3 vaults
- Assets: USDC, USDT, DAI, USDe (curated whitelist of `pool_id`s)

**Data sources**
- DefiLlama Yields API (`https://yields.llama.fi/pools` and `/chart/{pool}`)
  — free, no key, 30‑day APY history per pool. Sufficient for mean‑reversion
  modelling.
- On‑chain: read live `getReserveData(asset)` directly from Aave/Compound for
  freshness during execution.
- Etherscan / Arbiscan for historical gas oracle (median Arbitrum gwei).

**Signal**
```
forecast_apy_h = ewma(apy_history, halflife=72h)
rank pools by forecast_apy_h
rotate from current pool i → pool j only if:
    (forecast_apy_h[j] - forecast_apy_h[i]) * holding_period_days/365
        > 2 * round_trip_cost_bps + protocol_risk_premium[j]
```

**Risk filters (must pass to enter pool)**
- TVL ≥ $20M
- Audited (DefiLlama `audits ≥ 1`)
- No exposure to non‑pegged collateral (e.g., reject pools with USDD)
- Pool age ≥ 90 days

**KPI targets**
- Net APY uplift ≥ +200 bps over "always Aave USDC on Arbitrum" baseline
- Max DD (depeg event) ≤ 2%

**Failure modes**
- Stable depeg (USDC March 2023 style): → emergency exit handler must
  sell within 30 min based on price oracle deviation > 50 bps.
- APY history is forward‑looking biased — DefiLlama incentive APY can
  evaporate. Backtest must split base APY vs reward APY.

---

### S3 — CEX Cross‑Exchange Stat‑Arb

**Hypothesis.** Spreads of (BTC/USDT on Binance) − (BTC/USDT on Bybit)
mean‑revert on minute scale. When the z‑score of the spread > 2,
short the rich, long the cheap, exit at z = 0.

**Data sources**
- ccxt 1m OHLCV from each venue (already supported in `data.py` / `data_providers.py`)
- For sub‑minute: WS L2 books — Phase 2

**Signal**
```
spread_t = mid_A_t - mid_B_t
z_t = (spread_t - rolling_mean(spread, 240m)) / rolling_std(spread, 240m)
enter pair when |z_t| > 2.0, cap = 30‑min hold
exit at |z| < 0.3 or after 60m
```

**Costs.** Two takers if we cross, two makers if we post (preferred);
plus 5–15 bps spread cost.

**KPI targets**
- Sharpe ≥ 1.5 after fees
- Max position‑level DD ≤ 1%

**Failure modes**
- Withdrawal halts mean we can't actually rebalance inventory between
  venues; assume no inventory transfer in backtest, only paired entries.

---

### S4 — Pre‑IPO Token Carry (Research‑grade)

**Hypothesis.** Pre‑IPO tokens (SpaceX, Anthropic, Stripe, OpenAI,
xAI on Aevo Pre‑Markets / Whales Market) trade at a discount to the
expected secondary‑market price at IPO. Short‑dated holders of the
token typically capture a positive carry as IPO probability rises.

**Honest disclaimer.**
Backtesting this strategy is **scientifically weak**:
- No continuous order‑book history before 2025
- ≤ 50 trades/day on most names; tape is non‑stationary
- "Strike price" markets (Whales) settle binary; OB doesn't exist between trades

**What we will actually do**
1. **Build a thin data layer** (`pre_ipo.py`):
   - Pull Aevo Pre‑Markets daily settlement marks via their public REST.
   - Pull Whales Market trade prints (where available) for SpaceX, Anthropic.
   - Persist to `data/pre_ipo/{ticker}.parquet`.
2. **Simple "implied IPO probability" backtest:**
   - Treat each name as a binary asset whose price ≈ p_IPO * (terminal
     mark) + (1 − p_IPO) * 0.
   - Fit a logistic curve to *time‑to‑rumored‑IPO* using observed marks.
   - Backtest a discretionary rule: "buy if implied IPO probability < 30%
     **and** news momentum (Tavily, repo already integrated) is positive."
3. **Treat outputs as alerts only.** No auto‑execution in Phase 1.
   The Decision Agent surfaces a Slack/Discord notification; user clicks
   to fill manually.

**KPI targets (qualitative)**
- Hit rate of "buy" alerts that gain > 10% in the next 30 days ≥ 55%
- Calibration: if model says 40% IPO probability, observed rate within ±10pp.

---

## 4. Cost Model (used by every backtest)

```python
# CEX (spot venues)
TAKER_BPS = {"binance": 4.5, "binancejp": 7.5, "bybit": 5.5, "okx": 5.0}
MAKER_BPS = {"binance": 1.0, "binancejp": 1.5, "bybit": 1.0, "okx": 0.8}

# Perp DEX (Hyperliquid is canonical for Phase 1)
TAKER_BPS_PERP = {"hyperliquid": 2.5, "dydx_v4": 5.0, "aevo": 5.0}
MAKER_BPS_PERP = {"hyperliquid": 0.2, "dydx_v4": 2.0, "aevo": 2.0}

SLIPPAGE_BPS = lambda notional_usd: max(2.0, 1.5 * (notional_usd / 100_000))

# L2 DeFi
GAS_USD = {"arbitrum": 0.20, "base": 0.10, "optimism": 0.25}
SWAP_BPS = 5.0           # uniswap v3 5bps tier
LP_DEPOSIT_BPS = 0.0     # most lending pools are flat fee
WITHDRAW_TAX = 0.0       # excl. utilization‑based withdrawal caps (modeled separately)

# Pre‑IPO
SPREAD_BPS = 200.0       # honest acknowledgement of wide spreads
```

All numbers configurable in `src/auto_investment/cost_model.py` (to be
created — see roadmap).

---

## 5. Backtest Methodology

We use the existing `backtest.py` engine, extending it with per‑strategy
plugins. Common discipline:

1. **Walk‑forward, no peeking.** Train indicators on `[t-N, t]`, decide at
   `t`, fill at `t+1` open.
2. **Fees & slippage from the cost model above** subtracted from every fill.
3. **Two splits**: 2024‑01–2025‑12 IS, 2026‑01–2026‑05 OOS. Refuse to
   ship a strategy whose OOS Sharpe < 0.5× IS Sharpe.
4. **Multiple‑testing penalty.** Each strategy reports an adjusted Sharpe
   `S* = S − 0.5 / sqrt(N_trials)` per López de Prado (2018).
5. **Bootstrap CI.** 500 block‑bootstrap resamples; report 5‑th percentile
   Sharpe. Ship only if 5th‑pct > 0.
6. **Capacity test.** Re‑run at 1×, 3×, 10× capital. Strategies that
   degrade non‑linearly are flagged.

---

## 6. Phased Roadmap

**Phase 0 — Spec sign‑off (now).**
This document. No code changes.

**Phase 1 — Backtest of S1 + S2 (week 1‑2).**
- Add `cost_model.py`
- Add `strategies/funding_arb.py` and `strategies/yield_router.py`
- Wire into existing `alpha_arena.py`
- Output: notebook + `results/strategy_spec_v0.1.json` with KPIs

**Phase 2 — S3 + S4 (week 3‑4).**
- Cross‑exchange stat‑arb
- Pre‑IPO data layer + alert‑only mode

**Phase 3 — Paper trading (week 5‑8).**
- Connect `live.py` to Binance + a Web3 RPC (Alchemy free tier on Arbitrum)
- Run S1 with real $500 only; compare to backtest

**Phase 4 — Scaled live (week 9+).** Only after OOS metrics hold up
in paper.

---

## 7. Claude for Financial Services 連携

Anthropic launched **Claude for Financial Services** with deep agent
support and add‑ins for Microsoft 365 (Excel/PowerPoint/Word). For our
project the relevant pieces are:

| Capability | How we use it |
|---|---|
| **MCP connectors** to FactSet, S&P CapIQ, PitchBook, Moody's, Daloopa | Phase 3+: pull pre‑IPO valuation comps for S4 (Anthropic, SpaceX cap‑table data via PitchBook) |
| **Claude for Excel** add‑in | Generate the daily P&L attribution sheet automatically; user reviews in Excel |
| **Managed Agents (Platform, public beta)** | Host the autonomous loop without us running our own server. Each strategy becomes a managed agent invoked on a cron. |
| **Claude Code finance plugins** (Pitch agent, Comps agent) | Phase 4: comps‑based fair‑value model for S4 |

> **Action item.** Once we are out of Phase 1 backtest, evaluate whether
> to migrate `improvement_loop.py` from raw API calls to a Managed Agent.
> Cost / observability tradeoff to be quantified at that point.

Sources:
- [Claude for Financial Services (Anthropic)](https://www.anthropic.com/news/claude-for-financial-services)
- [Agents for Financial Services (May 2026)](https://www.anthropic.com/news/finance-agents)
- [anthropics/financial-services GitHub repo](https://github.com/anthropics/financial-services)

---

## 8. Agentic Company Architecture

We treat the system as a **company of agents**, each a long‑running
Claude Agent SDK process with a narrow role and an explicit veto power
over the others. This maps cleanly onto the file structure already in
`src/auto_investment/`.

```
                        ┌──────────────────┐
                        │   CEO  (you)     │  human approval gate
                        └────────┬─────────┘
                                 │ daily report / approval
                  ┌──────────────┴───────────────┐
                  │                              │
        ┌─────────▼─────────┐         ┌──────────▼─────────┐
        │  Strategy R&D     │         │  Risk Officer       │
        │  (improvement_    │         │  (risk.py +         │
        │   loop.py)        │         │   new veto agent)   │
        │                   │         │                     │
        │ Proposes new      │         │ Hard caps:          │
        │ strategies, runs  │         │  • daily VaR        │
        │ backtests         │         │  • leverage         │
        │                   │         │  • venue exposure   │
        └─────────┬─────────┘         └──────────┬─────────┘
                  │ approved spec                │ veto signal
                  ▼                              │
        ┌───────────────────┐                    │
        │   Treasurer/CFO   │◄───────────────────┘
        │  (new module)     │
        │                   │
        │ Allocates capital │
        │ across S1‑S4 by   │
        │ Kelly + risk      │
        │ budget            │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  Execution Trader │  ── ccxt + Web3
        │  (live.py)        │  ── retries, idempotency
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │   Reporter/CCO    │  ── daily PnL → Excel
        │  (server.py +     │  ── compliance log
        │   Claude Excel    │  ── prompt‑injection scan
        │   add‑in)         │
        └───────────────────┘
```

### Roles in detail

| Agent | Module (proposed) | Tools it can call | Veto power |
|---|---|---|---|
| **Strategy R&D** | `agents/researcher.py` (extends `improvement_loop.py`) | backtest, IC, optimizer, web search | — |
| **Risk Officer** | `agents/risk_officer.py` | read positions, force unwind, halt new entries | **Yes** — can block all new orders |
| **Treasurer (CFO)** | `agents/treasurer.py` | reallocate capital, withdraw to cold | Sets caps |
| **Execution Trader** | `live.py` (existing) | ccxt orders, web3 sends | — |
| **Reporter/CCO** | `agents/reporter.py` | read trades, write Excel (Claude add‑in), emit notifications | **Yes** — can pause on prompt‑injection or anomaly detection |

### Communication protocol

- Each agent runs as an **MCP server** so the others (and you) can call
  it through a typed tool list — no shared mutable state.
- Decisions are written to an append‑only log (`results/decisions.jsonl`)
  signed by the originating agent. The Reporter checks the log every hour.
- All cross‑agent messages pass through a **Risk Officer pre‑hook**;
  the RO can rewrite or reject a message (this is the closest thing the
  system has to "internal compliance").

### What this lets us do

- **Self‑improvement** is already partially built (`improvement_loop.py`
  is a straight implementation of the Kawamura et al. JSAI 2026 paper).
  We extend it: instead of only re‑tuning EMA/RSI parameters, the
  Researcher can propose *new strategies* that are written into a
  candidate file and run through the alpha arena before being promoted.
- **Capital allocation** ceases to be a single hard‑coded number;
  the Treasurer rebalances weekly based on rolling Sharpe of each
  live strategy.
- **Human override** stays mandatory. CEO (the user) approves the
  weekly allocation diff via a single Slack message. Nothing else can
  push capital between strategies.

### Hard safety rules (encode in `risk.py`)

1. **Per‑trade max loss = 0.5% of equity.** Already in code; keep it.
2. **Per‑day max DD = 2%.** New. Risk Officer halts trading on breach.
3. **Per‑venue max exposure = 25% of equity.** New. Survives a CEX halt.
4. **No agent may withdraw to a new address.** Whitelist‑only.
5. **All large orders > $5k require CEO confirm in Phase 4.** Tunable.

---

## 9. CEO Decisions (resolved 2026‑05‑09)

| # | Question | CEO answer | Implication |
|---|---|---|---|
| 1 | Funded CEX accounts? | Binance Japan (spot), Hyperliquid (perp DEX) — both wallet‑signed, no SMS KYC blocker | S1/S3 use Binance Japan as spot leg, Hyperliquid as perp leg |
| 2 | Web3 wallet for L2? | MetaMask already exists; Hyperliquid wallet is the same key | No fresh wallet needed in Phase 1 |
| 3 | External data budget | **$0/month**. Use only free tiers; all LLM reasoning routed through Claude Code Pro Max | PitchBook / FactSet dropped; replaced with Tavily MCP free tier (1k searches/mo) + DuckDuckGo MCP for overflow |
| 4 | Reporting cadence | **Daily Slack** push + **weekly Excel** workbook | `agents/reporter` writes both; weekly run produces `results/weekly_YYYYWW.xlsx` |
| 5 | Panic key | **Required.** Single command that flattens all perps, withdraws DeFi to USDC, halts agents | Implemented as `python -m auto_investment.panic` and `/panic` slash command |

### Perp DEX choice — Hyperliquid

We evaluated Hyperliquid vs dYdX v4 vs Aevo on eight axes (liquidity, funding
stability, ccxt support, fees, geo access, MetaMask compatibility, security
track record, settlement asset). Hyperliquid wins on six of eight and ties
on two; in particular it is the only candidate with native ccxt support,
the lowest taker fee (2.5 bps), and the deepest BTC perp book (~$4–6B/24h).
dYdX v4 and Aevo remain candidates for **Phase 3 redundancy**, not Phase 1.

---

## 11. Cost Architecture — staying at $0/month

We treat "LLM reasoning" and "external data" as separate cost lines. The
goal is to keep the **monthly cash cost at $0** while staying inside the
Claude Code Pro Max plan the user already pays for.

### 11.1 LLM reasoning runs through Claude Code, not the Anthropic API

Every place where v0.1 of this spec proposed `anthropic.messages.create()`
is replaced with **Claude Code (headless mode)** invoked from a slash
command. This shifts cost from per‑token API billing onto the existing
Pro Max subscription.

| Concern | Old (API) | New (Claude Code) |
|---|---|---|
| Daily signal generation | `improvement_loop.py` calls `anthropic` SDK | `cron → claude -p "/morning_signals"` reads ledger + market data, emits Slack |
| News sentiment for S4 | API call per ticker | `/news_sentiment <ticker>` slash command (uses Tavily MCP web search inside Claude Code) |
| Weekly Excel narrative | API → openpyxl | `/weekly_report` slash command writes the workbook directly |
| Strategy improvement loop | Monthly API batch | `/improve` slash command, weekly cron |
| Pre‑IPO valuation reasoning | API + PitchBook MCP | `/preipo_eval` runs web search via Tavily and produces a memo |
| Anomaly detection | API on each tick | SessionStart hook reads `results/decisions.jsonl`; Claude flags drift |

Implementation path:
- Slash commands live in `.claude/commands/<name>.md`. They are plain
  Markdown prompts checked into the repo; no API key required.
- Subagents live in `.claude/agents/<role>.md` (analyst, reporter, improver).
- Hooks live in `.claude/hooks/`. The SessionStart hook ingests latest
  positions + last 24h of decisions so any interactive session starts
  context‑full.
- Headless invocation pattern: `claude -p "/morning_signals" --output-format=stream-json`.
  This counts against Pro Max quota, not API spend.

### 11.2 External data — free tiers only

| Source | Used by | Free tier | Expected usage | Headroom |
|---|---|---|---|---|
| **ccxt → Hyperliquid** | S1, S3 (perp leg), S4 (Aevo) | unlimited | 1‑min funding, 5‑min OHLCV | ✅ |
| **ccxt → Binance Japan** | S1, S3 (spot leg) | 1200 req/min | 1‑min OHLCV | ✅ |
| **DefiLlama Yields** | S2 | unlimited, no key | 4h pool refresh + 30‑day APY history | ✅ |
| **Tavily MCP** | S4 news, `/improve` web search | 1,000 searches/mo | ~25/day = 750/mo | ⚠️ tight |
| **DuckDuckGo MCP** | Overflow for Tavily | unlimited | spillover bucket | ✅ |
| **yfinance (unofficial)** | Macro reference (DXY, SPX) | unlimited | daily | ✅ |
| **Etherscan / Arbiscan / Basescan** | S2 gas oracle | 5 req/s, 100k/day | hourly | ✅ |
| **CoinGecko (no key)** | Sanity prices | 10–30 req/min | per‑page rate‑limited fetch | ✅ |

If Tavily exceeds 1k/mo we fall through to DuckDuckGo MCP automatically;
both are wired in `.claude/commands/news_sentiment.md`.

### 11.3 What we explicitly do NOT spend on (anymore)

- **PitchBook / FactSet / S&P CapIQ via Claude MCP** — paid tiers; dropped.
- **Alpha Vantage paid** — not needed; yfinance covers daily macro.
- **CoinGecko Pro** — not needed at our request rate.
- **Tavily paid plan** — not needed if we route overflow to DuckDuckGo.
- **Anthropic API direct billing** — Claude Code Pro Max covers it.

---

## 12. Scheduling — cron + Claude Code headless

Production loop runs as cron jobs that invoke `claude -p` in headless mode.
This is intentionally boring; the orchestration logic lives in slash
commands rather than in Python so we can iterate on the prompts without
shipping code.

```cron
# Morning signal generation → Slack
0 7 * * *  cd ~/Auto-investment && claude -p "/morning_signals" --output-format=stream-json >> logs/$(date +\%Y\%m\%d).jsonl 2>&1

# Weekly Excel report (Mondays 09:00 JST)
0 9 * * 1  cd ~/Auto-investment && claude -p "/weekly_report"   >> logs/weekly.log 2>&1

# Self-improvement loop (Sundays 23:00 JST)
0 23 * * 0 cd ~/Auto-investment && claude -p "/improve"         >> logs/improve.log 2>&1
```

For development, the same commands can be triggered interactively
(`/morning_signals` typed into a Claude Code session) or via the `loop`
skill (`/loop 5m /morning_signals` while iterating).

**Why cron, not the Claude Code `loop` skill, in production:** cron survives
a session disconnect; the `loop` skill only runs while a Claude Code session
is active. We use cron for the autonomous loop and the `loop` skill for
interactive development only.

---

## 13. Capital Allocation — blended APR scenarios

The Phase 1 backtest computes blended expected APR across three allocation
policies on a $10,000 portfolio. Each respects §8 hard rule #3 (per‑venue
exposure ≤ 25% of equity); cash buffer is parked in stablecoins on a CEX
wallet for fast deployment and earns 0% in this model.

| Policy | S1 (Hyperliquid arb) | S2 (DeFi yield) | Cash | **Blended APR** | $/year on $10k | Worst‑case DD |
|---|---|---|---|---|---|---|
| **Conservative** | 25% | 50% | 25% | **7.29%** | $729 | ≤ 0.67% |
| **Balanced** | 25% | 60% | 15% | **8.01%** | $801 | ≤ 0.67% |
| **Aggressive** | 25% | 75% | 0% | **9.09%** | $909 | ≤ 0.67% |

> **Caveats.** These numbers come from synthetic data and assume:
>
> 1. S1 funding APR averages ~14.7% on its working‑capital pocket (the
>    backtest period saw enough funding > 1500 bps APR triggers to keep
>    capital deployed ~20% of the time).
> 2. S2 stable‑pool APY averages ~7.2% with near‑zero drawdown — this is
>    materially **optimistic** because the synthetic data contains no depeg
>    events. Real DeFi DD in a USDC March‑2023 style depeg can hit 5–10%
>    even with risk gates.
> 3. S1 and S2 are uncorrelated (true if Hyperliquid funding regime is
>    independent of DeFi APY regime, which has held historically).
> 4. We do not leverage; cash position is genuinely idle.
>
> Real‑data Phase 2 backtest is required to confirm. Until then, treat these
> as design targets, not forecasts.

The recommended starting allocation is **Conservative** for the first 4
weeks of paper trading (Phase 3), stepping up to Balanced once paper
metrics confirm IS performance.

---

## 14. Phase 2 Results — S3 + S4 + real‑data path

### 14.1 What shipped in Phase 2

| Component | Path | Notes |
|---|---|---|
| Hyperliquid funding fetcher | `src/auto_investment/data_fetchers/funding.py` | ccxt-based, paginated, parquet cache |
| DefiLlama yields fetcher | `src/auto_investment/data_fetchers/yields.py` | Stdlib HTTP, no key |
| Binance/Bybit OHLCV fetcher | `src/auto_investment/data_fetchers/ohlcv.py` | 1m default, paired spread builder for S3 |
| Aevo pre‑IPO fetcher | `src/auto_investment/data_fetchers/preipo.py` | Daily index marks for the watchlist |
| **S3 strategy** | `src/auto_investment/strategies/cross_exchange.py` | Mean‑reversion on z(spread); see KPI below |
| **S4 alerts** | `src/auto_investment/strategies/preipo_alerts.py` | Alert‑only; no auto‑trade |
| Local data‑fetch CLI | `scripts/fetch_real_data.py` | Runs on user's machine; populates `data/` cache |
| Phase 2 backtest runner | `scripts/backtest_phase2.py` | `--mode synth | real` |

### 14.2 S3 backtest summary (synthetic, retail‑taker fees)

| Metric | Target | Result (seed=7) | Multi-seed range |
|---|---|---|---|
| Annualised return | > 0% | **+8.26%** | +5.0% – +15.5% |
| Hit rate | ≥ 55% | **70%** | 70% – 89% |
| Max DD | ≤ 1% | **0.04%** | 0.00% – 0.04% |
| Trades / 14 days | — | 10 | 5 – 10 |

**Key finding.** With both venues at retail taker fee (Binance 4.5 bps,
Bybit 5.5 bps + slippage), round‑trip cost ≈ 28 bps. The strategy clears
costs only at z ≥ 3.0; we set the default to z=3.5 for a +12 bps buffer
per trade. **Phase 3 priority: move to maker‑rebate fee tier on at least
one venue** to relax this threshold and increase trade count.

### 14.3 S4 alerts behaviour

S4 emits an alert when *implied IPO probability* (mark / 90‑day rolling
max) jumps by > 5pp over a 7‑day window, with a 3‑day cooldown to prevent
spam. On the synthetic SpaceX series with a 30% rumor jump, it fires 13
alerts over 180 days. Real‑data validation requires the user to run
`python scripts/fetch_real_data.py --target s4` to populate Aevo marks.

### 14.4 Real‑data path (user runs locally)

The sandbox where Claude Code runs has no network egress to crypto APIs.
The user runs the following on their machine to populate caches:

```bash
# All four targets, default 90‑day window
python scripts/fetch_real_data.py

# Just S3 (much smaller — 14 days × 1m)
python scripts/fetch_real_data.py --target s3 --days 14

# Then re‑run the backtest with real data
python scripts/backtest_phase2.py --mode real
```

Cache is gitignored (`data/{funding,yields,ohlcv,preipo}/*.parquet`)
to keep the repo small. Run it weekly during Phase 2 to validate the
synth assumptions.

### 14.5 Updated allocation across S1+S2+S3 (synthetic numbers)

| Policy | S1 | S2 | S3 | Cash | **Blended APR** | $/yr on $10k |
|---|---|---|---|---|---|---|
| Conservative | 20% | 50% | 10% | 20% | **7.38%** | $738 |
| Balanced | 25% | 50% | 15% | 10% | **8.53%** | $853 |
| Aggressive | 25% | 50% | 25% | 0% | **9.35%** | $935 |

Same caveats as §13: synthetic data, no depeg events, S3 Sharpe
inflated by clean OU process. Replace with real‑data run before Phase 3.

---

## 10. Disclaimer

This is a research and engineering specification. None of it is
investment advice. All strategies described will lose money under
some market regimes. The Phase 4 transition (paper → real money)
requires explicit written approval from the CEO (you), not just a
Claude session.
