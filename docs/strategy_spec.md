# Auto‑Investment Strategy Specification (v0.1)

**Status:** Draft for review (2026‑05‑09)
**Branch:** `claude/crypto-portfolio-automation-gEv4B`
**Author:** Claude Code (auto‑drafted, awaiting human review)

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
- Sharpe ≥ 2.0 net of fees
- Max DD ≤ 5%
- Hit rate ≥ 65%

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
# CEX
TAKER_BPS = {"binance": 4.5, "bybit": 5.5, "okx": 5.0}
MAKER_BPS = {"binance": 1.0, "bybit": 1.0, "okx": 0.8}
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

## 9. Open Questions for the User (CEO) before Phase 1 starts

1. Which **CEX** do you actually have funded accounts on? (drives S1 & S3 venue list)
2. Do you have a **Web3 wallet** ready for L2 DeFi, or do we provision a fresh one with `cast wallet new` on Phase 3 day 1?
3. Are you willing to fund the **Tavily**, **Alpha Vantage**, **PitchBook (via Claude MCP)** keys for Phase 2/3? Approximate monthly ≤ $200.
4. **Reporting cadence**: daily Slack? Weekly Excel via Claude for Excel? Both?
5. **Stop‑the‑world rule.** Do you want a "panic key" command that liquidates everything to USDC and halts all agents? (recommended)

---

## 10. Disclaimer

This is a research and engineering specification. None of it is
investment advice. All strategies described will lose money under
some market regimes. The Phase 4 transition (paper → real money)
requires explicit written approval from the CEO (you), not just a
Claude session.
