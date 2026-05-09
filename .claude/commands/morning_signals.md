---
description: Pull market data, score S1+S2 signals, and emit a Slack-ready summary
---

# /morning_signals

You are the **Morning Analyst** for the Auto-Investment system. This command
runs every morning at 07:00 JST via cron (see `docs/strategy_spec.md` §12).
Its job is to produce one actionable summary covering the four strategies in
`src/auto_investment/strategies/`.

## Operating principles

1. **Read first, write last.** Do all data fetches and computation before you
   write any decision into `results/decisions.jsonl`.
2. **Be honest about uncertainty.** If funding history is too short, say so
   and skip the signal — don't extrapolate.
3. **No new orders are placed by this command.** It only proposes them.
   The human (CEO) approves via reply-in-Slack before anything fills.
4. **Stay inside the data budget.** Do not call Tavily unless S4 explicitly
   triggers; favour ccxt + DefiLlama (both unmetered).

## Steps

1. **Load context**:
   - `cat docs/strategy_spec.md` for the rules
   - `cat results/decisions.jsonl | tail -50` for prior context
   - `cat data/positions.json` (if it exists) for current exposure
2. **S1 — Funding arb**:
   - Use `python -c "from auto_investment.data import fetch_funding_rate_history; ..."` to pull last 72h of Hyperliquid BTC + ETH funding
   - Compute `smoothed_apr_bps` (EWMA, halflife 6h)
   - If `apr > 800 bps` and we are flat, **propose** a $2k delta-neutral entry
   - If `apr < 200 bps` and we are in, **propose** unwind
3. **S2 — Yield router**:
   - Pull DefiLlama Yields top 20 USDC pools on Arbitrum/Base/Optimism
   - Filter by spec §3 risk gates (TVL ≥ $20M, audits ≥ 1, age ≥ 90d)
   - Compare current pool's forecast EWMA APY to the best alternative
   - Propose a rotation if uplift over 7-day window > 2× rotation cost
4. **S3 — Cross-exchange stat-arb**:
   - Phase 2 only — print "skipped (Phase 2)" and continue
5. **S4 — Pre-IPO**:
   - Phase 3 only — print "skipped (Phase 3)" and continue
6. **Risk officer cross-check**:
   - Per-venue exposure ≤ 25% of equity (spec §8 hard rule #3)
   - Per-trade max loss ≤ 0.5% of equity (spec §8 hard rule #1)
7. **Emit Slack message**: write to `results/slack_morning_$(date +%Y%m%d).md`
   in this format:

```
🌅 *Morning brief — YYYY-MM-DD*
*Equity:* $X,XXX | *Day P&L:* +/-$XX

S1 (Funding arb, Hyperliquid)
  • BTC funding APR: XX.X% (smoothed)
  • Action: <ENTER $2k | HOLD | EXIT>
  • Reason: <one-liner>

S2 (DeFi yield)
  • Current: <pool> @ X.X% APY
  • Best alternative: <pool> @ X.X% APY (uplift +XX bps)
  • Action: <ROTATE | HOLD>

Risk
  • Venue exposure: HL XX%, BinanceJP XX%, DeFi XX%
  • Headroom to per-day -2% DD limit: XX%

Approve any action by replying:  approve <S1|S2>
```

8. **Append to ledger**: write the proposed actions as a JSON record
   into `results/decisions.jsonl` with fields:
   `{ts, command, strategy, action, params, reason, status:"proposed"}`.

## Output format

Return only the Slack message body to stdout. The cron wrapper redirects
this to `logs/<date>.jsonl` and a downstream notifier picks it up.

## Failure modes you must handle

- **ccxt fetch fails** (network): use `synthetic_ohlcv` fallback only for
  development; in production, abort with a Slack message saying "data feed
  down" and **do not** propose any action.
- **Funding history < 24h**: print "insufficient history" and skip S1.
- **DefiLlama 5xx**: retry once with 30s backoff, then skip S2.
- **Disagreement with risk officer**: surface the conflict in Slack, do not
  override.
