"""Strategy plug-ins for the alpha arena.

Each strategy module exposes:
  - `generate_signals(...)` returning a DataFrame with entry/exit decisions
  - `backtest(...)` returning a uniform stats record

Why split this out of the existing `strategy.py`: that module is the single
EMA/RSI strategy; this package is for the four strategies described in
`docs/strategy_spec.md` §3.
"""
