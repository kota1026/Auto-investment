"""Information Coefficient (IC) — signal quality metric.

The IC is the rank correlation between a strategy's predicted signal at time
t and the realized forward return at t+1. It's the standard quant-fund
quality metric for an alpha signal:

  - IC > 0  → signal has positive predictive power
  - IC = 0  → signal is noise
  - IC < 0  → signal is wrong-way (consider flipping it)

ICIR (Information Coefficient Information Ratio) is the mean IC divided by
the IC standard deviation — higher means the signal is consistently good
rather than occasionally lucky.

The JSAI paper "大規模言語モデルを用いた株式投資戦略の自動生成における
フィードバック設計" (Kawamura, Kubo, Nakagawa 2026) showed that adding IC
and ICIR to the LLM feedback prompt (their P2 condition) caused Claude to
implement style-factor neutralization more often. We expose IC/ICIR via this
module so the AI advisor can reference them.

We compute IC on a univariate price series since the rest of our system is
single-symbol. For a true cross-sectional IC (which is what the paper uses),
extend `cross_sectional_ic` to multi-asset panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICReport:
    """Aggregated IC statistics for a strategy signal."""

    mean_ic: float
    std_ic: float
    icir: float
    n_obs: int

    def to_dict(self) -> dict:
        return {
            "mean_ic": self.mean_ic,
            "std_ic": self.std_ic,
            "icir": self.icir,
            "n_obs": self.n_obs,
        }

    def summary(self) -> str:
        if self.n_obs == 0:
            return "no IC data"
        return (
            f"IC mean={self.mean_ic:+.4f}, std={self.std_ic:.4f}, "
            f"ICIR={self.icir:+.3f} ({self.n_obs} obs)"
        )


def rolling_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    window: int = 30,
    method: str = "pearson",
) -> pd.Series:
    """Rolling correlation between signal and next-bar forward returns.

    Args:
        signal: A real-valued signal series indexed by timestamp. For an
            EMA-cross strategy this might be `ema_fast - ema_slow`.
        forward_returns: A returns series of the same index. Should be
            *forward* returns (the return realized AFTER the signal time).
        window: Rolling window size in bars.
        method: "pearson" (default — fast, no scipy needed). "spearman"
            requires scipy.

    Note:
        pandas rolling().corr() does not accept a `method` parameter — it
        always uses Pearson. For Spearman, we'd need a manual loop with
        scipy.stats.spearmanr; we omit that here to keep dependencies light.
    """
    if len(signal) != len(forward_returns):
        raise ValueError("signal and forward_returns must have the same length")
    if len(signal) < window:
        return pd.Series(dtype=float)

    s = pd.Series(signal.values)
    r = pd.Series(forward_returns.values)
    # pandas rolling Pearson correlation
    return s.rolling(window).corr(r)


def ic_report(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "pearson",
) -> ICReport:
    """Compute mean IC, std IC, and ICIR over a per-bar IC series.

    The "per-bar IC" here is computed as a single batch correlation rather
    than rolling — for a single-symbol time series, the rolling alternative
    introduces look-ahead artifacts.

    For a multi-asset cross-sectional IC, use `cross_sectional_ic` below.
    """
    df = pd.DataFrame({"sig": signal, "ret": forward_returns}).dropna()
    if len(df) < 10:
        return ICReport(mean_ic=0.0, std_ic=0.0, icir=0.0, n_obs=len(df))

    # Bootstrap IC distribution by chunking the series into 30-bar blocks
    # and computing IC per block. This gives us mean/std/ICIR.
    block = 30
    blocks = [df.iloc[i : i + block] for i in range(0, len(df) - block + 1, block)]
    if len(blocks) < 2:
        # Just compute a single batch IC
        ic = df["sig"].corr(df["ret"], method=method)
        return ICReport(mean_ic=float(ic) if not np.isnan(ic) else 0.0,
                        std_ic=0.0, icir=0.0, n_obs=len(df))

    block_ics = []
    for b in blocks:
        if len(b) < 10:
            continue
        c = b["sig"].corr(b["ret"], method=method)
        if not np.isnan(c):
            block_ics.append(float(c))
    if len(block_ics) < 2:
        return ICReport(mean_ic=0.0, std_ic=0.0, icir=0.0, n_obs=len(df))

    arr = np.asarray(block_ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1))
    icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
    return ICReport(mean_ic=round(mean_ic, 4), std_ic=round(std_ic, 4),
                    icir=round(icir, 3), n_obs=len(df))


def signal_from_indicators(df: pd.DataFrame) -> pd.Series:
    """Build a real-valued signal series from indicator-augmented OHLCV.

    Used as the input to `rolling_ic` / `ic_report`. The signal here is the
    normalized EMA distance — positive when fast EMA is above slow EMA, and
    scaled by ATR for stability across volatility regimes.
    """
    if not {"ema_fast", "ema_slow", "atr"}.issubset(df.columns):
        raise ValueError("DataFrame must have ema_fast, ema_slow, atr columns")
    raw = (df["ema_fast"] - df["ema_slow"]) / df["atr"].replace(0, np.nan)
    return raw.fillna(0.0)


def forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute forward returns over `horizon` bars.

    A horizon of 1 means "the return realized over the bar after the signal
    was observed" — the standard IC convention.
    """
    return close.pct_change(periods=horizon).shift(-horizon)


def cross_sectional_ic(
    signals_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """Cross-sectional IC for a multi-asset panel.

    Both inputs are wide DataFrames (rows=time, cols=assets). For each row,
    the function computes the rank correlation across assets between the
    signal vector and the next-period return vector. Returns a per-time-step
    IC series — its mean and std give you the standard cross-sectional ICIR.

    Use this when you extend the system to a basket of symbols (the JSAI
    paper does this on TOPIX 500 minus financials).
    """
    if signals_panel.shape != returns_panel.shape:
        raise ValueError("signals_panel and returns_panel must have the same shape")
    ic_series = pd.Series(index=signals_panel.index, dtype=float)
    for ts in signals_panel.index:
        s = signals_panel.loc[ts]
        r = returns_panel.loc[ts]
        valid = s.notna() & r.notna()
        if valid.sum() < 5:
            continue
        ic_series.loc[ts] = s[valid].corr(r[valid], method=method)
    return ic_series.dropna()
