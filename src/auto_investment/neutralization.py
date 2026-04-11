"""Style factor neutralization — hedge a strategy signal against common factors.

The JSAI paper showed that giving an LLM the strategy's factor exposure (their
P2 condition) caused models to implement neutralization more often. The paper
neutralizes against 17 Barra-style factors: BPR, EarningsYield, Size, MidCap,
ShortTermMomentum, MidTermMomentum, LongTermMomentum, Beta, ResidualVolatility,
EarningsQuality, EarningsVariability, InvestmentQuality, Leverage, Profitability,
DividendYield, Growth, Liquidity.

For our single-symbol crypto setting, "style factor neutralization" doesn't
quite apply, but the idea generalizes: regress the signal against a set of
explanatory series (e.g. recent volatility, recent volume, time-of-day) and
keep only the residual. The residual is the part of the signal that isn't
explained by those factors — i.e., the actual alpha.

For a true multi-asset Barra-style neutralization, extend this to take a
factor-loading panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeutralizationReport:
    """Result of a neutralization run."""

    raw_std: float
    residual_std: float
    explained_fraction: float  # 1 - var(residual) / var(raw)
    factor_loadings: dict

    def to_dict(self) -> dict:
        return {
            "raw_std": self.raw_std,
            "residual_std": self.residual_std,
            "explained_fraction": self.explained_fraction,
            "factor_loadings": self.factor_loadings,
        }


def build_factor_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small set of explanatory factors from an OHLCV+indicator frame.

    Each factor is a per-bar series. The factors here are deliberately simple
    and computable from local data:

      - vol_20:  20-bar realized volatility of returns
      - mom_20:  20-bar momentum (close pct change)
      - volume_z: 20-bar z-score of volume
      - atr_pct: ATR as a fraction of price (volatility regime)

    For a real cross-sectional Barra style, replace this with sized vectors
    against (Size, Value, Momentum, Quality, ...).
    """
    out = pd.DataFrame(index=df.index)
    rets = df["close"].pct_change()
    out["vol_20"] = rets.rolling(20).std()
    out["mom_20"] = df["close"].pct_change(20)
    vol = df["volume"].astype(float)
    out["volume_z"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    if "atr" in df.columns:
        out["atr_pct"] = df["atr"] / df["close"]
    return out.fillna(0.0)


def neutralize(signal: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, NeutralizationReport]:
    """Regress `signal` against `factors` (OLS) and return the residual.

    Uses the closed-form OLS solution: β = (XᵀX)⁻¹ Xᵀy. No intercept term —
    factors should already be demeaned per row.

    Returns a tuple of (residual_signal, report). The report tells you how
    much of the signal's variance was explained by the factor panel — large
    explained_fraction means the raw signal was mostly noise from common
    factors and the residual carries the real alpha.
    """
    df = pd.concat([signal.rename("y"), factors], axis=1).dropna()
    if len(df) < 30:
        # Not enough data — return raw signal unchanged
        return signal, NeutralizationReport(
            raw_std=float(signal.std() or 0.0),
            residual_std=float(signal.std() or 0.0),
            explained_fraction=0.0,
            factor_loadings={},
        )

    y = df["y"].to_numpy(dtype=np.float64)
    X = df[factors.columns].to_numpy(dtype=np.float64)

    # Demean for stability (we don't add a separate intercept)
    y_centered = y - y.mean()
    X_centered = X - X.mean(axis=0)

    try:
        beta, *_ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
    except np.linalg.LinAlgError:
        return signal, NeutralizationReport(
            raw_std=float(signal.std() or 0.0),
            residual_std=float(signal.std() or 0.0),
            explained_fraction=0.0,
            factor_loadings={},
        )

    fitted = X_centered @ beta
    residual = y_centered - fitted

    raw_std = float(np.std(y_centered))
    res_std = float(np.std(residual))
    explained = 1.0 - (res_std**2 / raw_std**2) if raw_std > 0 else 0.0

    loadings = {col: float(round(b, 6)) for col, b in zip(factors.columns, beta)}

    # Reindex residual back onto the original signal index
    residual_series = pd.Series(residual, index=df.index, name=signal.name)
    residual_series = residual_series.reindex(signal.index).fillna(0.0)

    return residual_series, NeutralizationReport(
        raw_std=round(raw_std, 6),
        residual_std=round(res_std, 6),
        explained_fraction=round(explained, 4),
        factor_loadings=loadings,
    )
