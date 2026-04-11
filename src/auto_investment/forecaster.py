"""Optional time-series forecaster — Google TimesFM with graceful fallback.

TimesFM (https://github.com/google-research/timesfm) is a 200M-parameter
foundation model for zero-shot time-series forecasting. It needs no training
on your data — feed it a univariate context, get a forecast + quantiles.

Install (optional, heavy):
    pip install timesfm

If `timesfm` isn't installed, this module falls back to a naive ARIMA-style
projection (last value + drift) so the rest of the system stays runnable.

Usage:
    from auto_investment.forecaster import forecast_close
    fc = forecast_close(df["close"], horizon=24)
    print(fc.point, fc.lower, fc.upper)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Forecast:
    """Forecast result with central estimate + 80% confidence interval."""

    point: list[float]
    lower: list[float]
    upper: list[float]
    horizon: int
    backend: str  # "timesfm" or "naive"

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "horizon": self.horizon,
            "backend": self.backend,
        }

    def summary(self) -> str:
        """Human-readable single-line summary for the AI prompt."""
        if not self.point:
            return "no forecast available"
        first, last = self.point[0], self.point[-1]
        change_pct = ((last - first) / first * 100.0) if first else 0.0
        return (
            f"{self.backend} forecast over {self.horizon} bars: "
            f"{first:.2f} → {last:.2f} ({change_pct:+.2f}%), "
            f"80% CI at horizon: [{self.lower[-1]:.2f}, {self.upper[-1]:.2f}]"
        )


_timesfm_model: Optional[object] = None  # cached singleton


def _load_timesfm():
    """Lazily load and cache the TimesFM model. Returns None on failure."""
    global _timesfm_model
    if _timesfm_model is not None:
        return _timesfm_model
    try:
        import timesfm  # noqa: PLC0415

        # 200M checkpoint, CPU-friendly defaults. Users with a GPU can rebuild
        # this with backend="gpu".
        _timesfm_model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=128,
                num_layers=20,
                context_len=512,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
            ),
        )
        logger.info("TimesFM loaded")
    except Exception as exc:  # noqa: BLE001
        logger.info("TimesFM unavailable (%s) — using naive forecaster", exc)
        _timesfm_model = None
    return _timesfm_model


def forecast_close(close: pd.Series, horizon: int = 24) -> Forecast:
    """Forecast the next `horizon` close values using TimesFM if available.

    Falls back to a naive drift forecast (last-N-bar mean change extrapolation)
    when TimesFM isn't installed. Both paths return the same `Forecast` shape.
    """
    if len(close) < 32:
        raise ValueError(f"Need at least 32 bars of context, got {len(close)}")

    model = _load_timesfm()
    if model is not None:
        return _timesfm_forecast(model, close, horizon)
    return _naive_forecast(close, horizon)


def _timesfm_forecast(model, close: pd.Series, horizon: int) -> Forecast:
    """Run TimesFM zero-shot forecast on the close series."""
    series = close.to_numpy(dtype=np.float64)
    # TimesFM expects a list of 1-D arrays + a frequency hint (0 = high freq).
    point_forecast, quantile_forecast = model.forecast(
        [series],
        freq=[0],
    )
    point = point_forecast[0][:horizon].tolist()
    # quantile_forecast shape: (n_series, horizon, n_quantiles)
    # Default quantiles include 0.1 and 0.9 — use those as 80% CI.
    q = quantile_forecast[0]
    lower = q[:horizon, 1].tolist()  # 0.1 quantile
    upper = q[:horizon, -2].tolist()  # 0.9 quantile
    return Forecast(point=point, lower=lower, upper=upper, horizon=horizon, backend="timesfm")


def _naive_forecast(close: pd.Series, horizon: int) -> Forecast:
    """Fallback: last value + recent drift, with widening confidence band.

    Not a real forecaster — just a placeholder so downstream code (server,
    frontend, AI advisor) keeps working when TimesFM isn't installed.
    """
    last = float(close.iloc[-1])
    recent = close.iloc[-30:]
    drift_per_bar = float((recent.iloc[-1] - recent.iloc[0]) / max(len(recent) - 1, 1))
    vol = float(recent.pct_change().std() * last)  # rough sigma per bar in price units

    point = []
    lower = []
    upper = []
    for i in range(1, horizon + 1):
        p = last + drift_per_bar * i
        band = 1.28 * vol * np.sqrt(i)  # ~80% interval, widening with sqrt(t)
        point.append(p)
        lower.append(p - band)
        upper.append(p + band)
    return Forecast(point=point, lower=lower, upper=upper, horizon=horizon, backend="naive")
