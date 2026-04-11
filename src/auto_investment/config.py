"""Runtime configuration loaded from environment variables.

Keep secrets in `.env` (gitignored). Never commit API keys.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    ai_enabled: bool = Field(default=True, alias="AI_ENABLED")
    ai_model: str = Field(default="claude-opus-4-6", alias="AI_MODEL")
    ai_min_confidence: float = Field(default=0.6, alias="AI_MIN_CONFIDENCE")

    # --- Exchange ---
    exchange_id: str = Field(default="binance", alias="EXCHANGE_ID")
    exchange_api_key: str = Field(default="", alias="EXCHANGE_API_KEY")
    exchange_api_secret: str = Field(default="", alias="EXCHANGE_API_SECRET")
    exchange_testnet: bool = Field(default=True, alias="EXCHANGE_TESTNET")

    # --- Trading parameters ---
    symbol: str = Field(default="BTC/USDT", alias="SYMBOL")
    timeframe: str = Field(default="1h", alias="TIMEFRAME")
    equity_usd: float = Field(default=10_000.0, alias="EQUITY_USD")
    risk_per_trade: float = Field(default=0.01, alias="RISK_PER_TRADE")

    # --- Server ---
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")


settings = Settings()
