"""
macro_feeds.py — fetch helpers for FRED, Alpaca, and EODHD.

IMPORTANT: these were written and syntax-checked but never executed against
the live APIs — the sandbox this was built in has no network access. Test
them locally before relying on them. All three need API keys read from
environment variables (see .env / README.md) — never hardcode keys here.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # pulls FRED_API_KEY / ALPACA_API_KEY / ALPACA_API_SECRET / EODHD_API_KEY from .env

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"
EODHD_BASE = "https://eodhd.com/api"


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing environment variable {name}. Put it in your .env file.")
    return val


# ──────────────────────────────────────────────────────────────────────────
# FRED — macro series (rates, credit spreads, VIX-adjacent series, etc.)
# ──────────────────────────────────────────────────────────────────────────
# Useful series IDs for the StructuralFilterEngine's proxy inputs:
#   VIXCLS   - CBOE Volatility Index (front-month VIX)
#   BAMLH0A0HYM2 - ICE BofA US High Yield OAS  (-> hy_oas)
#   T10Y2Y   - 10Y-2Y Treasury spread (general macro stress)
#   NFCI     - Chicago Fed National Financial Conditions Index (liquidity proxy)
FRED_SERIES = {
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
    "t10y2y": "T10Y2Y",
    "nfci": "NFCI",
}


def fetch_fred_series(series_id: str, start: str | None = None, end: str | None = None,
                       api_key: str | None = None) -> pd.DataFrame:
    api_key = api_key or _require_env("FRED_API_KEY")
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    return df[["date", "value"]].rename(columns={"value": series_id})


def fetch_fred_macro_bundle(start: str | None = None, end: str | None = None,
                             api_key: str | None = None) -> pd.DataFrame:
    """Fetches all FRED_SERIES and joins them on date — a ready-made input
    table for StructuralProxyInputs (hy_oas) and general macro context."""
    frames = []
    for _, series_id in FRED_SERIES.items():
        try:
            frames.append(fetch_fred_series(series_id, start, end, api_key))
        except Exception as e:
            print(f"[macro_feeds] WARNING: failed to fetch FRED series {series_id}: {e}")
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Alpaca — historical OHLCV bars (equities/ETFs; NQ/ES futures are not on
# Alpaca — use your futures data vendor for those, as you've already done)
# ──────────────────────────────────────────────────────────────────────────
def fetch_alpaca_bars(symbol: str, timeframe: str = "5Min",
                       start: str | None = None, end: str | None = None,
                       api_key: str | None = None, api_secret: str | None = None,
                       limit: int = 10_000) -> pd.DataFrame:
    """timeframe examples: '1Min', '5Min', '30Min', '1Hour', '1Day'."""
    api_key = api_key or _require_env("ALPACA_API_KEY")
    api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"timeframe": timeframe, "limit": limit, "adjustment": "raw"}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    bars = []
    url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        bars.extend(data.get("bars", []))
        page_token = data.get("next_page_token")
        if not page_token:
            break

    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                             "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ──────────────────────────────────────────────────────────────────────────
# EODHD — historical EOD options chains
# ──────────────────────────────────────────────────────────────────────────
def fetch_eodhd_option_chain(symbol: str, trade_date: str | date,
                              api_key: str | None = None) -> pd.DataFrame:
    """symbol e.g. 'QQQ.US'. trade_date 'YYYY-MM-DD'. Returns the same shape
    as load_option_chain_csv() produces, for drop-in compatibility."""
    api_key = api_key or _require_env("EODHD_API_KEY")
    url = f"{EODHD_BASE}/options/{symbol}"
    params = {"api_token": api_key, "from": str(trade_date), "to": str(trade_date), "fmt": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for snap in data.get("data", []):
        for opt in snap.get("options", {}).get("CALL", []) + snap.get("options", {}).get("PUT", []):
            rows.append(opt)
    return pd.DataFrame(rows)


def fetch_eodhd_option_chain_range(symbol: str, start: str | date, end: str | date,
                                    api_key: str | None = None) -> pd.DataFrame:
    """Fetches and concatenates daily EOD chain snapshots across a date
    range — this is what you need to build a Φ_t-ready, bar-aligned options
    history rather than the single-day snapshots in the current upload batch."""
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for d in dates:
        try:
            frames.append(fetch_eodhd_option_chain(symbol, d.strftime("%Y-%m-%d"), api_key))
        except Exception as e:
            print(f"[macro_feeds] WARNING: failed {symbol} chain for {d.date()}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────
# Polygon.io — historical underlying bars + options reference/aggregates.
# NOTE: deep historical EOD options-chain history (the thing actually
# needed to align with your 2026-06 bar window) is most reliably gotten via
# Polygon's "Flat Files" S3 bulk-download product, not these REST endpoints
# — REST aggregates are per-contract and require already knowing the
# contract ticker. Worth checking your plan tier before building a full
# backfill pipeline on top of fetch_polygon_option_aggs below.
# ──────────────────────────────────────────────────────────────────────────
POLYGON_BASE = "https://api.polygon.io"


def fetch_polygon_bars(ticker: str, multiplier: int, timespan: str,
                        start: str, end: str, api_key: str | None = None,
                        adjusted: bool = True, limit: int = 50_000) -> pd.DataFrame:
    """timespan: 'minute' | 'hour' | 'day'. ticker: 'SPY', 'I:SPX', 'X:..' etc."""
    api_key = api_key or _require_env("POLYGON_API_KEY")
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": limit, "apiKey": api_key}
    rows = []
    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        url, params = next_url, {"apiKey": api_key}
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"t": "ts_ms", "o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_polygon_option_contracts(underlying: str, as_of: str | None = None,
                                    api_key: str | None = None) -> pd.DataFrame:
    """Lists option contracts for an underlying (optionally as of a past
    date) — gives you contract tickers to then pull aggregates for."""
    api_key = api_key or _require_env("POLYGON_API_KEY")
    url = f"{POLYGON_BASE}/v3/reference/options/contracts"
    params = {"underlying_ticker": underlying, "limit": 1000, "apiKey": api_key}
    if as_of:
        params["as_of"] = as_of
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json().get("results", []))


def fetch_polygon_option_aggs(option_ticker: str, start: str, end: str,
                               api_key: str | None = None) -> pd.DataFrame:
    """option_ticker format: 'O:SPY230721C00400000'."""
    return fetch_polygon_bars(option_ticker, 1, "day", start, end, api_key)


# ──────────────────────────────────────────────────────────────────────────
# Tradier — live/near-term option chains. Tradier's standard market-data
# API is oriented at current/live chains, not deep multi-year EOD history —
# don't expect this to backfill 2021-2023 the way Polygon flat files or
# EODHD's historical endpoint can.
# ──────────────────────────────────────────────────────────────────────────
TRADIER_BASE = "https://api.tradier.com/v1"


def fetch_tradier_option_chain(symbol: str, expiration: str,
                                api_key: str | None = None) -> pd.DataFrame:
    """expiration: 'YYYY-MM-DD'. Returns the live/current chain for that
    expiration — see the module docstring re: historical depth limits."""
    api_key = api_key or _require_env("TRADIER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    params = {"symbol": symbol, "expiration": expiration, "greeks": "true"}
    resp = requests.get(f"{TRADIER_BASE}/markets/options/chains",
                         headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    options = resp.json().get("options", {}) or {}
    return pd.DataFrame(options.get("option", []))


def fetch_tradier_expirations(symbol: str, api_key: str | None = None) -> list[str]:
    api_key = api_key or _require_env("TRADIER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    resp = requests.get(f"{TRADIER_BASE}/markets/options/expirations",
                         headers=headers, params={"symbol": symbol}, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("expirations", {}) or {}
    return data.get("date", [])
