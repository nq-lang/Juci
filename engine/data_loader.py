"""
data_loader.py — robust OHLCV / option-chain ingestion for the HybridSigma
quant terminal.

Handles the multiple vendor dialects observed in the user's uploaded samples:
  • Standard Alpaca-style CSV: timestamp,open,high,low,close[,volume]
  • SPX-style (no volume column, index has no traded volume)
  • European-format export ("Time Series;<SYM>;..." header, ';'-delimited,
    period-as-thousands / comma-as-decimal numbers, "M/D/YYYY H:MM AM/PM"
    timestamps, newest-row-first ordering)

None of the bar files include real buy/sell order-flow. A bar-level proxy
is synthesized from OHLC (documented below) — this is an approximation,
not real tape data, and is labeled as such everywhere it surfaces in the UI.
"""
from __future__ import annotations

import io
import re
import csv as _csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STANDARD_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# ──────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────
def _sniff_delimiter(sample_text: str) -> str:
    # European exports use ';' throughout (header AND data rows).
    first_lines = sample_text.splitlines()[:3]
    semi = sum(line.count(";") for line in first_lines)
    comma = sum(line.count(",") for line in first_lines)
    return ";" if semi > comma else ","


def _looks_european_numeric(series_sample: list[str]) -> bool:
    # European: "25.640,75" (period=thousands, comma=decimal).
    # US:       "25640.75" or "25,640.75"
    pattern = re.compile(r"^\d{1,3}(\.\d{3})*,\d+$")
    hits = sum(1 for v in series_sample if pattern.match(v.strip()))
    return hits >= max(1, len(series_sample) // 2)


def _to_european_float(x: str) -> float:
    if x is None or x == "":
        return np.nan
    return float(x.replace(".", "").replace(",", "."))


# ──────────────────────────────────────────────────────────────────────────
# Main bar loader
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class LoadedBars:
    df: pd.DataFrame          # standardized columns, ascending timestamp, UTC ms in df['ts_ms']
    source_path: str
    vendor_format: str        # 'standard' | 'no_volume' | 'european'
    n_rows: int
    start: pd.Timestamp
    end: pd.Timestamp


def load_ohlcv_csv(path: str | Path) -> LoadedBars:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Skip a leading vendor banner row like "Time Series;NQH26;;;;;"
    lines = raw.splitlines()
    header_idx = 0
    for i, line in enumerate(lines[:5]):
        low = line.lower()
        if ("date" in low or "timestamp" in low or "time" in low) and (
            "open" in low or "close" in low
        ):
            header_idx = i
            break
    trimmed = "\n".join(lines[header_idx:])

    delim = _sniff_delimiter(trimmed)
    df = pd.read_csv(io.StringIO(trimmed), sep=delim, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize column names across vendors
    rename_map = {}
    for c in df.columns:
        if c in ("date", "time", "datetime"):
            rename_map[c] = "timestamp"
        elif c == "symbol":
            rename_map[c] = "symbol"
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: could not find a date/timestamp column "
                          f"(found columns: {list(df.columns)})")

    vendor_format = "standard"

    # Detect European numeric formatting on the 'open' column (or 'close' if missing)
    probe_col = "open" if "open" in df.columns else df.columns[1]
    sample_vals = df[probe_col].astype(str).head(20).tolist()
    if _looks_european_numeric(sample_vals):
        vendor_format = "european"
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(str).map(_to_european_float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %I:%M %p",
                                          errors="coerce")
    else:
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)

    if "volume" not in df.columns:
        vendor_format = "no_volume" if vendor_format == "standard" else vendor_format
        df["volume"] = 0.0

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(
        subset=["timestamp", "open", "high", "low", "close"]
    )
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    # Robust against pandas' internal datetime64 resolution (ns/us/s — version-
    # dependent as of pandas 3.x): force ms resolution via numpy before casting.
    df["ts_ms"] = df["timestamp"].values.astype("datetime64[ms]").astype("int64")

    return LoadedBars(
        df=df, source_path=str(path), vendor_format=vendor_format,
        n_rows=len(df),
        start=df["timestamp"].iloc[0] if len(df) else pd.NaT,
        end=df["timestamp"].iloc[-1] if len(df) else pd.NaT,
    )


# ──────────────────────────────────────────────────────────────────────────
# Order-flow proxy (NOT real tape data — documented approximation)
# ──────────────────────────────────────────────────────────────────────────
def add_orderflow_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds buy_vol / sell_vol / cum_delta using the bar's close-location-value
    (where price closed within the bar's high-low range) as a proxy for
    buy/sell pressure — the same logic behind Chaikin Money Flow / Williams
    Accumulation-Distribution. This is an OHLC-derived approximation, not
    real trade-by-trade tape data. cum_delta here is the PER-BAR net delta
    (buy_vol - sell_vol for that bar), matching the C++ engine's semantics —
    not a running cumulative total across bars.
    """
    out = df.copy()
    rng = (out["high"] - out["low"]).clip(lower=1e-9)
    clv = ((out["close"] - out["low"]) / rng).clip(0.0, 1.0)
    out["buy_vol"] = out["volume"] * clv
    out["sell_vol"] = out["volume"] * (1.0 - clv)
    out["cum_delta"] = out["buy_vol"] - out["sell_vol"]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Option chain loader (EODHD-style EOD snapshot export)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class LoadedChain:
    df: pd.DataFrame
    source_path: str
    trade_dates: list[str]
    n_rows: int


def load_option_chain_csv(path: str | Path) -> LoadedChain:
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    rename = {
        "trade_date": "trade_date", "strike": "strike", "expiry_date": "expiry_date",
        "call_put": "call_put", "last_trade_price": "last_price",
        "bid_price": "bid", "ask_price": "ask",
        "bid_implied_volatility": "bid_iv", "ask_implied_volatility": "ask_iv",
        "open_interest": "open_interest", "volume": "volume",
        "delta": "delta", "gamma": "gamma", "vega": "vega", "theta": "theta", "rho": "rho",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    df["call_put"] = df["call_put"].astype(str).str.lower().str[0]  # 'c'/'p'
    df = df.dropna(subset=["trade_date", "strike", "expiry_date"])
    return LoadedChain(
        df=df, source_path=str(path),
        trade_dates=sorted(df["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        n_rows=len(df),
    )


def chain_overlaps_bars(chain: LoadedChain, bars: LoadedBars) -> bool:
    if not chain.trade_dates:
        return False
    chain_min, chain_max = chain.trade_dates[0], chain.trade_dates[-1]
    return not (pd.Timestamp(chain_max) < bars.start or pd.Timestamp(chain_min) > bars.end)


# ──────────────────────────────────────────────────────────────────────────
# Parquet bar loader — for the SPY 2021-2023 EOD data once it arrives in an
# extractable form (raw .parquet, or a .zip you can actually open). Needs
# `pyarrow` (in requirements.txt) — not installed/testable in the sandbox
# this was built in, so this is syntax-checked but not run against a real
# file yet. Column names are guessed from the same conventions as the CSV
# loader; adjust `rename_map` below once you see the actual columns.
# ──────────────────────────────────────────────────────────────────────────
def load_option_chain_parquet(path: str | Path) -> LoadedChain:
    """Mirrors load_option_chain_csv()'s column normalization for parquet
    sources. Column-name guesses will need adjusting once the real schema
    is visible — this is untested against an actual file (see module note)."""
    path = Path(path)
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    rename = {
        "trade_date": "trade_date", "date": "trade_date", "strike": "strike",
        "expiry_date": "expiry_date", "expiration": "expiry_date",
        "call_put": "call_put", "type": "call_put", "option_type": "call_put",
        "last_trade_price": "last_price", "last": "last_price",
        "bid_price": "bid", "bid": "bid", "ask_price": "ask", "ask": "ask",
        "bid_implied_volatility": "bid_iv", "ask_implied_volatility": "ask_iv", "iv": "iv",
        "open_interest": "open_interest", "oi": "open_interest", "volume": "volume",
        "delta": "delta", "gamma": "gamma", "vega": "vega", "theta": "theta", "rho": "rho",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "trade_date" not in df.columns or "strike" not in df.columns:
        raise ValueError(f"{path.name}: doesn't look like an option chain "
                          f"(columns found: {list(df.columns)}) — try load_ohlcv_parquet "
                          f"instead, or fix up rename_map in load_option_chain_parquet().")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
    if "call_put" in df.columns:
        df["call_put"] = df["call_put"].astype(str).str.lower().str[0]
    df = df.dropna(subset=["trade_date", "strike"])
    return LoadedChain(
        df=df, source_path=str(path),
        trade_dates=sorted(df["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        n_rows=len(df),
    )


def load_parquet_auto(path: str | Path):
    """Peeks at column names to decide bars vs option-chain, then loads
    accordingly. Returns (kind, loaded) where kind is 'bars', 'chain', or
    'wide_chain' (the historicaloptiondata.com-style schema)."""
    path = Path(path)
    try:
        import pyarrow.parquet as pq  # noqa: avoids reading the full file just to see column names
        cols = {c.strip().lower() for c in pq.ParquetFile(path).schema.names}
    except Exception:
        cols = {c.strip().lower() for c in pd.read_parquet(path).columns}
    if "[strike]" in cols or "[c_delta]" in cols or "[p_delta]" in cols:
        return "wide_chain", load_historicaloptiondata_parquet(path)
    if "strike" in cols or "expiry_date" in cols or "expiration" in cols:
        return "chain", load_option_chain_parquet(path)
    return "bars", load_ohlcv_parquet(path)


# ──────────────────────────────────────────────────────────────────────────
# historicaloptiondata.com-style wide EOD options schema — verified against
# your real spy_eod_2021/2022/2023.parquet files via direct Parquet footer
# inspection (no pyarrow needed for that part). One row per
# (QUOTE_DATE, EXPIRE_DATE, STRIKE), with BOTH call and put quotes/Greeks
# in the same row (C_* / P_* prefixed columns) — melted here into the same
# long format as load_option_chain_csv()'s output for drop-in use elsewhere.
#
# IMPORTANT: this schema has NO open_interest column, only per-day volume.
# The C++ engine's GEX calc (m3::GreeksEngine) is written against
# open_interest, not volume — they measure different things (standing
# dealer exposure vs. that day's trading activity). open_interest is left
# as NaN by default; pass use_volume_as_oi_proxy=True only if you
# understand that's an approximation, not a real OI-based GEX signal.
# ──────────────────────────────────────────────────────────────────────────
_WIDE_WANT = [
    "[QUOTE_UNIXTIME]", "[QUOTE_DATE]", "[UNDERLYING_LAST]", "[EXPIRE_DATE]",
    "[EXPIRE_UNIX]", "[DTE]", "[STRIKE]", "[STRIKE_DISTANCE]", "[STRIKE_DISTANCE_PCT]",
    "[C_DELTA]", "[C_GAMMA]", "[C_VEGA]", "[C_THETA]", "[C_RHO]", "[C_IV]",
    "[C_VOLUME]", "[C_LAST]", "[C_BID]", "[C_ASK]",
    "[P_DELTA]", "[P_GAMMA]", "[P_VEGA]", "[P_THETA]", "[P_RHO]", "[P_IV]",
    "[P_VOLUME]", "[P_LAST]", "[P_BID]", "[P_ASK]",
]


def _melt_wide_options(df: pd.DataFrame, use_volume_as_oi_proxy: bool = False) -> pd.DataFrame:
    shared = {
        "trade_date": pd.to_datetime(df["[QUOTE_DATE]"], errors="coerce"),
        "underlying_last": pd.to_numeric(df["[UNDERLYING_LAST]"], errors="coerce"),
        "expiry_date": pd.to_datetime(df["[EXPIRE_DATE]"], errors="coerce"),
        "dte": pd.to_numeric(df["[DTE]"], errors="coerce"),
        "strike": pd.to_numeric(df["[STRIKE]"], errors="coerce"),
    }
    calls = pd.DataFrame({
        **shared, "call_put": "c",
        "bid": pd.to_numeric(df["[C_BID]"], errors="coerce"),
        "ask": pd.to_numeric(df["[C_ASK]"], errors="coerce"),
        "last_price": pd.to_numeric(df["[C_LAST]"], errors="coerce"),
        "iv": pd.to_numeric(df["[C_IV]"], errors="coerce"),
        "delta": pd.to_numeric(df["[C_DELTA]"], errors="coerce"),
        "gamma": pd.to_numeric(df["[C_GAMMA]"], errors="coerce"),
        "vega": pd.to_numeric(df["[C_VEGA]"], errors="coerce"),
        "theta": pd.to_numeric(df["[C_THETA]"], errors="coerce"),
        "rho": pd.to_numeric(df["[C_RHO]"], errors="coerce"),
        "volume": pd.to_numeric(df["[C_VOLUME]"], errors="coerce"),
    })
    puts = pd.DataFrame({
        **shared, "call_put": "p",
        "bid": pd.to_numeric(df["[P_BID]"], errors="coerce"),
        "ask": pd.to_numeric(df["[P_ASK]"], errors="coerce"),
        "last_price": pd.to_numeric(df["[P_LAST]"], errors="coerce"),
        "iv": pd.to_numeric(df["[P_IV]"], errors="coerce"),
        "delta": pd.to_numeric(df["[P_DELTA]"], errors="coerce"),
        "gamma": pd.to_numeric(df["[P_GAMMA]"], errors="coerce"),
        "vega": pd.to_numeric(df["[P_VEGA]"], errors="coerce"),
        "theta": pd.to_numeric(df["[P_THETA]"], errors="coerce"),
        "rho": pd.to_numeric(df["[P_RHO]"], errors="coerce"),
        "volume": pd.to_numeric(df["[P_VOLUME]"], errors="coerce"),
    })
    long_df = pd.concat([calls, puts], ignore_index=True)
    long_df["open_interest"] = long_df["volume"] if use_volume_as_oi_proxy else np.nan
    return long_df.dropna(subset=["trade_date", "strike"])


def load_historicaloptiondata_parquet(path: str | Path,
                                       use_volume_as_oi_proxy: bool = False) -> LoadedChain:
    """Loads and melts an ENTIRE file (one full year, ~1-1.3M wide rows ->
    ~2-2.6M long rows for this vendor's data) into memory. Fine for
    inspection/visualization; for feeding the C++ engine bar-by-bar, prefer
    load_historicaloptiondata_chain_for_date() below, which only reads the
    one day you actually need via Parquet predicate pushdown."""
    path = Path(path)
    df = pd.read_parquet(path)  # requires pyarrow or fastparquet
    missing = [c for c in ("[QUOTE_DATE]", "[STRIKE]", "[C_DELTA]") if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected wide-chain columns {missing} "
                          f"(found: {list(df.columns)[:10]}...)")
    long_df = _melt_wide_options(df, use_volume_as_oi_proxy)
    return LoadedChain(
        df=long_df, source_path=str(path),
        trade_dates=sorted(long_df["trade_date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        n_rows=len(long_df),
    )


def load_historicaloptiondata_chain_for_date(path: str | Path, trade_date: str,
                                              use_volume_as_oi_proxy: bool = False) -> pd.DataFrame:
    """Efficiently loads just one day's chain using Parquet predicate
    pushdown (pyarrow filters) — reads only the matching row group(s)/pages
    rather than the whole ~1.2M-row file. This is what per-bar GEX feeding
    into HybridSigmaStrategy.OnBarUpdate's chain_for_current_bar should
    actually call, once that integration is wired up."""
    import pyarrow.parquet as pq
    table = pq.read_table(str(path), filters=[("[QUOTE_DATE]", "=", trade_date)])
    df = table.to_pandas()
    return _melt_wide_options(df, use_volume_as_oi_proxy)


def load_ohlcv_parquet(path: str | Path) -> LoadedBars:
    path = Path(path)
    df = pd.read_parquet(path)  # requires pyarrow or fastparquet installed
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        if c in ("date", "time", "datetime", "t"):
            rename_map[c] = "timestamp"
        elif c in ("o",):
            rename_map[c] = "open"
        elif c in ("h",):
            rename_map[c] = "high"
        elif c in ("l",):
            rename_map[c] = "low"
        elif c in ("c", "close_price"):
            rename_map[c] = "close"
        elif c in ("v", "vol"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: no recognizable timestamp column "
                          f"(found: {list(df.columns)}) — update rename_map in "
                          f"load_ohlcv_parquet() once you see the real schema.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep = [c for c in STANDARD_COLS if c in df.columns]
    df = df[keep].dropna(subset=[c for c in ("timestamp", "open", "high", "low", "close") if c in keep])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    df["ts_ms"] = df["timestamp"].values.astype("datetime64[ms]").astype("int64")

    return LoadedBars(
        df=df, source_path=str(path), vendor_format="parquet",
        n_rows=len(df),
        start=df["timestamp"].iloc[0] if len(df) else pd.NaT,
        end=df["timestamp"].iloc[-1] if len(df) else pd.NaT,
    )


# ──────────────────────────────────────────────────────────────────────────
# Dealer-positioning "levels" snapshots (call_wall/put_wall/gamma_flip/
# max_pain/vol_trigger) — a derived analytics product, distinct from a raw
# option chain. Filename convention observed: "<SYMBOL>_<YYYY-MM-DD>_levels.csv"
# Schema: symbol,exp,spot,call_wall,put_wall,gamma_flip,max_pain,vol_trigger
# ──────────────────────────────────────────────────────────────────────────
LEVELS_COLS = ["symbol", "date", "spot", "call_wall", "put_wall", "gamma_flip",
               "max_pain", "vol_trigger"]


def load_levels_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"exp": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing = [c for c in LEVELS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected levels columns {missing}")
    return df[LEVELS_COLS]


def load_all_levels(data_dir: str | Path) -> pd.DataFrame:
    """Scans data_dir for '*_levels.csv' files and concatenates them into
    one date-indexed table, one row per (symbol, date) snapshot."""
    data_dir = Path(data_dir)
    frames = []
    for p in sorted(data_dir.glob("*_levels.csv")):
        try:
            frames.append(load_levels_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=LEVELS_COLS)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
