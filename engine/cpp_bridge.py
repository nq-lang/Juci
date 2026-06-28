"""
cpp_bridge.py — ctypes wrapper around the REAL compiled hybrid_sigma_strategy
C ABI (libhybridsigma.so). This calls the actual, unmodified C++ engine —
it does not reimplement any strategy logic in Python.
"""
from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ASSET_MAP = {"MNQ": 0}  # C++ Asset enum is MNQ-only now; kept as a dict for API stability
ASSET_MULT_DEFAULTS = {"MNQ": 2.0}


class _CTrade(ct.Structure):
    _fields_ = [
        ("trade_id", ct.c_int64), ("entry_ts_ms", ct.c_int64), ("direction", ct.c_int32),
        ("entry_price", ct.c_double), ("stop_price", ct.c_double), ("target_price", ct.c_double),
        ("exit_price", ct.c_double), ("contracts", ct.c_double), ("pnl", ct.c_double),
        ("mae", ct.c_double), ("mfe", ct.c_double), ("rr", ct.c_double), ("kelly", ct.c_double),
        ("ml_p_success", ct.c_double), ("bars_held", ct.c_int32),
        ("stopped", ct.c_int32), ("target_hit", ct.c_int32),
    ]


@dataclass
class BacktestParams:
    asset: str = "MNQ"
    mult: float = 2.0           # $ per point per contract (MNQ=2, NQ=20, ES=50, SPX=100, SPY=1 share-equiv)
    equity: float = 100_000.0
    risk_pct: float = 0.006     # hard-capped at 0.006 inside the engine regardless of this value
    kelly_cap: float = 0.25
    stop_atr_mult: float = 1.5
    target_sigma_mult: float = 2.0
    min_rr: float = 1.5
    ml_threshold: float = 0.55


def _find_library() -> Path:
    here = Path(__file__).resolve().parent.parent
    candidates = [
        here / "libhybridsigma.so",
        here / "build" / "libhybridsigma.so",
        Path("/home/claude/libhybridsigma.so"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "libhybridsigma.so not found. Run build.sh first to compile it "
        "for your platform (see README.md)."
    )


class HybridSigmaEngine:
    """One backtest run = one instance. Not thread-safe; create a new
    instance per run if you need concurrency."""

    def __init__(self, lib_path: str | Path | None = None):
        self._lib_path = Path(lib_path) if lib_path else _find_library()
        self.lib = ct.CDLL(str(self._lib_path))
        self._bind_signatures()
        self._handle = self.lib.hybrid_sigma_create()
        if not self._handle:
            raise RuntimeError("hybrid_sigma_create() returned a null handle")

    def _bind_signatures(self):
        lib = self.lib
        lib.hybrid_sigma_create.restype = ct.c_void_p
        lib.hybrid_sigma_destroy.argtypes = [ct.c_void_p]
        lib.hybrid_sigma_initialize.argtypes = [
            ct.c_void_p, ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int64, ct.c_int64,
        ]
        lib.hybrid_sigma_reset.argtypes = [ct.c_void_p]
        lib.hybrid_sigma_on_bar.argtypes = [
            ct.c_void_p, ct.c_int64, ct.c_double, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ]
        lib.hybrid_sigma_finalize.argtypes = [ct.c_void_p]
        lib.hybrid_sigma_trade_count.argtypes = [ct.c_void_p]
        lib.hybrid_sigma_trade_count.restype = ct.c_int
        lib.hybrid_sigma_equity.argtypes = [ct.c_void_p]
        lib.hybrid_sigma_equity.restype = ct.c_double
        lib.hybrid_sigma_get_trade.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(_CTrade)]
        lib.hybrid_sigma_get_trade.restype = ct.c_int

    def initialize(self, params: BacktestParams,
                    start_date_ms: int = 0, end_date_ms: int = 0):
        self.lib.hybrid_sigma_initialize(
            self._handle, ASSET_MAP[params.asset.upper()], params.mult, params.equity,
            params.risk_pct, params.kelly_cap, params.stop_atr_mult,
            params.target_sigma_mult, params.min_rr, params.ml_threshold,
            int(start_date_ms), int(end_date_ms),
        )

    def reset(self):
        self.lib.hybrid_sigma_reset(self._handle)

    def on_bar(self, ts_ms: int, o: float, h: float, l: float, c: float,
               volume: float, buy_vol: float, sell_vol: float, cum_delta: float):
        self.lib.hybrid_sigma_on_bar(self._handle, int(ts_ms), o, h, l, c,
                                      volume, buy_vol, sell_vol, cum_delta)

    def finalize(self):
        self.lib.hybrid_sigma_finalize(self._handle)

    def run_bars(self, df: pd.DataFrame, params: BacktestParams,
                 progress_cb=None) -> pd.DataFrame:
        """df must have columns: ts_ms, open, high, low, close, volume,
        buy_vol, sell_vol, cum_delta (ascending time order)."""
        self.reset()
        self.initialize(params)
        n = len(df)
        cols = df[["ts_ms", "open", "high", "low", "close", "volume",
                   "buy_vol", "sell_vol", "cum_delta"]].to_numpy()
        report_every = max(1, n // 100)
        for i in range(n):
            ts, o, h, l, c, v, bv, sv, cd = cols[i]
            self.on_bar(int(ts), float(o), float(h), float(l), float(c),
                        float(v), float(bv), float(sv), float(cd))
            if progress_cb and (i % report_every == 0 or i == n - 1):
                progress_cb((i + 1) / n)
        self.finalize()
        return self.trades()

    def trades(self) -> pd.DataFrame:
        n = self.lib.hybrid_sigma_trade_count(self._handle)
        t = _CTrade()
        rows = []
        for i in range(n):
            ok = self.lib.hybrid_sigma_get_trade(self._handle, i, ct.byref(t))
            if not ok:
                continue
            rows.append({
                "trade_id": t.trade_id, "entry_ts_ms": t.entry_ts_ms,
                "direction": "LONG" if t.direction > 0 else ("SHORT" if t.direction < 0 else "FLAT"),
                "entry_price": t.entry_price, "stop_price": t.stop_price,
                "target_price": t.target_price, "exit_price": t.exit_price,
                "contracts": t.contracts, "pnl": t.pnl, "mae": t.mae, "mfe": t.mfe,
                "rr": t.rr, "kelly": t.kelly, "ml_p_success": t.ml_p_success,
                "bars_held": t.bars_held, "stopped": bool(t.stopped),
                "target_hit": bool(t.target_hit),
            })
        df = pd.DataFrame(rows)
        if len(df):
            df["entry_time"] = pd.to_datetime(df["entry_ts_ms"], unit="ms")
        return df

    def equity_base(self) -> float:
        return self.lib.hybrid_sigma_equity(self._handle)

    def close(self):
        if getattr(self, "_handle", None):
            self.lib.hybrid_sigma_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
