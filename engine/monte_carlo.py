"""
monte_carlo.py — bootstrap resampling of a completed backtest's trade
sequence to show the distribution of plausible equity-curve outcomes,
not just the single realized path.

Two resampling modes:
  • "iid"   — draw n trades with replacement from the realized pnl
              distribution (destroys trade-order/serial-correlation effects,
              shows pure variance from trade-outcome randomness)
  • "block" — resample contiguous blocks of trades (preserves local
              streakiness / serial correlation in the realized sequence)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    paths: np.ndarray            # shape (n_sims, n_trades+1), equity curves incl. starting equity
    final_equity: np.ndarray     # shape (n_sims,)
    max_drawdown: np.ndarray     # shape (n_sims,), positive = drawdown magnitude in $
    percentiles: pd.DataFrame    # equity percentile bands over trade index
    prob_of_ruin: float          # fraction of sims that ever hit equity <= ruin_level
    summary: dict


def _max_drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    return float(dd.max())


def run_monte_carlo(
    trade_pnls: np.ndarray,
    starting_equity: float,
    n_sims: int = 2000,
    mode: str = "iid",
    block_size: int = 10,
    ruin_fraction: float = 0.5,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    trade_pnls: realized per-trade pnl ($), in original chronological order.
    ruin_fraction: ruin is defined as equity falling to this fraction of
                   starting_equity (e.g. 0.5 = a 50% drawdown from start).
    """
    rng = np.random.default_rng(seed)
    n_trades = len(trade_pnls)
    if n_trades == 0:
        raise ValueError("No trades to simulate — run a backtest with at least one trade first.")

    paths = np.empty((n_sims, n_trades + 1), dtype=float)
    paths[:, 0] = starting_equity

    if mode == "iid":
        draws = rng.choice(trade_pnls, size=(n_sims, n_trades), replace=True)
    elif mode == "block":
        n_blocks_needed = int(np.ceil(n_trades / block_size))
        draws = np.empty((n_sims, n_trades), dtype=float)
        max_start = max(1, n_trades - block_size)
        for s in range(n_sims):
            chunks = []
            for _ in range(n_blocks_needed):
                start = rng.integers(0, max_start + 1)
                chunks.append(trade_pnls[start:start + block_size])
            draws[s, :] = np.concatenate(chunks)[:n_trades]
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'iid' or 'block')")

    paths[:, 1:] = starting_equity + np.cumsum(draws, axis=1)

    final_equity = paths[:, -1]
    max_dd = np.array([_max_drawdown(paths[s]) for s in range(n_sims)])
    ruin_level = starting_equity * ruin_fraction
    prob_of_ruin = float((paths.min(axis=1) <= ruin_level).mean())

    pct_levels = [5, 25, 50, 75, 95]
    pct_table = np.percentile(paths, pct_levels, axis=0)
    percentiles = pd.DataFrame(pct_table.T, columns=[f"p{p}" for p in pct_levels])
    percentiles.insert(0, "trade_idx", np.arange(n_trades + 1))

    summary = {
        "n_sims": n_sims, "n_trades": n_trades, "mode": mode,
        "median_final_equity": float(np.median(final_equity)),
        "p5_final_equity": float(np.percentile(final_equity, 5)),
        "p95_final_equity": float(np.percentile(final_equity, 95)),
        "median_max_drawdown": float(np.median(max_dd)),
        "p95_max_drawdown": float(np.percentile(max_dd, 95)),
        "prob_of_ruin_pct": prob_of_ruin * 100.0,
        "ruin_level": ruin_level,
        "realized_final_equity": float(starting_equity + trade_pnls.sum()),
    }

    return MonteCarloResult(
        paths=paths, final_equity=final_equity, max_drawdown=max_dd,
        percentiles=percentiles, prob_of_ruin=prob_of_ruin, summary=summary,
    )
