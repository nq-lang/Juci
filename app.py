"""
HybridSigma Quant Terminal — Streamlit backtesting UI for the real,
compiled C++ HybridSigmaStrategy engine.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from engine.cpp_bridge import ASSET_MULT_DEFAULTS, BacktestParams, HybridSigmaEngine
from engine.data_loader import (add_orderflow_proxy, chain_overlaps_bars,
                                 load_all_levels, load_historicaloptiondata_parquet,
                                 load_ohlcv_csv, load_ohlcv_parquet,
                                 load_option_chain_csv, load_option_chain_parquet,
                                 load_parquet_auto)
from engine.monte_carlo import run_monte_carlo

st.set_page_config(page_title="HybridSigma Quant Terminal", layout="wide",
                    initial_sidebar_state="expanded")

DATA_DIR = Path(__file__).resolve().parent / "data"


# ──────────────────────────────────────────────────────────────────────────
# Cached loaders — dispatch by extension; parquet files are auto-classified
# as bars vs. chain by inspecting their columns (see load_parquet_auto).
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_load_bars(path_str: str):
    lb = load_ohlcv_parquet(path_str) if path_str.endswith(".parquet") else load_ohlcv_csv(path_str)
    df = add_orderflow_proxy(lb.df)
    return df, lb.vendor_format, lb.start, lb.end


@st.cache_data(show_spinner=False)
def _cached_load_chain(path_str: str, use_volume_as_oi_proxy: bool = False):
    if path_str.endswith(".parquet"):
        kind, loaded = load_parquet_auto(path_str)
        if kind == "wide_chain":
            # re-load with the actual proxy flag requested (load_parquet_auto's
            # internal call used the default; cheap to redo, these are cached)
            return load_historicaloptiondata_parquet(path_str, use_volume_as_oi_proxy), "wide_chain"
        return loaded, kind
    return load_option_chain_csv(path_str), "chain"


@st.cache_data(show_spinner=False)
def _cached_load_levels(data_dir_str: str):
    return load_all_levels(data_dir_str)


@st.cache_data(show_spinner=False)
def _classify_parquet(path_str: str) -> str:
    """'bars', 'chain', 'wide_chain', or 'unknown' — see load_parquet_auto."""
    try:
        kind, _ = load_parquet_auto(path_str)
        return kind
    except Exception:
        return "unknown"


def _discover_data_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.parquet")))


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — data + config
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("HybridSigma Terminal")

st.sidebar.subheader("1. Bar data")
local_files = _discover_data_files()
uploaded = st.sidebar.file_uploader("Upload OHLCV CSV/Parquet/Zip (or pick from /data below)",
                                     type=["csv", "parquet", "zip"])

# Bucket local files: CSVs by filename convention (as before), parquet by
# actually inspecting their schema since we can't infer from the name alone.
csv_files = [p for p in local_files if p.suffix == ".csv"]
parquet_files = [p for p in local_files if p.suffix == ".parquet"]
parquet_bar_files = [p for p in parquet_files if _classify_parquet(str(p)) == "bars"]
parquet_chain_files = [p for p in parquet_files if _classify_parquet(str(p)) in ("chain", "wide_chain")]

bar_candidates = [p for p in csv_files
                   if "option_chain" not in p.name.lower() and "_levels" not in p.name.lower()
                   ] + parquet_bar_files
chain_candidates = [p for p in csv_files if "option_chain" in p.name.lower()] + parquet_chain_files

bar_source_path = None
if uploaded is not None and uploaded.name.endswith(".zip"):
    import zipfile
    extract_dir = Path("/tmp") / f"upload_{Path(uploaded.name).stem}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(uploaded) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".csv", ".parquet"))
                   and not m.startswith("__MACOSX")]
        zf.extractall(extract_dir, members=members)
    if not members:
        st.sidebar.error(f"{uploaded.name}: no .csv or .parquet files found inside the zip.")
    else:
        member_choice = st.sidebar.selectbox(f"File inside {uploaded.name}", members)
        bar_source_path = str(extract_dir / member_choice)
elif uploaded is not None:
    tmp_path = Path("/tmp") / uploaded.name
    tmp_path.write_bytes(uploaded.getvalue())
    bar_source_path = str(tmp_path)
elif bar_candidates:
    choice = st.sidebar.selectbox("Local data files", [p.name for p in bar_candidates])
    bar_source_path = str(DATA_DIR / choice)
else:
    st.sidebar.info("No bar CSV/Parquet found in /data and nothing uploaded yet.")

st.sidebar.subheader("2. Option chain (optional)")
chain_path = None
use_volume_as_oi_proxy = False
if chain_candidates:
    chain_choice = st.sidebar.selectbox(
        "Option chain file", ["(none)"] + [p.name for p in chain_candidates])
    if chain_choice != "(none)":
        chain_path = str(DATA_DIR / chain_choice)
        if chain_path.endswith(".parquet") and _classify_parquet(chain_path) == "wide_chain":
            use_volume_as_oi_proxy = st.sidebar.checkbox(
                "Use volume as open-interest proxy (NOT real OI — approximation only)",
                value=False,
                help="This dataset has no open_interest column, only per-day volume. "
                     "The C++ engine's GEX calc is written against open_interest "
                     "(standing dealer exposure), not volume (that day's trading "
                     "activity) — they measure different things. Leave off unless "
                     "you specifically want to explore this approximation.")

st.sidebar.subheader("2b. VIX reference (optional)")
vix_csvs = [p for p in csv_files if p.name.upper().startswith("VIX")]
vix_path = None
if vix_csvs:
    vix_choice = st.sidebar.selectbox("VIX series", ["(none)"] + [p.name for p in vix_csvs])
    if vix_choice != "(none)":
        vix_path = str(DATA_DIR / vix_choice)

st.sidebar.subheader("3. Asset & sizing")
asset = st.sidebar.selectbox("Asset", list(ASSET_MULT_DEFAULTS.keys()))
mult = st.sidebar.number_input("$ per point per contract", value=ASSET_MULT_DEFAULTS[asset], step=1.0)
equity0 = st.sidebar.number_input("Starting equity ($)", value=100_000.0, step=10_000.0)


with st.sidebar.expander("Alpha engine config (BacktestConfig)", expanded=False):
    risk_pct = st.slider("risk_pct (hard-capped at 0.006 inside the engine)", 0.001, 0.006, 0.006, 0.0005)
    kelly_cap = st.slider("kelly_cap", 0.05, 0.50, 0.25, 0.01)
    stop_atr_mult = st.slider("stop_atr_mult", 1.0, 3.0, 1.5, 0.1)
    target_sigma_mult = st.slider("target_sigma_mult", 1.0, 3.5, 2.0, 0.1)
    min_rr = st.slider("min_rr", 1.0, 3.0, 1.5, 0.1)
    ml_threshold = st.slider("ml_threshold", 0.40, 0.80, 0.55, 0.01)

st.sidebar.subheader("4. Macro filter overlay (Φ_t)")
st.sidebar.caption(
    "StructuralFilterEngine needs macro time-series aligned to your bar dates "
    "to compute Φ_t for real — not available yet for the data on hand. Use "
    "this panel to *manually* scale position size for what-if analysis until "
    "aligned macro data is wired in."
)
phi_mode = st.sidebar.radio("Mode", ["Off (Φ=1, full size)", "Constant Φ", "Φ from CSV (timestamp,phi)"])
phi_constant = 1.0
phi_csv_path = None
if phi_mode == "Constant Φ":
    phi_constant = st.sidebar.slider("Φ (position-size multiplier)", 0.0, 1.0, 1.0, 0.05)
elif phi_mode == "Φ from CSV (timestamp,phi)":
    phi_upload = st.sidebar.file_uploader("Φ_t CSV", type=["csv"], key="phi_csv")
    if phi_upload is not None:
        phi_csv_path = Path("/tmp") / phi_upload.name
        phi_csv_path.write_bytes(phi_upload.getvalue())

st.sidebar.subheader("5. Backtest window")
max_bars_default = 20_000
n_bars_limit = st.sidebar.number_input(
    "Max bars to backtest (most-recent N within range)", value=max_bars_default,
    min_value=500, step=1000)

run_clicked = st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
st.title("HybridSigma Quant Terminal")

if bar_source_path is None:
    st.warning("Upload a bar CSV or add files to the /data folder to get started.")
    st.stop()

bars_df, vendor_fmt, data_start, data_end = _cached_load_bars(bar_source_path)

# ── Data completeness panel ──────────────────────────────────────────────
with st.expander("📋 Data completeness", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Bars loaded", f"{len(bars_df):,}")
    c2.metric("Date range", f"{data_start.date()} → {data_end.date()}")
    c3.metric("Vendor format detected", vendor_fmt)
    st.caption(
        "⚠️ buy_vol / sell_vol / cum_delta are a **bar-close-location proxy** "
        "(Chaikin-style), not real tape data — order-flow gates run on an "
        "approximation until real tick/quote data is supplied."
    )
    if chain_path:
        chain, chain_kind = _cached_load_chain(chain_path, use_volume_as_oi_proxy)
        from engine.data_loader import LoadedBars
        lb_for_check = LoadedBars(df=bars_df, source_path=bar_source_path,
                                   vendor_format=vendor_fmt, n_rows=len(bars_df),
                                   start=data_start, end=data_end)
        overlaps = chain_overlaps_bars(chain, lb_for_check)
        if overlaps:
            st.success(f"✅ Option chain dates ({chain.trade_dates[0]}…{chain.trade_dates[-1]}) "
                       f"overlap the bar window — GEX/vanna features will be populated.")
        else:
            st.error(
                f"❌ Option chain data dated {chain.trade_dates[0]}…{chain.trade_dates[-1]} "
                f"does NOT overlap the bar window ({data_start.date()}…{data_end.date()}). "
                f"Running price-only — GEX/vanna/charm features stay at neutral defaults."
            )
        if chain_kind == "wide_chain":
            if use_volume_as_oi_proxy:
                st.warning("⚠️ Using **volume as an open-interest proxy** for this chain — "
                           "this is an approximation, not real dealer positioning data.")
            else:
                st.caption("This chain has no open_interest column (only per-day volume) — "
                          "GEX/vanna stay at neutral defaults regardless of date overlap "
                          "unless you enable the volume-as-OI-proxy checkbox above.")
    else:
        st.info("No option chain selected — running price-only (no GEX/vanna/charm features).")

    levels_df = _cached_load_levels(str(DATA_DIR))
    matching_levels = levels_df[levels_df["symbol"] == asset] if len(levels_df) else levels_df
    if len(matching_levels):
        in_window = matching_levels[(matching_levels["date"] >= data_start) &
                                     (matching_levels["date"] <= data_end)]
        if len(in_window):
            st.success(f"✅ {len(in_window)} dealer-positioning level snapshot(s) for {asset} "
                       f"fall inside the bar window — shown as chart overlays below.")
        else:
            st.warning(f"{len(matching_levels)} level snapshot(s) found for {asset}, but none "
                       f"fall inside {data_start.date()}…{data_end.date()}.")
    elif len(levels_df):
        st.caption(f"Level snapshots found for: {sorted(levels_df['symbol'].unique())} "
                   f"— none for {asset}.")

# ── Slice to backtest window ─────────────────────────────────────────────
sliced = bars_df.tail(int(n_bars_limit)).reset_index(drop=True)

if run_clicked:
    params = BacktestParams(asset=asset, mult=mult, equity=equity0, risk_pct=risk_pct,
                             kelly_cap=kelly_cap, stop_atr_mult=stop_atr_mult,
                             target_sigma_mult=target_sigma_mult, min_rr=min_rr,
                             ml_threshold=ml_threshold)
    progress = st.progress(0.0, text="Running C++ engine...")
    eng = HybridSigmaEngine()
    trades = eng.run_bars(sliced, params, progress_cb=lambda f: progress.progress(f, text=f"Running C++ engine... {f*100:.0f}%"))
    eng.close()
    progress.empty()

    # ── Macro filter overlay (Φ_t × BaseSize) — pure pass-through math,
    # matches PositionSizer::FinalSize exactly: final = phi * base_size ──
    if phi_mode == "Constant Φ":
        trades["phi"] = phi_constant
    elif phi_mode == "Φ from CSV (timestamp,phi)" and phi_csv_path is not None:
        phi_df = pd.read_csv(phi_csv_path)
        phi_df.columns = [c.strip().lower() for c in phi_df.columns]
        phi_df["timestamp"] = pd.to_datetime(phi_df["timestamp"])
        phi_df = phi_df.sort_values("timestamp")
        trades["phi"] = np.interp(
            trades["entry_ts_ms"], phi_df["timestamp"].values.astype("datetime64[ms]").astype("int64"),
            phi_df["phi"].values, left=1.0, right=1.0,
        )
    else:
        trades["phi"] = 1.0
    trades["final_contracts"] = trades["contracts"] * trades["phi"]
    trades["final_pnl"] = trades["pnl"] * trades["phi"]  # pnl scales linearly with size

    # True exit timestamp = entry + bars_held * (actual bar interval of this
    # dataset) — NOT a fixed 1-minute assumption, which would silently
    # mis-plot exits for any 5min/30min/1h/1day series.
    bar_interval_ms = int(sliced["ts_ms"].diff().median())
    trades["exit_ts_ms"] = trades["entry_ts_ms"] + trades["bars_held"] * bar_interval_ms
    trades["exit_time"] = pd.to_datetime(trades["exit_ts_ms"], unit="ms")

    st.session_state["trades"] = trades
    st.session_state["sliced"] = sliced
    st.session_state["equity0"] = equity0

if "trades" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    st.stop()

trades: pd.DataFrame = st.session_state["trades"]
sliced: pd.DataFrame = st.session_state["sliced"]
equity0: float = st.session_state["equity0"]

# ── KPI row ───────────────────────────────────────────────────────────────
st.subheader("Results")
if len(trades) == 0:
    st.warning("No trades were generated over this window — try a larger window or relax the gate thresholds.")
else:
    pnl = trades["final_pnl"]
    win_rate = (pnl > 0).mean() * 100
    total_pnl = pnl.sum()
    equity_curve = equity0 + pnl.cumsum()
    peak = equity_curve.cummax()
    max_dd = (peak - equity_curve).max()
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Trades", f"{len(trades)}")
    k2.metric("Win rate", f"{win_rate:.1f}%")
    k3.metric("Total P&L", f"${total_pnl:,.2f}")
    k4.metric("Max drawdown", f"${max_dd:,.2f}")
    k5.metric("Sharpe (ann., trade-level)", f"{sharpe:.2f}")

    # ── Candlestick chart with entries/exits ─────────────────────────────
    st.subheader("Price action with entries / exits")
    max_chart_bars = st.slider("Bars to display on chart", 200, min(10_000, len(sliced)),
                                value=min(2000, len(sliced)), step=200)
    chart_df = sliced.tail(max_chart_bars)
    chart_start_ts = chart_df["timestamp"].iloc[0]
    chart_end_ts = chart_df["timestamp"].iloc[-1]

    vix_df = None
    if vix_path:
        vix_lb, _, _, _ = _cached_load_bars(vix_path)
        vix_df = vix_lb[(vix_lb["timestamp"] >= chart_start_ts) & (vix_lb["timestamp"] <= chart_end_ts)]

    n_rows = 3 if vix_df is not None and len(vix_df) else 2
    row_heights = [0.65, 0.20, 0.15] if n_rows == 3 else [0.75, 0.25]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=row_heights,
                         vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=chart_df["timestamp"], open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"], close=chart_df["close"], name="price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Dealer-positioning levels overlay — only the snapshot(s) matching this
    # asset and falling within the visible window get drawn.
    levels_df = _cached_load_levels(str(DATA_DIR))
    asset_levels = levels_df[(levels_df["symbol"] == asset) &
                              (levels_df["date"] >= chart_start_ts) &
                              (levels_df["date"] <= chart_end_ts)] if len(levels_df) else levels_df
    level_styles = {
        "call_wall":   ("#ff1744", "Call wall (resistance)"),
        "put_wall":    ("#00e676", "Put wall (support)"),
        "gamma_flip":  ("#ffd600", "Gamma flip"),
        "max_pain":    ("#b388ff", "Max pain"),
        "vol_trigger": ("#40c4ff", "Vol trigger"),
    }
    for _, lvl_row in asset_levels.iterrows():
        for col, (color, label) in level_styles.items():
            fig.add_hline(y=lvl_row[col], line_dash="dot", line_color=color, line_width=1.5,
                          annotation_text=f"{label} {lvl_row[col]:.0f} ({lvl_row['date'].date()})",
                          annotation_position="right", row=1, col=1)

    visible_trades = trades[pd.to_datetime(trades["entry_ts_ms"], unit="ms") >= chart_start_ts]

    long_entries = visible_trades[visible_trades["direction"] == "LONG"]
    short_entries = visible_trades[visible_trades["direction"] == "SHORT"]
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(long_entries["entry_ts_ms"], unit="ms"), y=long_entries["entry_price"],
        mode="markers", name="Long entry",
        marker=dict(symbol="triangle-up", size=11, color="#00e676", line=dict(width=1, color="black")),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(short_entries["entry_ts_ms"], unit="ms"), y=short_entries["entry_price"],
        mode="markers", name="Short entry",
        marker=dict(symbol="triangle-down", size=11, color="#ff1744", line=dict(width=1, color="black")),
    ), row=1, col=1)

    win_exits = visible_trades[visible_trades["final_pnl"] > 0]
    loss_exits = visible_trades[visible_trades["final_pnl"] <= 0]
    fig.add_trace(go.Scatter(
        x=win_exits["exit_time"], y=win_exits["exit_price"],
        mode="markers", name="Exit (win)",
        marker=dict(symbol="circle", size=8, color="#69f0ae", line=dict(width=1, color="black")),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=loss_exits["exit_time"], y=loss_exits["exit_price"],
        mode="markers", name="Exit (loss)",
        marker=dict(symbol="x", size=8, color="#ff5252", line=dict(width=1, color="black")),
    ), row=1, col=1)

    fig.add_trace(go.Bar(x=chart_df["timestamp"], y=chart_df["volume"], name="volume",
                          marker_color="#90a4ae"), row=2, col=1)

    if n_rows == 3:
        fig.add_trace(go.Scatter(x=vix_df["timestamp"], y=vix_df["close"], name="VIX",
                                  line=dict(color="#ab47bc", width=1.5)), row=3, col=1)

    fig.update_layout(height=650 if n_rows == 2 else 780, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", y=1.02), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Equity curve ───────────────────────────────────────────────────
    st.subheader("Realized equity curve")
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=trades["entry_time"], y=equity_curve, mode="lines",
                                 name="Equity", line=dict(color="#42a5f5")))
    eq_fig.update_layout(height=350, margin=dict(t=20, b=10))
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Monte Carlo ─────────────────────────────────────────────────────
    st.subheader("Monte Carlo simulation")
    mc1, mc2, mc3 = st.columns(3)
    n_sims = mc1.number_input("Simulations", 200, 20_000, 2000, 200)
    mc_mode = mc2.selectbox("Resampling mode", ["iid", "block"],
                             help="iid: draw trades independently. block: resample contiguous "
                                  "chunks, preserving streakiness/serial correlation.")
    ruin_frac = mc3.slider("Ruin level (fraction of starting equity)", 0.1, 0.9, 0.5, 0.05)

    if st.button("Run Monte Carlo"):
        mc = run_monte_carlo(pnl.to_numpy(), starting_equity=equity0, n_sims=int(n_sims),
                              mode=mc_mode, ruin_fraction=ruin_frac, seed=None)
        s = mc.summary
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median final equity", f"${s['median_final_equity']:,.0f}")
        m2.metric("5th–95th pct range", f"${s['p5_final_equity']:,.0f} – ${s['p95_final_equity']:,.0f}")
        m3.metric("Median max drawdown", f"${s['median_max_drawdown']:,.0f}")
        m4.metric("P(ruin)", f"{s['prob_of_ruin_pct']:.2f}%")

        fan = go.Figure()
        pct = mc.percentiles
        fan.add_trace(go.Scatter(x=pct["trade_idx"], y=pct["p95"], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=pct["trade_idx"], y=pct["p5"], fill="tonexty",
                                  fillcolor="rgba(66,165,245,0.15)", line=dict(width=0), name="5–95%"))
        fan.add_trace(go.Scatter(x=pct["trade_idx"], y=pct["p75"], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=pct["trade_idx"], y=pct["p25"], fill="tonexty",
                                  fillcolor="rgba(66,165,245,0.35)", line=dict(width=0), name="25–75%"))
        fan.add_trace(go.Scatter(x=pct["trade_idx"], y=pct["p50"], line=dict(color="#1565c0", width=2),
                                  name="Median"))
        fan.add_trace(go.Scatter(x=np.arange(len(equity_curve)+1), y=[equity0]+equity_curve.tolist(),
                                  line=dict(color="#ff9800", width=2, dash="dot"), name="Realized path"))
        fan.add_hline(y=s["ruin_level"], line_dash="dash", line_color="red",
                       annotation_text="Ruin level")
        fan.update_layout(height=450, title="Equity-curve distribution across simulations",
                           xaxis_title="Trade #", yaxis_title="Equity ($)", margin=dict(t=40))
        st.plotly_chart(fan, use_container_width=True)

        hist = go.Figure(go.Histogram(x=mc.final_equity, nbinsx=60, marker_color="#42a5f5"))
        hist.add_vline(x=s["realized_final_equity"], line_color="orange", line_dash="dot",
                        annotation_text="Realized")
        hist.update_layout(height=300, title="Distribution of final equity", margin=dict(t=40))
        st.plotly_chart(hist, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────
    st.subheader("Trade log")
    display_cols = ["entry_time", "direction", "entry_price", "exit_price", "contracts",
                     "phi", "final_contracts", "pnl", "final_pnl", "rr", "kelly",
                     "stopped", "target_hit", "bars_held"]
    st.dataframe(trades[display_cols], use_container_width=True, height=350)
    st.download_button("Download trades as CSV", trades.to_csv(index=False),
                        file_name="hybridsigma_trades.csv", mime="text/csv")
