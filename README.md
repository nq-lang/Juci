# HybridSigma Quant Terminal

A personal Streamlit backtesting terminal that drives the **real, compiled
C++ `HybridSigmaStrategy` engine** — not a Python reimplementation — via a
ctypes bridge to a shared library built straight from `cpp/hybrid_sigma_strategy.cpp`.

## ⚠️ MNQ-only as of this version

The C++ engine's `Asset` enum was stripped down to **MNQ only** —
NQ/ES/SPX/SPY are no longer selectable in either the C++ engine or this
terminal. This was a purely mechanical refactor: the now-permanently-dead
SPX/SPY "direct trade" branches and the ES/NQ-specific switch cases were
removed, but every formula, gate, threshold, and the live futures↔QQQ
ratio computation itself are byte-for-byte unchanged — verified by running
the exact same backtest window before and after the strip and confirming
identical trade-by-trade output (same entries, same contracts, same P&L).
MNQ uses the same 40x futures/QQQ ratio as NQ did, since MNQ tracks the
identical Nasdaq-100 index level — only the $2/point multiplier differs
from NQ's $20/point. Your existing NQ price-data files are still the
correct input (MNQ has no separate price series; it moves tick-for-tick
with NQ), just paired with the now-default $2/point contract multiplier.

## Setup

```bash
cd quant_terminal
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Compile the real engine for YOUR machine (the .so shipped here was built
# in an offline Linux sandbox and will not run on macOS/Windows/ARM):
./build.sh

streamlit run app.py
```

If `build.sh` fails with an "illegal instruction" later when you run the
app, recompile without `-march=native` (edit `build.sh`) — that flag tunes
for the exact CPU it's built on, which may not match the CPU you run on.

## What's real vs. approximated right now

| Component | Status |
|---|---|
| Alpha engine (signal generation, gates, Kelly/risk sizing) | **Real** — the actual compiled C++ engine, called via ctypes, unmodified |
| OHLCV bars | **Real** — your uploaded data, multi-vendor format auto-detected |
| Order flow (buy_vol/sell_vol/cum_delta) | **Approximated** — synthesized from each bar's close-location-value (Chaikin-style proxy). You don't have real tick/quote data yet, so order-flow gates run on this proxy, not real tape. |
| Options/GEX/vanna/charm | **Inactive for your current bar data, but real data now exists** — `spy_eod_2021/2022/2023.parquet` is genuine, verified, full-year SPY EOD options chains (~3.4M rows total), but it's 2021-2023 and your bar data is all 2026-06; they don't overlap. It also has no `open_interest` column (only volume), which the GEX calc needs — see below. The single-day snapshots (QQQ/SPX/SPY/etc., dated 2023-07-03) remain too thin and equally non-overlapping. The moment bar data and chain data share a date range, the loader detects it automatically (see "Data completeness" in the app). |
| Dealer-positioning levels overlay (call wall / put wall / gamma flip / max pain / vol trigger) | **Real, and active** — `SPX_2026-06-18_levels.csv` *does* fall inside your bar window. These are pre-computed dealer-positioning analytics (not a raw chain), so they're drawn as horizontal reference lines on the candlestick chart rather than fed into the C++ GreeksEngine, which expects raw per-strike chain rows. |
| VIX reference series | **Real** — your VIX bars (1min/30min/1h/1day) align with the same 2026-06-10→06-25 window as everything else. Selectable in the sidebar as a subplot under the main chart for visual context; not yet wired into the Macro Structural Filter Engine (that still needs the C ABI extended — see below). |
| Macro Structural Filter Engine (Φ_t) | **Built in C++, not wired to live data yet** (see `cpp/hybrid_sigma_strategy.cpp`'s `m8::StructuralFilterEngine`). The sidebar's "Macro filter overlay" panel lets you apply a constant or CSV-supplied Φ_t as a what-if position-size multiplier in the meantime — this is the exact `PositionSizer::FinalSize` math, just done in Python since it has no internal state worth crossing the ABI for. |
| Monte Carlo | **Real** — bootstrap resampling (iid or block) of your actual realized trade-pnl sequence |
| Candlestick + entries/exits | **Real** — plotted from actual trade records read out of the C++ engine |

## API keys (.env)

Your FRED / Alpaca / EODHD keys are in `.env`. **Rotate all three** — they
were shared in a chat session, which should be treated as a compromised
channel for credentials. `macro_feeds.py` reads them from the environment
(via `python-dotenv`); they're never hardcoded into source beyond this
local `.env` file, which is git-ignored.

`macro_feeds.py`'s FRED/Alpaca/EODHD calls were written carefully but
**never executed against the live APIs** — they were built in a sandbox
with no network access. Test them locally before depending on them; the
most likely failure points are exact response-shape assumptions (EODHD's
options endpoint schema in particular — verify against a live response and
adjust `fetch_eodhd_option_chain`'s parsing if needed).

Note: Alpaca needs **both** an API Key ID and an API Secret Key. You only
sent the Key ID (`PKVCGB5...`) — add the secret to `ALPACA_API_SECRET` in
`.env` before calling `fetch_alpaca_bars`.

## Data you've sent so far

- `data/NQ_5Min.csv` is the standout — **12 years of real 5-min NQ data**
  (2014-01-02 → 2026-01-30, 850k+ bars), in a different vendor format
  (semicolon-delimited, European decimal notation) than your other files.
  The loader auto-detects and parses it correctly.
- Most other `*_sample.csv` files (NQ/SPY/QQQ/SPX/VIX) cover the same
  2026-06-10 → 2026-06-25 window across multiple timeframes — these are
  genuinely cross-consistent and useful together.
- `QQQ_option_chain.csv` (7,220 rows), `SPX_option_chain.csv` (19,216 rows),
  `SPY_option_chain.csv` (7,798 rows) are single EOD snapshots, all dated
  2023-07-03, with real Greeks columns — useful once you send chains dated
  to match your bar windows. `QQQA/QQQE/QQQJ/QQQM/SPRU/VIV` option chains
  are much thinner (sub-400 rows, VIV only 48) — likely auxiliary tickers,
  not core instruments for this strategy.
- `SPX_2026-06-18_levels.csv` is a **pre-computed dealer-positioning
  snapshot** (call wall / put wall / gamma flip / max pain / vol trigger) —
  different in kind from a raw chain, and the one piece of options-derived
  data that actually falls inside your bar window. It's drawn as chart
  overlay lines (see table above) rather than fed into the GreeksEngine.
  Send more of these (any `<SYMBOL>_<YYYY-MM-DD>_levels.csv`) and they'll
  all show up automatically.
- Two `NQ_1hour_sample*.csv`-named pairs are **not duplicates** — they cover
  contiguous, non-overlapping date ranges (May 24–Jun 8 vs Jun 11–Jun 25).
  Both are kept; pick either from the sidebar dropdown.
- **`spy_eod_2021/2022/2023.parquet` — resolved, real, and verified.** You
  re-sent these as individual `.zip` files, which extracted cleanly. Since
  `pyarrow` isn't installable in this build sandbox (no network), I read
  the Parquet footer metadata directly (Thrift compact-protocol, hand-
  parsed — see the diagnostic technique noted below) to verify the actual
  schema and date ranges **without needing pyarrow at all**:
  - **This is full-year, real, wide-format EOD options chain data** —
    `historicaloptiondata.com`-style schema: one row per
    `(QUOTE_DATE, EXPIRE_DATE, STRIKE)`, with both call and put quotes/Greeks
    in the same row (`[C_DELTA]`, `[P_DELTA]`, etc., brackets included in
    the real column names).
  - Verified row counts and date coverage straight from column statistics:
    **2021**: 1,277,698 rows, 2021-01-04 → 2021-12-31, SPY $368.84–$477.48.
    **2022**: 1,146,980 rows, 2022-01-03 → 2022-12-30, SPY $356.58–$477.77.
    **2023**: 972,162 rows, 2023-01-03 → 2023-12-29, SPY $379.38–$476.73.
    All three ranges are realistic for SPY in those years — this is good data.
  - **No `open_interest` column** — only per-day `volume` per side. The C++
    engine's GEX calc (`m3::GreeksEngine`) is written against
    `open_interest` (standing dealer exposure), not `volume` (that day's
    trading activity) — substituting one for the other would distort the
    signal, not just approximate it. `engine/data_loader.py`'s new
    `load_historicaloptiondata_parquet()` leaves `open_interest` as `NaN`
    by default; the app's sidebar has an explicit, off-by-default
    "use volume as OI proxy" checkbox if you want to experiment anyway,
    clearly labeled as an approximation.
  - **Still doesn't overlap your bar data** — this is 2021-2023; your bar
    files are all 2026-06. If you have (or can pull via your Alpaca key)
    SPY bars for 2021-2023, that would finally give you a fully aligned
    price+options dataset. Worth doing — this is the best options data
    in the project so far by a wide margin.
  - `load_parquet_auto()` now auto-detects this exact schema (via the
    `[STRIKE]`/`[C_DELTA]`/`[P_DELTA]` bracketed column signature) and
    routes to the right loader automatically. There's also
    `load_historicaloptiondata_chain_for_date()`, which uses Parquet
    predicate pushdown to read just one day's chain efficiently — the
    function to actually call once per-bar GEX feeding into
    `HybridSigmaStrategy.OnBarUpdate` gets wired up (not done yet — see
    "What's real vs. approximated" table above).
  - Melting logic (wide → long, call+put split) was tested against a
    synthetic row built from the verified real schema — it works. Reading
    the actual 60-85MB files via `pd.read_parquet()` itself is **not**
    tested end-to-end here, since `pyarrow` isn't installable in this
    sandbox; it should work (pyarrow is far more robust than my hand-rolled
    footer parser, which already confirmed the files are well-formed), but
    test it locally before trusting it blindly.

## New API keys this round

- **Polygon.io** and **Tradier** fetch helpers added to `engine/macro_feeds.py`
  (`fetch_polygon_bars`, `fetch_polygon_option_contracts`,
  `fetch_polygon_option_aggs`, `fetch_tradier_option_chain`,
  `fetch_tradier_expirations`) — written but **not tested live** (no network
  in the build sandbox).
- ⚠️ **The Polygon key you sent is byte-for-byte identical to the EODHD key
  sent earlier.** Polygon and EODHD have different key formats, so this is
  very likely a copy-paste duplication — double-check your Polygon
  dashboard for the real key before relying on `POLYGON_API_KEY`.
- Polygon's REST aggregates are per-contract (you need a contract ticker
  first via `fetch_polygon_option_contracts`); for bulk historical EOD
  chains across 2021-2023, Polygon's **Flat Files** (S3 bulk download) is
  the more direct path than these REST endpoints — worth checking your
  plan tier.
- Tradier's market-data API is oriented around live/current chains, not
  deep multi-year history — don't expect it to backfill 2021-2023.

## Project layout

```
quant_terminal/
├── app.py                  # Streamlit UI
├── build.sh                # compiles cpp/ → libhybridsigma.(so|dylib)
├── cpp/hybrid_sigma_strategy.cpp
├── engine/
│   ├── cpp_bridge.py        # ctypes wrapper around the compiled engine
│   ├── data_loader.py       # multi-vendor CSV ingestion + order-flow proxy
│   ├── monte_carlo.py        # bootstrap simulation
│   └── macro_feeds.py        # FRED/Alpaca/EODHD fetchers (untested live)
├── data/                    # your uploaded CSVs
├── requirements.txt
└── .env                     # your API keys — rotate them, see above
```

## Known limitations to keep in mind

- Backtests run a real per-bar Python loop calling into C++ via ctypes —
  roughly 15–20k bars/sec on the sandbox machine this was built on. The
  full 850k-bar NQ file takes well under a minute; use the sidebar's "Max
  bars" control to iterate faster while tuning parameters.
- The candlestick chart renders at most a few thousand bars at a time
  (browser rendering limit) — it's a viewer for a window of your backtest,
  not the entire multi-year history at once.
- `risk_pct` in the sidebar is cosmetic above 0.006 — the C++ engine hard-
  clamps it to 0.006 (0.60%) internally regardless of what you set, by
  design (see `Optimizing.txt`'s account-risk constraint).
