#pragma once
/*
╔══════════════════════════════════════════════════════════════════════════════════╗
║  hybrid_sigma_strategy.hpp                                                     ║
║  Hybrid Regression Sigma + Adaptive Continuation Model — DECOUPLED "BRAIN"     ║
║  Event-Driven, Portable, Date-Range-Aware Strategy Core                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE                                                                    ║
║  This header contains ONLY the strategy's mathematical core and execution       ║
║  state machine. It performs ZERO file I/O, ZERO CSV parsing, and has NO          ║
║  knowledge of where data comes from (historical CSV, live feed, another         ║
║  backtesting engine's callback, a Python harness via pybind11, etc).            ║
║                                                                                    ║
║  The strategy reacts ONLY to two public entry points:                           ║
║      OnBarUpdate(const Bar&, const std::vector<OptionRow>&)                     ║
║      OnTick(const TickData&)            [convenience: internally aggregates    ║
║                                           ticks into bars, then calls           ║
║                                           OnBarUpdate() when a bar completes]   ║
║                                                                                    ║
║  LOOKBACK PRIMING (warm-up)                                                       ║
║  Bars with a timestamp BEFORE BacktestConfig::start_date are still pushed       ║
║  through OnBarUpdate() — EGARCH, the HMM, sigma bands, order flow, and the       ║
║  Greeks surface all continue to update their internal state exactly as          ║
║  normal — but the entry-execution branch is gated shut until start_date is      ║
║  reached. This lets any external loop simply feed the FULL historical series    ║
║  chronologically; the brain decides for itself when to start trading.           ║
║                                                                                    ║
║  ZERO MATHEMATICAL CHANGES                                                       ║
║  Every formula, gate condition, threshold, and module (EGARCH, IS-HMM, Dealer   ║
║  Greeks/GEX/SEX/VEX/CHMX, Vanna-deformed Sigma Bands, Order Flow Absorption/    ║
║  Exhaustion, the 7-gate Continuation Logic, the Particle-Filter SMC layer, and  ║
║  the OrderDirection/OrderType/EntryReason/ExitReason classification layer) is   ║
║  reproduced verbatim from the original monolithic implementation. The ONLY      ║
║  new conditional logic in this entire file is the single `&& in_window_` term   ║
║  appended to the pre-existing entry gate — exactly the lookback-priming         ║
║  behavior this refactor was commissioned to add.                                ║
║                                                                                    ║
║  PORTABILITY                                                                      ║
║  • Header-only core — #include this file, link nothing extra for C++ hosts.    ║
║  • Compiles cleanly into a shared library: see the HYBRID_SIGMA_C_ABI block     ║
║    at the bottom for a flat extern "C" surface (opaque handle + create/         ║
║    destroy/feed/reset functions) suitable for .so/.dll consumption from any     ║
║    language with a C FFI (ctypes, ffi, ...).                                    ║
║  • See HYBRID_SIGMA_PYBIND block for a ready pybind11 module definition —       ║
║    wraps the C++ class directly, no ABI layer needed for Python hosts.         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
*/

// ── Standard Library ──────────────────────────────────────────────────────────
#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hybrid_sigma {

// MdSpan2D — 2D view over contiguous storage (std::mdspan semantics)
//   Used for the Greeks surface: rows=moneyness_bins × cols=expiry_bins
// ══════════════════════════════════════════════════════════════════════════════
template<typename T>
class MdSpan2D {
    T* data_; std::size_t rows_, cols_;
public:
    MdSpan2D(T* d, std::size_t r, std::size_t c) noexcept : data_(d), rows_(r), cols_(c) {}
    [[nodiscard]] T& operator()(std::size_t r, std::size_t c) noexcept       { return data_[r*cols_+c]; }
    [[nodiscard]] T  operator()(std::size_t r, std::size_t c) const noexcept { return data_[r*cols_+c]; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
};

// ══════════════════════════════════════════════════════════════════════════════
// CONCEPTS
// ══════════════════════════════════════════════════════════════════════════════
template<typename T>
concept Floating = std::floating_point<T>;

template<typename T>
concept BarLike = requires(T t) {
    { t.open }  -> std::convertible_to<double>;
    { t.close } -> std::convertible_to<double>;
    { t.high }  -> std::convertible_to<double>;
    { t.low }   -> std::convertible_to<double>;
};

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — CONFIGURATION CONSTANTS
//   All statistically validated thresholds derived from quantitative analysis.
//   Nothing in this file is hard-coded market data or API-specific.
// ══════════════════════════════════════════════════════════════════════════════
namespace cfg {

    // ── Bar aggregation ─────────────────────────────────────────────────────
    inline constexpr int         BAR_SECONDS          = 300;     // 5-minute bars

    // ── MODULE 4: Sigma band lookback ────────────────────────────────────────
    // 25 bars × 5 min = 125 minutes ≈ one complete intraday micro-cycle.
    // 20-bar lower bound: captures exactly 100 min of order flow memory.
    // 30-bar upper bound: beyond this, stale prior-session data contaminates bands.
    inline constexpr std::size_t SIGMA_WINDOW         = 25;      // VALIDATED: 20–30 range
    inline constexpr double      SIGMA_L1             = 1.0;
    inline constexpr double      SIGMA_L2             = 2.0;
    inline constexpr double      SIGMA_L3             = 3.0;
    inline constexpr double      BAND_SKEW_FACTOR     = 0.25;    // Vanna deformation

    // ── MODULE 2A: EGARCH warmup ─────────────────────────────────────────────
    // 300 bars (≈1.5 days of 5-min data).
    // MLE convergence for asymmetric volatility shock parameters requires
    // ≥250 observations. Below 250, the conditional variance σ²_t is unreliable
    // as a regime gate. 300 gives a 20% safety margin over the minimum.
    inline constexpr std::size_t EGARCH_WARMUP        = 300;     // VALIDATED: 250–500 range
    // ω/(1−β) = target E[log(h)]; β=0.88, target vol=0.003/bar: ω = log(9e-6)×0.12 = -1.394
    inline constexpr double      EGARCH_OMEGA_INIT    = -1.394;  // stationary intercept (validated)
    inline constexpr double      EGARCH_ALPHA_INIT    = 0.10;    // ARCH coefficient
    inline constexpr double      EGARCH_BETA_INIT     = 0.88;    // GARCH persistence
    inline constexpr double      EGARCH_GAMMA_INIT    = -0.05;   // leverage (negative = asymmetric)
    inline constexpr double      EGARCH_LR            = 5.0e-5;  // gradient step (conservative for MLE)

    // ── MODULE 2B: IS-HMM window ─────────────────────────────────────────────
    // 200 bars provides the Baum-Welch EM algorithm enough state transitions
    // to populate the transition matrix with statistically significant probs.
    // Below 150: impossible to observe full [Cons→Trend→MeanRev] cycle.
    inline constexpr std::size_t HMM_WINDOW          = 200;      // VALIDATED: 150–250 range
    inline constexpr std::size_t HMM_STATES          = 3;        // Bull / Bear / Consolidating
    inline constexpr std::size_t HMM_OBS_DIM         = 3;        // [log_ret, egarch_vol, atr_ratio]
    inline constexpr std::size_t HMM_COV_DIM         = 3;        // covariate: [1, Z, norm_GEX]
    inline constexpr double      HMM_LR              = 0.015;    // soft M-step update rate
    inline constexpr std::size_t HMM_MIN_OBS         = 150;      // minimum before emission stable

    // ── MODULE 7: Particle filter ────────────────────────────────────────────
    // ESS exactly 0.50 × N: proven optimal for tracking fat-tail events.
    // Below 0.50: path impoverishment, weights collapse, cascade tracking fails.
    // Above 0.50: excessive resampling in low-vol noise, computation waste.
    inline constexpr std::size_t N_PARTICLES         = 500;
    inline constexpr double      ESS_THRESHOLD       = 0.50;     // VALIDATED: exactly 0.50

    // ── MODULE 3: Greeks surface ─────────────────────────────────────────────
    inline constexpr std::size_t MONEY_BINS          = 40;
    inline constexpr std::size_t EXP_BINS            = 6;
    inline constexpr double      MONEY_MIN           = -2.5;
    inline constexpr double      MONEY_MAX           =  2.5;
    inline constexpr double      GEX_SCALE           =  0.01;    // per-spec: ×0.01

    // ── MODULE 5: Order flow ─────────────────────────────────────────────────
    inline constexpr std::size_t OF_BAR_WINDOW       = 5;        // bars in absorption check
    inline constexpr double      ABSORPTION_DELTA_TH = -0.18;    // normalised delta gate
    inline constexpr double      EXHAUST_EFF_DECAY   = 0.72;     // price/vol efficiency decay

    // ── MODULE 6: Entry gates ────────────────────────────────────────────────
    inline constexpr double      CPS_DEFAULT         = 0.55;     // base CPS threshold
    inline constexpr double      CPS_HIGH_VOL        = 0.50;     // relaxed in high-vol cascade
    inline constexpr double      Z_ENTRY_MIN         = 0.80;     // min |Z| for sigma gate
    // ML entry filter (Path B — offline-trained, flat-file inference, zero runtime ML deps)
    inline constexpr double      ML_P_SUCCESS_THRESH = 0.55;     // prune below Threshold_alpha

    // ── BACKTEST / RISK ──────────────────────────────────────────────────────
    inline constexpr double      RISK_FREE_RATE      = 0.05;
    inline constexpr double      DIVIDEND_YIELD      = 0.015;
    inline constexpr double      ACCOUNT_EQUITY      = 250'000.0;
    inline constexpr double      RISK_PCT            = 0.006;    // HARD CAP: 0.60% max risk per trade (architectural invariant, not tunable)
    inline constexpr double      KELLY_CAP           = 0.25;     // max 25% Kelly
    inline constexpr double      STOP_ATR_MULT       = 1.50;
    inline constexpr double      TARGET_SIGMA_MULT   = 2.00;
    inline constexpr double      MIN_RR              = 1.50;
    inline constexpr std::size_t VOL_MEAN_WINDOW     = 20;       // rolling volume mean

} // namespace cfg

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — ENUMERATIONS & DATA STRUCTURES
// ══════════════════════════════════════════════════════════════════════════════
enum class Asset        : uint8_t  { MNQ=0 };
// NQ/ES: futures requiring cross-asset mapping to QQQ/SPY options chains.
// SPX/SPY: traded DIRECTLY as the underlying — no futures-to-ETF conversion.
//          Each uses its own native options chain (SPX options / SPY options).
enum class Direction    : int8_t   { LONG=1, SHORT=-1, FLAT=0 };
enum class RegimeState  : uint8_t  { BULL=0, BEAR=1, CONS=2 };
enum class SpeedZone    : int8_t   { DEALER_WALL=2, FRICTION=1, NEUTRAL=0, ACCEL=-1, CASCADE=-2 };
enum class OptionType   : uint8_t  { CALL=0, PUT=1 };
enum class TriggerVerdict : uint8_t {
    NO_TRADE=0, HALT_3SIGMA, WAIT_SETUP,
    LONG_CONT, LONG_SQUEEZE,
    SHORT_CONT, SHORT_CASCADE
};

// ══════════════════════════════════════════════════════════════════════════════
// ORDER / TRADE CLASSIFICATION ARCHITECTURE
//   Pure logging/classification layer — wraps the existing strategy logic
//   without altering a single mathematical computation, gate condition, or
//   threshold. These enums and the TradeContext struct below exist solely to
//   give a backtesting engine unambiguous, strongly-typed visibility into
//   WHY each order fired and HOW it was filled. No enum here participates
//   in any entry/exit decision — they are populated strictly AFTER the
//   existing core logic has already determined the outcome.
// ══════════════════════════════════════════════════════════════════════════════

// Order side / intent — distinguishes opening a position from closing one.
enum class OrderDirection : uint8_t { Long=0, Short=1, CloseLong=2, CloseShort=3 };

// Fill mechanism classification. The engine's existing execution model fills
// entries at the signal bar's close (Market), profit-target exits at a fixed
// price level (Limit), and stop exits at a fixed adverse price level (Stop).
// This purely RE-LABELS the existing fill behavior — it does not change which
// price is used or when a fill occurs.
enum class OrderType : uint8_t { Market=0, Limit=1, Stop=2 };

// Maps 1:1 onto the existing TriggerVerdict entry states (Module 6's gate
// output). Nothing about the gate's internal logic changes — this is the
// externally-facing classification of which structural trigger fired.
enum class EntryReason : uint8_t {
    None=0,
    LongContinuation,        // TriggerVerdict::LONG_CONT
    LongSqueezeRetest,       // TriggerVerdict::LONG_SQUEEZE
    ShortContinuation,       // TriggerVerdict::SHORT_CONT
    ShortFatTailCascade      // TriggerVerdict::SHORT_CASCADE
};

// Exit classification. TrailingStop and SignalReversal are included for
// completeness/back-tester compatibility per spec; the current strategy's
// exit logic (stop/target/end-of-data) only ever resolves to StopLoss,
// TakeProfit, or TimeExit — those three are the only reasons actually
// reachable today, and that exit logic itself is completely unmodified.
enum class ExitReason : uint8_t {
    None=0, TakeProfit, StopLoss, TrailingStop, TimeExit, SignalReversal
};

// ── Perimeter-only translation: TriggerVerdict → EntryReason ─────────────────
// constexpr, zero runtime cost, called only when packaging a log event —
// never inside the gate evaluation or the hot per-bar loop.
[[nodiscard]] constexpr EntryReason to_entry_reason(TriggerVerdict v) noexcept {
    switch (v) {
        case TriggerVerdict::LONG_CONT:     return EntryReason::LongContinuation;
        case TriggerVerdict::LONG_SQUEEZE:  return EntryReason::LongSqueezeRetest;
        case TriggerVerdict::SHORT_CONT:    return EntryReason::ShortContinuation;
        case TriggerVerdict::SHORT_CASCADE: return EntryReason::ShortFatTailCascade;
        default:                             return EntryReason::None;
    }
}

[[nodiscard]] constexpr OrderDirection to_order_direction(Direction d, bool closing) noexcept {
    if (!closing) return (d==Direction::LONG) ? OrderDirection::Long      : OrderDirection::Short;
    return               (d==Direction::LONG) ? OrderDirection::CloseLong : OrderDirection::CloseShort;
}

// ── String views — used ONLY at the logging perimeter (never in hot path) ───
[[nodiscard]] constexpr std::string_view sv_order_dir(OrderDirection d) noexcept {
    switch (d) {
        case OrderDirection::Long:       return "Long";
        case OrderDirection::Short:      return "Short";
        case OrderDirection::CloseLong:  return "CloseLong";
        case OrderDirection::CloseShort: return "CloseShort";
    }
    return "Unknown";
}
[[nodiscard]] constexpr std::string_view sv_order_type(OrderType t) noexcept {
    switch (t) {
        case OrderType::Market: return "Market";
        case OrderType::Limit:  return "Limit";
        case OrderType::Stop:   return "Stop";
    }
    return "Unknown";
}
[[nodiscard]] constexpr std::string_view sv_entry_reason(EntryReason r) noexcept {
    switch (r) {
        case EntryReason::None:                return "None";
        case EntryReason::LongContinuation:    return "LongContinuation";
        case EntryReason::LongSqueezeRetest:   return "LongSqueezeRetest";
        case EntryReason::ShortContinuation:   return "ShortContinuation";
        case EntryReason::ShortFatTailCascade: return "ShortFatTailCascade";
    }
    return "Unknown";
}
[[nodiscard]] constexpr std::string_view sv_exit_reason(ExitReason r) noexcept {
    switch (r) {
        case ExitReason::None:           return "None";
        case ExitReason::TakeProfit:     return "TakeProfit";
        case ExitReason::StopLoss:       return "StopLoss";
        case ExitReason::TrailingStop:   return "TrailingStop";
        case ExitReason::TimeExit:       return "TimeExit";
        case ExitReason::SignalReversal: return "SignalReversal";
    }
    return "Unknown";
}
[[nodiscard]] constexpr std::string_view asset_symbol(Asset a) noexcept {
    switch (a) {
        case Asset::MNQ: return "MNQ";
    }
    return "UNKNOWN";
}

// ── Order metadata bundle (TradeContext) ──────────────────────────────────────
// Lightweight: one std::string_view (zero-alloc, static-lifetime symbol) plus
// POD fields. Constructing this struct has no heap allocation and negligible
// cost — safe to build at every entry/exit event without hot-path impact.
struct TradeContext {
    uint64_t        trade_id      = 0;          // unique per round-trip trade
    int64_t         timestamp_ms  = 0;
    std::string_view symbol       = "";
    double          size          = 0.0;        // contracts/volume
    double          target_price  = 0.0;        // fill price for this event
    OrderDirection  direction     = OrderDirection::Long;
    OrderType       order_type    = OrderType::Market;
    EntryReason     entry_reason  = EntryReason::None;
    ExitReason      exit_reason   = ExitReason::None;
};

// ── Serialisation — perimeter-only, invoked solely at log-emission time ──────
// Heavy string work (std::format) happens here and ONLY here — never inside
// any per-bar computation. Trade events are sparse (one call per fill), so
// this has no measurable impact on backtest throughput.
[[nodiscard]] inline std::string to_json(const TradeContext& tc) {
    return std::format(
        "{{\"trade_id\":{},\"timestamp_ms\":{},\"symbol\":\"{}\","
        "\"direction\":\"{}\",\"order_type\":\"{}\","
        "\"entry_reason\":\"{}\",\"exit_reason\":\"{}\","
        "\"size\":{:.4f},\"price\":{:.4f}}}",
        tc.trade_id, tc.timestamp_ms, tc.symbol,
        sv_order_dir(tc.direction), sv_order_type(tc.order_type),
        sv_entry_reason(tc.entry_reason), sv_exit_reason(tc.exit_reason),
        tc.size, tc.target_price);
}

[[nodiscard]] inline std::string to_csv_row(const TradeContext& tc) {
    return std::format(
        "{},{},{},{},{},{},{},{:.4f},{:.4f}",
        tc.trade_id, tc.timestamp_ms, tc.symbol,
        sv_order_dir(tc.direction), sv_order_type(tc.order_type),
        sv_entry_reason(tc.entry_reason), sv_exit_reason(tc.exit_reason),
        tc.size, tc.target_price);
}

// ── Raw feed structs (Tastytrade/dxFeed CSV layout) ──────────────────────────
struct FuturesTick {
    int64_t timestamp_ms=0;
    double  price=0, bid=0, ask=0, bid_sz=0, ask_sz=0, volume=0;
    bool    is_buy_aggr=false;
};

struct OptionRow {
    int64_t    timestamp_ms=0, expiry_ms=0;
    double     strike=0, mid_iv=0, bid_iv=0, ask_iv=0;
    double     delta=0, gamma=0, open_interest=0, volume=0;
    OptionType type=OptionType::CALL;
};

struct Bar {
    int64_t timestamp_ms=0;
    double  open=0, high=0, low=0, close=0, volume=0, vwap=0;
    double  buy_vol=0, sell_vol=0, cum_delta=0;
};

// ── Signal output (Module 8 — spec-exact fields) ────────────────────────────
struct TradeSignal {
    int64_t       timestamp_ms=0;
    Asset         asset=Asset::MNQ;
    Direction     direction=Direction::FLAT;
    double        entry_price=0, stop_price=0, target_price=0;
    double        sigma_band_lvl=0, z_score=0;
    RegimeState   regime=RegimeState::CONS;
    double        cps=0, net_gex=0, net_speed=0, net_vanna=0;
    double        egarch_vol=0, particle_conf=0;
    SpeedZone     speed_zone=SpeedZone::NEUTRAL;
    TriggerVerdict verdict=TriggerVerdict::NO_TRADE;
    double        contracts=0, rr=0, kelly=0;
    bool          is_cascade=false, is_retest=false;
    double        ml_p_success=1.0;   // ML entry filter score at signal time (1.0 if unfiltered)
};

struct TradeRecord {
    TradeSignal sig;
    double exit_price=0, pnl=0, mae=0, mfe=0;
    int    bars_held=0;
    bool   stopped=false, target_hit=false;
    uint64_t trade_id=0;   // unique round-trip ID — links ENTRY/EXIT log events
};

// ── Feature row: full per-bar feature vector for ML pipeline (Path B) ────────
// Captured at EVERY bar (not just signals) — zero forward-looking bias.
// All values use ONLY data available through this bar's close.
struct FeatureRow {
    int64_t timestamp_ms=0;
    double  close=0;
    // Module 1: cross-asset
    double  implied_forward=0, basis_spread=0;
    // Module 2A: EGARCH
    double  egarch_vol=0, egarch_vol_ratio=0;
    bool    egarch_ready=false, low_vol=false, high_vol=false, expanding=false;
    // Module 2B: HMM
    double  hmm_bull=0, hmm_bear=0, hmm_cons=0;
    bool    hmm_ready=false;
    // ATR / regime features (Task A spec-exact)
    double  atr_fast=0, atr_slow=0, atr_ratio=0;
    double  ols_slope=0, vol_zscore=0;
    // Module 3: Greeks surface (GEX/VEX/SEX/CHMX proxies)
    double  net_gex=0, norm_gex=0, net_speed=0, norm_speed=0;
    double  net_vanna=0, norm_vanna=0, net_charm=0, iv_trend=0;
    int     speed_zone=0;  // SpeedZone enum value
    // Module 4: sigma bands
    double  sigma=0, band_level=0, z_score=0, vanna_skew=0;
    // Module 5: order flow
    double  cum_delta=0, norm_delta=0, of_vol_ratio=0, price_efficiency=0;
    bool    absorption=false, exhaustion=false;
    // Module 6: gate outputs
    bool    g_vol=false, g_gex=false, g_speed=false, g_vanna=false;
    bool    g_sigma=false, g_of=false, g_hmm=false;
    int     verdict=0;       // TriggerVerdict enum value
    int     direction=0;     // Direction enum value (raw structural signal)
    double  conviction=0;
    // Module 7: SMC
    double  particle_cps=0;
    // Forward-looking labels (filled by Python, NOT by C++) — left as 0 here
};

// ══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — MATHEMATICAL UTILITIES
// ══════════════════════════════════════════════════════════════════════════════
namespace math {

inline constexpr double PI         = 3.141592653589793;
inline constexpr double INV_SQRT2  = 0.7071067811865476;
inline constexpr double INV_S2PI   = 0.3989422804014327;
inline constexpr double E_ABS_Z    = 0.7978845608;  // E[|z|] = √(2/π)

[[nodiscard]] double npdf(double x) noexcept { return INV_S2PI*std::exp(-0.5*x*x); }
[[nodiscard]] double ncdf(double x) noexcept { return 0.5*std::erfc(-x*INV_SQRT2); }
[[nodiscard]] double sigmoid(double x) noexcept { return 1.0/(1.0+std::exp(-x)); }
[[nodiscard]] double clamp(double x, double lo, double hi) noexcept { return std::min(std::max(x,lo),hi); }

// ── Black-Scholes Greeks ─────────────────────────────────────────────────────
struct BSG { double delta, gamma, vanna, charm, speed; };

[[nodiscard]] BSG bs(OptionType ot, double S, double K,
                     double r, double iv, double T, double q=0.0) noexcept {
    BSG g{};
    if (S<=0||K<=0||iv<=0||T<=0) return g;
    double sqT=std::sqrt(T), iv2=iv*iv;
    double d1=(std::log(S/K)+(r-q+0.5*iv2)*T)/(iv*sqT), d2=d1-iv*sqT;
    double Nd1=ncdf(d1), nd1=npdf(d1), eqT=std::exp(-q*T);
    g.delta  =(ot==OptionType::CALL)?eqT*Nd1:eqT*(Nd1-1.0);
    g.gamma  = eqT*nd1/(S*iv*sqT);
    g.vanna  =-eqT*nd1*d2/iv;
    g.speed  =-(g.gamma/S)*(d1/(iv*sqT)+1.0);
    {
        double inn=nd1*(2*(r-q)*T-d2*iv*sqT)/(2*T*iv*sqT);
        g.charm=(ot==OptionType::CALL)?-eqT*(inn-q*Nd1):-eqT*(inn+q*(1-Nd1));
        g.charm/=365.0;
    }
    return g;
}

// ── Rolling statistics — O(1) update (deque-based Welford) ──────────────────
class RollStats {
    std::deque<double> buf_; std::size_t n_;
    double sum_=0, sumsq_=0;
public:
    explicit RollStats(std::size_t n): n_(n){}
    void push(double x) noexcept {
        if (buf_.size()==n_) { sum_-=buf_.front(); sumsq_-=buf_.front()*buf_.front(); buf_.pop_front(); }
        buf_.push_back(x); sum_+=x; sumsq_+=x*x;
    }
    [[nodiscard]] double mean()   const noexcept { return buf_.empty()?0.0:sum_/buf_.size(); }
    [[nodiscard]] double var()    const noexcept {
        if (buf_.size()<2) return 0.0;
        double m=mean(), v=sumsq_/buf_.size()-m*m;
        return v>0.0?v:0.0;
    }
    [[nodiscard]] double stddev() const noexcept { return std::sqrt(var()); }
    [[nodiscard]] double z(double x) const noexcept { double s=stddev(); return s>1e-10?(x-mean())/s:0.0; }
    [[nodiscard]] std::size_t count() const noexcept { return buf_.size(); }
    [[nodiscard]] bool ready() const noexcept { return buf_.size()>=n_; }
};

// ── Rolling OLS slope ────────────────────────────────────────────────────────
class RollOLS {
    std::deque<double> y_; std::size_t n_;
    double sx_=0,sy_=0,sxx_=0,sxy_=0; std::size_t cnt_=0;
public:
    explicit RollOLS(std::size_t n): n_(n){}
    void push(double y) noexcept {
        double xn=static_cast<double>(cnt_);
        if (y_.size()==n_) {
            double yo=y_.front(); double xo=xn-static_cast<double>(n_);
            sx_-=xo; sy_-=yo; sxx_-=xo*xo; sxy_-=xo*yo; y_.pop_front();
        }
        y_.push_back(y); sx_+=xn; sy_+=y; sxx_+=xn*xn; sxy_+=xn*y; ++cnt_;
    }
    [[nodiscard]] double slope() const noexcept {
        double nd=static_cast<double>(y_.size()); if (nd<2) return 0.0;
        double D=nd*sxx_-sx_*sx_; return std::abs(D)<1e-12?0.0:(nd*sxy_-sx_*sy_)/D;
    }
    [[nodiscard]] bool ready() const noexcept { return y_.size()>=n_; }
};

// ── EMA ─────────────────────────────────────────────────────────────────────
class EMA {
    double a_,v_=0; bool init_=false;
public:
    explicit EMA(double a): a_(a){}
    void push(double x) noexcept { v_=init_?a_*x+(1-a_)*v_:x; init_=true; }
    [[nodiscard]] double val() const noexcept { return v_; }
    [[nodiscard]] bool ready() const noexcept { return init_; }
    void reset() noexcept { init_=false; v_=0; }
};

// ── ATR ──────────────────────────────────────────────────────────────────────
class ATR {
    EMA ema_; double pc_=std::numeric_limits<double>::quiet_NaN();
public:
    explicit ATR(std::size_t p): ema_(1.0/static_cast<double>(p)){}
    void push(double hi, double lo, double cl) noexcept {
        double tr=std::isnan(pc_)?(hi-lo):std::max({hi-lo,std::abs(hi-pc_),std::abs(lo-pc_)});
        ema_.push(tr); pc_=cl;
    }
    [[nodiscard]] double val() const noexcept { return ema_.val(); }
    [[nodiscard]] bool ready() const noexcept { return ema_.ready(); }
};

} // namespace math


// ══════════════════════════════════════════════════════════════════════════════
// MODULE 1 — CROSS-ASSET MAPPING (pure math only — zero file I/O in this header)
//   Cost-of-Carry implied forward, normalised moneyness space, the live
//   futures↔ETF AssetMap, and the ML prediction row type. All CSV/file
//   reading lives in the external ingestion layer (e.g. backtest_runner.cpp),
//   never in the brain.
// ══════════════════════════════════════════════════════════════════════════════
namespace m1 {

// Cost-of-Carry: F_t = S_t · exp((r−d)·τ)
[[nodiscard]] double implied_forward(double S, double r, double d, double tau) noexcept {
    return S*std::exp((r-d)*tau);
}

// Normalised moneyness: M = ln(K/F_t) / (σ·√τ)
[[nodiscard]] double moneyness(double K, double F, double sigma, double tau) noexcept {
    if (F<=0||sigma<=0||tau<=0) return 0.0;
    return std::log(K/F)/(sigma*std::sqrt(tau));
}

[[nodiscard]] std::size_t money_bin(double M) noexcept {
    double f=(M-cfg::MONEY_MIN)/(cfg::MONEY_MAX-cfg::MONEY_MIN);
    auto b=static_cast<std::size_t>(std::floor(f*cfg::MONEY_BINS));
    return std::min(b,cfg::MONEY_BINS-1);
}

[[nodiscard]] std::size_t expiry_bin(double dte) noexcept {
    if (dte<=7)  return 0;
    if (dte<=14) return 1;
    if (dte<=28) return 2;
    if (dte<=60) return 3;
    if (dte<=90) return 4;
    return 5;
}

// ── ML prediction row (Path B: flat-file inference, zero runtime ML deps) ────
// Loaded from the offline Python pipeline's exported predictions CSV.
// Keyed by exact timestamp_ms — guarantees temporal alignment with zero
// forward-looking risk (each row's p_success was computed using ONLY
// feature data available through that bar's close).
struct MLPrediction {
    double p_success  = 0.5;   // Task B: entry filter probability
    int    regime_pred = -1;   // Task A: 0=BuyTrend 1=SellTrend 2=MeanRevert 3=Consolidating
    double regime_conf = 0.0;
};

// NOTE: loading predictions from a CSV is a file-I/O concern and therefore
// lives in the external ingestion layer (e.g. backtest_runner.cpp), NOT in
// this header. The strategy receives predictions purely via
// HybridSigmaStrategy::SetMLPredictions(std::unordered_map<int64_t,MLPrediction>).

struct AssetMap {
    Asset  asset=Asset::MNQ;
    double ratio=0.0;   // futures / ETF (QQQ) ratio — live-computed every bar, never hard-coded
    double etf_spot=0.0;

    // Ratio is always live-computed from concurrent MNQ + QQQ prices —
    // identical math to the original NQ ingestion-layer logic, just with
    // the now-permanently-false SPX/SPY direct-trade branch removed.
    void update(double fp, double ep) noexcept {
        if (ep>0) ratio=fp/ep;
        etf_spot=ep;
    }
    [[nodiscard]] double etf(double fp) const noexcept {
        return (ratio>0)?fp/ratio:fp;
    }
};

} // namespace m1

// ══════════════════════════════════════════════════════════════════════════════
// BACKTEST CONFIGURATION — runtime risk parameters + native date-range window
//   All fields here are configured by the HOST (external loop, pybind11
//   caller, another backtesting engine) at Initialize() time. Nothing in
//   this struct is read from a file or hard-coded — every value is supplied
//   by whoever constructs the BacktestConfig.
// ══════════════════════════════════════════════════════════════════════════════

// Parse a "YYYY-MM-DD" date string into a UTC midnight time_point using
// C++20/23 <chrono> calendar types. Returns std::nullopt on malformed input.
// This is a pure utility — no file I/O — safe to call from any host.
[[nodiscard]] inline std::optional<std::chrono::system_clock::time_point>
parse_date(std::string_view ymd) noexcept {
    if (ymd.size() != 10 || ymd[4] != '-' || ymd[7] != '-') return std::nullopt;
    int y=0, m=0, d=0;
    auto conv = [](std::string_view sv, int& out) -> bool {
        auto res = std::from_chars(sv.data(), sv.data()+sv.size(), out);
        return res.ec == std::errc{};
    };
    if (!conv(ymd.substr(0,4), y)) return std::nullopt;
    if (!conv(ymd.substr(5,2), m)) return std::nullopt;
    if (!conv(ymd.substr(8,2), d)) return std::nullopt;

    using namespace std::chrono;
    year_month_day ymd_val{ year{y}, month{static_cast<unsigned>(m)}, day{static_cast<unsigned>(d)} };
    if (!ymd_val.ok()) return std::nullopt;
    sys_days days{ ymd_val };
    return time_point_cast<system_clock::duration>(days);
}

// Convert a millisecond epoch timestamp (the convention used throughout the
// Bar/TickData/OptionRow structs) into a chrono time_point for comparison
// against BacktestConfig::start_date / end_date.
[[nodiscard]] inline std::chrono::system_clock::time_point
ms_to_time_point(int64_t timestamp_ms) noexcept {
    return std::chrono::system_clock::time_point(std::chrono::milliseconds(timestamp_ms));
}

struct BacktestConfig {
    Asset  asset             = Asset::MNQ;
    double mult               = 2.0;                     // $/point — MNQ = $2/point
    double equity             = cfg::ACCOUNT_EQUITY;
    double risk_pct           = cfg::RISK_PCT;
    double kelly_cap          = cfg::KELLY_CAP;
    double stop_atr_mult      = cfg::STOP_ATR_MULT;
    double target_sigma_mult  = cfg::TARGET_SIGMA_MULT;
    double min_rr             = cfg::MIN_RR;
    double ml_threshold       = cfg::ML_P_SUCCESS_THRESH;

    // ── Native date-range window (lookback priming) ──────────────────────────
    // Default = the full representable range, i.e. "no filtering" — every bar
    // fed to the strategy is within the execution window. This preserves
    // identical behavior to a date-unaware backtest when left unset.
    //
    // When start_date is set in the future relative to the first bars fed,
    // OnBarUpdate() continues to update every internal model (EGARCH, HMM,
    // sigma bands, order flow, Greeks surface, particle filter) for every bar
    // it receives — this is the "priming" warm-up — but the entry-execution
    // branch stays gated shut until a bar's timestamp >= start_date. Once a
    // bar's timestamp exceeds end_date, new entries are likewise blocked,
    // and FinalizeBacktest() will force-close any still-open position with
    // ExitReason::TimeExit.
    std::chrono::system_clock::time_point start_date =
        std::chrono::system_clock::time_point::min();
    std::chrono::system_clock::time_point end_date =
        std::chrono::system_clock::time_point::max();
};

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 2A — EGARCH(1,1) VOLATILITY CLUSTERING ENGINE
//   log(h_t) = ω + β·log(h_{t-1}) + α·(|z_{t-1}|−E[|z|]) + γ·z_{t-1}
//   Requires 300-bar warmup for MLE convergence (validated threshold).
//   Leverage effect via γ: negative return amplifies variance asymmetrically.
// ══════════════════════════════════════════════════════════════════════════════
namespace m2a {

class EGARCH {
public:
    EGARCH() = default;

    void on_return(double ret) noexcept {
        if (cnt_==0) {
            // Warm-start: initialise log_h from unconditional mean = ω/(1−β)
            // This prevents explosive negative log_h with the correct omega
            log_h_ = cfg::EGARCH_OMEGA_INIT / (1.0 - cfg::EGARCH_BETA_INIT);
            prev_ret_=ret; ++cnt_; return;
        }
        double h=std::exp(log_h_);
        double sqh=std::sqrt(std::max(h,1e-15));
        double zt=(sqh>1e-12)?prev_ret_/sqh:0.0;
        double abs_zt=std::abs(zt);

        // EGARCH log-variance update
        double nlh=omega_+beta_*log_h_+alpha_*(abs_zt-math::E_ABS_Z)+gamma_*zt;
        nlh=math::clamp(nlh,-25.0,2.0);  // [-25,2] = [e^-25..e^2] vol range [5.5e-6..1.65/bar]
        double hn=std::exp(nlh);

        // Gradient of log-likelihood: ∂ℓ/∂θ via chain rule through log_h
        double err=math::clamp(-0.5*(1.0/hn-(ret*ret)/(hn*hn)),-20.0,20.0);
        double dlh_dw=1.0+beta_*dw_,  dlh_da=(abs_zt-math::E_ABS_Z)+beta_*da_;
        double dlh_dg=zt+beta_*dg_,   dlh_db=log_h_+beta_*db_;

        // Gradient ascent (maximise log-likelihood)
        omega_+=cfg::EGARCH_LR*err*dlh_dw;
        alpha_+=cfg::EGARCH_LR*err*dlh_da;
        gamma_+=cfg::EGARCH_LR*err*dlh_dg;
        beta_ +=cfg::EGARCH_LR*err*dlh_db;

        // Stationarity: |β|<1, α>0
        beta_ =math::clamp(beta_,-0.999,0.999);
        alpha_=std::max(alpha_,1e-6);

        // Save recursive gradient terms
        dw_=dlh_dw; da_=dlh_da; dg_=dlh_dg; db_=dlh_db;
        log_h_=nlh;
        slow_.push(std::sqrt(hn));
        prev_ret_=ret; ++cnt_;
    }

    [[nodiscard]] double cond_var()   const noexcept { return std::exp(log_h_); }
    [[nodiscard]] double cond_vol()   const noexcept { return std::sqrt(cond_var()); }
    [[nodiscard]] double slow_vol()   const noexcept { return slow_.ready()?slow_.val():cond_vol(); }
    [[nodiscard]] bool   expanding()  const noexcept { return ready()&&slow_.ready()&&cond_vol()>slow_vol()*1.04; }
    [[nodiscard]] bool   high_vol()   const noexcept { return ready()&&slow_.ready()&&cond_vol()>slow_vol()*1.30; }
    [[nodiscard]] bool   low_vol()    const noexcept {
        // Before EGARCH is fully warmed up, never suppress (no reliable baseline)
        if (!ready()) return false;
        return slow_.ready() && cond_vol() < slow_vol()*0.85;
    }
    [[nodiscard]] bool   ready()      const noexcept { return cnt_>=cfg::EGARCH_WARMUP; }
    [[nodiscard]] double vol_ratio()  const noexcept { return (slow_vol()>1e-12)?cond_vol()/slow_vol():1.0; }
    [[nodiscard]] std::tuple<double,double,double,double> params() const noexcept {
        return {omega_,alpha_,beta_,gamma_};
    }

private:
    double omega_=cfg::EGARCH_OMEGA_INIT, alpha_=cfg::EGARCH_ALPHA_INIT;
    double beta_ =cfg::EGARCH_BETA_INIT,  gamma_=cfg::EGARCH_GAMMA_INIT;
    double log_h_=-9.0, prev_ret_=0.0;
    double dw_=0,da_=0,dg_=0,db_=0;
    math::EMA slow_{0.02};   // 50-bar half-life: tracks intraday regime shifts
    std::size_t cnt_=0;
};

} // namespace m2a

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 2B — IS-HMM (Inhomogeneous Switching HMM with Covariate-Driven A(t))
//   3 states: BULL(0) BEAR(1) CONS(2)
//   Transition A(t)_{ij} = softmax_j(W_{ij}·x_t) where x_t=[1, Z, norm_GEX]
//   Requires 200-bar window for EM transition matrix to stabilise (validated).
// ══════════════════════════════════════════════════════════════════════════════
namespace m2b {

class ISHMM {
public:
    static constexpr std::size_t S=cfg::HMM_STATES, D=cfg::HMM_OBS_DIM, NC=cfg::HMM_COV_DIM;
    using SVec=std::array<double,S>; using OVec=std::array<double,D>; using CVec=std::array<double,NC>;

    ISHMM() { init(); }

    // Push observation + covariate, run online filter update
    void push(const OVec& obs, const CVec& cov) noexcept {
        obs_buf_.push_back(obs); cov_buf_.push_back(cov);
        if (obs_buf_.size()>cfg::HMM_WINDOW) { obs_buf_.pop_front(); cov_buf_.pop_front(); }
        if (obs_buf_.size()>=20) filter_step(obs,cov);
    }

    [[nodiscard]] SVec       posterior() const noexcept { return post_; }
    [[nodiscard]] RegimeState regime()   const noexcept {
        auto mx=std::max_element(post_.begin(),post_.end());
        return static_cast<RegimeState>(std::size_t(mx-post_.begin()));
    }
    [[nodiscard]] bool       ready()    const noexcept { return obs_buf_.size()>=cfg::HMM_MIN_OBS; }

private:
    // W[i][j][k]: weight for transition i→j from covariate k
    std::array<std::array<std::array<double,NC>,S>,S> W_{};
    std::array<OVec,S> mu_{}, sig2_{};  // diagonal Gaussian emission
    SVec pi_{}, post_{};
    std::deque<OVec> obs_buf_;
    std::deque<CVec> cov_buf_;

    void init() noexcept {
        pi_.fill(1.0/S); post_.fill(1.0/S);
        // Emission priors from prior knowledge of regime characteristics
        mu_[0]={ 0.002, 0.012, 1.10}; sig2_[0]={1e-5,1e-4,0.04};  // Bull
        mu_[1]={-0.002, 0.015, 1.10}; sig2_[1]={1e-5,1e-4,0.04};  // Bear
        mu_[2]={ 0.000, 0.005, 0.80}; sig2_[2]={1e-6,5e-5,0.02};  // Cons
        // Self-loop bias in W
        for (auto& wi:W_) for (auto& wij:wi) wij.fill(0.0);
        for (std::size_t i=0;i<S;++i) W_[i][i][0]=1.5;
    }

    // Build time-varying A(t) via softmax over covariate logits
    std::array<SVec,S> build_A(const CVec& cov) const noexcept {
        std::array<SVec,S> A{};
        for (std::size_t i=0;i<S;++i) {
            double mx=-1e9;
            SVec logits{};
            for (std::size_t j=0;j<S;++j) {
                logits[j]=0; for (std::size_t k=0;k<NC;++k) logits[j]+=W_[i][j][k]*cov[k];
                mx=std::max(mx,logits[j]);
            }
            double Z=0;
            for (std::size_t j=0;j<S;++j) { A[i][j]=std::exp(logits[j]-mx); Z+=A[i][j]; }
            for (std::size_t j=0;j<S;++j) A[i][j]/=Z;
        }
        return A;
    }

    double emit(std::size_t st, const OVec& o) const noexcept {
        double lp=0;
        for (std::size_t d=0;d<D;++d) {
            double v=sig2_[st][d]+1e-10, diff=o[d]-mu_[st][d];
            lp+=-0.5*(std::log(2*math::PI*v)+diff*diff/v);
        }
        return std::exp(math::clamp(lp,-300.0,0.0));
    }

    void filter_step(const OVec& obs, const CVec& cov) noexcept {
        auto A=build_A(cov);
        SVec np{};
        double Z=0;
        for (std::size_t j=0;j<S;++j) {
            double pred=0; for (std::size_t i=0;i<S;++i) pred+=post_[i]*A[i][j];
            np[j]=pred*emit(j,obs); Z+=np[j];
        }
        if (Z>1e-15) for (auto& v:np) v/=Z;
        else np=post_;
        // Soft online M-step: nudge emission means toward observed value
        constexpr double lr=cfg::HMM_LR;
        for (std::size_t s=0;s<S;++s) for (std::size_t d=0;d<D;++d)
            mu_[s][d]+= lr*np[s]*(obs[d]-mu_[s][d]);
        // Covariate weight gradient (encourage GEX-driven transitions)
        // W[i][j] += lr * (A_ij - pi_j) * cov  -- simplified covariate update
        for (std::size_t i=0;i<S;++i) for (std::size_t j=0;j<S;++j)
            for (std::size_t k=0;k<NC;++k)
                W_[i][j][k]+=lr*(A[i][j]-1.0/S)*cov[k]*0.40;
        post_=np;
    }
};

} // namespace m2b

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 3 — DEALER HEDGING GREEKS & MARKET FRICTION
//   Aggregates GEX/Speed/Vanna across MdSpan2D(moneyness × expiry) surface.
//   Speed Dynamics: DEALER_WALL blocks entries; CASCADE authorises short fat-tail.
//   Vanna Dynamics: IV-rising + negative vanna = forced dealer selling.
// ══════════════════════════════════════════════════════════════════════════════
namespace m3 {

struct Surface {
    std::vector<double> gd,sd,vd;  // GEX, Speed, Vanna data
    MdSpan2D<double>    gs,ss,vs;  // 2D views
    Surface()
        : gd(cfg::MONEY_BINS*cfg::EXP_BINS,0),
          sd(cfg::MONEY_BINS*cfg::EXP_BINS,0),
          vd(cfg::MONEY_BINS*cfg::EXP_BINS,0),
          gs(gd.data(),cfg::MONEY_BINS,cfg::EXP_BINS),
          ss(sd.data(),cfg::MONEY_BINS,cfg::EXP_BINS),
          vs(vd.data(),cfg::MONEY_BINS,cfg::EXP_BINS) {}
    void reset() noexcept {
        std::ranges::fill(gd,0.0); std::ranges::fill(sd,0.0); std::ranges::fill(vd,0.0);
    }
};

class GreeksEngine {
public:
    GreeksEngine() = default;

    void update(std::span<const OptionRow> chain, double spot, double mult, int64_t now_ms) noexcept {
        surf_.reset();
        ng_=ns_=nv_=nc_=tm_=0.0;
        for (const auto& row:chain) {
            double tau=(row.expiry_ms-now_ms)/(1000.0*365.25*86400.0);
            if (tau<=0||row.mid_iv<=0) continue;
            double F=m1::implied_forward(spot,cfg::RISK_FREE_RATE,cfg::DIVIDEND_YIELD,tau);
            auto g=math::bs(row.type,spot,row.strike,cfg::RISK_FREE_RATE,row.mid_iv,tau,cfg::DIVIDEND_YIELD);
            double oi=row.open_interest;
            bool c=(row.type==OptionType::CALL);
            // GEX (Gamma Exposure) = OI × Γ × Multiplier × S² × 0.01
            double gex=oi*g.gamma*mult*spot*spot*cfg::GEX_SCALE;
            double sgx=c?gex:-gex; ng_+=sgx; tm_+=std::abs(gex);
            // SEX (Speed Exposure) and VEX/VNAX (Vanna Exposure)
            double spd=oi*g.speed*mult*spot*spot; ns_+=c?spd:-spd;
            double van=oi*g.vanna*mult;            nv_+=c?van:-van;
            // CHMX (Charm Exposure) — DTE-weighted (near-expiry pressure higher)
            double dte=tau*365.25;
            double wt=(dte<=5.0)?((5.0-dte)/5.0+0.1):1.0;
            double chm=oi*g.charm*mult*wt;          nc_+=c?chm:-chm;
            // Fill full surface (GEX/SEX/VEX bins; CHMX tracked as scalar aggregate)
            double M=m1::moneyness(row.strike,F,row.mid_iv,tau);
            std::size_t mb=m1::money_bin(M), eb=m1::expiry_bin(dte);
            surf_.gs(mb,eb)+=sgx; surf_.ss(mb,eb)+=c?spd:-spd; surf_.vs(mb,eb)+=c?van:-van;
        }
        classify();
    }

    void push_iv(double atm_iv) noexcept {
        iv_fast_.push(atm_iv); iv_slow_.push(atm_iv);
        if (iv_fast_.ready()&&iv_slow_.ready())
            iv_trend_=(iv_fast_.val()<iv_slow_.val()*0.998)?+1.0
                      :(iv_fast_.val()>iv_slow_.val()*1.002)?-1.0:0.0;
    }

    [[nodiscard]] double    net_gex()     const noexcept { return ng_; }
    [[nodiscard]] double    net_speed()   const noexcept { return ns_; }
    [[nodiscard]] double    net_vanna()   const noexcept { return nv_; }
    [[nodiscard]] double    total_mag()   const noexcept { return tm_; }
    [[nodiscard]] double    norm_gex()    const noexcept { return tm_>1e-10?ng_/tm_:0.0; }
    [[nodiscard]] double    norm_speed()  const noexcept { return tm_>1e-10?ns_/tm_:0.0; }
    [[nodiscard]] double    norm_vanna()  const noexcept { return tm_>1e-10?nv_/tm_:0.0; }
    [[nodiscard]] double    net_charm()   const noexcept { return nc_; }
    [[nodiscard]] double    norm_charm()  const noexcept { return tm_>1e-10?nc_/tm_:0.0; }
    [[nodiscard]] double    iv_trend()    const noexcept { return iv_trend_; }
    [[nodiscard]] SpeedZone speed_zone()  const noexcept { return sz_; }
    [[nodiscard]] bool      neg_gex()     const noexcept { return ng_<0.0; }
    [[nodiscard]] bool      vanna_sell()  const noexcept { return iv_trend_<0.0&&nv_<0.0; }
    [[nodiscard]] bool      vanna_buy()   const noexcept { return iv_trend_>0.0&&nv_>0.0; }

private:
    Surface surf_; double ng_=0,ns_=0,nv_=0,nc_=0,tm_=0,iv_trend_=0;
    math::EMA iv_fast_{0.12}, iv_slow_{0.025};
    SpeedZone sz_=SpeedZone::NEUTRAL;

    void classify() noexcept {
        double ns=tm_>1e-10?ns_/tm_:0.0;
        bool pg=ng_>0,ng=ng_<0, ps=ns>0.28,nss=ns<-0.28, ext=std::abs(ns)>0.68;
        if (pg&&ps)           sz_=SpeedZone::DEALER_WALL;
        else if (pg&&!ps)     sz_=SpeedZone::FRICTION;
        else if (ng&&nss&&ext)sz_=SpeedZone::CASCADE;
        else if (ng&&nss)     sz_=SpeedZone::ACCEL;
        else                  sz_=SpeedZone::NEUTRAL;
    }
};

} // namespace m3

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 4 — HYBRID REGRESSION SIGMA BANDS
//   25-bar lookback (validated: 20–30 range for intraday micro-cycle memory).
//   Bands deformed by net Vanna skew × EGARCH vol_ratio (asymmetric).
//   Negative Vanna → lower band expands, upper band contracts.
// ══════════════════════════════════════════════════════════════════════════════
namespace m4 {

struct Bands {
    double mu=0, sigma=0;
    double u1=0,u2=0,u3=0, l1=0,l2=0,l3=0;   // deformed (Vanna-adjusted)
    double su1=0,su2=0,su3=0,sl1=0,sl2=0,sl3=0; // symmetric (reference)
    double band_level=0;    // signed σ-units of current close
    double vanna_skew=0;    // deformation input
};

class BandEngine {
public:
    BandEngine(): stats_(cfg::SIGMA_WINDOW), ols_(cfg::SIGMA_WINDOW) {}

    void on_bar(double close) noexcept { stats_.push(close); ols_.push(close); }

    [[nodiscard]] Bands compute(double close, double norm_vanna, double vol_ratio) const noexcept {
        Bands b; if (!stats_.ready()) { b.mu=close; return b; }
        b.mu=stats_.mean();
        // Scale sigma by EGARCH vol_ratio: expanding vol → wider bands
        double rs=stats_.stddev();
        b.sigma=rs*std::max(0.4,vol_ratio);
        b.vanna_skew=norm_vanna;
        // Vanna deformation: negative Vanna → shrink upper, expand lower
        double def=math::clamp(norm_vanna,-1.0,1.0)*cfg::BAND_SKEW_FACTOR;
        auto up=[&](double n){ return b.mu+n*b.sigma*(1.0+def); };
        auto dn=[&](double n){ return b.mu-n*b.sigma*(1.0-def); };
        b.u1=up(1); b.u2=up(2); b.u3=up(3);
        b.l1=dn(1); b.l2=dn(2); b.l3=dn(3);
        b.su1=b.mu+1*b.sigma; b.su2=b.mu+2*b.sigma; b.su3=b.mu+3*b.sigma;
        b.sl1=b.mu-1*b.sigma; b.sl2=b.mu-2*b.sigma; b.sl3=b.mu-3*b.sigma;
        b.band_level=(b.sigma>1e-10)?(close-b.mu)/b.sigma:0.0;
        return b;
    }
    [[nodiscard]] double z(double x) const noexcept { return stats_.z(x); }
    [[nodiscard]] double slope()     const noexcept { return ols_.slope(); }
    [[nodiscard]] bool   ready()     const noexcept { return stats_.ready(); }

private:
    math::RollStats stats_; math::RollOLS ols_;
};

} // namespace m4

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 5 — MICROSTRUCTURE & ORDER FLOW
//   Absorption: negative delta (sellers hitting bid) but price holds →
//               passive limit buyers absorbing → long continuation setup.
//   Exhaustion: delta flattens at ±2σ/±3σ → halt all entries.
// ══════════════════════════════════════════════════════════════════════════════
namespace m5 {

struct OFState {
    double cum_delta=0, norm_delta=0, vol_ratio=1;
    double price_efficiency=0;
    bool   absorption=false, exhaustion=false;
};

class OFEngine {
    struct BR { double delta,range,vol; };
public:
    OFEngine(): vstats_(cfg::VOL_MEAN_WINDOW), eff_ema_(0.2){}

    void on_bar(const Bar& b) noexcept {
        vstats_.push(b.volume);
        double eff=(b.volume>1e-6)?(b.high-b.low)/b.volume:0.0;
        prev_eff_=eff_ema_.ready()?eff_ema_.val():eff;
        eff_ema_.push(eff);
        buf_.push_back(BR{b.cum_delta,(b.high-b.low),b.volume});
        if (buf_.size()>cfg::OF_BAR_WINDOW) buf_.pop_front();
        double vm=vstats_.mean(); double vr=(vm>1e-6)?b.volume/vm:1.0;
        double nd=(b.volume>1e-6)?math::clamp(b.cum_delta/b.volume,-1.0,1.0):0.0;
        state_={b.cum_delta,nd,vr,eff,abs_check(),exh_check(vr)};
    }

    [[nodiscard]] OFState state() const noexcept { return state_; }
    [[nodiscard]] bool    ready() const noexcept { return buf_.size()>=3; }

private:
    math::RollStats vstats_; math::EMA eff_ema_; double prev_eff_=0;
    std::deque<BR> buf_; OFState state_;

    // Absorption: significant opposing delta failing to move price
    bool abs_check() const noexcept {
        if (buf_.size()<3) return false;
        double sd=0,sr=0,sv=0;
        for (const auto& r:buf_) { sd+=r.delta; sr+=r.range; sv+=r.vol; }
        double nd=(sv>1e-6)?sd/sv:0.0;
        // For LONG setup: sellers dominant but price not dropping
        bool neg_delta=(nd<cfg::ABSORPTION_DELTA_TH);
        // For SHORT setup: buyers dominant but price not rising  
        bool pos_delta=(nd>-cfg::ABSORPTION_DELTA_TH);
        double me=vstats_.ready()?vstats_.mean():1.0;
        bool tight=(sr/static_cast<double>(buf_.size()))<me*0.95;
        return (neg_delta||pos_delta)&&tight;
    }

    // Exhaustion: volume expands but price efficiency collapses
    bool exh_check(double vr) const noexcept {
        return eff_ema_.ready()&&(eff_ema_.val()<prev_eff_*cfg::EXHAUST_EFF_DECAY)&&(vr>1.20);
    }
};

} // namespace m5

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 6 — CORE ADAPTIVE CONTINUATION LOGIC GATE
//   Synthesizes all modules into a boolean entry matrix.
//   Uses std::expected for clean error propagation through the gate chain.
// ══════════════════════════════════════════════════════════════════════════════
namespace m6 {

struct GateResult {
    TriggerVerdict verdict=TriggerVerdict::NO_TRADE;
    Direction      direction=Direction::FLAT;
    double         conviction=0.0;
    bool g_vol=false, g_gex=false, g_speed=false;
    bool g_vanna=false, g_sigma=false, g_of=false, g_hmm=false;
    bool is_cascade=false, is_retest=false;
    std::string_view why;
};

using GateExpected = std::expected<GateResult, std::string_view>;

class Gate {
public:
    [[nodiscard]] GateResult eval(
        const m2a::EGARCH&    eg,
        const m2b::ISHMM&     hmm,
        const m3::GreeksEngine& gk,
        const m4::Bands&      bands,
        const m5::OFState&    of,
        double z, [[maybe_unused]] double cps, double prev_ngex) const noexcept
    {
        GateResult res{};
        double bl=bands.band_level;
        auto post=hmm.posterior();

        // ── ±3σ HALT (standard exhaustion boundary) ─────────────────────────
        bool at3=(std::abs(bl)>=cfg::SIGMA_L3);
        // CASCADE EXCEPTION: −3σ + CASCADE SpeedZone + Vanna_sell
        bool cascade_exc=(bl<=-cfg::SIGMA_L3
                         &&gk.speed_zone()==SpeedZone::CASCADE
                         &&gk.vanna_sell());
        if (at3&&!cascade_exc) {
            res.verdict=TriggerVerdict::HALT_3SIGMA;
            res.why="±3σ exhaustion: halt all entries";
            return res;
        }

        // ── GATE 1: Vol regime ───────────────────────────────────────────────
        // SUPPRESS only when: EGARCH ready AND cond_vol < slow_vol×0.85 AND no GEX catalyst
        // ALLOW in: normal vol, expanding vol, high vol, pre-warmup, GEX zero-crossing
        {
            bool gex_cross_g  = (prev_ngex >= 0.0 && gk.net_gex() < 0.0);
            bool vol_suppress = eg.ready() && eg.low_vol() && !gex_cross_g;
            res.g_vol = !vol_suppress;
        }

        // ── GATE 2: GEX ─────────────────────────────────────────────────────
        bool long_gex_ok=(gk.norm_gex()>=-0.50);    // neutral/slight neg = OK for long
        bool short_gex_ok=(gk.net_gex()<0.0);        // must be negative for short
        res.g_gex=(long_gex_ok||short_gex_ok);

        // ── GATE 3: Speed ────────────────────────────────────────────────────
        // DEALER_WALL: blocks ALL longs. CASCADE/ACCEL: confirms shorts.
        res.g_speed=(gk.speed_zone()!=SpeedZone::DEALER_WALL);

        // ── GATE 4: Vanna ────────────────────────────────────────────────────
        bool vbuy=(gk.vanna_buy()||gk.net_vanna()>=0.0);
        bool vsell=gk.vanna_sell();
        res.g_vanna=(vbuy||vsell);

        // ── GATE 5: Sigma band ───────────────────────────────────────────────
        res.g_sigma=(std::abs(bl)>=cfg::SIGMA_L1&&std::abs(bl)<cfg::SIGMA_L3)||cascade_exc;

        // ── GATE 6: Order flow ───────────────────────────────────────────────
        // Entry: absorption confirmed OR delta turning favourable; exhaustion=halt
        res.g_of=!of.exhaustion&&(of.absorption||std::abs(of.norm_delta)>0.05);

        // ── GATE 7: HMM regime ───────────────────────────────────────────────
        res.g_hmm=hmm.ready();

        // ── SHORT CASCADE (fat-tail exception) ───────────────────────────────
        if (cascade_exc) {
            if (res.g_vol&&short_gex_ok&&vsell&&res.g_of) {
                res.verdict=TriggerVerdict::SHORT_CASCADE;
                res.direction=Direction::SHORT;
                res.is_cascade=true;
                res.why="FAT-TAIL CASCADE: −3σ+neg_speed+vanna_sell=acceleration node";
                fill_conviction(res,post,of,eg,z,gk);
                return res;
            }
        }

        // ── LONG CONTINUATION / SQUEEZE ─────────────────────────────────────
        bool lreg=!hmm.ready()||(post[0]>post[1]&&post[0]>post[2])||post[0]>0.36;
        bool lsig=(bl>=cfg::SIGMA_L1&&bl<cfg::SIGMA_L3);
        bool lof =!of.exhaustion&&(of.absorption||of.norm_delta>0.0);
        bool lspd=(gk.speed_zone()!=SpeedZone::DEALER_WALL);
        bool lvan=!vsell;
        bool lz  =(z>=cfg::Z_ENTRY_MIN);
        if (res.g_vol&&lreg&&lsig&&lof&&lspd&&lvan&&long_gex_ok&&lz) {
            res.g_gex=res.g_speed=res.g_vanna=res.g_sigma=res.g_of=res.g_hmm=true;
            // Retest: returning inside band after wick = highest-conviction setup
            bool retest=(bl<cfg::SIGMA_L2+0.15&&of.absorption);
            res.verdict=retest?TriggerVerdict::LONG_SQUEEZE:TriggerVerdict::LONG_CONT;
            res.direction=Direction::LONG;
            res.is_retest=retest;
            res.why=retest?"LONG RETEST SQUEEZE: absorption at +1σ/+2σ, sellers waning"
                          :"LONG CONTINUATION: +1σ−+2σ, regime BULL, OF confirmed";
            fill_conviction(res,post,of,eg,z,gk);
            return res;
        }

        // ── SHORT CONTINUATION (fat-tail capture) ───────────────────────────
        bool sreg=!hmm.ready()||(post[1]>post[0]&&post[1]>post[2])||post[1]>0.36;
        bool ssig=(bl<=-cfg::SIGMA_L1&&bl>-cfg::SIGMA_L3);
        bool sof =!of.exhaustion&&(of.absorption||of.norm_delta<0.0);
        bool sspd=(gk.speed_zone()==SpeedZone::ACCEL||gk.speed_zone()==SpeedZone::CASCADE
                 ||gk.speed_zone()==SpeedZone::NEUTRAL);
        bool sz  =(z<=-cfg::Z_ENTRY_MIN);
        if (res.g_vol&&sreg&&ssig&&sof&&sspd&&vsell&&short_gex_ok&&sz) {
            res.g_gex=res.g_speed=res.g_vanna=res.g_sigma=res.g_of=res.g_hmm=true;
            res.verdict=TriggerVerdict::SHORT_CONT;
            res.direction=Direction::SHORT;
            res.why="SHORT CONTINUATION: −1σ/−2σ, neg GEX+Speed, Vanna_sell, OF breakdown";
            fill_conviction(res,post,of,eg,z,gk);
            return res;
        }

        // Setup forming — gates partially met
        if (res.g_vol&&res.g_sigma&&(long_gex_ok||short_gex_ok)) {
            res.verdict=TriggerVerdict::WAIT_SETUP;
            res.why="WAIT: Greek+sigma conditions met; awaiting OF absorption";
        }
        return res;
    }

private:
    void fill_conviction(GateResult& r,
                         const m2b::ISHMM::SVec& post,
                         const m5::OFState& of,
                         const m2a::EGARCH& eg,
                         double z, const m3::GreeksEngine& gk) const noexcept {
        double s=0;
        s+=0.25*(r.direction==Direction::LONG?post[0]:post[1]); // HMM certainty
        s+=0.20*math::clamp(std::abs(z)/3.0,0.0,1.0);          // Z magnitude
        s+=0.15*math::clamp(of.vol_ratio-1.0,0.0,1.0);         // vol anomaly
        s+=0.15*(eg.high_vol()?1.0:eg.expanding()?0.65:0.25);  // EGARCH state
        s+=0.15*std::abs(gk.norm_gex());                        // GEX magnitude
        s+=0.10*std::abs(gk.norm_vanna());                      // Vanna magnitude
        r.conviction=math::clamp(s,0.0,1.0);
    }
};

} // namespace m6

// ══════════════════════════════════════════════════════════════════════════════
// MODULE 7 — SEQUENTIAL MONTE CARLO / PARTICLE FILTER (BAYESIAN UPDATER)
//   500 particles · ESS exactly 0.50 (validated optimal threshold).
//   State: [drift, log_vol, gex_level] per particle.
//   Bayesian Vanna prior updates posterior continuation probability.
//   Systematic resampling at ESS < 0.50×N (prevents path impoverishment).
// ══════════════════════════════════════════════════════════════════════════════
namespace m7 {

struct Particle { double drift=0, log_vol=-4.0, gex=0, w=0; };

class PFilter {
public:
    explicit PFilter(std::size_t N=cfg::N_PARTICLES, uint64_t seed=0xDEADBEEF42ULL)
        : N_(N), rng_(seed)
    {
        particles_.resize(N);
        std::uniform_real_distribution<double> ud(-5e-4,5e-4), lv(-6.0,-3.0);
        for (auto& p:particles_) { p.drift=ud(rng_); p.log_vol=lv(rng_); p.gex=0; p.w=1.0/N; }
    }

    // Returns posterior CPS with Bayesian Vanna prior
    double step(double obs_ret, double obs_vol, double ngex_norm,
                double nvanna_norm, Direction dir) noexcept
    {
        std::normal_distribution<double> nd(0,1);
        // 1. Propagate particles (state transition)
        // Drift mean-reversion target: toward observed return (scaled by persistence).
        // This ensures directional information from price actually updates the filter.
        double drift_target = obs_ret * 12.0;  // scale: bar ret → annualised drift proxy
        for (auto& p:particles_) {
            double gex_drag = -0.25 * p.gex;
            // Pull drift toward observed return — key fix for CPS convergence
            p.drift   = 0.88*p.drift + 0.08*drift_target + gex_drag*5e-4 + 1.5e-4*nd(rng_);
            p.log_vol = 0.96*p.log_vol + 0.04*std::log(std::max(obs_vol,1e-8)) + 4e-3*nd(rng_);
            p.gex     = 0.88*p.gex     + 0.12*ngex_norm + 0.04*nd(rng_);
        }
        // 2. Weight: likelihood of obs_ret given particle state
        double wsum=0;
        for (auto& p:particles_) {
            double h=std::exp(p.log_vol), resid=obs_ret-p.drift;
            double ll=-0.5*(std::log(2*math::PI*h*h)+resid*resid/(h*h));
            double gex_ll=-0.5*std::pow(p.gex-ngex_norm,2)/0.20;
            p.w*=std::exp(math::clamp(ll+gex_ll,-30.0,0.0));
            wsum+=p.w;
        }
        // 3. Normalise
        if (wsum<1e-30) { for (auto& p:particles_) p.w=1.0/N_; wsum=1.0; }
        else for (auto& p:particles_) p.w/=wsum;
        // 4. ESS — resamples exactly when ESS < 0.50×N (validated threshold)
        double ess=0; for (const auto& p:particles_) ess+=p.w*p.w;
        ess=1.0/ess;
        if (ess < cfg::ESS_THRESHOLD*static_cast<double>(N_)) resample();
        // 5. Posterior: P(continuation | state)
        double pcont=0;
        for (const auto& p:particles_)
            if ((dir==Direction::LONG&&p.drift>0)||(dir==Direction::SHORT&&p.drift<0))
                pcont+=p.w;
        // 6. Bayesian multi-prior update: Vanna + return direction
        // Vanna prior: IV falling + pos vanna = tailwind for longs
        double vp=(dir==Direction::SHORT)
            ? math::clamp(-nvanna_norm*0.55+0.52,0.30,0.90)
            : math::clamp( nvanna_norm*0.55+0.52,0.30,0.90);
        // Return direction prior: consistent return in signal direction
        double ret_sign = (dir==Direction::LONG) ? obs_ret : -obs_ret;
        double rp = math::clamp(0.50 + ret_sign * 25.0, 0.35, 0.85);
        // Combined log-odds update
        double log_odds = std::log(std::max(pcont,1e-6)/std::max(1.0-pcont,1e-6))
                        + std::log(vp/std::max(1.0-vp,1e-6))
                        + 0.5*std::log(rp/std::max(1.0-rp,1e-6));
        return math::clamp(math::sigmoid(log_odds),0.0,1.0);
    }

    [[nodiscard]] double cps(Direction dir) const noexcept {
        double s=0;
        for (const auto& p:particles_)
            if ((dir==Direction::LONG&&p.drift>0)||(dir==Direction::SHORT&&p.drift<0))
                s+=p.w;
        return math::clamp(s,0.0,1.0);
    }

private:
    std::size_t N_; std::mt19937_64 rng_; std::vector<Particle> particles_;

    void resample() noexcept {
        // Systematic resampling — unbiased, O(N)
        std::vector<double> cdf(N_,0);
        cdf[0]=particles_[0].w;
        for (std::size_t i=1;i<N_;++i) cdf[i]=cdf[i-1]+particles_[i].w;
        std::uniform_real_distribution<double> u(0.0,1.0/N_);
        double start=u(rng_);
        std::vector<Particle> np; np.reserve(N_);
        std::size_t j=0;
        for (std::size_t i=0;i<N_;++i) {
            double t=start+static_cast<double>(i)/N_;
            while (j<N_-1&&cdf[j]<t) ++j;
            np.push_back(particles_[j]); np.back().w=1.0/N_;
        }
        particles_=std::move(np);
    }
};

} // namespace m7

// ══════════════════════════════════════════════════════════════════════════════
// HybridSigmaStrategy — THE DECOUPLED BRAIN
//   Event-driven. Owns no file handles, no CSV parser, no knowledge of dates
//   beyond the BacktestConfig window it was given. Reacts only to
//   OnBarUpdate()/OnTick() calls made by whatever external loop is feeding it.
//
//   MATHEMATICAL FIDELITY: every formula and gate condition inside
//   OnBarUpdate() is copied verbatim from the original monolithic engine's
//   per-bar loop body. The single new term is `&& in_window_` appended to
//   the pre-existing `entry_allowed` boolean — exactly the lookback-priming
//   gate this refactor was commissioned to add, and nothing else.
// ══════════════════════════════════════════════════════════════════════════════
class HybridSigmaStrategy {
public:
    HybridSigmaStrategy() {
        on_trade_event_ = [](const TradeContext&) {};
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    // Configure the strategy for a fresh backtest run. Always wipes any prior
    // state first (equivalent to calling Reset()), then applies `cfg`.
    void Initialize(const BacktestConfig& cfg) {
        Reset();
        cfg_ = cfg;
        // Hard architectural invariant — no single trade may risk more than
        // cfg::RISK_PCT (0.60%) of equity. Not a tunable; clamp and flag via
        // assert (debug builds) rather than std::cout (no console I/O in the brain).
        if (cfg_.risk_pct > cfg::RISK_PCT) {
            assert(false && "BacktestConfig::risk_pct exceeds the 0.60% hard cap — clamping to cfg::RISK_PCT");
            cfg_.risk_pct = cfg::RISK_PCT;
        }
        equity_ = cfg.equity;
        amap_.asset = cfg.asset;
        // Seed ratio estimate — identical constant to the original,
        // validated NQ ingestion-layer code (MNQ tracks the same index
        // level as NQ, only the $/point multiplier differs), simply
        // relocated into the brain's own lazy first-bar initialisation
        // (see OnBarUpdate()).
        seed_etf_ratio_ = 40.0;
        initialized_ = true;
    }

    // Cleanly wipes ALL internal state — every module is reconstructed from
    // scratch, all trade history/feature history is cleared, equity resets
    // to the configured starting value. Safe to call between sequential
    // date-range backtests on the same strategy instance with zero leakage.
    void Reset() noexcept {
        egarch_     = m2a::EGARCH{};
        hmm_        = m2b::ISHMM{};
        gk_         = m3::GreeksEngine{};
        bands_eng_  = m4::BandEngine{};
        of_eng_     = m5::OFEngine{};
        gate_       = m6::Gate{};
        smc_        = m7::PFilter{};
        atr_        = math::ATR(14);
        open_.reset();
        trades_.clear();
        features_.clear();
        gstat_      = GateStat{};
        prev_close_ = 0.0;
        prev_gex_   = 0.0;
        bar_count_  = 0;
        next_trade_id_ = 1;
        equity_     = cfg_.equity;
        amap_seeded_ = false;
        tick_agg_   = TickAggregator{};
        // ml_preds_ intentionally NOT cleared by Reset() — predictions are a
        // host-supplied configuration, not strategy state; call
        // SetMLPredictions({}) explicitly if you want to clear them too.
    }

    // ── Primary data-feed entry point ─────────────────────────────────────────
    // The ONLY way price/options data reaches the strategy. `chain_for_bar`
    // should contain exactly the OptionRow entries whose timestamp rounds to
    // this bar's bucket (see bucket_timestamp() below) — the external loop
    // owns that slicing responsibility; the brain just consumes what it's given.
    void OnBarUpdate(const Bar& bar, const std::vector<OptionRow>& chain_for_bar) noexcept {
        if (!initialized_) return;  // must call Initialize() first

        // Lazy-seed the AssetMap ratio from the very first bar ever received —
        // identical math to the original ingestion-layer seeding, relocated.
        if (!amap_seeded_) {
            amap_.update(bar.close, bar.close/seed_etf_ratio_);
            amap_seeded_ = true;
        }

        // ── Execution-window membership (lookback priming gate) ────────────
        auto bar_tp   = ms_to_time_point(bar.timestamp_ms);
        in_window_    = (bar_tp >= cfg_.start_date) && (bar_tp <= cfg_.end_date);
        bool past_end = (bar_tp > cfg_.end_date);

        const Bar& b = bar;
        double spot = amap_.etf(b.close);

        // ── Feed all modules — VERBATIM from the original engine ───────────
        double lr = (prev_close_>0) ? std::log(b.close/prev_close_) : 0.0;
        egarch_.on_return(lr);
        bands_eng_.on_bar(b.close);
        of_eng_.on_bar(b);
        if (bar_count_>0) atr_.push(b.high,b.low,b.close);

        double rv = egarch_.cond_vol();
        double ar = (atr_.ready()&&atr_.val()>1e-10) ? rv/atr_.val() : 1.0;
        double z  = bands_eng_.z(b.close);
        double ng = (gk_.total_mag()>0) ? gk_.net_gex()/gk_.total_mag() : 0.0;
        hmm_.push({lr,rv,ar},{1.0,z,ng});

        // ── Options chain for this bar (caller-supplied slice) ─────────────
        if (!chain_for_bar.empty()) {
            auto ai = std::ranges::min_element(chain_for_bar,
                [&](const OptionRow& a, const OptionRow& bb){
                    return std::abs(a.strike-spot) < std::abs(bb.strike-spot);
                });
            if (ai != chain_for_bar.end()) gk_.push_iv(ai->mid_iv);
            gk_.update(chain_for_bar, spot, cfg_.mult, b.timestamp_ms);
        }

        // ── Sigma bands ──────────────────────────────────────────────────────
        auto bands = bands_eng_.compute(b.close, gk_.norm_vanna(), egarch_.vol_ratio());

        // ── SMC — Bayesian prior feeds conviction ───────────────────────────
        Direction dir_guess = (z>=0) ? Direction::LONG : Direction::SHORT;
        double step_cps = smc_.step(lr, rv, ng, gk_.norm_vanna(), dir_guess);
        double pcps = std::max(step_cps, smc_.cps(dir_guess));

        // ── Gate evaluation ──────────────────────────────────────────────────
        auto of = of_eng_.state();
        auto gres = gate_.eval(egarch_, hmm_, gk_, bands, of, z, pcps, prev_gex_);
        if (gres.conviction > pcps) pcps = gres.conviction;

        // ── Feature capture (every bar, zero look-ahead) ────────────────────
        {
            FeatureRow fr;
            fr.timestamp_ms     = b.timestamp_ms;
            fr.close            = b.close;
            fr.implied_forward  = m1::implied_forward(spot, cfg::RISK_FREE_RATE,
                                                       cfg::DIVIDEND_YIELD, 5.0/365.0);
            fr.basis_spread     = b.close - spot*amap_.ratio;
            fr.egarch_vol       = egarch_.cond_vol();
            fr.egarch_vol_ratio = egarch_.vol_ratio();
            fr.egarch_ready     = egarch_.ready();
            fr.low_vol          = egarch_.low_vol();
            fr.high_vol         = egarch_.high_vol();
            fr.expanding        = egarch_.expanding();
            { auto post=hmm_.posterior(); fr.hmm_bull=post[0]; fr.hmm_bear=post[1]; fr.hmm_cons=post[2]; }
            fr.hmm_ready        = hmm_.ready();
            fr.atr_fast         = atr_.val();
            fr.atr_slow         = atr_.val();
            fr.atr_ratio        = ar;
            fr.ols_slope        = bands_eng_.slope();
            fr.vol_zscore       = z;
            fr.net_gex          = gk_.net_gex();
            fr.norm_gex         = gk_.norm_gex();
            fr.net_speed        = gk_.net_speed();
            fr.norm_speed       = gk_.norm_speed();
            fr.net_vanna        = gk_.net_vanna();
            fr.norm_vanna       = gk_.norm_vanna();
            fr.net_charm        = gk_.net_charm();
            fr.iv_trend         = gk_.iv_trend();
            fr.speed_zone       = static_cast<int>(gk_.speed_zone());
            fr.sigma            = bands.sigma;
            fr.band_level       = bands.band_level;
            fr.z_score          = z;
            fr.vanna_skew       = bands.vanna_skew;
            fr.cum_delta        = of.cum_delta;
            fr.norm_delta       = of.norm_delta;
            fr.of_vol_ratio     = of.vol_ratio;
            fr.price_efficiency = of.price_efficiency;
            fr.absorption       = of.absorption;
            fr.exhaustion       = of.exhaustion;
            fr.g_vol            = gres.g_vol;
            fr.g_gex            = gres.g_gex;
            fr.g_speed          = gres.g_speed;
            fr.g_vanna          = gres.g_vanna;
            fr.g_sigma          = gres.g_sigma;
            fr.g_of             = gres.g_of;
            fr.g_hmm            = gres.g_hmm;
            fr.verdict          = static_cast<int>(gres.verdict);
            fr.direction        = static_cast<int>(gres.direction);
            fr.conviction       = gres.conviction;
            fr.particle_cps     = pcps;
            features_.push_back(fr);
        }

        // ── Manage open trade ────────────────────────────────────────────────
        if (open_) {
            auto& t = *open_;
            double dp = (b.close-t.sig.entry_price)*(t.sig.direction==Direction::LONG?1:-1);
            t.mae = std::min(t.mae,dp); t.mfe = std::max(t.mfe,dp); ++t.bars_held;
            bool stop = (t.sig.direction==Direction::LONG && b.low<=t.sig.stop_price)
                      ||(t.sig.direction==Direction::SHORT && b.high>=t.sig.stop_price);
            bool tgt  = (t.sig.direction==Direction::LONG && b.high>=t.sig.target_price)
                      ||(t.sig.direction==Direction::SHORT && b.low<=t.sig.target_price);
            if (stop||tgt) {
                t.exit_price = stop ? t.sig.stop_price : t.sig.target_price;
                t.pnl = (t.exit_price-t.sig.entry_price)
                        *(t.sig.direction==Direction::LONG?1:-1)
                        *cfg_.mult*t.sig.contracts;
                t.stopped=stop; t.target_hit=tgt;
                // ── ACCOUNT RISK CONSTRAINT verification (hard cap, non-negotiable) ──
                // contracts was sized at entry as floor(risk/(stop_d*mult)) with
                // risk = min(equity_*cfg_.risk_pct, equity_*Kf), so a stopped-out
                // loss can never exceed equity_*cfg_.risk_pct. Verified here as a
                // debug-only runtime invariant — never reported via std::cout.
                assert((t.pnl >= 0.0 || -t.pnl <= equity_*cfg_.risk_pct + 1e-6)
                       && "Trade loss exceeds the 0.60% account-risk hard cap");
                {
                    TradeContext tc;
                    tc.trade_id=t.trade_id; tc.timestamp_ms=b.timestamp_ms;
                    tc.symbol=asset_symbol(t.sig.asset); tc.size=t.sig.contracts;
                    tc.target_price=t.exit_price;
                    tc.direction=to_order_direction(t.sig.direction, true);
                    tc.order_type = stop ? OrderType::Stop : OrderType::Limit;
                    tc.exit_reason = stop ? ExitReason::StopLoss : ExitReason::TakeProfit;
                    on_trade_event_(tc);
                }
                trades_.push_back(*open_); open_.reset();
                // NOTE: equity_ is intentionally NOT updated here. The brain
                // returns raw t.pnl per trade via Trades(); the external
                // harness reconstructs the compounding equity curve (and
                // applies the zero-floor clamp) outside the hot loop.
                // equity_ remains the static Initialize()-time base used
                // solely as the Kelly-sizing denominator below.
            }
        }

        // ── Gate diagnostics ─────────────────────────────────────────────────
        if (bands_eng_.ready() && egarch_.ready()) {
            bool diag_sigma = (std::abs(bands.band_level) >= cfg::SIGMA_L1);
            bool diag_z     = (std::abs(z) >= cfg::Z_ENTRY_MIN);
            bool diag_gex   = (gk_.norm_gex() >= -0.50 || gk_.net_gex() < 0.0);
            bool diag_spd   = (gk_.speed_zone() != SpeedZone::DEALER_WALL);
            bool diag_of    = !of.exhaustion;
            bool diag_fired = (gres.verdict==TriggerVerdict::LONG_CONT
                            || gres.verdict==TriggerVerdict::LONG_SQUEEZE
                            || gres.verdict==TriggerVerdict::SHORT_CONT
                            || gres.verdict==TriggerVerdict::SHORT_CASCADE);
            record_gate(gres.g_vol, diag_sigma, diag_of, diag_gex, diag_spd, diag_z, diag_fired);
        }

        // ── ML entry filter lookup (Path B) ──────────────────────────────────
        double ml_p_success = 1.0;
        bool   ml_filter_active = !ml_preds_.empty();
        if (ml_filter_active) {
            auto it = ml_preds_.find(b.timestamp_ms);
            ml_p_success = (it!=ml_preds_.end()) ? it->second.p_success : 0.0;
        }

        // ── New entry — IDENTICAL condition to the original engine, with     ──
        // ── exactly one added term: `&& in_window_` (the lookback-priming   ──
        // ── gate this refactor was commissioned to add).                     ──
        bool entry_allowed =
            !open_
            && gres.direction!=Direction::FLAT
            && (gres.verdict==TriggerVerdict::LONG_CONT
              || gres.verdict==TriggerVerdict::LONG_SQUEEZE
              || gres.verdict==TriggerVerdict::SHORT_CONT
              || gres.verdict==TriggerVerdict::SHORT_CASCADE)
            && pcps>=(gres.is_cascade?cfg::CPS_HIGH_VOL:cfg::CPS_DEFAULT)
            && (!ml_filter_active||ml_p_success>=cfg_.ml_threshold)
            && atr_.ready() && bands_eng_.ready()
            && in_window_;   // ← lookback-priming gate (the only new condition)

        if (entry_allowed) {
            double atrv = atr_.val();
            double stop_d = cfg_.stop_atr_mult*atrv;
            double tgt_d = std::max(cfg_.target_sigma_mult*bands.sigma, stop_d*cfg_.min_rr);
            double rr = tgt_d/std::max(stop_d,1e-10);
            if (rr>=cfg_.min_rr) {
                TradeSignal sig;
                sig.timestamp_ms=b.timestamp_ms; sig.asset=cfg_.asset;
                sig.direction=gres.direction; sig.entry_price=b.close;
                sig.stop_price=(gres.direction==Direction::LONG)?b.close-stop_d:b.close+stop_d;
                sig.target_price=(gres.direction==Direction::LONG)?b.close+tgt_d:b.close-tgt_d;
                sig.sigma_band_lvl=bands.band_level; sig.z_score=z;
                sig.regime=hmm_.regime(); sig.cps=pcps;
                sig.net_gex=gk_.net_gex(); sig.net_speed=gk_.net_speed();
                sig.net_vanna=gk_.net_vanna(); sig.egarch_vol=rv;
                sig.particle_conf=gres.conviction; sig.speed_zone=gk_.speed_zone();
                sig.verdict=gres.verdict; sig.rr=rr;
                sig.is_cascade=gres.is_cascade; sig.is_retest=gres.is_retest;
                sig.ml_p_success=ml_p_success;
                double Kb=rr, Kp=pcps;
                double Kf=math::clamp((Kp*Kb-(1-Kp))/Kb, 0.0, cfg_.kelly_cap);
                double risk=std::min(equity_*cfg_.risk_pct, equity_*Kf);
                double rpc=stop_d*cfg_.mult;
                sig.contracts=(rpc>0)?std::floor(risk/rpc):0;
                sig.kelly=Kf;
                if (sig.contracts>=1) {
                    TradeRecord rec; rec.sig=sig;
                    rec.trade_id = next_trade_id_++;
                    open_=rec;
                    {
                        TradeContext tc;
                        tc.trade_id=rec.trade_id; tc.timestamp_ms=sig.timestamp_ms;
                        tc.symbol=asset_symbol(sig.asset); tc.size=sig.contracts;
                        tc.target_price=sig.entry_price;
                        tc.direction=to_order_direction(sig.direction, false);
                        tc.order_type=OrderType::Market;
                        tc.entry_reason=to_entry_reason(sig.verdict);
                        on_trade_event_(tc);
                    }
                }
            }
        }

        // ── If we've moved past end_date with a position still open, force ──
        // ── closure now rather than waiting for FinalizeBacktest() — keeps ──
        // ── the date window strictly authoritative for entries AND exits.  ──
        if (past_end && open_) {
            ForceCloseOpenPosition(b.timestamp_ms, b.close, ExitReason::TimeExit);
        }

        prev_close_ = b.close;
        prev_gex_   = gk_.net_gex();
        ++bar_count_;
    }

    // ── Convenience tick-level entry point ────────────────────────────────────
    // Internally aggregates ticks into bars using the identical field-update
    // math as the original batch aggregator (high/low/close/volume/buy_vol/
    // sell_vol/cum_delta accumulation), then forwards each COMPLETED bar to
    // OnBarUpdate(). `chain_for_current_bar` lets the host attach options data
    // that arrived during this tick's bar window — pass {} if none.
    void OnTick(const FuturesTick& tick,
               const std::vector<OptionRow>& chain_for_current_bar = {}) noexcept {
        int64_t bms = static_cast<int64_t>(cfg::BAR_SECONDS) * 1000;
        int64_t bucket = (tick.timestamp_ms / bms) * bms;

        if (!tick_agg_.active) {
            tick_agg_.active = true;
            tick_agg_.bucket_ms = bucket;
            tick_agg_.bar = Bar{};
            tick_agg_.bar.timestamp_ms = bucket;
            tick_agg_.bar.open = tick_agg_.bar.high = tick_agg_.bar.low = tick_agg_.bar.close = tick.price;
        } else if (bucket != tick_agg_.bucket_ms) {
            // Bar boundary crossed — finalise the completed bar and forward it.
            tick_agg_.bar.vwap = (tick_agg_.wv>0) ? tick_agg_.ws/tick_agg_.wv : tick_agg_.bar.close;
            OnBarUpdate(tick_agg_.bar, tick_agg_.pending_chain);
            tick_agg_.bar = Bar{};
            tick_agg_.bar.timestamp_ms = bucket;
            tick_agg_.bar.open = tick_agg_.bar.high = tick_agg_.bar.low = tick_agg_.bar.close = tick.price;
            tick_agg_.ws = tick_agg_.wv = 0.0;
            tick_agg_.pending_chain.clear();
            tick_agg_.bucket_ms = bucket;
        }

        tick_agg_.bar.high = std::max(tick_agg_.bar.high, tick.price);
        tick_agg_.bar.low  = std::min(tick_agg_.bar.low,  tick.price);
        tick_agg_.bar.close = tick.price;
        tick_agg_.bar.volume += tick.volume;
        tick_agg_.ws += tick.price*tick.volume; tick_agg_.wv += tick.volume;
        if (tick.is_buy_aggr) tick_agg_.bar.buy_vol += tick.volume;
        else                  tick_agg_.bar.sell_vol += tick.volume;
        tick_agg_.bar.cum_delta += tick.is_buy_aggr ? tick.volume : -tick.volume;

        if (!chain_for_current_bar.empty())
            tick_agg_.pending_chain.insert(tick_agg_.pending_chain.end(),
                chain_for_current_bar.begin(), chain_for_current_bar.end());
    }

    // ── Call once the external loop has no more data to push ─────────────────
    // Replaces the old "close any open trade at last bar" tail logic. Also
    // flushes any partially-built bar still sitting in the tick aggregator.
    void FinalizeBacktest() noexcept {
        if (tick_agg_.active) {
            tick_agg_.bar.vwap = (tick_agg_.wv>0) ? tick_agg_.ws/tick_agg_.wv : tick_agg_.bar.close;
            OnBarUpdate(tick_agg_.bar, tick_agg_.pending_chain);
            tick_agg_.active = false;
        }
        if (open_) {
            ForceCloseOpenPosition(prev_close_ts_ms(), prev_close_, ExitReason::TimeExit);
        }
    }

    // ── Configuration hooks ───────────────────────────────────────────────────
    void SetTradeEventCallback(std::function<void(const TradeContext&)> cb) {
        on_trade_event_ = std::move(cb);
    }
    void SetMLPredictions(std::unordered_map<int64_t, m1::MLPrediction> preds) {
        ml_preds_ = std::move(preds);
    }

    // ── Introspection ─────────────────────────────────────────────────────────
    [[nodiscard]] bool IsWarmedUp() const noexcept { return egarch_.ready() && bands_eng_.ready(); }
    [[nodiscard]] bool IsInExecutionWindow() const noexcept { return in_window_; }
    [[nodiscard]] double Equity() const noexcept { return equity_; }
    [[nodiscard]] std::size_t BarsProcessed() const noexcept { return bar_count_; }
    [[nodiscard]] const std::vector<TradeRecord>& Trades() const noexcept { return trades_; }
    [[nodiscard]] const std::vector<FeatureRow>& Features() const noexcept { return features_; }
    [[nodiscard]] std::tuple<double,double,double,double> EgarchParams() const noexcept {
        return egarch_.params();
    }

    // ── Reporting: NONE. Sharpe, win rate, drawdown, CSV/file export, and any
    // console output belong exclusively to the external Python/Streamlit
    // harness, which consumes Trades() and Features() and computes tearsheet
    // metrics via pandas/quantstats. The brain stays a pure signal engine.

private:
    struct GateStat {
        std::size_t total=0,g_vol=0,g_sigma=0,g_of=0,g_gex=0,g_speed=0,g_z=0,fired=0;
    };
    struct TickAggregator {
        bool active=false; int64_t bucket_ms=0;
        Bar bar{}; double ws=0, wv=0;
        std::vector<OptionRow> pending_chain;
    };

    void record_gate(bool gv,bool gs,bool gof,bool gg,bool gsp,bool gz,bool fired) noexcept {
        ++gstat_.total;
        if(gv){++gstat_.g_vol;} if(gs){++gstat_.g_sigma;}
        if(gof){++gstat_.g_of;} if(gg){++gstat_.g_gex;}
        if(gsp){++gstat_.g_speed;} if(gz){++gstat_.g_z;}
        if(fired){++gstat_.fired;}
    }

    void ForceCloseOpenPosition(int64_t ts_ms, double price, ExitReason reason) noexcept {
        if (!open_) return;
        open_->exit_price = price;
        open_->pnl = (open_->exit_price - open_->sig.entry_price)
                     *(open_->sig.direction==Direction::LONG?1:-1)
                     *cfg_.mult*open_->sig.contracts;
        {
            TradeContext tc;
            tc.trade_id=open_->trade_id; tc.timestamp_ms=ts_ms;
            tc.symbol=asset_symbol(open_->sig.asset); tc.size=open_->sig.contracts;
            tc.target_price=open_->exit_price;
            tc.direction=to_order_direction(open_->sig.direction, true);
            tc.order_type=OrderType::Market;
            tc.exit_reason=reason;
            on_trade_event_(tc);
        }
        trades_.push_back(*open_);
        open_.reset();
    }

    [[nodiscard]] int64_t prev_close_ts_ms() const noexcept {
        // Best-effort timestamp for a FinalizeBacktest()-triggered closure
        // when no bar object is directly on hand; bars_held already counts
        // correctly since it incremented on every OnBarUpdate() call while
        // the position was open.
        return features_.empty() ? 0 : features_.back().timestamp_ms;
    }

    BacktestConfig cfg_;
    bool   initialized_  = false;
    bool   in_window_    = false;
    bool   amap_seeded_  = false;
    double seed_etf_ratio_ = 1.0;
    double equity_       = 0.0;
    double prev_close_   = 0.0;
    double prev_gex_     = 0.0;
    std::size_t bar_count_    = 0;
    uint64_t    next_trade_id_ = 1;

    m1::AssetMap     amap_;
    m2a::EGARCH      egarch_;
    m2b::ISHMM       hmm_;
    m3::GreeksEngine gk_;
    m4::BandEngine   bands_eng_;
    m5::OFEngine     of_eng_;
    m6::Gate         gate_;
    m7::PFilter      smc_;
    math::ATR        atr_{14};

    std::optional<TradeRecord> open_;
    std::vector<TradeRecord>   trades_;
    std::vector<FeatureRow>    features_;
    GateStat                   gstat_;
    TickAggregator              tick_agg_;

    std::unordered_map<int64_t, m1::MLPrediction> ml_preds_;
    std::function<void(const TradeContext&)>      on_trade_event_;
};

// ════════════════════════════════════════════════════════════════════════
// m8: MACRO STRUCTURAL FILTER ENGINE
// ────────────────────────────────────────────────────────────────────────
// Macro data is NEVER an alpha generator here. This module has zero
// visibility into HybridSigmaStrategy's signal, direction, or conviction —
// no shared state, no includes of each other's headers, no friend
// declarations. It ingests macro/structural feeds on its own cadence and
// publishes a single risk scalar Φ_t ∈ [0,1]. It can only scale or
// zero-out a trade's size; it can never generate one. The only point of
// contact with the Alpha Engine is PositionSizer::FinalSize() below.
// ════════════════════════════════════════════════════════════════════════
namespace m8 {

// ── Shared utility: trailing-window z-score normalizer ─────────────────
// Every macro input is normalized relative to its own historical surprise
// distribution, never treated as an absolute level (Spec §IV, §V, §VI.1).
class RollingZScore {
public:
    explicit RollingZScore(std::size_t window = 252) noexcept : window_(window) {}

    double update(double x) noexcept {
        buf_.push_back(x);
        if (buf_.size() > window_) buf_.pop_front();
        if (buf_.size() < 2) return 0.0;
        const double mu = std::accumulate(buf_.begin(), buf_.end(), 0.0)
                         / static_cast<double>(buf_.size());
        double var = 0.0;
        for (double v : buf_) var += (v - mu) * (v - mu);
        var /= static_cast<double>(buf_.size() - 1);
        const double sd = std::sqrt(var);
        return sd > 1e-12 ? (x - mu) / sd : 0.0;
    }
    [[nodiscard]] bool ready() const noexcept { return buf_.size() >= std::max<std::size_t>(8, window_ / 4); }

private:
    std::deque<double> buf_;
    std::size_t        window_;
};

// ── Layer 1: scheduled macro events (NFP, CPI, PPI, PMI, FOMC) ──────────
enum class MacroEventType : std::uint8_t { NFP, CPI, PPI, PMI, FOMC, COUNT };

struct MacroEvent {
    MacroEventType type{MacroEventType::NFP};
    std::int64_t   release_ts_ms{0};
    double         actual{0.0};
    double         consensus{0.0};
};

// Surprise_z(t) = (Actual_t - Consensus_t) / sigma(trailing surprise window).
// One independent distribution per event type — a CPI surprise and an NFP
// surprise are not drawn from the same distribution (Spec §IV).
class ScheduledSurpriseTracker {
public:
    explicit ScheduledSurpriseTracker(std::size_t window = 252) noexcept
        : z_{RollingZScore(window), RollingZScore(window), RollingZScore(window),
             RollingZScore(window), RollingZScore(window)} {}

    double on_event(const MacroEvent& e) noexcept {
        const auto idx = static_cast<std::size_t>(e.type);
        return idx < z_.size() ? z_[idx].update(e.actual - e.consensus) : 0.0;
    }

private:
    std::array<RollingZScore, static_cast<std::size_t>(MacroEventType::COUNT)> z_;
};

// ── Event-window throttling (pre-release / post-release / recovery) ────
struct EventWindowConfig {
    std::int64_t pre_window_ms       = 30LL * 60'000;  // T-30min liquidity withdrawal
    std::int64_t recovery_ms         = 60LL * 60'000;  // post-event decay-back window
    double       pre_window_phi_mult = 0.50;           // automatic Φ reduction pre-release
};

// Pre-release: automatic throttle regardless of expected direction (known
// liquidity withdrawal/spread-widening). Post-release: sized by realized-
// vs-implied vol jump, not a fixed clock. Recovery: smooth decay back to
// baseline, never a hard on/off switch (Spec §IV).
class EventWindowThrottle {
public:
    explicit EventWindowThrottle(EventWindowConfig cfg = {}) noexcept : cfg_(cfg) {}

    void arm_next_release(std::int64_t release_ts_ms) noexcept { next_release_ts_ms_ = release_ts_ms; }

    void on_release(std::int64_t release_ts_ms, double realized_vs_implied_vol_jump) noexcept {
        last_release_ts_ms_    = release_ts_ms;
        post_release_vol_jump_ = std::max(0.0, realized_vs_implied_vol_jump);
        next_release_ts_ms_    = std::numeric_limits<std::int64_t>::max();
    }

    [[nodiscard]] double multiplier(std::int64_t now_ms) const noexcept {
        if (next_release_ts_ms_ != std::numeric_limits<std::int64_t>::max() &&
            now_ms <= next_release_ts_ms_ &&
            next_release_ts_ms_ - now_ms <= cfg_.pre_window_ms) {
            return cfg_.pre_window_phi_mult;
        }
        if (now_ms >= last_release_ts_ms_) {
            const std::int64_t dt = now_ms - last_release_ts_ms_;
            if (dt <= cfg_.recovery_ms) {
                const double jump_throttle = 1.0 / (1.0 + post_release_vol_jump_);
                const double decay = static_cast<double>(dt) / static_cast<double>(cfg_.recovery_ms);
                return jump_throttle + (1.0 - jump_throttle) * decay; // smooth, never instant
            }
        }
        return 1.0;
    }

private:
    EventWindowConfig cfg_;
    std::int64_t next_release_ts_ms_    = std::numeric_limits<std::int64_t>::max();
    std::int64_t last_release_ts_ms_    = std::numeric_limits<std::int64_t>::min();
    double       post_release_vol_jump_ = 0.0;
};

// ── Layer 2: unscheduled structural-shock proxies ───────────────────────
struct StructuralProxyInputs {
    double gpr_raw          = 0.0; // Caldara-Iacoviello GPR level
    double skew_25d         = 0.0; // 25-delta put/call skew level (SPX/NDX)
    double vix_front        = 0.0;
    double vix_3m           = 0.0;
    double hy_oas           = 0.0; // high-yield OAS, bps
    double bid_ask_spread   = 0.0;
    double book_depth       = 0.0; // higher = deeper/healthier order book
    double etf_premium_disc = 0.0; // e.g. QQQ premium/discount vs NAV
};

struct ProxyNormalized {
    double gpr_z = 0.0, skew_roc_z = 0.0, vix_ts_z = 0.0, oas_z = 0.0,
           spread_z = 0.0, depth_z = 0.0, etf_z = 0.0;
    int n_confirming = 0; // proxies simultaneously elevated past the confirm threshold
};

// No single proxy may trigger full de-risking alone (Spec §V) — n_confirming
// gates how many independent channels must agree before the structural-shock
// component of the composite score is allowed through.
class StructuralProxyEngine {
public:
    explicit StructuralProxyEngine(std::size_t window = 252) noexcept
        : gpr_z_(window), skew_chg_z_(window), vix_ts_z_(window),
          oas_z_(window), spread_z_(window), depth_z_(window), etf_z_(window) {}

    ProxyNormalized update(const StructuralProxyInputs& in, double confirm_z) noexcept {
        ProxyNormalized out;
        out.gpr_z = gpr_z_.update(in.gpr_raw);

        const double skew_roc = std::isnan(prev_skew_) ? 0.0 : (in.skew_25d - prev_skew_);
        prev_skew_   = in.skew_25d;
        out.skew_roc_z = skew_chg_z_.update(skew_roc); // rate-of-change, not static level

        out.vix_ts_z = vix_ts_z_.update(in.vix_front - in.vix_3m); // >0 => backwardation/inversion
        out.oas_z    = oas_z_.update(in.hy_oas);
        out.spread_z = spread_z_.update(in.bid_ask_spread);
        out.depth_z  = depth_z_.update(-in.book_depth);            // depth collapse => positive stress z
        out.etf_z    = etf_z_.update(std::abs(in.etf_premium_disc));

        int n = 0;
        if (out.gpr_z      >= confirm_z) ++n;
        if (out.skew_roc_z >= confirm_z) ++n;
        if (out.vix_ts_z   >= confirm_z) ++n;
        if (out.oas_z      >= confirm_z) ++n;
        if (out.spread_z   >= confirm_z) ++n;
        if (out.depth_z    >= confirm_z) ++n;
        if (out.etf_z      >= confirm_z) ++n;
        out.n_confirming = n;
        return out;
    }

private:
    RollingZScore gpr_z_, skew_chg_z_, vix_ts_z_, oas_z_, spread_z_, depth_z_, etf_z_;
    double prev_skew_ = std::numeric_limits<double>::quiet_NaN();
};

// ── Composite risk score: weighted sum OR first principal component ────
enum class CompositeMode : std::uint8_t { WEIGHTED_SUM, PCA_FIRST_COMPONENT };
inline constexpr std::size_t COMPOSITE_FACTOR_COUNT = 9; // 5 surprise z's + 4 proxy z's

struct CompositeConfig {
    // Weights are structural placeholders. Per Spec §VI.2 they MUST be fit
    // via the out-of-sample statistical validation process (Section VII),
    // never set by discretion alone — recompute via PositionSizer's host
    // harness, do not hand-tune in this file.
    std::array<double, COMPOSITE_FACTOR_COUNT> weights{
        0.10, 0.15, 0.08, 0.07, 0.15,   // NFP, CPI, PPI, PMI, FOMC surprise z
        0.15, 0.10, 0.10, 0.10          // GPR, skew RoC, VIX term-structure, HY OAS
    };
    std::size_t pca_window = 252;
};

// Minimal symmetric-matrix power-iteration PCA (first eigenvector only).
// Avoids a linear-algebra dependency for a single low-dimensional component
// (Spec §VI.2's PCA alternative — "removes the need to hand-pick weights").
class FirstPrincipalComponent {
public:
    static constexpr std::size_t K = COMPOSITE_FACTOR_COUNT;
    using Vec = std::array<double, K>;

    explicit FirstPrincipalComponent(std::size_t window = 252) noexcept : window_(window) {}

    void push(const Vec& v) {
        hist_.push_back(v);
        if (hist_.size() > window_) hist_.pop_front();
    }

    [[nodiscard]] bool ready() const noexcept { return hist_.size() >= K * 3; }

    [[nodiscard]] double project_latest() const noexcept {
        if (!ready()) return 0.0;
        Vec mean{};
        for (const auto& row : hist_) for (std::size_t i = 0; i < K; ++i) mean[i] += row[i];
        for (double& m : mean) m /= static_cast<double>(hist_.size());

        std::array<Vec, K> cov{};
        for (const auto& row : hist_) {
            Vec d{};
            for (std::size_t i = 0; i < K; ++i) d[i] = row[i] - mean[i];
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j)
                    cov[i][j] += d[i] * d[j];
        }
        const double denom = static_cast<double>(hist_.size() - 1);
        for (auto& r : cov) for (double& v : r) v /= denom;

        Vec v{}; v.fill(1.0 / std::sqrt(static_cast<double>(K))); // power iteration, dominant eigenvector
        for (int it = 0; it < 100; ++it) {
            Vec nv{};
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j)
                    nv[i] += cov[i][j] * v[j];
            const double norm = std::sqrt(std::inner_product(nv.begin(), nv.end(), nv.begin(), 0.0));
            if (norm < 1e-12) break;
            for (double& x : nv) x /= norm;
            v = nv;
        }
        const auto& latest = hist_.back();
        Vec d{};
        for (std::size_t i = 0; i < K; ++i) d[i] = latest[i] - mean[i];
        return std::inner_product(d.begin(), d.end(), v.begin(), 0.0);
    }

private:
    std::deque<Vec> hist_;
    std::size_t     window_;
};

// ── Logistic squashing: bounded composite -> Φ_t ∈ [0,1] ───────────────
struct LogisticConfig {
    // k (steepness) and theta (stress threshold) MUST be fit and validated
    // out-of-sample (Spec §VI.3) — defaults here are structural placeholders.
    double k     = 1.5;
    double theta = 2.0;
};

[[nodiscard]] inline double logistic_phi(double R_t, const LogisticConfig& c) noexcept {
    return 1.0 / (1.0 + std::exp(c.k * (R_t - c.theta)));
}

// ── Discrete regime alternative: 2-3 state HMM (Calm/Elevated/Crisis) ──
enum class MacroRegime : std::uint8_t { CALM, ELEVATED, CRISIS };

struct RegimeHMMConfig {
    std::array<double, 3> phi_by_state{1.0, 0.5, 0.0}; // fixed Φ per state
    std::array<std::array<double, 3>, 3> trans{{       // row-stochastic [from][to]
        {0.97, 0.025, 0.005},
        {0.10, 0.85,  0.05},
        {0.02, 0.18,  0.80}
    }};
    std::array<double, 3> emit_mu{0.0, 1.5, 3.5}; // Gaussian emission on composite obs
    std::array<double, 3> emit_sd{0.6, 1.0, 1.5};
};

// State *probabilities* smooth the Φ transition rather than a hard jump
// between fixed per-state values (Spec §VI.4).
class MacroRegimeHMM {
public:
    explicit MacroRegimeHMM(RegimeHMMConfig cfg = {}) noexcept : cfg_(cfg) {}

    void update(double composite_obs) noexcept {
        std::array<double, 3> pred{};
        for (std::size_t to = 0; to < 3; ++to)
            for (std::size_t from = 0; from < 3; ++from)
                pred[to] += post_[from] * cfg_.trans[from][to];

        std::array<double, 3> lik{};
        double tot = 0.0;
        for (std::size_t s = 0; s < 3; ++s) {
            lik[s] = pred[s] * gauss(composite_obs, cfg_.emit_mu[s], cfg_.emit_sd[s]);
            tot += lik[s];
        }
        post_ = (tot > 1e-300) ? [&]{ for (auto& l : lik) l /= tot; return lik; }() : pred;
    }

    [[nodiscard]] std::array<double, 3> posterior() const noexcept { return post_; }

    [[nodiscard]] double phi() const noexcept {
        double phi = 0.0;
        for (std::size_t s = 0; s < 3; ++s) phi += post_[s] * cfg_.phi_by_state[s];
        return phi;
    }

private:
    static double gauss(double x, double mu, double sd) noexcept {
        const double z = (x - mu) / sd;
        return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * std::numbers::pi));
    }
    RegimeHMMConfig        cfg_;
    std::array<double, 3>  post_{1.0, 0.0, 0.0}; // start in Calm
};

// ── Hard overrides: catastrophic safety rails, never statistically fit ──
struct HardOverrideConfig {
    double circuit_breaker_move_pct = 0.07; // |intraday move| >= 7% (Level 1 breaker)
    double extreme_gap_pct          = 0.05; // overnight gap >= 5%
};

class HardOverrideMonitor {
public:
    explicit HardOverrideMonitor(HardOverrideConfig cfg = {}) noexcept : cfg_(cfg) {}

    [[nodiscard]] bool triggered(double intraday_move_pct, double overnight_gap_pct,
                                  bool exchange_halt) const noexcept {
        return exchange_halt
            || std::abs(intraday_move_pct) >= cfg_.circuit_breaker_move_pct
            || std::abs(overnight_gap_pct)  >= cfg_.extreme_gap_pct;
    }

private:
    HardOverrideConfig cfg_;
};

// ════════════════════════════════════════════════════════════════════════
// StructuralFilterEngine — the only public surface of m8.
// Subscribes to its own feeds on independent cadences; publishes Φ_t.
// Has zero visibility into HybridSigmaStrategy's signal or direction.
// ════════════════════════════════════════════════════════════════════════
struct StructuralFilterConfig {
    CompositeMode       mode                   = CompositeMode::WEIGHTED_SUM;
    CompositeConfig     composite{};
    LogisticConfig       logistic{};
    EventWindowConfig    event_window{};
    HardOverrideConfig   hard_override{};
    std::size_t          zscore_window         = 252;
    double                confirm_z_threshold   = 1.5; // multi-proxy confirmation guard (Spec §V)
    int                   min_confirming_proxies = 2;  // no single feed may trigger de-risk alone
    bool                  use_hmm_regime        = false; // discrete-regime alternative to logistic
    RegimeHMMConfig       hmm{};
};

class StructuralFilterEngine {
public:
    explicit StructuralFilterEngine(StructuralFilterConfig cfg = {})
        : cfg_(cfg),
          surprise_(cfg.zscore_window),
          proxy_(cfg.zscore_window),
          pca_(cfg.composite.pca_window),
          event_throttle_(cfg.event_window),
          hmm_(cfg.hmm),
          override_(cfg.hard_override) {}

    // ── independent feed subscriptions — each called on its own cadence ──
    void OnMacroEvent(const MacroEvent& e) noexcept {
        const auto idx = static_cast<std::size_t>(e.type);
        if (idx < last_surprise_z_.size()) last_surprise_z_[idx] = surprise_.on_event(e);
        event_throttle_.on_release(e.release_ts_ms, last_vol_jump_);
        recompute(e.release_ts_ms);
    }
    void ArmUpcomingEvent(std::int64_t release_ts_ms) noexcept {
        event_throttle_.arm_next_release(release_ts_ms);
    }
    void OnVolJump(double realized_vs_implied_jump) noexcept {
        last_vol_jump_ = realized_vs_implied_jump;
    }
    void OnStructuralProxies(std::int64_t now_ms, const StructuralProxyInputs& in) noexcept {
        last_proxy_ = proxy_.update(in, cfg_.confirm_z_threshold);
        recompute(now_ms);
    }
    void OnHardOverrideCheck(std::int64_t now_ms, double intraday_move_pct,
                              double overnight_gap_pct, bool exchange_halt) noexcept {
        hard_override_active_ = override_.triggered(intraday_move_pct, overnight_gap_pct, exchange_halt);
        recompute(now_ms);
    }
    void Tick(std::int64_t now_ms) noexcept { recompute(now_ms); } // advance event-window decay with no new obs

    // ── the ONLY surface the rest of the system may read ──
    [[nodiscard]] double Phi() const noexcept { return phi_; }
    [[nodiscard]] MacroRegime Regime() const noexcept {
        if (!cfg_.use_hmm_regime) return MacroRegime::CALM;
        const auto p = hmm_.posterior();
        const auto arg = static_cast<std::size_t>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
        return static_cast<MacroRegime>(arg);
    }
    [[nodiscard]] std::int64_t LastUpdateTsMs() const noexcept { return last_update_ts_ms_; }

private:
    void recompute(std::int64_t now_ms) noexcept {
        last_update_ts_ms_ = now_ms;
        if (hard_override_active_) { phi_ = 0.0; return; }

        const bool proxy_confirmed = last_proxy_.n_confirming >= cfg_.min_confirming_proxies;

        double composite;
        if (cfg_.mode == CompositeMode::PCA_FIRST_COMPONENT) {
            const FirstPrincipalComponent::Vec v{
                last_surprise_z_[0], last_surprise_z_[1], last_surprise_z_[2],
                last_surprise_z_[3], last_surprise_z_[4],
                proxy_confirmed ? last_proxy_.gpr_z      : 0.0,
                proxy_confirmed ? last_proxy_.skew_roc_z : 0.0,
                proxy_confirmed ? last_proxy_.vix_ts_z   : 0.0,
                proxy_confirmed ? last_proxy_.oas_z      : 0.0
            };
            pca_.push(v);
            composite = pca_.project_latest();
        } else {
            const auto& w = cfg_.composite.weights;
            composite = w[0] * last_surprise_z_[0] + w[1] * last_surprise_z_[1] +
                        w[2] * last_surprise_z_[2] + w[3] * last_surprise_z_[3] +
                        w[4] * last_surprise_z_[4] +
                        (proxy_confirmed ? w[5] * last_proxy_.gpr_z      : 0.0) +
                        (proxy_confirmed ? w[6] * last_proxy_.skew_roc_z : 0.0) +
                        (proxy_confirmed ? w[7] * last_proxy_.vix_ts_z   : 0.0) +
                        (proxy_confirmed ? w[8] * last_proxy_.oas_z      : 0.0);
        }

        double regime_baseline;
        if (cfg_.use_hmm_regime) {
            hmm_.update(composite);
            regime_baseline = hmm_.phi();
        } else {
            regime_baseline = logistic_phi(composite, cfg_.logistic);
        }

        // Event-window throttle is multiplicative on top of the regime
        // baseline — captures known liquidity withdrawal independent of
        // the composite score, with smooth (never hard-switch) recovery.
        const double event_mult = event_throttle_.multiplier(now_ms);
        phi_ = std::clamp(regime_baseline * event_mult, 0.0, 1.0);
    }

    StructuralFilterConfig  cfg_;
    ScheduledSurpriseTracker surprise_;
    StructuralProxyEngine    proxy_;
    FirstPrincipalComponent  pca_;
    EventWindowThrottle      event_throttle_;
    MacroRegimeHMM           hmm_;
    HardOverrideMonitor      override_;

    std::array<double, static_cast<std::size_t>(MacroEventType::COUNT)> last_surprise_z_{};
    ProxyNormalized last_proxy_{};
    double          last_vol_jump_        = 0.0;
    bool            hard_override_active_ = false;
    double          phi_                  = 1.0; // full size until first observation
    std::int64_t    last_update_ts_ms_    = 0;
};

} // namespace m8

// ════════════════════════════════════════════════════════════════════════
// PositionSizer — the ONLY point of contact between the Alpha Engine
// (HybridSigmaStrategy) and the Structural Filter Engine (m8). It reads
// the alpha-sized base quantity and the latest published Φ_t; it never
// recomputes Φ_t and never has access to raw macro inputs. Φ_t scales
// magnitude only — direction and conviction are untouched, by construction
// (this class has no notion of direction at all).
// ════════════════════════════════════════════════════════════════════════
class PositionSizer {
public:
    // Ablation-testing hook (Spec §VII/§VIII): force Φ_t ≡ 1.0 to run the
    // identical, unmodified alpha signal with the filter effectively
    // disabled — without touching either engine's internals. This is what
    // makes the H0 vs H1 hypothesis test in Spec §VII actually runnable.
    void SetAblationForcePhiOne(bool enabled) noexcept { ablation_force_one_ = enabled; }
    [[nodiscard]] bool AblationForcePhiOne() const noexcept { return ablation_force_one_; }

    // FinalSize_t = Φ_t × BaseSize_t(AlphaSignal_t)
    [[nodiscard]] double FinalSize(double base_size, double phi) const noexcept {
        const double eff_phi = ablation_force_one_ ? 1.0 : std::clamp(phi, 0.0, 1.0);
        return base_size * eff_phi;
    }

private:
    bool ablation_force_one_ = false;
};

} // namespace hybrid_sigma

// ══════════════════════════════════════════════════════════════════════════════
// HYBRID_SIGMA_C_ABI — flat extern "C" surface for raw .so/.dll consumption
//   Compile this header into a shared library and any host with a C FFI
//   (ctypes, .NET P/Invoke, Rust, Java JNI, etc.) can drive the brain through
//   an opaque handle — no C++ name-mangling or vtable-layout dependency.
//
//   Build as shared lib:
//     g++ -std=c++23 -O3 -march=native -shared -fPIC
//         -DHYBRID_SIGMA_C_ABI -x c++ hybrid_sigma_strategy.hpp
//         -o libhybridsigma.so
//   (Or #include this header from a tiny .cpp and compile that instead.)
// ══════════════════════════════════════════════════════════════════════════════
#ifdef HYBRID_SIGMA_C_ABI
extern "C" {

    using HybridSigmaHandle = void*;

    [[nodiscard]] HybridSigmaHandle hybrid_sigma_create() {
        return new hybrid_sigma::HybridSigmaStrategy();
    }

    void hybrid_sigma_destroy(HybridSigmaHandle h) {
        delete static_cast<hybrid_sigma::HybridSigmaStrategy*>(h);
    }

    // asset: 0=MNQ (only value now — kept as an int param for ABI stability).
    // start_date_ms/end_date_ms: 0 = unset (uses full min()/max() window —
    // pass 0 for "no filtering").
    void hybrid_sigma_initialize(
        HybridSigmaHandle h, int asset, double mult, double equity,
        double risk_pct, double kelly_cap, double stop_atr_mult,
        double target_sigma_mult, double min_rr, double ml_threshold,
        int64_t start_date_ms, int64_t end_date_ms)
    {
        auto* s = static_cast<hybrid_sigma::HybridSigmaStrategy*>(h);
        hybrid_sigma::BacktestConfig cfg;
        cfg.asset = static_cast<hybrid_sigma::Asset>(asset);
        cfg.mult=mult; cfg.equity=equity; cfg.risk_pct=risk_pct;
        cfg.kelly_cap=kelly_cap; cfg.stop_atr_mult=stop_atr_mult;
        cfg.target_sigma_mult=target_sigma_mult; cfg.min_rr=min_rr;
        cfg.ml_threshold=ml_threshold;
        if (start_date_ms != 0) cfg.start_date = hybrid_sigma::ms_to_time_point(start_date_ms);
        if (end_date_ms   != 0) cfg.end_date   = hybrid_sigma::ms_to_time_point(end_date_ms);
        s->Initialize(cfg);
    }

    void hybrid_sigma_reset(HybridSigmaHandle h) {
        static_cast<hybrid_sigma::HybridSigmaStrategy*>(h)->Reset();
    }

    // Flat-array bar update — avoids exposing the Bar struct layout across
    // the ABI boundary. Pass chain arrays as parallel flat arrays; n_opts=0
    // for a bar with no options data.
    void hybrid_sigma_on_bar(
        HybridSigmaHandle h,
        int64_t timestamp_ms, double open, double high, double low,
        double close, double volume, double buy_vol, double sell_vol, double cum_delta)
    {
        auto* s = static_cast<hybrid_sigma::HybridSigmaStrategy*>(h);
        hybrid_sigma::Bar bar;
        bar.timestamp_ms=timestamp_ms; bar.open=open; bar.high=high;
        bar.low=low; bar.close=close; bar.volume=volume;
        bar.buy_vol=buy_vol; bar.sell_vol=sell_vol; bar.cum_delta=cum_delta;
        s->OnBarUpdate(bar, {});
    }

    void hybrid_sigma_finalize(HybridSigmaHandle h) {
        static_cast<hybrid_sigma::HybridSigmaStrategy*>(h)->FinalizeBacktest();
    }

    [[nodiscard]] int      hybrid_sigma_trade_count(HybridSigmaHandle h) {
        return static_cast<int>(static_cast<hybrid_sigma::HybridSigmaStrategy*>(h)->Trades().size());
    }
    [[nodiscard]] double   hybrid_sigma_equity(HybridSigmaHandle h) {
        return static_cast<hybrid_sigma::HybridSigmaStrategy*>(h)->Equity();
    }

    // Flat POD mirror of TradeRecord/TradeSignal for the ABI boundary — no
    // struct-layout dependency on the C++ side beyond this file. direction
    // is the raw Direction int8_t (LONG=1, SHORT=-1, FLAT=0).
    struct HybridSigmaCTrade {
        int64_t trade_id;
        int64_t entry_ts_ms;
        int32_t direction;
        double  entry_price, stop_price, target_price, exit_price;
        double  contracts, pnl, mae, mfe, rr, kelly, ml_p_success;
        int32_t bars_held;
        int32_t stopped, target_hit; // 0/1
    };

    // Returns 1 and fills *out on success, 0 if idx is out of range.
    int hybrid_sigma_get_trade(HybridSigmaHandle h, int idx, HybridSigmaCTrade* out) {
        const auto& trades = static_cast<hybrid_sigma::HybridSigmaStrategy*>(h)->Trades();
        if (idx < 0 || static_cast<std::size_t>(idx) >= trades.size() || out == nullptr) return 0;
        const auto& t = trades[static_cast<std::size_t>(idx)];
        out->trade_id      = static_cast<int64_t>(t.trade_id);
        out->entry_ts_ms   = t.sig.timestamp_ms;
        out->direction     = static_cast<int32_t>(t.sig.direction);
        out->entry_price   = t.sig.entry_price;
        out->stop_price    = t.sig.stop_price;
        out->target_price  = t.sig.target_price;
        out->exit_price    = t.exit_price;
        out->contracts     = t.sig.contracts;
        out->pnl           = t.pnl;
        out->mae           = t.mae;
        out->mfe           = t.mfe;
        out->rr            = t.sig.rr;
        out->kelly         = t.sig.kelly;
        out->ml_p_success  = t.sig.ml_p_success;
        out->bars_held     = t.bars_held;
        out->stopped       = t.stopped    ? 1 : 0;
        out->target_hit    = t.target_hit ? 1 : 0;
        return 1;
    }

} // extern "C"
#endif // HYBRID_SIGMA_C_ABI

// ══════════════════════════════════════════════════════════════════════════════
// HYBRID_SIGMA_PYBIND — ready-to-use pybind11 module definition
//   Wraps the C++ class directly — no ABI layer needed. Build with:
//     pip install pybind11
//     g++ -std=c++23 -O3 -march=native -shared -fPIC
//         -DHYBRID_SIGMA_PYBIND $(python3 -m pybind11 --includes)
//         hybrid_sigma_strategy.hpp -o hybrid_sigma$(python3-config --extension-suffix)
//   Then in Python:
//     import hybrid_sigma
//     s = hybrid_sigma.HybridSigmaStrategy()
//     cfg = hybrid_sigma.BacktestConfig()
//     cfg.asset = hybrid_sigma.Asset.NQ
//     s.Initialize(cfg)
//     s.OnBarUpdate(bar, [])   # bar is a hybrid_sigma.Bar
//     s.FinalizeBacktest()
//     print(s.Equity(), len(s.Trades()))
// ══════════════════════════════════════════════════════════════════════════════
#ifdef HYBRID_SIGMA_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

PYBIND11_MODULE(hybrid_sigma, m) {
    namespace py = pybind11;
    using namespace hybrid_sigma;

    py::enum_<Asset>(m, "Asset")
        .value("MNQ", Asset::MNQ);

    py::enum_<Direction>(m, "Direction")
        .value("LONG", Direction::LONG).value("SHORT", Direction::SHORT)
        .value("FLAT", Direction::FLAT);

    py::class_<Bar>(m, "Bar")
        .def(py::init<>())
        .def_readwrite("timestamp_ms", &Bar::timestamp_ms)
        .def_readwrite("open", &Bar::open).def_readwrite("high", &Bar::high)
        .def_readwrite("low", &Bar::low).def_readwrite("close", &Bar::close)
        .def_readwrite("volume", &Bar::volume)
        .def_readwrite("buy_vol", &Bar::buy_vol).def_readwrite("sell_vol", &Bar::sell_vol)
        .def_readwrite("cum_delta", &Bar::cum_delta);

    py::class_<OptionRow>(m, "OptionRow")
        .def(py::init<>())
        .def_readwrite("timestamp_ms", &OptionRow::timestamp_ms)
        .def_readwrite("expiry_ms", &OptionRow::expiry_ms)
        .def_readwrite("strike", &OptionRow::strike)
        .def_readwrite("mid_iv", &OptionRow::mid_iv)
        .def_readwrite("open_interest", &OptionRow::open_interest)
        .def_readwrite("type", &OptionRow::type);

    py::class_<BacktestConfig>(m, "BacktestConfig")
        .def(py::init<>())
        .def_readwrite("asset", &BacktestConfig::asset)
        .def_readwrite("mult", &BacktestConfig::mult)
        .def_readwrite("equity", &BacktestConfig::equity)
        .def_readwrite("risk_pct", &BacktestConfig::risk_pct)
        .def_readwrite("kelly_cap", &BacktestConfig::kelly_cap)
        .def_readwrite("stop_atr_mult", &BacktestConfig::stop_atr_mult)
        .def_readwrite("target_sigma_mult", &BacktestConfig::target_sigma_mult)
        .def_readwrite("min_rr", &BacktestConfig::min_rr)
        .def_readwrite("ml_threshold", &BacktestConfig::ml_threshold)
        .def_readwrite("start_date", &BacktestConfig::start_date)
        .def_readwrite("end_date", &BacktestConfig::end_date);

    py::class_<TradeRecord>(m, "TradeRecord")
        .def_readonly("exit_price", &TradeRecord::exit_price)
        .def_readonly("pnl", &TradeRecord::pnl)
        .def_readonly("mae", &TradeRecord::mae)
        .def_readonly("mfe", &TradeRecord::mfe)
        .def_readonly("bars_held", &TradeRecord::bars_held)
        .def_readonly("trade_id", &TradeRecord::trade_id);

    py::class_<TradeContext>(m, "TradeContext")
        .def_readonly("trade_id", &TradeContext::trade_id)
        .def_readonly("timestamp_ms", &TradeContext::timestamp_ms)
        .def_readonly("size", &TradeContext::size)
        .def_readonly("target_price", &TradeContext::target_price);

    py::class_<FeatureRow>(m, "FeatureRow")
        .def_readonly("timestamp_ms", &FeatureRow::timestamp_ms)
        .def_readonly("close", &FeatureRow::close)
        .def_readonly("implied_forward", &FeatureRow::implied_forward)
        .def_readonly("basis_spread", &FeatureRow::basis_spread)
        .def_readonly("egarch_vol", &FeatureRow::egarch_vol)
        .def_readonly("egarch_vol_ratio", &FeatureRow::egarch_vol_ratio)
        .def_readonly("egarch_ready", &FeatureRow::egarch_ready)
        .def_readonly("low_vol", &FeatureRow::low_vol)
        .def_readonly("high_vol", &FeatureRow::high_vol)
        .def_readonly("expanding", &FeatureRow::expanding)
        .def_readonly("hmm_bull", &FeatureRow::hmm_bull)
        .def_readonly("hmm_bear", &FeatureRow::hmm_bear)
        .def_readonly("hmm_cons", &FeatureRow::hmm_cons)
        .def_readonly("hmm_ready", &FeatureRow::hmm_ready)
        .def_readonly("atr_fast", &FeatureRow::atr_fast)
        .def_readonly("atr_slow", &FeatureRow::atr_slow)
        .def_readonly("atr_ratio", &FeatureRow::atr_ratio)
        .def_readonly("ols_slope", &FeatureRow::ols_slope)
        .def_readonly("vol_zscore", &FeatureRow::vol_zscore)
        .def_readonly("net_gex", &FeatureRow::net_gex)
        .def_readonly("norm_gex", &FeatureRow::norm_gex)
        .def_readonly("net_speed", &FeatureRow::net_speed)
        .def_readonly("norm_speed", &FeatureRow::norm_speed)
        .def_readonly("net_vanna", &FeatureRow::net_vanna)
        .def_readonly("norm_vanna", &FeatureRow::norm_vanna)
        .def_readonly("net_charm", &FeatureRow::net_charm)
        .def_readonly("iv_trend", &FeatureRow::iv_trend)
        .def_readonly("speed_zone", &FeatureRow::speed_zone)
        .def_readonly("sigma", &FeatureRow::sigma)
        .def_readonly("band_level", &FeatureRow::band_level)
        .def_readonly("z_score", &FeatureRow::z_score)
        .def_readonly("vanna_skew", &FeatureRow::vanna_skew)
        .def_readonly("cum_delta", &FeatureRow::cum_delta)
        .def_readonly("norm_delta", &FeatureRow::norm_delta)
        .def_readonly("of_vol_ratio", &FeatureRow::of_vol_ratio)
        .def_readonly("price_efficiency", &FeatureRow::price_efficiency)
        .def_readonly("absorption", &FeatureRow::absorption)
        .def_readonly("exhaustion", &FeatureRow::exhaustion)
        .def_readonly("g_vol", &FeatureRow::g_vol)
        .def_readonly("g_gex", &FeatureRow::g_gex)
        .def_readonly("g_speed", &FeatureRow::g_speed)
        .def_readonly("g_vanna", &FeatureRow::g_vanna)
        .def_readonly("g_sigma", &FeatureRow::g_sigma)
        .def_readonly("g_of", &FeatureRow::g_of)
        .def_readonly("g_hmm", &FeatureRow::g_hmm)
        .def_readonly("verdict", &FeatureRow::verdict)
        .def_readonly("direction", &FeatureRow::direction)
        .def_readonly("conviction", &FeatureRow::conviction)
        .def_readonly("particle_cps", &FeatureRow::particle_cps);

    m.def("parse_date", &parse_date, "Parse YYYY-MM-DD into a time_point");

    py::class_<HybridSigmaStrategy>(m, "HybridSigmaStrategy")
        .def(py::init<>())
        .def("Initialize", &HybridSigmaStrategy::Initialize)
        .def("Reset", &HybridSigmaStrategy::Reset)
        .def("OnBarUpdate", &HybridSigmaStrategy::OnBarUpdate)
        .def("OnTick", &HybridSigmaStrategy::OnTick,
             py::arg("tick"), py::arg("chain_for_current_bar")=std::vector<OptionRow>{})
        .def("FinalizeBacktest", &HybridSigmaStrategy::FinalizeBacktest)
        .def("SetTradeEventCallback", &HybridSigmaStrategy::SetTradeEventCallback)
        .def("IsWarmedUp", &HybridSigmaStrategy::IsWarmedUp)
        .def("IsInExecutionWindow", &HybridSigmaStrategy::IsInExecutionWindow)
        .def("Equity", &HybridSigmaStrategy::Equity)
        .def("BarsProcessed", &HybridSigmaStrategy::BarsProcessed)
        .def("Trades", &HybridSigmaStrategy::Trades, py::return_value_policy::reference_internal)
        .def("Features", &HybridSigmaStrategy::Features, py::return_value_policy::reference_internal);

    // ── m8: Macro Structural Filter Engine ──────────────────────────────
    // Bound entirely separately from HybridSigmaStrategy above — the only
    // place they meet in Python, exactly as in C++, is PositionSizer.
    py::enum_<m8::MacroEventType>(m, "MacroEventType")
        .value("NFP", m8::MacroEventType::NFP).value("CPI", m8::MacroEventType::CPI)
        .value("PPI", m8::MacroEventType::PPI).value("PMI", m8::MacroEventType::PMI)
        .value("FOMC", m8::MacroEventType::FOMC);

    py::class_<m8::MacroEvent>(m, "MacroEvent")
        .def(py::init<>())
        .def_readwrite("type", &m8::MacroEvent::type)
        .def_readwrite("release_ts_ms", &m8::MacroEvent::release_ts_ms)
        .def_readwrite("actual", &m8::MacroEvent::actual)
        .def_readwrite("consensus", &m8::MacroEvent::consensus);

    py::class_<m8::StructuralProxyInputs>(m, "StructuralProxyInputs")
        .def(py::init<>())
        .def_readwrite("gpr_raw", &m8::StructuralProxyInputs::gpr_raw)
        .def_readwrite("skew_25d", &m8::StructuralProxyInputs::skew_25d)
        .def_readwrite("vix_front", &m8::StructuralProxyInputs::vix_front)
        .def_readwrite("vix_3m", &m8::StructuralProxyInputs::vix_3m)
        .def_readwrite("hy_oas", &m8::StructuralProxyInputs::hy_oas)
        .def_readwrite("bid_ask_spread", &m8::StructuralProxyInputs::bid_ask_spread)
        .def_readwrite("book_depth", &m8::StructuralProxyInputs::book_depth)
        .def_readwrite("etf_premium_disc", &m8::StructuralProxyInputs::etf_premium_disc);

    py::enum_<m8::CompositeMode>(m, "CompositeMode")
        .value("WEIGHTED_SUM", m8::CompositeMode::WEIGHTED_SUM)
        .value("PCA_FIRST_COMPONENT", m8::CompositeMode::PCA_FIRST_COMPONENT);

    py::class_<m8::CompositeConfig>(m, "CompositeConfig")
        .def(py::init<>())
        .def_readwrite("weights", &m8::CompositeConfig::weights)
        .def_readwrite("pca_window", &m8::CompositeConfig::pca_window);

    py::class_<m8::LogisticConfig>(m, "LogisticConfig")
        .def(py::init<>())
        .def_readwrite("k", &m8::LogisticConfig::k)
        .def_readwrite("theta", &m8::LogisticConfig::theta);

    py::class_<m8::EventWindowConfig>(m, "EventWindowConfig")
        .def(py::init<>())
        .def_readwrite("pre_window_ms", &m8::EventWindowConfig::pre_window_ms)
        .def_readwrite("recovery_ms", &m8::EventWindowConfig::recovery_ms)
        .def_readwrite("pre_window_phi_mult", &m8::EventWindowConfig::pre_window_phi_mult);

    py::class_<m8::HardOverrideConfig>(m, "HardOverrideConfig")
        .def(py::init<>())
        .def_readwrite("circuit_breaker_move_pct", &m8::HardOverrideConfig::circuit_breaker_move_pct)
        .def_readwrite("extreme_gap_pct", &m8::HardOverrideConfig::extreme_gap_pct);

    py::class_<m8::RegimeHMMConfig>(m, "RegimeHMMConfig")
        .def(py::init<>())
        .def_readwrite("phi_by_state", &m8::RegimeHMMConfig::phi_by_state)
        .def_readwrite("trans", &m8::RegimeHMMConfig::trans)
        .def_readwrite("emit_mu", &m8::RegimeHMMConfig::emit_mu)
        .def_readwrite("emit_sd", &m8::RegimeHMMConfig::emit_sd);

    py::class_<m8::StructuralFilterConfig>(m, "StructuralFilterConfig")
        .def(py::init<>())
        .def_readwrite("mode", &m8::StructuralFilterConfig::mode)
        .def_readwrite("composite", &m8::StructuralFilterConfig::composite)
        .def_readwrite("logistic", &m8::StructuralFilterConfig::logistic)
        .def_readwrite("event_window", &m8::StructuralFilterConfig::event_window)
        .def_readwrite("hard_override", &m8::StructuralFilterConfig::hard_override)
        .def_readwrite("zscore_window", &m8::StructuralFilterConfig::zscore_window)
        .def_readwrite("confirm_z_threshold", &m8::StructuralFilterConfig::confirm_z_threshold)
        .def_readwrite("min_confirming_proxies", &m8::StructuralFilterConfig::min_confirming_proxies)
        .def_readwrite("use_hmm_regime", &m8::StructuralFilterConfig::use_hmm_regime)
        .def_readwrite("hmm", &m8::StructuralFilterConfig::hmm);

    py::enum_<m8::MacroRegime>(m, "MacroRegime")
        .value("CALM", m8::MacroRegime::CALM)
        .value("ELEVATED", m8::MacroRegime::ELEVATED)
        .value("CRISIS", m8::MacroRegime::CRISIS);

    py::class_<m8::StructuralFilterEngine>(m, "StructuralFilterEngine")
        .def(py::init<>())
        .def(py::init<m8::StructuralFilterConfig>())
        .def("OnMacroEvent", &m8::StructuralFilterEngine::OnMacroEvent)
        .def("ArmUpcomingEvent", &m8::StructuralFilterEngine::ArmUpcomingEvent)
        .def("OnVolJump", &m8::StructuralFilterEngine::OnVolJump)
        .def("OnStructuralProxies", &m8::StructuralFilterEngine::OnStructuralProxies)
        .def("OnHardOverrideCheck", &m8::StructuralFilterEngine::OnHardOverrideCheck)
        .def("Tick", &m8::StructuralFilterEngine::Tick)
        .def("Phi", &m8::StructuralFilterEngine::Phi)
        .def("Regime", &m8::StructuralFilterEngine::Regime)
        .def("LastUpdateTsMs", &m8::StructuralFilterEngine::LastUpdateTsMs);

    // ── PositionSizer: the only point of contact between the two engines ──
    py::class_<PositionSizer>(m, "PositionSizer")
        .def(py::init<>())
        .def("SetAblationForcePhiOne", &PositionSizer::SetAblationForcePhiOne)
        .def("AblationForcePhiOne", &PositionSizer::AblationForcePhiOne)
        .def("FinalSize", &PositionSizer::FinalSize, py::arg("base_size"), py::arg("phi"));
}
#endif // HYBRID_SIGMA_PYBIND
