# config.py — P2-ETF-FACTOR-TILT
# Cross-Sectional Factor Momentum (FACTOR-TILT) Engine
# All constants in one place — edit here only.

# ── Universe ──────────────────────────────────────────────────────────────────
FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
                   "XLY", "XLP", "XLU", "GDX", "XME", "IWM"]
ALL_TICKERS = FI_TICKERS + EQUITY_TICKERS

BENCHMARKS = {
    "fi":       "AGG",
    "equity":   "SPY",
    "combined": "SPY",
}

# ── HuggingFace data source ───────────────────────────────────────────────────
HF_MASTER_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_MASTER_FILE   = "master_data.parquet"
HF_RESULTS_REPO  = "P2SAMAPA/p2-etf-factor-tilt-results"
HF_RESULTS_FILE  = "factor_tilt_results.parquet"

# ── Fama-French factors ───────────────────────────────────────────────────────
# Downloaded via pandas_datareader from Kenneth French's Data Library.
# Using the 5-factor model + momentum (UMD) = 6 factors total.
FF_DATASET      = "F-F_Research_Data_5_Factors_2x3_daily"   # Mkt-RF, SMB, HML, RMW, CMA
MOM_DATASET     = "F-F_Momentum_Factor_daily"                # UMD (Up Minus Down)
RISK_FREE_COL   = "RF"

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
# Human-readable display names
FACTOR_NAMES = {
    "Mkt-RF": "Market",
    "SMB":    "Size",
    "HML":    "Value",
    "RMW":    "Profitability",
    "CMA":    "Investment",
    "Mom":    "Momentum",
}

# ── Rolling OLS windows (trading days) ───────────────────────────────────────
# Short / medium / long — chosen to be CPU-light and statistically meaningful
ROLLING_WINDOWS = [21, 63, 126]      # ~1M, ~3M, ~6M
MIN_OBS         = 15                 # minimum obs to fit a window (avoid noise)

# ── Tilt momentum lookbacks ───────────────────────────────────────────────────
# How far back to compare current beta vs. past beta
TILT_LOOKBACKS = [21, 42, 63]        # delta measured over 1M, 2M, 3M

# ── Scoring weights ───────────────────────────────────────────────────────────
# Weight each factor's tilt contribution to the composite score.
# Momentum (UMD) and Market beta tilts weighted higher — strongest
# predictive tilts for ETF rotation in the literature.
FACTOR_WEIGHTS = {
    "Mkt-RF": 0.25,
    "SMB":    0.10,
    "HML":    0.15,
    "RMW":    0.15,
    "CMA":    0.10,
    "Mom":    0.25,
}

# Window weights (longer window = more stable signal, weighted higher)
WINDOW_WEIGHTS = {21: 0.20, 63: 0.45, 126: 0.35}

# Tilt lookback weights (medium lookback weighted highest)
TILT_LOOKBACK_WEIGHTS = {21: 0.25, 42: 0.45, 63: 0.30}

# ── Regime integration (optional — requires HHMM results on HF) ───────────────
USE_REGIME_FILTER    = True
HF_HHMM_REPO         = "P2SAMAPA/P2-ETF-HHMM-REGIME"
HF_HHMM_FILE         = "hhmm_results.parquet"
# In stress regime, downweight momentum tilt; upweight value/profitability
REGIME_FACTOR_OVERRIDES = {
    "stress":  {"Mom": 0.10, "Mkt-RF": 0.15, "HML": 0.25, "RMW": 0.25, "CMA": 0.15, "SMB": 0.10},
    "neutral": FACTOR_WEIGHTS,  # use defaults
    "calm":    {"Mom": 0.30, "Mkt-RF": 0.30, "HML": 0.10, "RMW": 0.10, "CMA": 0.10, "SMB": 0.10},
}

# ── Output columns ────────────────────────────────────────────────────────────
OUTPUT_COLS = [
    "run_date",
    "universe",
    "ticker",
    "csfm_score",          # final composite cross-sectional z-score
    "csfm_rank",           # rank within universe (1 = strongest tilt momentum)
    "dominant_factor",     # factor with largest absolute tilt contribution
    "dominant_direction",  # "strengthening" or "weakening"
    # Per-factor delta betas (medium window, medium lookback)
    "delta_Mkt-RF",
    "delta_SMB",
    "delta_HML",
    "delta_RMW",
    "delta_CMA",
    "delta_Mom",
    # Raw betas (126d window, most recent)
    "beta_Mkt-RF",
    "beta_SMB",
    "beta_HML",
    "beta_RMW",
    "beta_CMA",
    "beta_Mom",
    "alpha",               # annualised Jensen's alpha (126d window)
    "r_squared",           # goodness of fit (126d window)
    "regime",              # stress / neutral / calm (from HHMM if available)
]

# ── CPU budget guard ──────────────────────────────────────────────────────────
# GitHub free tier: 2 vCPU, 7GB RAM, 6h limit.
# Rolling OLS on 20 ETFs x 6 factors x 3 windows x ~4500 days ≈ 8-12 min.
# Total expected runtime: ~20 min including data load + push.
MAX_RUNTIME_MINUTES = 300            # hard ceiling — fail fast if exceeded
