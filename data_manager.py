# data_manager.py — P2-ETF-FACTOR-TILT
import io
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from config import (
    ALL_TICKERS, FI_TICKERS, EQUITY_TICKERS,
    HF_MASTER_REPO, HF_MASTER_FILE,
    FF_DATASET, MOM_DATASET, RISK_FREE_COL,
    FACTOR_COLS, USE_REGIME_FILTER,
    HF_HHMM_REPO, HF_HHMM_FILE,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Master data (OHLCV + log returns) from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def load_master_data() -> pd.DataFrame:
    """
    Load master_data.parquet from HF.
    Returns a DataFrame indexed by date with MultiIndex columns (field, ticker)
    OR wide format — we normalise to wide log_return columns here.
    """
    log.info(f"Loading master data from {HF_MASTER_REPO}")
    try:
        path = hf_hub_download(
            repo_id=HF_MASTER_REPO,
            filename=HF_MASTER_FILE,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
    except Exception as e:
        log.error(f"Failed to load master data: {e}")
        raise

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    log.info(f"Master data loaded: {df.shape}, {df.index.min()} → {df.index.max()}")
    return df


def extract_log_returns(master: pd.DataFrame) -> pd.DataFrame:
    """
    Extract log return columns for all tickers.
    Handles both flat column names (e.g. 'log_return_SPY') and
    MultiIndex columns (('log_return', 'SPY')).
    Returns: DataFrame[date x ticker] of daily log returns.
    """
    if isinstance(master.columns, pd.MultiIndex):
        returns = master["log_return"][ALL_TICKERS].copy()
    else:
        cols = {t: f"log_return_{t}" for t in ALL_TICKERS if f"log_return_{t}" in master.columns}
        if not cols:
            # Fallback: compute from close prices
            log.warning("log_return columns not found — computing from close prices")
            close_cols = {t: f"close_{t}" for t in ALL_TICKERS if f"close_{t}" in master.columns}
            closes = master[[v for v in close_cols.values()]].rename(
                columns={v: k for k, v in close_cols.items()}
            )
            returns = np.log(closes / closes.shift(1))
        else:
            returns = master[[v for v in cols.values()]].rename(
                columns={v: k for k, v in cols.items()}
            )

    returns = returns.dropna(how="all")
    log.info(f"Log returns extracted: {returns.shape}")
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fama-French factors from Kenneth French Data Library
# ─────────────────────────────────────────────────────────────────────────────

def load_ff_factors(start: str = "2007-01-01", end: str = None) -> pd.DataFrame:
    """
    Download Fama-French 5 factors + Momentum from Kenneth French's library.
    Returns daily factor returns as decimals (divide by 100).
    Columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    log.info("Downloading Fama-French 5-factor daily data from Kenneth French library...")
    try:
        ff5 = web.DataReader(FF_DATASET, "famafrench", start=start, end=end)[0]
        ff5 = ff5 / 100.0  # convert from percent to decimal
        ff5.index = pd.to_datetime(ff5.index)
        log.info(f"FF5 factors: {ff5.shape}")
    except Exception as e:
        log.error(f"Failed to download FF5 factors: {e}")
        raise

    log.info("Downloading Momentum factor daily data...")
    try:
        mom = web.DataReader(MOM_DATASET, "famafrench", start=start, end=end)[0]
        mom = mom / 100.0
        mom.index = pd.to_datetime(mom.index)
        mom.columns = ["Mom"]
        log.info(f"Momentum factor: {mom.shape}")
    except Exception as e:
        log.warning(f"Momentum factor download failed: {e} — filling with zeros")
        mom = pd.DataFrame({"Mom": 0.0}, index=ff5.index)

    factors = ff5.join(mom, how="left")
    factors["Mom"] = factors["Mom"].fillna(0.0)

    # Ensure all FACTOR_COLS are present
    for col in FACTOR_COLS:
        if col not in factors.columns:
            log.warning(f"Factor column '{col}' missing — filling with zeros")
            factors[col] = 0.0

    factors = factors.sort_index()
    log.info(f"FF factors ready: {factors.shape}, {factors.index.min()} → {factors.index.max()}")
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# 3. Date alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_returns_and_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner-join returns and factors on the date index.
    Returns aligned (returns, factors) with no NaN rows.
    """
    common = returns.index.intersection(factors.index)
    returns_aligned = returns.loc[common]
    factors_aligned = factors.loc[common]

    # Drop dates where ALL ETFs are NaN (market holidays not in FF data)
    valid = returns_aligned.notna().any(axis=1) & factors_aligned.notna().all(axis=1)
    returns_aligned = returns_aligned.loc[valid]
    factors_aligned = factors_aligned.loc[valid]

    log.info(f"Aligned data: {len(common)} common dates → {valid.sum()} valid rows")
    return returns_aligned, factors_aligned


# ─────────────────────────────────────────────────────────────────────────────
# 4. Excess returns (subtract risk-free rate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_excess_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Subtract daily risk-free rate (RF) from each ETF's log return
    to get excess returns for regression.
    """
    rf = factors[RISK_FREE_COL] if RISK_FREE_COL in factors.columns else pd.Series(0.0, index=factors.index)
    excess = returns.sub(rf, axis=0)
    log.info("Excess returns computed (log_return - RF)")
    return excess


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optional: HHMM regime data for factor weight overrides
# ─────────────────────────────────────────────────────────────────────────────

def load_hhmm_regime() -> pd.DataFrame | None:
    """
    Load HHMM regime results from HF if available.
    Returns DataFrame with columns [date, macro_regime] where
    macro_regime ∈ {stress, neutral, calm}.
    Returns None if not available (engine runs with default factor weights).
    """
    if not USE_REGIME_FILTER:
        return None
    try:
        path = hf_hub_download(
            repo_id=HF_HHMM_REPO,
            filename=HF_HHMM_FILE,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        # Normalise regime column name
        for col in ["macro_regime", "regime", "hhmm_regime"]:
            if col in df.columns:
                df = df.rename(columns={col: "macro_regime"})
                break
        log.info(f"HHMM regime data loaded: {df.shape}")
        return df[["macro_regime"]]
    except Exception as e:
        log.warning(f"HHMM regime not available: {e} — using default factor weights")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Universe helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_universe_tickers(universe: str) -> list[str]:
    if universe == "fi":
        return FI_TICKERS
    elif universe == "equity":
        return EQUITY_TICKERS
    else:
        return ALL_TICKERS
