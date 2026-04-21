# data_manager.py — P2-ETF-FACTOR-TILT
# Fixed: replaced pandas_datareader (broken in Python 3.11) with direct
# HTTP fetch of Fama-French CSV files from Kenneth French's Data Library.
import io
import logging
import warnings
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from config import (
    ALL_TICKERS, FI_TICKERS, EQUITY_TICKERS,
    HF_MASTER_REPO, HF_MASTER_FILE,
    FACTOR_COLS, USE_REGIME_FILTER,
    HF_HHMM_REPO, HF_HHMM_FILE,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Kenneth French Data Library — direct ZIP URLs (no API key, no third-party lib)
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
REQUEST_TIMEOUT = 60


# ─────────────────────────────────────────────────────────────────────────────
# 1. Master data from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def load_master_data() -> pd.DataFrame:
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
    log.info(f"Master data loaded: {df.shape}, {df.index.min().date()} to {df.index.max().date()}")
    return df


def extract_log_returns(master: pd.DataFrame) -> pd.DataFrame:
    """
    master_data.parquet has bare ticker price columns (SPY, TLT, GLD, etc.).
    Compute log returns directly from those prices.
    """
    log.info(f"Master columns ({len(master.columns)}): {list(master.columns)}")
    available = [t for t in ALL_TICKERS if t in master.columns]
    if not available:
        raise ValueError(
            f"No expected tickers found in master_data. Columns: {list(master.columns)}"
        )
    prices  = master[available].copy().apply(pd.to_numeric, errors="coerce")
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    log.info(f"Log returns computed: {returns.shape}, tickers: {available}")
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fama-French factors — direct HTTP fetch (no pandas_datareader)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_ff_zip(url: str) -> pd.DataFrame:
    """
    Download a Kenneth French ZIP, extract the CSV, parse the daily factor table.
    Detects data start by finding the first line whose first token is an 8-digit date.
    """
    log.info(f"Fetching {url}")
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

    lines = raw_text.splitlines()

    # Find first data line — 8-digit YYYYMMDD date
    data_start = None
    for i, line in enumerate(lines):
        token = line.strip().split(",")[0].strip()
        if token.isdigit() and len(token) == 8:
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Could not find data start in {url}")

    # Header is the non-blank line immediately before the data
    header_idx = data_start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1

    # Find last data line (stop at blank or non-date line after data starts)
    data_end = data_start
    for i in range(data_start, len(lines)):
        token = lines[i].strip().split(",")[0].strip()
        if token.isdigit() and len(token) == 8:
            data_end = i
        elif not token.isdigit() and i > data_start:
            break

    header_line = lines[header_idx].strip()
    data_block  = "\n".join([header_line] + lines[data_start: data_end + 1])

    df = pd.read_csv(io.StringIO(data_block), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df = df.sort_index()
    log.info(f"  Parsed {len(df)} rows, cols: {list(df.columns)}")
    return df


def load_ff_factors(start: str = "2007-01-01", end: str = None) -> pd.DataFrame:
    """
    Load FF5 + Momentum daily factors directly from Kenneth French.
    Returns columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF  (decimal form).
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    try:
        ff5 = _fetch_ff_zip(FF5_URL)
    except Exception as e:
        log.error(f"FF5 download failed: {e}")
        raise

    try:
        mom = _fetch_ff_zip(MOM_URL)
        mom.columns = ["Mom"]
    except Exception as e:
        log.warning(f"Momentum download failed: {e} — filling zeros")
        mom = pd.DataFrame({"Mom": 0.0}, index=ff5.index)

    factors = ff5.join(mom, how="left")
    factors["Mom"] = factors["Mom"].fillna(0.0)

    for col in FACTOR_COLS:
        if col not in factors.columns:
            log.warning(f"Factor column '{col}' missing — filling zeros")
            factors[col] = 0.0

    factors = factors.loc[start:end]
    log.info(f"FF factors ready: {factors.shape}, {factors.index.min().date()} to {factors.index.max().date()}")
    return factors


# ─────────────────────────────────────────────────────────────────────────────
# 3. Date alignment + excess returns
# ─────────────────────────────────────────────────────────────────────────────

def align_returns_and_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = returns.index.intersection(factors.index)
    ret_a  = returns.loc[common]
    fac_a  = factors.loc[common]
    valid  = ret_a.notna().any(axis=1) & fac_a[FACTOR_COLS].notna().all(axis=1)
    log.info(f"Aligned: {len(common)} common dates, {valid.sum()} valid rows")
    return ret_a.loc[valid], fac_a.loc[valid]


def compute_excess_returns(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0.0, index=factors.index)
    return returns.sub(rf, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Optional HHMM regime
# ─────────────────────────────────────────────────────────────────────────────

def load_hhmm_regime():
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
        for col in ["macro_regime", "regime", "hhmm_regime"]:
            if col in df.columns:
                df = df.rename(columns={col: "macro_regime"})
                break
        log.info(f"HHMM regime loaded: {df.shape}")
        return df[["macro_regime"]]
    except Exception as e:
        log.warning(f"HHMM regime not available: {e} — using default weights")
        return None


def get_universe_tickers(universe: str) -> list[str]:
    if universe == "fi":
        return FI_TICKERS
    elif universe == "equity":
        return EQUITY_TICKERS
    return ALL_TICKERS
