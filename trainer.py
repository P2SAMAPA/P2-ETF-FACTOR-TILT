# trainer.py — P2-ETF-FACTOR-TILT
# Orchestrates the full pipeline end-to-end.
# Run: python trainer.py
# Expected runtime: ~15-20 min on GitHub Actions 2-vCPU CPU runner.
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    ALL_TICKERS, FI_TICKERS, EQUITY_TICKERS,
    ROLLING_WINDOWS, TILT_LOOKBACKS,
    FACTOR_COLS, FACTOR_NAMES, FACTOR_WEIGHTS,
    OUTPUT_COLS, MAX_RUNTIME_MINUTES,
    USE_REGIME_FILTER,
)
from data_manager import (
    load_master_data,
    extract_log_returns,
    load_ff_factors,
    align_returns_and_factors,
    compute_excess_returns,
    load_hhmm_regime,
    get_universe_tickers,
)
from factor_tilt_model import (
    compute_rolling_betas,
    compute_tilt_momentum,
    compute_composite_score,
    get_dominant_factor,
    get_factor_weights_for_regime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

UNIVERSES = {
    "fi":       FI_TICKERS,
    "equity":   EQUITY_TICKERS,
    "combined": ALL_TICKERS,
}


def run_universe(
    universe: str,
    tickers: list[str],
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    rolling_betas: dict,          # pre-computed — shared across universes
    tilt_results: dict,           # pre-computed — shared across universes
    regime_df: pd.DataFrame | None,
    run_date: str,
) -> pd.DataFrame:
    """
    For a given universe, compute composite CSFM scores and build output rows.
    """
    log.info(f"  Scoring universe: {universe} ({len(tickers)} ETFs)")

    # Determine factor weights — use regime override for the most recent date
    current_regime = None
    if regime_df is not None and not regime_df.empty:
        latest_regime_date = regime_df.index.max()
        current_regime = regime_df.loc[latest_regime_date, "macro_regime"]
        log.info(f"  Regime: {current_regime} (from {latest_regime_date.date()})")

    factor_weights = get_factor_weights_for_regime(current_regime)

    # Compute composite scores for this universe
    universe_tickers = [t for t in tickers if t in excess_returns.columns]
    scores_df = compute_composite_score(tilt_results, universe_tickers, factor_weights)

    # ── Build output rows for the most recent date ────────────────────────────
    # Use the last date with valid scores
    latest_date = scores_df.dropna(how="all").index.max()
    if pd.isna(latest_date):
        log.warning(f"  No valid scores for universe {universe}")
        return pd.DataFrame()

    score_row = scores_df.loc[latest_date]

    # Reference betas — use 126d window, latest date
    ref_window = max(ROLLING_WINDOWS)
    beta_df    = rolling_betas.get(ref_window, pd.DataFrame())

    rows = []
    for ticker in universe_tickers:
        score = score_row.get(ticker, np.nan)
        if np.isnan(score):
            continue

        row = {
            "run_date":  run_date,
            "universe":  universe,
            "ticker":    ticker,
            "csfm_score": round(float(score), 6),
        }

        # Per-factor delta betas (63d window, 42d lookback — the "medium" combo)
        tilt_key = (63, 42)
        if tilt_key in tilt_results and latest_date in tilt_results[tilt_key].index:
            tilt_row = tilt_results[tilt_key].loc[latest_date]
            for factor in FACTOR_COLS:
                col_key = (ticker, factor)
                val = tilt_row.get(col_key, np.nan)
                row[f"delta_{factor}"] = round(float(val), 6) if not pd.isna(val) else np.nan

        # Raw betas — 126d window, latest date
        if not beta_df.empty and latest_date in beta_df.index:
            beta_row = beta_df.loc[latest_date]
            for factor in FACTOR_COLS:
                col_key = (ticker, factor)
                val = beta_row.get(col_key, np.nan)
                row[f"beta_{factor}"] = round(float(val), 6) if not pd.isna(val) else np.nan

            alpha_key  = (ticker, "alpha")
            r2_key     = (ticker, "r_squared")
            alpha_val  = beta_row.get(alpha_key, np.nan)
            r2_val     = beta_row.get(r2_key, np.nan)
            # Annualise alpha (daily → annual)
            row["alpha"]     = round(float(alpha_val) * 252, 6) if not pd.isna(alpha_val) else np.nan
            row["r_squared"] = round(float(r2_val), 4) if not pd.isna(r2_val) else np.nan

        # Dominant factor
        dom_factor, dom_dir = get_dominant_factor(tilt_results, ticker, latest_date)
        row["dominant_factor"]    = dom_factor
        row["dominant_direction"] = dom_dir
        row["regime"]             = current_regime if current_regime else "unknown"

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows)

    # Add rank within universe
    out_df = out_df.sort_values("csfm_score", ascending=False)
    out_df["csfm_rank"] = range(1, len(out_df) + 1)

    log.info(f"  {universe}: {len(out_df)} ETFs scored. "
             f"Top 3: {out_df.head(3)['ticker'].tolist()}")

    return out_df


def main():
    t_start = time.time()
    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    log.info(f"=== P2-ETF-FACTOR-TILT | Run date: {run_date} ===")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("Step 1/5: Loading data...")
    master = load_master_data()
    returns = extract_log_returns(master)
    factors = load_ff_factors()
    returns, factors = align_returns_and_factors(returns, factors)
    excess = compute_excess_returns(returns, factors)
    regime_df = load_hhmm_regime() if USE_REGIME_FILTER else None

    # ── 2. Rolling OLS — compute once, shared across all universes ────────────
    log.info("Step 2/5: Computing rolling OLS factor betas...")
    log.info(f"  Windows: {ROLLING_WINDOWS} days | Tickers: {len(ALL_TICKERS)}")
    t_ols = time.time()
    rolling_betas = compute_rolling_betas(excess, factors, windows=ROLLING_WINDOWS)
    log.info(f"  Rolling OLS complete in {(time.time() - t_ols) / 60:.1f} min")

    # ── 3. Tilt momentum ──────────────────────────────────────────────────────
    log.info("Step 3/5: Computing tilt momentum (delta betas)...")
    tilt_results = compute_tilt_momentum(rolling_betas, ALL_TICKERS, lookbacks=TILT_LOOKBACKS)
    log.info(f"  Tilt combinations: {len(tilt_results)} ({len(ROLLING_WINDOWS)} windows × {len(TILT_LOOKBACKS)} lookbacks)")

    # ── 4. Score all universes ────────────────────────────────────────────────
    log.info("Step 4/5: Scoring all universes...")
    all_results = []

    for universe, tickers in UNIVERSES.items():
        universe_tickers = [t for t in tickers if t in excess.columns]
        df = run_universe(
            universe=universe,
            tickers=universe_tickers,
            excess_returns=excess,
            factors=factors,
            rolling_betas=rolling_betas,
            tilt_results=tilt_results,
            regime_df=regime_df,
            run_date=run_date,
        )
        if not df.empty:
            all_results.append(df)

    if not all_results:
        log.error("No results produced — aborting push")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # Ensure all output columns exist
    for col in OUTPUT_COLS:
        if col not in final_df.columns:
            final_df[col] = np.nan

    final_df = final_df[OUTPUT_COLS]
    log.info(f"Final output: {final_df.shape} rows")

    # ── 5. Push to HuggingFace ────────────────────────────────────────────────
    log.info("Step 5/5: Pushing results to HuggingFace...")
    from push_results import push_to_hf
    push_to_hf(final_df, run_date)

    elapsed = (time.time() - t_start) / 60
    log.info(f"=== COMPLETE | Total runtime: {elapsed:.1f} min ===")

    if elapsed > MAX_RUNTIME_MINUTES:
        log.warning(f"Runtime exceeded budget of {MAX_RUNTIME_MINUTES} min!")


if __name__ == "__main__":
    main()
