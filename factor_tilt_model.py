# factor_tilt_model.py — P2-ETF-FACTOR-TILT
# Core engine: rolling OLS factor exposures → tilt momentum → cross-sectional score
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    FACTOR_COLS, FACTOR_NAMES, FACTOR_WEIGHTS,
    ROLLING_WINDOWS, TILT_LOOKBACKS, MIN_OBS,
    WINDOW_WEIGHTS, TILT_LOOKBACK_WEIGHTS,
    REGIME_FACTOR_OVERRIDES,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rolling OLS — fit factor model in each window
# ─────────────────────────────────────────────────────────────────────────────

def _ols_one_window(
    y: np.ndarray,
    X: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """
    Fit OLS: y = alpha + X @ beta + e
    Returns (alpha, betas[n_factors], r_squared).
    Returns NaNs if regression fails.
    """
    try:
        if len(y) < MIN_OBS or np.isnan(y).any() or np.isnan(X).any():
            return np.nan, np.full(X.shape[1], np.nan), np.nan
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return reg.intercept_, reg.coef_, r2
    except Exception:
        return np.nan, np.full(X.shape[1], np.nan), np.nan


def compute_rolling_betas(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
) -> dict[int, pd.DataFrame]:
    """
    For each rolling window, compute factor betas for every ETF on every date.

    Returns dict: {window_size → DataFrame}
    Each DataFrame has a MultiIndex columns: (ticker, factor_col)
    Plus (ticker, 'alpha') and (ticker, 'r_squared').
    Index = dates.

    CPU note: sklearn LinearRegression on small arrays (21-126 obs, 6 features)
    is near-instant. 20 ETFs × 3 windows × 4500 dates ≈ 270k regressions, ~8 min.
    """
    factor_matrix = factors[FACTOR_COLS].values
    factor_dates  = factors.index
    tickers       = [t for t in excess_returns.columns if t in excess_returns.columns]

    results = {}

    for window in windows:
        log.info(f"  Rolling OLS — window={window}d, {len(tickers)} ETFs ...")

        # Pre-allocate output arrays
        n_dates   = len(factor_dates)
        n_factors = len(FACTOR_COLS)
        n_tickers = len(tickers)

        alpha_arr = np.full((n_dates, n_tickers), np.nan)
        beta_arr  = np.full((n_dates, n_tickers, n_factors), np.nan)
        r2_arr    = np.full((n_dates, n_tickers), np.nan)

        for t_idx, ticker in enumerate(tickers):
            if ticker not in excess_returns.columns:
                continue
            ret_series = excess_returns[ticker].reindex(factor_dates).values

            for d_idx in range(window - 1, n_dates):
                start = d_idx - window + 1
                y_win = ret_series[start : d_idx + 1]
                X_win = factor_matrix[start : d_idx + 1]

                if np.isnan(y_win).sum() > window * 0.3:  # >30% missing → skip
                    continue

                # Drop NaN rows together
                mask = ~(np.isnan(y_win) | np.isnan(X_win).any(axis=1))
                if mask.sum() < MIN_OBS:
                    continue

                alpha, betas, r2 = _ols_one_window(y_win[mask], X_win[mask])
                alpha_arr[d_idx, t_idx]      = alpha
                beta_arr[d_idx, t_idx, :]    = betas
                r2_arr[d_idx, t_idx]         = r2

        # Pack into DataFrame with MultiIndex columns
        col_tuples = (
            [(t, f) for t in tickers for f in FACTOR_COLS] +
            [(t, "alpha") for t in tickers] +
            [(t, "r_squared") for t in tickers]
        )
        col_index = pd.MultiIndex.from_tuples(col_tuples)

        data_matrix = np.hstack([
            beta_arr.reshape(n_dates, n_tickers * n_factors),
            alpha_arr,
            r2_arr,
        ])
        df_window = pd.DataFrame(data_matrix, index=factor_dates, columns=col_index)
        results[window] = df_window
        log.info(f"  Window {window}d done — {(~np.isnan(alpha_arr)).sum()} valid fits")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tilt momentum — delta beta over lookback periods
# ─────────────────────────────────────────────────────────────────────────────

def compute_tilt_momentum(
    rolling_betas: dict[int, pd.DataFrame],
    tickers: list[str],
    lookbacks: list[int] = TILT_LOOKBACKS,
) -> dict[tuple[int, int], pd.DataFrame]:
    """
    For each (window, lookback) combination, compute:
        delta_beta(t) = beta(t) - beta(t - lookback)

    This is the "tilt momentum" — how much each factor exposure
    has changed over the lookback period.

    Returns dict: {(window, lookback) → DataFrame[dates × (ticker, factor)]}
    """
    tilt_results = {}

    for window, beta_df in rolling_betas.items():
        for lookback in lookbacks:
            delta_dict = {}
            for ticker in tickers:
                for factor in FACTOR_COLS:
                    key = (ticker, factor)
                    if key not in beta_df.columns:
                        continue
                    series = beta_df[key]
                    delta = series - series.shift(lookback)
                    delta_dict[key] = delta

            if delta_dict:
                delta_df = pd.DataFrame(delta_dict)
                delta_df.columns = pd.MultiIndex.from_tuples(delta_df.columns)
                tilt_results[(window, lookback)] = delta_df
                log.debug(f"  Tilt computed: window={window}, lookback={lookback}")

    return tilt_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-sectional z-score normalisation
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """
    Normalise a cross-sectional series (values across ETFs on one date)
    to z-scores. Returns NaN if std == 0.
    """
    mean = series.mean()
    std  = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composite CSFM score
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_score(
    tilt_results: dict[tuple[int, int], pd.DataFrame],
    tickers: list[str],
    factor_weights: dict[str, float] = FACTOR_WEIGHTS,
) -> pd.DataFrame:
    """
    Aggregate tilt momentum signals across all (window, lookback) combos
    and all factors into a single composite cross-sectional z-score per ETF.

    Steps:
      1. For each (window, lookback, factor): z-score delta_betas cross-sectionally
      2. Weighted sum across factors using factor_weights
      3. Weighted sum across windows using WINDOW_WEIGHTS
      4. Weighted sum across lookbacks using TILT_LOOKBACK_WEIGHTS
      5. Final cross-sectional z-score normalisation

    Returns DataFrame[dates × tickers] of composite CSFM scores.
    """
    # Accumulator: date × ticker
    all_dates = sorted(set(
        date for df in tilt_results.values() for date in df.index
    ))
    score_accum = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    weight_accum = pd.DataFrame(0.0, index=all_dates, columns=tickers)

    for (window, lookback), delta_df in tilt_results.items():
        w_window   = WINDOW_WEIGHTS.get(window, 1.0 / len(ROLLING_WINDOWS))
        w_lookback = TILT_LOOKBACK_WEIGHTS.get(lookback, 1.0 / len(TILT_LOOKBACKS))
        combo_weight = w_window * w_lookback

        # For each date, compute cross-sectional z-score per factor then combine
        for date in delta_df.index:
            row = delta_df.loc[date]

            # Weighted sum of factor z-scores for this date
            factor_score = pd.Series(0.0, index=tickers)
            factor_weight_sum = 0.0

            for factor in FACTOR_COLS:
                f_weight = factor_weights.get(factor, 0.0)
                if f_weight == 0:
                    continue

                # Extract delta_beta for this factor across all tickers on this date
                ticker_deltas = {}
                for ticker in tickers:
                    key = (ticker, factor)
                    if key in row.index:
                        val = row[key]
                        if not np.isnan(val):
                            ticker_deltas[ticker] = val

                if len(ticker_deltas) < 3:  # need at least 3 for meaningful z-score
                    continue

                delta_series = pd.Series(ticker_deltas)
                z_series     = cross_sectional_zscore(delta_series)

                for ticker in tickers:
                    if ticker in z_series.index and not np.isnan(z_series[ticker]):
                        factor_score[ticker] += f_weight * z_series[ticker]
                        factor_weight_sum += f_weight

            if factor_weight_sum > 0:
                factor_score /= factor_weight_sum
                score_accum.loc[date]  += combo_weight * factor_score
                weight_accum.loc[date] += combo_weight

    # Normalise by accumulated weights
    valid_mask = weight_accum > 0
    composite = score_accum.copy()
    composite[valid_mask] = score_accum[valid_mask] / weight_accum[valid_mask]
    composite[~valid_mask] = np.nan

    # Final cross-sectional z-score normalisation per date
    final_scores = composite.apply(
        lambda row: cross_sectional_zscore(row.dropna()).reindex(row.index),
        axis=1,
    )

    log.info(f"Composite CSFM scores computed: {final_scores.shape}")
    return final_scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dominant factor identification
# ─────────────────────────────────────────────────────────────────────────────

def get_dominant_factor(
    tilt_results: dict[tuple[int, int], pd.DataFrame],
    ticker: str,
    date: pd.Timestamp,
    window: int = 63,
    lookback: int = 42,
) -> tuple[str, str]:
    """
    For a given ETF on a given date, find which factor has the largest
    absolute tilt delta, and whether it is strengthening or weakening.

    Returns (factor_name, direction) e.g. ("Momentum", "strengthening")
    """
    key = (window, lookback)
    if key not in tilt_results:
        return "Unknown", "unknown"

    delta_df = tilt_results[key]
    if date not in delta_df.index:
        return "Unknown", "unknown"

    row = delta_df.loc[date]
    best_factor = None
    best_abs    = -1.0

    for factor in FACTOR_COLS:
        col_key = (ticker, factor)
        if col_key in row.index:
            val = row[col_key]
            if not np.isnan(val) and abs(val) > best_abs:
                best_abs    = abs(val)
                best_factor = factor
                best_val    = val

    if best_factor is None:
        return "Unknown", "unknown"

    direction = "strengthening" if best_val > 0 else "weakening"
    return FACTOR_NAMES.get(best_factor, best_factor), direction


# ─────────────────────────────────────────────────────────────────────────────
# 6. Apply regime-based factor weight override
# ─────────────────────────────────────────────────────────────────────────────

def get_factor_weights_for_regime(regime: Optional[str]) -> dict[str, float]:
    """
    Returns factor weights adjusted for the current macro regime.
    Falls back to default FACTOR_WEIGHTS if regime is None or unrecognised.
    """
    if regime is None:
        return FACTOR_WEIGHTS
    regime_clean = str(regime).lower().strip()
    return REGIME_FACTOR_OVERRIDES.get(regime_clean, FACTOR_WEIGHTS)
