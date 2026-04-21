# P2-ETF-FACTOR-TILT

**Cross-Sectional Factor Momentum — Ranking ETFs by the Momentum of Their Factor Tilts**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-FACTOR-TILT/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-FACTOR-TILT/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-p2--etf--factor--tilt--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-factor-tilt-results)

---

## Overview

`P2-ETF-FACTOR-TILT` ranks ETFs not by their raw returns, but by how their **factor exposures are changing** over time. An ETF whose momentum beta is accelerating from 0.42 → 0.78 is attracting momentum-factor capital — this engine detects that rotation signal before it shows up in raw price momentum.

This is fundamentally different from every other engine in the P2Quant suite:

| Engine type | What it ranks on |
|---|---|
| Return-based (ARIMA, HAR-RV, Wavelet-SVR, CNN-LSTM…) | Past returns / volatility patterns |
| Flow-based (FLOW-POSITIONING) | COT / AUM / short interest |
| Regime-based (HHMM, REGIMEFLOW…) | Macro state |
| **FACTOR-TILT (this engine)** | **Momentum of factor exposure change** |

---

## Methodology

### Step 1 — Rolling OLS factor regressions

For each ETF and each rolling window (21d, 63d, 126d), fit:

```
r_etf(t) - RF = alpha + β₁·MKT + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + β₆·Mom + ε
```

Factor data from **Kenneth French Data Library** (free, daily, 2008–present).
Six factors: Market (MKT-RF), Size (SMB), Value (HML), Profitability (RMW),
Investment (CMA), Momentum (Mom/UMD).

### Step 2 — Tilt momentum (delta beta)

For each (window, lookback) combination (3 windows × 3 lookbacks = 9 combos):

```
delta_beta(t) = beta(t) - beta(t - lookback)
```

A positive `delta_Mom` means the ETF's momentum-factor loading is increasing —
capital is rotating into momentum-style exposure in this ETF.

### Step 3 — Cross-sectional z-score normalisation

On each date, standardise `delta_beta` across all ETFs in the universe.
This removes market-wide factor drift and isolates *relative* tilt changes.

### Step 4 — Composite CSFM score

Weighted aggregation across factors, windows, and lookbacks:

```
csfm_score = Σ_factor  Σ_window  Σ_lookback  w_factor · w_window · w_lookback · z(delta_beta)
```

**Default factor weights:** MKT 25%, Mom 25%, HML 15%, RMW 15%, SMB 10%, CMA 10%

**Regime override:** When HHMM-REGIME signals `stress`, momentum weight drops to 10%
and value/profitability weights increase — reducing exposure to momentum crashes.

### Step 5 — Ranking

ETFs ranked 1→N by `csfm_score` within each universe (FI, Equity, Combined).
Rank 1 = strongest positive factor tilt momentum = buy signal.

---

## Output schema

Results pushed daily to `P2SAMAPA/p2-etf-factor-tilt-results`:

| Column | Description |
|---|---|
| `run_date` | Date of run (YYYY-MM-DD) |
| `universe` | `fi`, `equity`, or `combined` |
| `ticker` | ETF ticker |
| `csfm_score` | Composite cross-sectional z-score |
| `csfm_rank` | Rank within universe (1 = highest score) |
| `dominant_factor` | Factor with largest absolute tilt contribution |
| `dominant_direction` | `strengthening` or `weakening` |
| `delta_Mkt-RF` … `delta_Mom` | Per-factor delta beta (63d window, 42d lookback) |
| `beta_Mkt-RF` … `beta_Mom` | Raw factor loadings (126d window) |
| `alpha` | Annualised Jensen's alpha (126d window) |
| `r_squared` | Regression R² (126d window) |
| `regime` | Macro regime from HHMM (stress/neutral/calm) |

---

## Integration with P2Quant suite

| Engine | Integration |
|---|---|
| **HHMM-REGIME** | Regime-conditional factor weights (stress → reduce Mom weight) |
| **FACTOR-AE** | Latent factors vs. explicit FF6 factors — complementary views |
| **REGIME-HRP** | CSFM scores can be used as return forecasts in the HRP allocation |
| **FLOW-POSITIONING** | Orthogonal signal — combine for higher-conviction rankings |

---

## CPU budget

| Step | Time |
|---|---|
| Data load (HF + FF) | ~2 min |
| Rolling OLS (20 ETFs × 3 windows × ~4500 days) | ~8–10 min |
| Tilt momentum + scoring | ~3 min |
| Push to HuggingFace | ~1 min |
| **Total** | **~15–20 min** |

Well within the 6-hour GitHub Actions free tier limit.

---

## Setup

### 1. Add HF_TOKEN secret to GitHub repo

`Settings → Secrets → Actions → New repository secret`
- Name: `HF_TOKEN`
- Value: your HuggingFace write token

### 2. Local run

```bash
pip install -r requirements.txt
python trainer.py
```

---

## File structure

```
P2-ETF-FACTOR-TILT/
├── config.py              # All constants — edit here only
├── data_manager.py        # Data loading (HF master data + FF factors)
├── factor_tilt_model.py   # Core engine: rolling OLS, delta-beta, scoring
├── trainer.py             # Pipeline orchestrator
├── push_results.py        # HuggingFace push logic
├── requirements.txt
└── .github/workflows/
    └── daily_run.yml      # GitHub Actions — 22:30 UTC Mon–Fri
```
