"""
Streamlit Dashboard — P2-ETF-FACTOR-TILT
Cross-Sectional Factor Momentum Engine
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download

from us_calendar import USMarketCalendar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2Quant Factor Tilt",
    page_icon="📐",
    layout="wide",
)

# ── Styling — matches REGIME-HRP aesthetic ────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size: 2.4rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.2rem; }
.sub-header  { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
.hero-card {
    background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%);
    border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
    margin-bottom: 0.5rem;
}
.hero-label { font-size: 0.78rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.2rem; }
.hero-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.hero-sub   { font-size: 0.82rem; opacity: 0.8; margin-top: 0.3rem; }
.hero-card-green {
    background: linear-gradient(135deg, #276749 0%, #1E4D35 100%);
    border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
    margin-bottom: 0.5rem;
}
.hero-card-amber {
    background: linear-gradient(135deg, #92400E 0%, #78350F 100%);
    border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
    margin-bottom: 0.5rem;
}
.hero-card-purple {
    background: linear-gradient(135deg, #5B21B6 0%, #3B0764 100%);
    border-radius: 16px; padding: 1.4rem 1.6rem; color: white;
    margin-bottom: 0.5rem;
}
.rank-badge-1 { background:#1f77b4; color:white; border-radius:8px; padding:2px 10px; font-weight:700; font-size:1rem; }
.rank-badge-2 { background:#4a9eca; color:white; border-radius:8px; padding:2px 10px; font-weight:600; }
.rank-badge-3 { background:#7abde0; color:white; border-radius:8px; padding:2px 10px; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-factor-tilt-results"
HF_RESULTS_FILE = "factor_tilt_results.parquet"

FACTOR_DISPLAY = {
    "Mkt-RF": "Market",
    "SMB":    "Size",
    "HML":    "Value",
    "RMW":    "Profitability",
    "CMA":    "Investment",
    "Mom":    "Momentum",
}
FACTOR_COLORS = {
    "Market":       "#1f77b4",
    "Size":         "#ff7f0e",
    "Value":        "#2ca02c",
    "Profitability":"#d62728",
    "Investment":   "#9467bd",
    "Momentum":     "#8c564b",
}

UNIVERSE_LABELS = {"fi": "FI / Commodities", "equity": "Equity Sectors", "combined": "Combined"}
UNIVERSE_ICONS  = {"fi": "💰", "equity": "📈", "combined": "🌐"}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_results() -> pd.DataFrame | None:
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=HF_RESULTS_FILE,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        df["run_date"] = pd.to_datetime(df["run_date"])
        return df
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None


def get_latest(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    sub = df[df["universe"] == universe]
    if sub.empty:
        return sub
    latest = sub["run_date"].max()
    return sub[sub["run_date"] == latest].sort_values("csfm_rank")


# ── Hero card helper ──────────────────────────────────────────────────────────

def hero_card(label: str, value: str, sub: str = "", style: str = "hero-card") -> str:
    return f"""
    <div class="{style}">
        <div class="hero-label">{label}</div>
        <div class="hero-value">{value}</div>
        {"<div class='hero-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

cal = USMarketCalendar()
next_td = cal.next_trading_day()
is_today_trading = cal.is_trading_day()

st.sidebar.markdown("## ⚙️ P2Quant — Factor Tilt")
st.sidebar.markdown(
    f"**📅 Next trading day:** {next_td.strftime('%a %d %b %Y')}"
)
if is_today_trading:
    st.sidebar.success("Today is a trading day")
else:
    st.sidebar.info("Market closed today")

st.sidebar.divider()
st.sidebar.markdown("### Engine parameters")
st.sidebar.markdown("- **Factors:** MKT, SMB, HML, RMW, CMA, Mom")
st.sidebar.markdown("- **Windows:** 21d / 63d / 126d")
st.sidebar.markdown("- **Tilt lookbacks:** 21d / 42d / 63d")
st.sidebar.markdown("- **Universes:** FI, Equity, Combined")
st.sidebar.divider()
st.sidebar.markdown("### Links")
st.sidebar.markdown("[GitHub repo](https://github.com/P2SAMAPA/P2-ETF-FACTOR-TILT)")
st.sidebar.markdown("[HF dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-factor-tilt-results)")

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📐 P2Quant Factor Tilt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Sectional Factor Momentum — Ranking ETFs by the momentum of their factor exposures</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

df = load_results()

if df is None or df.empty:
    st.warning("No results available yet. The engine runs daily at 22:30 UTC after market close.")
    st.stop()

latest_run_date = df["run_date"].max()
n_dates = df["run_date"].nunique()

# ── Top-level hero cards ──────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(hero_card(
        "Last run date",
        latest_run_date.strftime("%d %b %Y"),
        f"{n_dates} trading days in history",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(hero_card(
        "Next trading day",
        next_td.strftime("%d %b %Y"),
        "NYSE calendar",
        style="hero-card-green",
    ), unsafe_allow_html=True)
with c3:
    combined_latest = get_latest(df, "combined")
    top_combined = combined_latest.iloc[0]["ticker"] if not combined_latest.empty else "—"
    top_score    = combined_latest.iloc[0]["csfm_score"] if not combined_latest.empty else 0
    st.markdown(hero_card(
        "Top combined ETF",
        top_combined,
        f"CSFM score: {top_score:.3f}",
        style="hero-card-amber",
    ), unsafe_allow_html=True)
with c4:
    regime_val = combined_latest.iloc[0]["regime"] if not combined_latest.empty else "unknown"
    regime_colors = {"stress": "hero-card-amber", "calm": "hero-card-green", "neutral": "hero-card-purple"}
    regime_style  = regime_colors.get(str(regime_val).lower(), "hero-card-purple")
    dominant_f = combined_latest.iloc[0]["dominant_factor"] if not combined_latest.empty else "—"
    dominant_d = combined_latest.iloc[0]["dominant_direction"] if not combined_latest.empty else ""
    st.markdown(hero_card(
        "Macro regime",
        str(regime_val).capitalize(),
        f"Dominant tilt: {dominant_f} ({dominant_d})",
        style=regime_style,
    ), unsafe_allow_html=True)

st.divider()

# ── Universe tabs ─────────────────────────────────────────────────────────────

tab_fi, tab_eq, tab_comb, tab_history = st.tabs([
    "💰 FI / Commodities",
    "📈 Equity Sectors",
    "🌐 Combined",
    "📊 History",
])

DELTA_COLS = ["delta_Mkt-RF", "delta_SMB", "delta_HML", "delta_RMW", "delta_CMA", "delta_Mom"]
BETA_COLS  = ["beta_Mkt-RF",  "beta_SMB",  "beta_HML",  "beta_RMW",  "beta_CMA",  "beta_Mom"]


def render_universe_tab(universe_key: str):
    latest = get_latest(df, universe_key)
    if latest.empty:
        st.info("No data for this universe yet.")
        return

    run_dt = latest["run_date"].iloc[0].strftime("%d %b %Y")
    st.caption(f"Data as of {run_dt}")

    # ── Ranking bar chart ─────────────────────────────────────────────────────
    st.markdown("#### CSFM score ranking")
    colors = ["#1f77b4" if s > 0 else "#d62728" for s in latest["csfm_score"]]
    fig_rank = go.Figure(go.Bar(
        x=latest["csfm_score"],
        y=latest["ticker"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in latest["csfm_score"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
    ))
    fig_rank.update_layout(
        height=max(300, len(latest) * 36),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis_title="CSFM score (cross-sectional z-score)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # ── Factor tilt heatmap ───────────────────────────────────────────────────
    st.markdown("#### Factor tilt delta (63d window, 42d lookback)")
    available_delta = [c for c in DELTA_COLS if c in latest.columns]
    if available_delta:
        heat_df = latest.set_index("ticker")[available_delta].copy()
        heat_df.columns = [FACTOR_DISPLAY.get(c.replace("delta_",""), c.replace("delta_","")) for c in available_delta]

        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=heat_df.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in heat_df.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b> — %{x}<br>Delta beta: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Δ Beta", thickness=12),
        ))
        fig_heat.update_layout(
            height=max(300, len(heat_df) * 36 + 60),
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top 5 detail cards ────────────────────────────────────────────────────
    st.markdown("#### Top 5 — factor tilt detail")
    top5 = latest.head(5)
    cols = st.columns(min(5, len(top5)))
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            score_str = f"{row['csfm_score']:.3f}"
            rank_str  = f"#{int(row['csfm_rank'])}"
            dom_f = row.get("dominant_factor", "—")
            dom_d = row.get("dominant_direction", "")
            arrow = "↑" if dom_d == "strengthening" else "↓"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1f77b4,#2C5282);border-radius:12px;
                        padding:1rem;color:white;text-align:center;margin-bottom:0.5rem;">
                <div style="font-size:0.7rem;opacity:0.75;text-transform:uppercase;letter-spacing:.05em">{rank_str}</div>
                <div style="font-size:1.6rem;font-weight:700">{row['ticker']}</div>
                <div style="font-size:1rem;opacity:0.9">{score_str}</div>
                <div style="font-size:0.75rem;opacity:0.75;margin-top:4px">{dom_f} {arrow}</div>
            </div>""", unsafe_allow_html=True)

    # ── Full table ────────────────────────────────────────────────────────────
    with st.expander("Full rankings table"):
        display_cols = ["ticker", "csfm_rank", "csfm_score", "dominant_factor",
                        "dominant_direction", "alpha", "r_squared", "regime"]
        show_cols = [c for c in display_cols if c in latest.columns]
        st.dataframe(
            latest[show_cols].style.format({
                "csfm_score": "{:.4f}",
                "alpha":      "{:.4f}",
                "r_squared":  "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


with tab_fi:
    render_universe_tab("fi")

with tab_eq:
    render_universe_tab("equity")

with tab_comb:
    render_universe_tab("combined")

# ── History tab ───────────────────────────────────────────────────────────────

with tab_history:
    st.markdown("#### CSFM score history — combined universe")

    hist = df[df["universe"] == "combined"].copy()
    if hist.empty:
        st.info("No history yet.")
    else:
        all_tickers = sorted(hist["ticker"].unique())
        selected = st.multiselect(
            "Select ETFs to compare",
            options=all_tickers,
            default=all_tickers[:6] if len(all_tickers) >= 6 else all_tickers,
        )

        if selected:
            pivot = hist[hist["ticker"].isin(selected)].pivot(
                index="run_date", columns="ticker", values="csfm_score"
            )
            fig_hist = go.Figure()
            for ticker in pivot.columns:
                fig_hist.add_trace(go.Scatter(
                    x=pivot.index,
                    y=pivot[ticker],
                    name=ticker,
                    mode="lines",
                    hovertemplate=f"<b>{ticker}</b><br>%{{x|%d %b %Y}}<br>Score: %{{y:.3f}}<extra></extra>",
                ))
            fig_hist.update_layout(
                height=420,
                xaxis_title="Date",
                yaxis_title="CSFM score",
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Dominant factor frequency over time ───────────────────────────────
        st.markdown("#### Dominant factor frequency (combined universe)")
        if "dominant_factor" in hist.columns:
            factor_counts = (
                hist.groupby(["run_date", "dominant_factor"])
                .size()
                .reset_index(name="count")
            )
            fig_factors = px.bar(
                factor_counts,
                x="run_date", y="count", color="dominant_factor",
                color_discrete_map=FACTOR_COLORS,
                labels={"run_date": "Date", "count": "ETF count", "dominant_factor": "Factor"},
                height=320,
            )
            fig_factors.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title="Dominant factor",
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_factors, use_container_width=True)

st.divider()
st.caption("P2Quant Factor Tilt Engine | P2SAMAPA | Data: Kenneth French Data Library + HuggingFace")
