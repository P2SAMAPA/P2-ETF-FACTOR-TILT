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

st.set_page_config(
    page_title="P2Quant Factor Tilt",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 600; color: #1f77b4; margin-bottom: 0; }
.sub-header  { font-size: 0.95rem; color: #888; margin-bottom: 0.5rem; }

/* Explanation box */
.csfm-explain {
    background: #f0f4fa;
    border-left: 4px solid #1f77b4;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #333;
    line-height: 1.6;
}
.csfm-explain b { color: #1f77b4; }

/* Universe hero sections */
.universe-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #888;
    margin-bottom: 0.4rem;
}

/* ETF rank cards */
.rank-card {
    border-radius: 12px;
    padding: 0.9rem 1rem;
    color: white;
    margin-bottom: 0.4rem;
    position: relative;
}
.rank-card .rc-rank  { font-size: 0.68rem; opacity: 0.75; text-transform: uppercase; letter-spacing: .05em; }
.rank-card .rc-ticker{ font-size: 1.55rem; font-weight: 700; line-height: 1.1; }
.rank-card .rc-score { font-size: 0.88rem; opacity: 0.85; margin-top: 1px; }
.rank-card .rc-tilt  { font-size: 0.75rem; opacity: 0.70; margin-top: 2px; }

.rc-1 { background: linear-gradient(135deg, #1a5fa8 0%, #2C5282 100%); }
.rc-2 { background: linear-gradient(135deg, #2d7dd2 0%, #1a5fa8 100%); }
.rc-3 { background: linear-gradient(135deg, #5499d8 0%, #2d7dd2 100%); }

.rc-eq-1 { background: linear-gradient(135deg, #1a6b3c 0%, #155534 100%); }
.rc-eq-2 { background: linear-gradient(135deg, #228b4e 0%, #1a6b3c 100%); }
.rc-eq-3 { background: linear-gradient(135deg, #2eaa63 0%, #228b4e 100%); }

.rc-fi-1 { background: linear-gradient(135deg, #7b2d8b 0%, #5c1f69 100%); }
.rc-fi-2 { background: linear-gradient(135deg, #9b3dab 0%, #7b2d8b 100%); }
.rc-fi-3 { background: linear-gradient(135deg, #b556c4 0%, #9b3dab 100%); }

.divider-col {
    border-left: 1px solid #e0e0e0;
    height: 100%;
}

/* Sidebar info rows */
.sb-row { font-size: 0.85rem; color: #444; margin-bottom: 0.3rem; }
.sb-val  { font-weight: 600; color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-factor-tilt-results"
HF_RESULTS_FILE = "factor_tilt_results.parquet"

FACTOR_DISPLAY = {
    "Mkt-RF": "Market", "SMB": "Size", "HML": "Value",
    "RMW": "Profitability", "CMA": "Investment", "Mom": "Momentum",
}
FACTOR_COLORS = {
    "Market": "#1f77b4", "Size": "#ff7f0e", "Value": "#2ca02c",
    "Profitability": "#d62728", "Investment": "#9467bd", "Momentum": "#8c564b",
}


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
    return sub[sub["run_date"] == sub["run_date"].max()].sort_values("csfm_rank")


# ── Sidebar ───────────────────────────────────────────────────────────────────

cal = USMarketCalendar()
next_td = cal.next_trading_day()
is_today = cal.is_trading_day()

df = load_results()
latest_run_date = df["run_date"].max() if df is not None and not df.empty else None
n_dates = df["run_date"].nunique() if df is not None and not df.empty else 0

st.sidebar.markdown("## 📐 P2Quant Factor Tilt")
st.sidebar.divider()

st.sidebar.markdown("### 📅 Market calendar")
st.sidebar.markdown(f"**Last run:** {latest_run_date.strftime('%a %d %b %Y') if latest_run_date else '—'}")
st.sidebar.markdown(f"**Next trading day:** {next_td.strftime('%a %d %b %Y')}")
if is_today:
    st.sidebar.success("Today is a trading day")
else:
    st.sidebar.info("Market closed today")
st.sidebar.markdown(f"*{n_dates} trading days in history*")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Engine parameters")
st.sidebar.markdown("- **Factors:** MKT, SMB, HML, RMW, CMA, Mom")
st.sidebar.markdown("- **Windows:** 21d / 63d / 126d rolling OLS")
st.sidebar.markdown("- **Tilt lookbacks:** 21d / 42d / 63d")
st.sidebar.markdown("- **Universes:** FI, Equity, Combined")
st.sidebar.divider()
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[GitHub repo](https://github.com/P2SAMAPA/P2-ETF-FACTOR-TILT)")
st.sidebar.markdown("[HF dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-factor-tilt-results)")

# ── Main header ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">📐 P2Quant Factor Tilt</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Sectional Factor Momentum — ranking ETFs by the acceleration of their factor exposures</div>', unsafe_allow_html=True)

# ── CSFM explanation ──────────────────────────────────────────────────────────

st.markdown("""
<div class="csfm-explain">
<b>What is the CSFM score?</b>&nbsp; Every other return engine asks <i>"which ETF performed best?"</i>
This engine asks something different: <i>"which ETF's factor tilts are <b>accelerating</b> right now?"</i>
The score measures how much each ETF's exposure to six Fama-French factors (Market, Size, Value,
Profitability, Investment, Momentum) has <b>changed</b> over the recent period, standardised cross-sectionally.
A high positive score means capital is actively rotating <i>into</i> this ETF's factor profile —
a signal that often leads raw price momentum by days to weeks.
Score = cross-sectional z-score (higher = stronger positive tilt momentum).
</div>
""", unsafe_allow_html=True)

# ── Check data ────────────────────────────────────────────────────────────────

if df is None or df.empty:
    st.warning("No results available yet. The engine runs daily at 22:30 UTC after market close.")
    st.stop()

# ── Hero section — top 3 per universe ────────────────────────────────────────

combined_latest = get_latest(df, "combined")
equity_latest   = get_latest(df, "equity")
fi_latest       = get_latest(df, "fi")

def universe_top3_html(rows: pd.DataFrame, style_prefix: str, universe_label: str) -> str:
    """Render a universe label + 3 stacked rank cards."""
    html = f'<div class="universe-header">{universe_label}</div>'
    for i, (_, row) in enumerate(rows.head(3).iterrows(), start=1):
        ticker   = row.get("ticker", "—")
        score    = row.get("csfm_score", 0.0)
        dom_f    = row.get("dominant_factor", "")
        dom_d    = row.get("dominant_direction", "")
        arrow    = "↑" if dom_d == "strengthening" else ("↓" if dom_d == "weakening" else "")
        tilt_str = f"{dom_f} {arrow}" if dom_f else ""
        html += f"""
        <div class="rank-card {style_prefix}-{i}">
            <div class="rc-rank">#{i}</div>
            <div class="rc-ticker">{ticker}</div>
            <div class="rc-score">Score: {score:.3f}</div>
            {"<div class='rc-tilt'>" + tilt_str + "</div>" if tilt_str else ""}
        </div>"""
    return html


col_comb, col_div1, col_eq, col_div2, col_fi = st.columns([3, 0.08, 3, 0.08, 3])

with col_comb:
    st.markdown(universe_top3_html(combined_latest, "rc", "🌐 Combined — top 3"), unsafe_allow_html=True)

with col_div1:
    st.markdown('<div style="border-left:1px solid #e0e0e0;height:260px;margin:0 auto;"></div>', unsafe_allow_html=True)

with col_eq:
    st.markdown(universe_top3_html(equity_latest, "rc-eq", "📈 Equity sectors — top 3"), unsafe_allow_html=True)

with col_div2:
    st.markdown('<div style="border-left:1px solid #e0e0e0;height:260px;margin:0 auto;"></div>', unsafe_allow_html=True)

with col_fi:
    st.markdown(universe_top3_html(fi_latest, "rc-fi", "💰 FI / Commodities — top 3"), unsafe_allow_html=True)

st.divider()

# ── Universe tabs ─────────────────────────────────────────────────────────────

tab_comb, tab_eq, tab_fi, tab_history = st.tabs([
    "🌐 Combined", "📈 Equity sectors", "💰 FI / Commodities", "📊 History",
])

DELTA_COLS = ["delta_Mkt-RF", "delta_SMB", "delta_HML", "delta_RMW", "delta_CMA", "delta_Mom"]


def render_universe_tab(latest: pd.DataFrame):
    if latest.empty:
        st.info("No data for this universe yet.")
        return

    run_dt = latest["run_date"].iloc[0].strftime("%d %b %Y")
    st.caption(f"Scores as of {run_dt}")

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
        hovertemplate="<b>%{y}</b><br>CSFM score: %{x:.4f}<extra></extra>",
    ))
    fig_rank.update_layout(
        height=max(300, len(latest) * 36),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=70, t=10, b=20),
        xaxis_title="CSFM score (cross-sectional z-score)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # ── Factor tilt heatmap ───────────────────────────────────────────────────
    st.markdown("#### Factor tilt delta (63d window, 42d lookback)")
    st.caption("Each cell = change in factor loading (beta) over the lookback period. Blue = tilt strengthening, Red = tilt weakening.")
    available_delta = [c for c in DELTA_COLS if c in latest.columns]
    if available_delta:
        heat_df = latest.set_index("ticker")[available_delta].copy()
        heat_df.columns = [FACTOR_DISPLAY.get(c.replace("delta_", ""), c.replace("delta_", "")) for c in available_delta]
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
            margin=dict(l=10, r=10, t=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

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


with tab_comb:
    render_universe_tab(combined_latest)

with tab_eq:
    render_universe_tab(equity_latest)

with tab_fi:
    render_universe_tab(fi_latest)

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
                    x=pivot.index, y=pivot[ticker], name=ticker, mode="lines",
                    hovertemplate=f"<b>{ticker}</b><br>%{{x|%d %b %Y}}<br>Score: %{{y:.3f}}<extra></extra>",
                ))
            fig_hist.update_layout(
                height=420,
                xaxis_title="Date", yaxis_title="CSFM score",
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Dominant factor frequency (combined universe)")
        if "dominant_factor" in hist.columns:
            factor_counts = (
                hist.groupby(["run_date", "dominant_factor"])
                .size().reset_index(name="count")
            )
            fig_factors = px.bar(
                factor_counts, x="run_date", y="count", color="dominant_factor",
                color_discrete_map=FACTOR_COLORS,
                labels={"run_date": "Date", "count": "ETF count", "dominant_factor": "Factor"},
                height=300,
            )
            fig_factors.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend_title="Dominant factor",
                margin=dict(l=10, r=10, t=10, b=20),
            )
            st.plotly_chart(fig_factors, use_container_width=True)

st.divider()
st.caption("P2Quant Factor Tilt Engine · P2SAMAPA · Factors: Kenneth French Data Library · Results: HuggingFace")
