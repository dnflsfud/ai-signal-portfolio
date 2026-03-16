# -*- coding: utf-8 -*-
"""
AI Signal Portfolio Dashboard
===============================
Streamlit - CSV 결과물 기반 시각화

실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Signal Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_DIR = PROJECT_ROOT / "outputs" / "csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
COLORS = {
    "fund": "#1f77b4",
    "benchmark": "#ff7f0e",
    "active": "#2ca02c",
    "negative": "#d62728",
    "purple": "#9467bd",
}

GROUP_COLORS = {
    "Accounting": "#1f77b4",
    "Price": "#ff7f0e",
    "Sellside": "#2ca02c",
    "Conditioning": "#d62728",
    "Factor": "#9467bd",
}


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_daily_performance():
    fp = CSV_DIR / "daily_performance.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


@st.cache_data(ttl=300)
def load_portfolio_weights():
    fp = CSV_DIR / "portfolio_weights.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


@st.cache_data(ttl=300)
def load_benchmark_weights():
    fp = CSV_DIR / "benchmark_weights.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


@st.cache_data(ttl=300)
def load_feature_importance():
    fp = CSV_DIR / "feature_importance.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


@st.cache_data(ttl=300)
def load_ic_series():
    fp = CSV_DIR / "ic_series.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_style_sector_tilt():
    fp = CSV_DIR / "style_sector_tilt.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_monthly_regime():
    fp = CSV_DIR / "monthly_regime.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


@st.cache_data(ttl=300)
def load_group_attribution():
    fp = CSV_DIR / "group_attribution.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


@st.cache_data(ttl=300)
def load_li_attribution():
    fp = CSV_DIR / "li_attribution.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"])


@st.cache_data(ttl=300)
def load_model_structure():
    fp = CSV_DIR / "model_structure.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


@st.cache_data(ttl=300)
def load_ow_explanations():
    fp = REPORT_DIR / "lightgbm_monthly_ow_explanations.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


@st.cache_data(ttl=300)
def load_stock_scores():
    fp = CSV_DIR / "stock_scores.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"], index_col="date")


@st.cache_data(ttl=300)
def load_stock_shap_attribution():
    fp = CSV_DIR / "stock_shap_attribution.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def compute_metrics(perf: pd.DataFrame) -> dict:
    port = perf["fund_daily_return"].dropna()
    bm = perf["bm_daily_return"].dropna()
    active = perf["active_daily_return"].dropna()
    ann = 252

    ann_ret = port.mean() * ann
    ann_vol = port.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    bm_ret = bm.mean() * ann
    bm_vol = bm.std() * np.sqrt(ann)
    bm_sharpe = bm_ret / bm_vol if bm_vol > 0 else 0

    active_ret = active.mean() * ann
    te = active.std() * np.sqrt(ann)
    ir = active_ret / te if te > 0 else 0

    cum = perf["fund_cumulative"]
    dd = (cum / cum.cummax()) - 1
    max_dd = dd.min()
    win_rate = (active > 0).mean()
    n_years = len(port) / ann
    total_ret = cum.iloc[-1] / cum.iloc[0] - 1
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "bm_ret": bm_ret, "bm_vol": bm_vol, "bm_sharpe": bm_sharpe,
        "active_ret": active_ret, "te": te, "ir": ir,
        "max_dd": max_dd, "win_rate": win_rate, "calmar": calmar,
        "n_years": n_years, "total_ret": total_ret,
        "total_bm_ret": perf["bm_cumulative"].iloc[-1] / perf["bm_cumulative"].iloc[0] - 1,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("AI Signal Portfolio")
    st.sidebar.markdown("---")

    perf = load_daily_performance()
    if perf is not None:
        min_date = perf.index.min().date()
        max_date = perf.index.max().date()
        st.sidebar.markdown(f"**Data:** {min_date} ~ {max_date}")

        date_range = st.sidebar.date_input(
            "Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            start, end = date_range
        else:
            start, end = min_date, max_date
    else:
        start, end = None, None

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Config A (Final)**")
    st.sidebar.markdown("""
    - Rebal: 10-day (bi-weekly)
    - Penalty: 0.3
    - Max Turnover: 15%
    - TE Cap: 12.5%
    - BM Floor: 50%
    - Active Share Cap: 50%
    - Benchmark: MCW
    """)

    if (CSV_DIR / "daily_performance.csv").exists():
        mtime = datetime.fromtimestamp((CSV_DIR / "daily_performance.csv").stat().st_mtime)
        st.sidebar.markdown(f"**Last Update:** {mtime.strftime('%Y-%m-%d %H:%M')}")

    return start, end


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview(start, end):
    st.header("Portfolio Overview")

    perf = load_daily_performance()
    if perf is None:
        st.error("daily_performance.csv not found. Run pipeline first.")
        return

    if start and end:
        mask = (perf.index >= pd.Timestamp(start)) & (perf.index <= pd.Timestamp(end))
        pf = perf[mask].copy()
        pf["fund_cumulative"] = (1 + pf["fund_daily_return"]).cumprod()
        pf["bm_cumulative"] = (1 + pf["bm_daily_return"]).cumprod()
    else:
        pf = perf

    m = compute_metrics(pf)

    # KPI cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Annual Return", f"{m['ann_ret']:.2%}", f"{m['active_ret']:+.2%} vs BM")
    with c2:
        st.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
    with c3:
        st.metric("Active Return", f"{m['active_ret']:.2%}")
    with c4:
        st.metric("Information Ratio", f"{m['ir']:.2f}")
    with c5:
        st.metric("Max Drawdown", f"{m['max_dd']:.2%}")
    with c6:
        ic = load_ic_series()
        avg_ic = ic["IC"].mean() if ic is not None else 0
        st.metric("Avg IC", f"{avg_ic:.4f}")

    st.markdown("---")

    # Cumulative returns + table
    col_l, col_r = st.columns([2, 1])

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pf.index, y=pf["fund_cumulative"],
            name="Fund", line=dict(color=COLORS["fund"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pf.index, y=pf["bm_cumulative"],
            name="Benchmark (MCW)", line=dict(color=COLORS["benchmark"], width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Cumulative Returns: Fund vs Benchmark",
            yaxis_title="Cumulative Return",
            hovermode="x unified", height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Performance Summary")
        summary = pd.DataFrame({
            "Metric": ["Annual Return", "Annual Vol", "Sharpe", "Total Return",
                        "Max Drawdown", "Calmar", "Win Rate", "Period (yrs)"],
            "Fund": [f"{m['ann_ret']:.2%}", f"{m['ann_vol']:.2%}", f"{m['sharpe']:.2f}",
                     f"{m['total_ret']:.2%}", f"{m['max_dd']:.2%}", f"{m['calmar']:.2f}",
                     f"{m['win_rate']:.1%}", f"{m['n_years']:.1f}"],
            "BM": [f"{m['bm_ret']:.2%}", f"{m['bm_vol']:.2%}", f"{m['bm_sharpe']:.2f}",
                   f"{m['total_bm_ret']:.2%}", "-", "-", "-", "-"],
        })
        st.dataframe(summary.set_index("Metric"), use_container_width=True)

        st.markdown("#### Active")
        active_df = pd.DataFrame({
            "": ["Active Return", "Tracking Error", "IR"],
            "Value": [f"{m['active_ret']:.2%}", f"{m['te']:.2%}", f"{m['ir']:.2f}"],
        })
        st.dataframe(active_df.set_index(""), use_container_width=True)

    # Drawdown
    cum = pf["fund_cumulative"]
    dd = (cum / cum.cummax()) - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(214,39,40,0.2)",
        line=dict(color=COLORS["negative"], width=1), name="Drawdown",
    ))
    fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown",
                          yaxis_tickformat=".1%", height=280, showlegend=False)
    st.plotly_chart(fig_dd, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Returns Analysis
# ---------------------------------------------------------------------------
def page_returns_analysis(start, end):
    st.header("Returns Analysis")

    perf = load_daily_performance()
    if perf is None:
        return

    if start and end:
        mask = (perf.index >= pd.Timestamp(start)) & (perf.index <= pd.Timestamp(end))
        perf = perf[mask]

    active = perf["active_daily_return"]

    # Rolling IR
    window = st.slider("Rolling Window (days)", 63, 504, 252, 21)
    rmean = active.rolling(window, min_periods=window // 2).mean() * 252
    rstd = active.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    rir = rmean / rstd.replace(0, np.nan)

    fig_ir = go.Figure()
    fig_ir.add_trace(go.Scatter(
        x=rir.index, y=rir.values, name=f"Rolling IR ({window}d)",
        line=dict(color=COLORS["fund"], width=1.5),
    ))
    fig_ir.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_ir.add_hline(y=0.5, line_dash="dash", line_color="green", opacity=0.3,
                     annotation_text="IR=0.5")
    fig_ir.update_layout(title=f"Rolling IR ({window}d)", yaxis_title="IR", height=400)
    st.plotly_chart(fig_ir, use_container_width=True)

    # Monthly heatmaps
    st.subheader("Monthly Returns Heatmap")
    c1, c2 = st.columns(2)
    for col, (rc, title) in zip([c1, c2], [("fund_daily_return", "Fund"), ("active_daily_return", "Active")]):
        with col:
            rets = perf[rc].copy()
            monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            pivot = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
            hm = pivot.pivot(index="year", columns="month", values="ret")
            hm.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            fig_hm = go.Figure(data=go.Heatmap(
                z=hm.values * 100, x=hm.columns, y=hm.index.astype(str),
                colorscale="RdYlGn", zmid=0,
                text=np.where(np.isnan(hm.values), "", np.char.add(np.char.mod("%.1f", hm.values * 100), "%")),
                texttemplate="%{text}", textfont={"size": 10},
            ))
            fig_hm.update_layout(title=f"{title} Monthly Returns (%)", height=400, yaxis_autorange="reversed")
            st.plotly_chart(fig_hm, use_container_width=True)

    # Distribution
    st.subheader("Active Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=active.values * 100, nbinsx=80,
                                      marker_color=COLORS["fund"], opacity=0.7))
    fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_dist.add_vline(x=active.mean() * 100, line_dash="dash", line_color="green",
                       annotation_text=f"Mean={active.mean()*100:.3f}%")
    fig_dist.update_layout(title="Daily Active Returns (%)", xaxis_title="%", yaxis_title="Freq", height=320)
    st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Portfolio
# ---------------------------------------------------------------------------
def page_portfolio(start, end):
    st.header("Portfolio Composition")

    pw = load_portfolio_weights()
    bw = load_benchmark_weights()
    if pw is None:
        st.error("No weights data.")
        return

    # Latest weights bar
    latest = pw.index[-1]
    lpw = pw.loc[latest].sort_values(ascending=False)
    lbw = bw.loc[latest] if bw is not None and latest in bw.index else pd.Series(1 / len(lpw), index=lpw.index)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=lpw.index, y=lpw.values * 100, name="Fund", marker_color=COLORS["fund"]))
    fig_bar.add_trace(go.Bar(x=lpw.index, y=[lbw.get(t, 0) * 100 for t in lpw.index],
                             name="BM", marker_color=COLORS["benchmark"], opacity=0.6))
    fig_bar.update_layout(title=f"Weights ({latest.strftime('%Y-%m-%d')})",
                           yaxis_title="%", barmode="group", height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Active weights horizontal bar
    st.subheader("Active Weights (Fund - BM)")
    aw = (lpw - lbw.reindex(lpw.index).fillna(0)).sort_values()
    fig_aw = go.Figure()
    fig_aw.add_trace(go.Bar(
        x=aw.values * 100, y=aw.index, orientation="h",
        marker_color=["green" if v > 0 else "red" for v in aw.values],
    ))
    fig_aw.update_layout(title="Active Weights (%)", xaxis_title="%", height=max(400, len(aw) * 18))
    st.plotly_chart(fig_aw, use_container_width=True)

    # Weight evolution
    st.subheader("Weight Evolution")
    top_n = st.slider("Stocks to show", 5, 50, 15)
    top = pw.iloc[-1].sort_values(ascending=False).head(top_n).index.tolist()
    fig_evo = go.Figure()
    for t in top:
        fig_evo.add_trace(go.Scatter(x=pw.index, y=pw[t] * 100, name=t, stackgroup="one"))
    fig_evo.update_layout(title=f"Top {top_n} Weights Over Time", yaxis_title="%",
                           hovermode="x unified", height=500)
    st.plotly_chart(fig_evo, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Sector & Style
# ---------------------------------------------------------------------------
def page_sector_style(start, end):
    st.header("Sector & Style Analysis")

    tilt = load_style_sector_tilt()
    if tilt is None:
        st.error("No sector/style data.")
        return

    if start and end:
        mask = (tilt["date"] >= pd.Timestamp(start)) & (tilt["date"] <= pd.Timestamp(end))
        tilt = tilt[mask]

    # Sector active weights
    st.subheader("Sector Active Weights")
    sec_cols = [c for c in tilt.columns if c.startswith("sector_") and not c.startswith("port_") and not c.startswith("bm_")]
    fig_sec = go.Figure()
    for c in sec_cols:
        fig_sec.add_trace(go.Scatter(x=tilt["date"], y=tilt[c] * 100, name=c.replace("sector_", ""), mode="lines"))
    fig_sec.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig_sec.update_layout(title="Sector Active Weights (%)", yaxis_title="%", hovermode="x unified", height=450)
    st.plotly_chart(fig_sec, use_container_width=True)

    # Sector pie charts
    c1, c2 = st.columns(2)
    latest = tilt.iloc[-1]
    with c1:
        ps = {c.replace("port_sector_", ""): latest[c] for c in tilt.columns if c.startswith("port_sector_")}
        fig_p = go.Figure(data=[go.Pie(labels=list(ps.keys()), values=list(ps.values()), hole=0.4)])
        fig_p.update_layout(title="Fund Sectors (Latest)", height=400)
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        bs = {c.replace("bm_sector_", ""): latest[c] for c in tilt.columns if c.startswith("bm_sector_")}
        fig_b = go.Figure(data=[go.Pie(labels=list(bs.keys()), values=list(bs.values()), hole=0.4)])
        fig_b.update_layout(title="BM Sectors (Latest)", height=400)
        st.plotly_chart(fig_b, use_container_width=True)

    # Style active weights
    st.subheader("Style Active Weights")
    sty_cols = [c for c in tilt.columns if c.startswith("style_") and not c.startswith("port_") and not c.startswith("bm_")]
    fig_sty = go.Figure()
    for c in sty_cols:
        fig_sty.add_trace(go.Scatter(x=tilt["date"], y=tilt[c] * 100, name=c.replace("style_", ""), mode="lines"))
    fig_sty.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
    fig_sty.update_layout(title="Style Active Weights (%)", yaxis_title="%", hovermode="x unified", height=450)
    st.plotly_chart(fig_sty, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model & Signal
# ---------------------------------------------------------------------------
def page_model_signal(start, end):
    st.header("Model & Signal Analysis")

    # IC
    ic_data = load_ic_series()
    if ic_data is not None:
        st.subheader("Information Coefficient (IC)")

        if start and end:
            mask = (ic_data["date"] >= pd.Timestamp(start)) & (ic_data["date"] <= pd.Timestamp(end))
            icf = ic_data[mask]
        else:
            icf = ic_data

        avg_ic = icf["IC"].mean()
        ic_std = icf["IC"].std()
        icir = avg_ic / ic_std if ic_std > 0 else 0
        hit = (icf["IC"] > 0).mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean IC", f"{avg_ic:.4f}")
        c2.metric("IC Std", f"{ic_std:.4f}")
        c3.metric("ICIR", f"{icir:.2f}")
        c4.metric("Hit Rate", f"{hit:.0%}")

        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(
            x=icf["date"], y=icf["IC"],
            marker_color=["green" if v > 0 else "red" for v in icf["IC"]], opacity=0.6, name="IC",
        ))
        ic_rm = icf.set_index("date")["IC"].rolling(20, min_periods=5).mean()
        fig_ic.add_trace(go.Scatter(x=ic_rm.index, y=ic_rm.values, name="MA(20)",
                                      line=dict(color="navy", width=2)))
        fig_ic.add_hline(y=avg_ic, line_dash="dash", line_color="blue",
                         annotation_text=f"Avg={avg_ic:.4f}")
        fig_ic.update_layout(title="IC Time Series", yaxis_title="IC (Spearman)", height=400)
        st.plotly_chart(fig_ic, use_container_width=True)

    # Feature importance
    fi = load_feature_importance()
    if fi is not None:
        st.subheader("Feature Importance")
        c1, c2 = st.columns([2, 1])
        with c1:
            top_n = st.slider("Top N", 10, 50, 30, key="fi")
            fi_t = fi.head(top_n)
            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=fi_t["importance"], y=fi_t["feature"], orientation="h",
                marker_color=[GROUP_COLORS.get(g, "#7f7f7f") for g in fi_t["group"]],
                text=fi_t["group"], textposition="inside", textfont_size=9,
            ))
            fig_fi.update_layout(title=f"Top {top_n} Features", xaxis_title="Importance (Gain)",
                                  height=max(400, top_n * 22), yaxis_autorange="reversed")
            st.plotly_chart(fig_fi, use_container_width=True)
        with c2:
            gp = fi.groupby("group")["importance"].sum()
            gp_pct = gp / gp.sum()
            fig_gp = go.Figure(data=[go.Pie(labels=gp_pct.index, values=gp_pct.values,
                                              marker_colors=[GROUP_COLORS.get(g, "#7f7f7f") for g in gp_pct.index],
                                              hole=0.4)])
            fig_gp.update_layout(title="Group Importance", height=400)
            st.plotly_chart(fig_gp, use_container_width=True)

            gc = fi.groupby("group")["feature"].count()
            st.markdown("#### Features per Group")
            for g, c in gc.items():
                st.write(f"- **{g}**: {c}")

    # Group attribution
    ga = load_group_attribution()
    if ga is not None:
        st.subheader("Group Attribution Over Time")
        fig_ga = go.Figure()
        for col in ga.columns:
            fig_ga.add_trace(go.Scatter(x=ga.index, y=ga[col], name=col, stackgroup="one"))
        fig_ga.update_layout(title="Feature Group SHAP", yaxis_title="Share",
                              yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig_ga, use_container_width=True)

    # Li attribution
    li = load_li_attribution()
    if li is not None:
        st.subheader("Linear vs Nonlinear (Li et al.)")
        fig_li = go.Figure()
        fig_li.add_trace(go.Bar(x=li["date"], y=li["linear_ratio"], name="Linear", marker_color="#1f77b4"))
        fig_li.add_trace(go.Bar(x=li["date"], y=li["marginal_nl_ratio"], name="Marginal NL", marker_color="#ff7f0e"))
        fig_li.add_trace(go.Bar(x=li["date"], y=li["interaction_ratio"], name="Interaction", marker_color="#2ca02c"))
        fig_li.update_layout(barmode="stack", title="3-Component Attribution",
                              yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig_li, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Linear", f"{li['linear_ratio'].mean():.2%}")
        c2.metric("Avg Marginal NL", f"{li['marginal_nl_ratio'].mean():.2%}")
        c3.metric("Avg Interaction", f"{li['interaction_ratio'].mean():.2%}")


# ---------------------------------------------------------------------------
# Page: Regime & Explanations
# ---------------------------------------------------------------------------
def page_regime(start, end):
    st.header("Monthly Regime & Position Explanations")

    regime = load_monthly_regime()
    ow_expl = load_ow_explanations()

    if regime is not None:
        st.subheader("Market Regime Timeline")
        dir_colors = {"Bullish": "green", "Bearish": "red", "Sideways": "gray"}
        fig_r = go.Figure()
        for _, row in regime.iterrows():
            fig_r.add_trace(go.Bar(
                x=[row["year_month"]], y=[row["ew_return_21d"] * 100],
                marker_color=dir_colors.get(row["market_direction"], "gray"),
                showlegend=False,
                hovertemplate=(
                    f"Month: {row['year_month']}<br>"
                    f"Dir: {row['market_direction']}<br>"
                    f"Vol: {row['volatility_regime']}<br>"
                    f"Rotation: {row['sector_rotation']}<br>"
                    f"Return: {row['ew_return_21d']:.2%}<br>"
                    f"Active Share: {row['total_active_share']:.2%}"
                ),
            ))
        fig_r.update_layout(title="Monthly Direction (21d EW Return %)", yaxis_title="%", height=350)
        st.plotly_chart(fig_r, use_container_width=True)

        st.dataframe(regime[["year_month", "market_direction", "volatility_regime",
                             "sector_rotation", "n_ow_stocks", "n_uw_stocks", "total_active_share"]],
                     use_container_width=True, hide_index=True)

    if ow_expl is not None:
        st.subheader("Position Explanations")
        months = ow_expl["year_month"].tolist()
        sel = st.selectbox("Month", months, index=len(months) - 1)
        row = ow_expl[ow_expl["year_month"] == sel].iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### Regime: {row['regime_label']}")
            st.markdown(f"**Date:** {row['rebal_date']}")
            st.markdown(f"**Reason:** {row['regime_reason']}")
            st.markdown(f"**Dominant:** {row['dominant_category_effects']}")
        with c2:
            st.markdown("#### Tilts")
            st.markdown(f"**Sector:** {row['sector_tilt']}")
            st.markdown(f"**Style:** {row['style_tilt']}")
            st.markdown(f"**OW:** {row['n_ow_stocks']} | **UW:** {row['n_uw_stocks']}")

        st.markdown("---")
        st.markdown("#### Overweight")
        for item in str(row.get("top_ow_details", "")).split(" | "):
            if item.strip():
                st.markdown(f"- {item}")

        st.markdown("#### Underweight")
        for item in str(row.get("top_uw_details", "")).split(" | "):
            if item.strip():
                st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# Page: Model Structure
# ---------------------------------------------------------------------------
def page_model_structure(start, end):
    st.header("Model Structure & Training")

    ms = load_model_structure()
    if ms is None:
        st.error("No model data.")
        return

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Trees per Retrain", "Avg Tree Depth"),
                        shared_xaxes=True)
    fig.add_trace(go.Bar(x=ms["retrain_date"], y=ms["n_trees"], name="Trees",
                         marker_color=COLORS["fund"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=ms["retrain_date"], y=ms["avg_tree_depth"], name="Depth",
                             mode="lines+markers", line=dict(color=COLORS["benchmark"])), row=2, col=1)
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Retrains", len(ms))
    c2.metric("Avg Trees", f"{ms['n_trees'].mean():.0f}")
    c3.metric("Avg Features Used", f"{ms['n_unique_features_used'].mean():.0f}")
    c4.metric("Avg Depth", f"{ms['avg_tree_depth'].mean():.1f}")

    st.subheader("Retrain History")
    st.dataframe(ms, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Stock Score & Attribution
# ---------------------------------------------------------------------------
TICKER_META = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication",
    "AMZN": "Consumer Disc.", "META": "Communication", "NVDA": "Semiconductors",
    "AVGO": "Semiconductors", "MU": "Semiconductors", "AMD": "Semiconductors",
    "000660": "Semiconductors", "005930": "Semiconductors",
    "TSLA": "Consumer Disc.", "PLTR": "Technology", "CRM": "Technology",
    "NFLX": "Communication", "GEV": "Industrials", "VRT": "Industrials",
    "BE": "Industrials", "LITE": "Technology",
    "UNH": "Healthcare", "LLY": "Healthcare", "ISRG": "Healthcare",
    "ABBV": "Healthcare", "REGN": "Healthcare",
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BLK": "Financials", "SPGI": "Financials", "GS": "Financials",
    "COST": "Consumer Staples", "HD": "Consumer Disc.", "PG": "Consumer Staples",
    "MCD": "Consumer Disc.", "WMT": "Consumer Staples",
    "CAT": "Industrials", "HON": "Industrials", "DE": "Industrials",
    "UNP": "Industrials", "LMT": "Industrials", "ETN": "Industrials",
    "XOM": "Energy", "LNG": "Energy", "FCX": "Materials", "LIN": "Materials",
    "NEE": "Utilities", "AMT": "Real Estate", "EQIX": "Real Estate",
    "TMUS": "Communication", "PLD": "Real Estate",
}

SHAP_GROUP_COLORS = {
    "Accounting": "#1f77b4",
    "Price": "#ff7f0e",
    "Sellside": "#2ca02c",
    "Conditioning": "#d62728",
    "Factor": "#9467bd",
}


def page_stock_score_attribution(start, end):
    st.header("Stock Score & Attribution")

    pw = load_portfolio_weights()
    bw = load_benchmark_weights()
    scores = load_stock_scores()
    shap_attr = load_stock_shap_attribution()

    if pw is None:
        st.error("No portfolio weights data. Run pipeline first.")
        return

    # --- 12-month cutoff ---
    latest_date = pw.index.max()
    cutoff_12m = latest_date - pd.DateOffset(months=12)

    # --- Stock selector ---
    all_tickers = sorted(pw.columns.tolist())

    # Default: top 5 by latest active weight
    if bw is not None and latest_date in bw.index:
        aw = (pw.loc[latest_date] - bw.loc[latest_date]).sort_values(ascending=False)
        default_tickers = aw.head(5).index.tolist()
    else:
        default_tickers = all_tickers[:5]

    selected = st.multiselect(
        "Select Stocks", all_tickers, default=default_tickers, key="score_stocks"
    )
    if not selected:
        st.info("Select at least one stock.")
        return

    # =====================================================================
    # Section 1: 12-month Weight Changes
    # =====================================================================
    st.subheader("12-Month Weight History")

    pw_12m = pw[pw.index >= cutoff_12m][selected]
    bw_12m = bw[bw.index >= cutoff_12m][selected] if bw is not None else None

    col1, col2 = st.columns(2)

    with col1:
        fig_pw = go.Figure()
        for t in selected:
            fig_pw.add_trace(go.Scatter(
                x=pw_12m.index, y=pw_12m[t] * 100,
                name=t, mode="lines+markers", marker=dict(size=4),
            ))
        fig_pw.update_layout(
            title="Portfolio Weight (%)", yaxis_title="%",
            hovermode="x unified", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_pw, use_container_width=True)

    with col2:
        if bw_12m is not None:
            fig_aw = go.Figure()
            aw_12m = (pw_12m - bw_12m.reindex(pw_12m.index).fillna(0)) * 100
            for t in selected:
                fig_aw.add_trace(go.Scatter(
                    x=aw_12m.index, y=aw_12m[t],
                    name=t, mode="lines+markers", marker=dict(size=4),
                ))
            fig_aw.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
            fig_aw.update_layout(
                title="Active Weight (Fund - BM) %", yaxis_title="%",
                hovermode="x unified", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_aw, use_container_width=True)
        else:
            st.info("Benchmark weights not available.")

    # =====================================================================
    # Section 2: Score History
    # =====================================================================
    st.subheader("Score History (Prediction Z-Score)")

    if scores is not None:
        scores_12m = scores[scores.index >= cutoff_12m]
        avail_tickers = [t for t in selected if t in scores_12m.columns]

        if avail_tickers:
            fig_score = go.Figure()
            for t in avail_tickers:
                s = scores_12m[t].dropna()
                fig_score.add_trace(go.Scatter(
                    x=s.index, y=s.values,
                    name=t, mode="lines+markers", marker=dict(size=4),
                ))
            fig_score.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
            fig_score.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3,
                                annotation_text="z=0.5")
            fig_score.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3,
                                annotation_text="z=-0.5")
            fig_score.update_layout(
                title="Model Prediction Score (Z-Score)",
                yaxis_title="Z-Score", hovermode="x unified", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_score, use_container_width=True)

            # Score summary table
            st.markdown("#### Latest Score Snapshot")
            latest_scores_date = scores_12m.index.max()
            latest_scores = scores_12m.loc[latest_scores_date, avail_tickers]
            latest_pw_val = pw.loc[pw.index.max(), avail_tickers] * 100
            latest_bw_val = bw.loc[bw.index.max(), avail_tickers] * 100 if bw is not None else pd.Series(0, index=avail_tickers)
            latest_aw_val = latest_pw_val - latest_bw_val.reindex(latest_pw_val.index).fillna(0)

            snap_df = pd.DataFrame({
                "Ticker": avail_tickers,
                "Sector": [TICKER_META.get(t, "-") for t in avail_tickers],
                "Score (z)": [f"{latest_scores.get(t, 0):.2f}" for t in avail_tickers],
                "Port Wt%": [f"{latest_pw_val.get(t, 0):.1f}" for t in avail_tickers],
                "BM Wt%": [f"{latest_bw_val.get(t, 0):.1f}" for t in avail_tickers],
                "Active Wt%": [f"{latest_aw_val.get(t, 0):.1f}" for t in avail_tickers],
            })

            def _score_signal(z_str):
                z = float(z_str)
                if z > 1.0:
                    return "Very Strong +"
                elif z > 0.5:
                    return "Strong +"
                elif z > 0:
                    return "Mild +"
                elif z > -0.5:
                    return "Mild -"
                elif z > -1.0:
                    return "Strong -"
                else:
                    return "Very Strong -"

            snap_df["Signal"] = snap_df["Score (z)"].apply(_score_signal)
            st.dataframe(snap_df.set_index("Ticker"), use_container_width=True)
        else:
            st.info("No score data for selected stocks.")
    else:
        st.warning("stock_scores.csv not found. Re-run pipeline to generate.")

    # =====================================================================
    # Section 3: Factor Attribution (SHAP Breakdown)
    # =====================================================================
    st.subheader("Score Factor Attribution (SHAP)")

    if shap_attr is not None:
        shap_dates = sorted(shap_attr["date"].unique())
        shap_avail_tickers = sorted(shap_attr["ticker"].unique())
        sel_shap_tickers = [t for t in selected if t in shap_avail_tickers]

        if sel_shap_tickers and shap_dates:
            groups = [c for c in shap_attr.columns if c not in ("date", "ticker", "total")]

            # --- Per-stock waterfall for latest SHAP date ---
            sel_shap_date = st.selectbox(
                "SHAP Date", shap_dates,
                index=len(shap_dates) - 1, key="shap_date",
                format_func=lambda d: d.strftime("%Y-%m-%d"),
            )

            shap_snap = shap_attr[shap_attr["date"] == sel_shap_date].set_index("ticker")

            n_cols = min(len(sel_shap_tickers), 3)
            cols = st.columns(n_cols)

            for idx, ticker in enumerate(sel_shap_tickers):
                with cols[idx % n_cols]:
                    if ticker not in shap_snap.index:
                        st.caption(f"{ticker}: No SHAP data")
                        continue

                    row = shap_snap.loc[ticker]
                    vals = {g: row.get(g, 0) for g in groups}
                    total_val = row.get("total", sum(vals.values()))

                    # Sort by absolute contribution
                    sorted_groups = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)

                    fig_wf = go.Figure(go.Waterfall(
                        orientation="v",
                        x=[g for g, _ in sorted_groups] + ["Total"],
                        y=[v for _, v in sorted_groups] + [0],
                        measure=["relative"] * len(sorted_groups) + ["total"],
                        connector={"line": {"color": "rgba(63,63,63,0.3)"}},
                        increasing={"marker": {"color": "#2ca02c"}},
                        decreasing={"marker": {"color": "#d62728"}},
                        totals={"marker": {"color": "#1f77b4"}},
                        textposition="outside",
                        text=[f"{v:+.3f}" for _, v in sorted_groups] + [f"{total_val:.3f}"],
                        textfont={"size": 10},
                    ))
                    fig_wf.update_layout(
                        title=f"{ticker} (z={total_val:.2f})",
                        yaxis_title="SHAP Contribution",
                        height=380, showlegend=False,
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

            # --- SHAP history for selected stocks ---
            if len(shap_dates) > 1 and sel_shap_tickers:
                st.markdown("#### Factor Attribution Over Time")

                sel_history_ticker = st.selectbox(
                    "Stock for SHAP History", sel_shap_tickers, key="shap_history_ticker"
                )

                t_shap = shap_attr[shap_attr["ticker"] == sel_history_ticker].set_index("date")

                if len(t_shap) > 0:
                    fig_shap_ts = go.Figure()
                    for g in groups:
                        if g in t_shap.columns:
                            fig_shap_ts.add_trace(go.Bar(
                                x=t_shap.index, y=t_shap[g],
                                name=g,
                                marker_color=SHAP_GROUP_COLORS.get(g, "#7f7f7f"),
                            ))
                    if "total" in t_shap.columns:
                        fig_shap_ts.add_trace(go.Scatter(
                            x=t_shap.index, y=t_shap["total"],
                            name="Total Score", mode="lines+markers",
                            line=dict(color="black", width=2),
                        ))
                    fig_shap_ts.update_layout(
                        barmode="relative",
                        title=f"{sel_history_ticker} - Factor Contributions Over Time",
                        yaxis_title="SHAP Value", height=420,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_shap_ts, use_container_width=True)

            # --- All-stock SHAP heatmap for latest date ---
            st.markdown("#### All-Stock Score Decomposition")
            shap_latest = shap_attr[shap_attr["date"] == shap_dates[-1]].set_index("ticker")

            if len(shap_latest) > 0:
                hm_data = shap_latest[groups].copy()
                hm_data = hm_data.sort_values(
                    by=groups[0] if groups else hm_data.columns[0],
                    key=lambda x: shap_latest.get("total", x),
                    ascending=False,
                )
                if "total" in shap_latest.columns:
                    hm_data = hm_data.loc[shap_latest["total"].sort_values(ascending=False).index]

                fig_hm = go.Figure(data=go.Heatmap(
                    z=hm_data.values,
                    x=hm_data.columns.tolist(),
                    y=hm_data.index.tolist(),
                    colorscale="RdBu", zmid=0,
                    text=np.char.mod("%.3f", hm_data.values),
                    texttemplate="%{text}", textfont={"size": 9},
                ))
                fig_hm.update_layout(
                    title=f"SHAP by Group ({shap_dates[-1].strftime('%Y-%m-%d')})",
                    height=max(400, len(hm_data) * 22),
                    yaxis_autorange="reversed",
                    xaxis_side="top",
                )
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("No SHAP data for selected stocks.")
    else:
        st.warning("stock_shap_attribution.csv not found. Re-run pipeline to generate.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start, end = sidebar()

    if not (CSV_DIR / "daily_performance.csv").exists():
        st.error("No CSV data. Run pipeline first:")
        st.code("python main.py --data_path ./data/ai_signal_data.xlsx --output_dir ./outputs/")
        return

    pages = {
        "Overview": page_overview,
        "Returns Analysis": page_returns_analysis,
        "Portfolio": page_portfolio,
        "Stock Score & Attribution": page_stock_score_attribution,
        "Sector & Style": page_sector_style,
        "Model & Signal": page_model_signal,
        "Regime & Explanations": page_regime,
        "Model Structure": page_model_structure,
    }

    page = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[page](start, end)


if __name__ == "__main__":
    main()
