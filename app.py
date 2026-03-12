"""
AI Signal Portfolio Dashboard (Streamlit)
실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="AI Signal Portfolio", layout="wide", page_icon="📊")

CSV_DIR = Path("./outputs/csv")
REPORT_DIR = Path("./outputs/reports")


# ─── Data Loading ──────────────────────────────────────────

@st.cache_data
def load_csv(name: str, base_dir: Path = CSV_DIR) -> pd.DataFrame:
    path = base_dir / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_all():
    return {
        "perf": load_csv("daily_performance.csv"),
        "weights": load_csv("portfolio_weights.csv"),
        "bm_weights": load_csv("benchmark_weights.csv"),
        "importance": load_csv("feature_importance.csv"),
        "group_attr": load_csv("group_attribution.csv"),
        "li_attr": load_csv("li_attribution.csv"),
        "ic": load_csv("ic_series.csv"),
        "model": load_csv("model_structure.csv"),
        "regime": load_csv("monthly_regime.csv"),
        "style_sector": load_csv("style_sector_tilt.csv"),
        "ow_explain": load_csv("lightgbm_monthly_ow_explanations.csv", REPORT_DIR),
    }


# ─── Sidebar & Navigation ─────────────────────────────────

st.sidebar.title("📊 AI Signal Portfolio")
page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "📈 Performance",
    "⚖️ Portfolio Weights",
    "🎯 Style / Sector Tilt",
    "🌲 Model Structure",
    "🔬 Attribution",
    "📅 Monthly Regime & OW Report",
])

data = load_all()


# ─── Page: Overview ────────────────────────────────────────

if page == "🏠 Overview":
    st.title("AI Signal Portfolio — Overview")

    perf = data["perf"]
    if perf.empty:
        st.warning("CSV 파일이 없습니다. 먼저 `python export_csv.py`를 실행하세요.")
        st.stop()

    # KPI Cards
    ann_ret = perf["fund_daily_return"].mean() * 252
    ann_vol = perf["fund_daily_return"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    active_ret = perf["active_daily_return"].mean() * 252
    te = perf["active_daily_return"].std() * np.sqrt(252)
    ir = active_ret / te if te > 0 else 0
    max_dd = (perf["fund_cumulative"] / perf["fund_cumulative"].cummax() - 1).min()

    ic_df = data["ic"]
    avg_ic = ic_df["IC"].mean() if not ic_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Active Return", f"{active_ret:.2%}")
    c3.metric("Information Ratio", f"{ir:.2f}")
    c4.metric("Avg IC", f"{avg_ic:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Annual Return", f"{ann_ret:.2%}")
    c6.metric("Annual Vol", f"{ann_vol:.2%}")
    c7.metric("Tracking Error", f"{te:.2%}")
    c8.metric("Max Drawdown", f"{max_dd:.2%}")

    st.markdown("---")

    # Cumulative Returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf["date"], y=perf["fund_cumulative"],
        name="Fund", line=dict(width=2, color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=perf["date"], y=perf["bm_cumulative"],
        name="Benchmark (MktCap)", line=dict(width=2, color="#ff7f0e", dash="dot"),
    ))
    fig.update_layout(
        title="Cumulative Returns: Fund vs Benchmark",
        yaxis_title="Cumulative Return",
        height=450, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Li et al. Attribution Summary
    li = data["li_attr"]
    if not li.empty:
        avg_lin = li["linear_ratio"].mean()
        avg_mnl = li["marginal_nl_ratio"].mean()
        avg_int = li["interaction_ratio"].mean()

        st.subheader("Li et al. 3-Component Attribution (Average)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Linear", f"{avg_lin:.1%}")
        col2.metric("Marginal Non-linear", f"{avg_mnl:.1%}")
        col3.metric("Interaction", f"{avg_int:.1%}")

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Linear", "Marginal Non-linear", "Interaction"],
            values=[avg_lin, avg_mnl, avg_int],
            hole=0.4,
            marker_colors=["#2ecc71", "#3498db", "#e74c3c"],
        )])
        fig_pie.update_layout(height=300, title="Non-linear Decomposition")
        st.plotly_chart(fig_pie, use_container_width=True)


# ─── Page: Performance ────────────────────────────────────

elif page == "📈 Performance":
    st.title("Performance Analysis")
    perf = data["perf"]
    if perf.empty:
        st.warning("daily_performance.csv not found.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Cumulative", "Drawdown", "Monthly Heatmap", "IC Series"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["fund_cumulative"], name="Fund"))
        fig.add_trace(go.Scatter(x=perf["date"], y=perf["bm_cumulative"], name="BM"))
        active_cum = (1 + perf["active_daily_return"]).cumprod()
        fig.add_trace(go.Scatter(x=perf["date"], y=active_cum, name="Active (Cumulative)"))
        fig.update_layout(height=500, template="plotly_white", title="Cumulative Returns")
        st.plotly_chart(fig, use_container_width=True)

        # Rolling IR
        active = perf.set_index("date")["active_daily_return"]
        roll_mean = active.rolling(252, min_periods=126).mean() * 252
        roll_std = active.rolling(252, min_periods=126).std() * np.sqrt(252)
        roll_ir = roll_mean / roll_std.replace(0, np.nan)

        fig_ir = go.Figure()
        fig_ir.add_trace(go.Scatter(x=roll_ir.index, y=roll_ir.values, name="Rolling IR (252d)"))
        fig_ir.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_ir.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="IR=1.0")
        fig_ir.update_layout(height=350, template="plotly_white", title="Rolling Information Ratio")
        st.plotly_chart(fig_ir, use_container_width=True)

    with tab2:
        dd = perf["fund_cumulative"] / perf["fund_cumulative"].cummax() - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=perf["date"], y=dd, fill="tozeroy",
            fillcolor="rgba(255,0,0,0.2)", line=dict(color="red"),
            name="Drawdown"
        ))
        fig_dd.update_layout(height=400, template="plotly_white", title="Strategy Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab3:
        monthly = perf.set_index("date")["fund_daily_return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }).pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig_hm = px.imshow(
            pivot.values, x=pivot.columns, y=pivot.index.astype(str),
            color_continuous_scale="RdYlGn", zmin=-0.10, zmax=0.10,
            text_auto=".1%", aspect="auto",
        )
        fig_hm.update_layout(height=500, title="Monthly Returns Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab4:
        ic_df = data["ic"]
        if not ic_df.empty:
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(x=ic_df["date"], y=ic_df["IC"], name="IC", opacity=0.6))
            avg_ic = ic_df["IC"].mean()
            fig_ic.add_hline(y=avg_ic, line_dash="dash", line_color="red",
                           annotation_text=f"Mean IC={avg_ic:.4f}")
            fig_ic.update_layout(height=400, template="plotly_white",
                               title="Information Coefficient Over Time")
            st.plotly_chart(fig_ic, use_container_width=True)


# ─── Page: Portfolio Weights ───────────────────────────────

elif page == "⚖️ Portfolio Weights":
    st.title("Portfolio Weights — Recent 6 Months")

    weights = data["weights"]
    bm_weights = data["bm_weights"]
    if weights.empty:
        st.warning("portfolio_weights.csv not found.")
        st.stop()

    weights["date"] = pd.to_datetime(weights["date"])
    tickers = [c for c in weights.columns if c != "date"]

    last_date = weights["date"].max()
    cutoff = last_date - pd.DateOffset(months=6)
    recent = weights[weights["date"] >= cutoff].copy()

    st.subheader(f"Recent Weights ({cutoff.strftime('%Y-%m')} ~ {last_date.strftime('%Y-%m')})")

    # Stacked Area Chart
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent[ticker],
            stackgroup="one", name=ticker,
        ))
    fig.update_layout(
        height=500, template="plotly_white",
        title="Portfolio Weight Allocation (Stacked)",
        yaxis_title="Weight", yaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Active Weights (vs BM)
    if not bm_weights.empty:
        bm_weights["date"] = pd.to_datetime(bm_weights["date"])
        recent_bm = bm_weights[bm_weights["date"] >= cutoff].copy()

        st.subheader("Active Weights (Portfolio - Benchmark)")

        rebal_dates = sorted(recent["date"].unique(), reverse=True)
        selected_date = st.selectbox(
            "Select Rebalance Date",
            [d.strftime("%Y-%m-%d") for d in rebal_dates],
        )
        sel_ts = pd.Timestamp(selected_date)

        port_w = recent[recent["date"] == sel_ts][tickers].iloc[0]
        bm_match = recent_bm[recent_bm["date"] == sel_ts]
        if not bm_match.empty:
            bm_w = bm_match[tickers].iloc[0]
        else:
            bm_w = pd.Series(1.0 / len(tickers), index=tickers)

        active_w = port_w - bm_w
        active_w = active_w.sort_values()

        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in active_w.values]
        fig_active = go.Figure(go.Bar(
            x=active_w.values, y=active_w.index,
            orientation="h", marker_color=colors,
            text=[f"{v:+.1%}" for v in active_w.values],
            textposition="outside",
        ))
        fig_active.update_layout(
            height=max(400, len(tickers) * 25),
            template="plotly_white",
            title=f"Active Weights @ {selected_date}",
            xaxis_title="Active Weight", xaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_active, use_container_width=True)

    # Weight History Table
    st.subheader("Weight History (Table)")
    display_df = recent.set_index("date")[tickers].T
    display_df.columns = [c.strftime("%Y-%m-%d") for c in display_df.columns]
    st.dataframe(display_df.style.format("{:.2%}").background_gradient(cmap="YlOrRd", axis=1), height=600)


# ─── Page: Style / Sector Tilt ─────────────────────────────

elif page == "🎯 Style / Sector Tilt":
    st.title("Style / Sector Active Tilt Analysis")

    ss = data["style_sector"]
    if ss.empty:
        st.warning("style_sector_tilt.csv not found. Re-run `python export_csv.py`.")
        st.stop()

    ss["date"] = pd.to_datetime(ss["date"])

    tab_sector, tab_style, tab_detail = st.tabs(["Sector Tilt", "Style Tilt", "Detailed Table"])

    # ── Sector Active Weight over time ──
    with tab_sector:
        sector_cols = [c for c in ss.columns if c.startswith("sector_")]
        sector_names = [c.replace("sector_", "") for c in sector_cols]

        st.subheader("Sector Active Weight (Portfolio - BM) Over Time")

        # 최근 기간 필터
        date_range = st.select_slider(
            "Period",
            options=["Full", "3Y", "1Y", "6M"],
            value="1Y",
            key="sector_period",
        )
        if date_range == "6M":
            ss_filt = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(months=6)]
        elif date_range == "1Y":
            ss_filt = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(years=1)]
        elif date_range == "3Y":
            ss_filt = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(years=3)]
        else:
            ss_filt = ss

        fig_sec = go.Figure()
        colors_sec = px.colors.qualitative.Set2
        for i, (col, name) in enumerate(zip(sector_cols, sector_names)):
            fig_sec.add_trace(go.Scatter(
                x=ss_filt["date"], y=ss_filt[col],
                name=name, mode="lines",
                line=dict(color=colors_sec[i % len(colors_sec)]),
            ))
        fig_sec.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sec.update_layout(
            height=500, template="plotly_white",
            title="Sector Active Weight Over Time",
            yaxis_title="Active Weight", yaxis=dict(tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

        # 최근 스냅샷 바 차트
        latest = ss_filt.iloc[-1]
        sec_vals = {name: latest[col] for col, name in zip(sector_cols, sector_names)}
        sec_sorted = sorted(sec_vals.items(), key=lambda x: x[1], reverse=True)

        fig_bar = go.Figure(go.Bar(
            x=[v for _, v in sec_sorted],
            y=[n for n, _ in sec_sorted],
            orientation="h",
            marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for _, v in sec_sorted],
            text=[f"{v:+.1%}" for _, v in sec_sorted],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=400, template="plotly_white",
            title=f"Sector Tilt Snapshot @ {latest['date'].strftime('%Y-%m-%d')}",
            xaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # 지배적 섹터 변화
        if "dominant_ow_sector" in ss.columns:
            st.subheader("Dominant OW / UW Sector History")
            dom_df = ss_filt[["date", "dominant_ow_sector", "dominant_uw_sector"]].copy()
            dom_df = dom_df.set_index("date")
            st.dataframe(dom_df.tail(30), use_container_width=True)

    # ── Style Active Weight over time ──
    with tab_style:
        style_cols = [c for c in ss.columns if c.startswith("style_")]
        style_names = [c.replace("style_", "") for c in style_cols]

        st.subheader("Style Active Weight (Portfolio - BM) Over Time")

        date_range_sty = st.select_slider(
            "Period",
            options=["Full", "3Y", "1Y", "6M"],
            value="1Y",
            key="style_period",
        )
        if date_range_sty == "6M":
            ss_filt_s = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(months=6)]
        elif date_range_sty == "1Y":
            ss_filt_s = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(years=1)]
        elif date_range_sty == "3Y":
            ss_filt_s = ss[ss["date"] >= ss["date"].max() - pd.DateOffset(years=3)]
        else:
            ss_filt_s = ss

        fig_sty = go.Figure()
        colors_sty = px.colors.qualitative.Pastel
        for i, (col, name) in enumerate(zip(style_cols, style_names)):
            fig_sty.add_trace(go.Scatter(
                x=ss_filt_s["date"], y=ss_filt_s[col],
                name=name, mode="lines",
                line=dict(color=colors_sty[i % len(colors_sty)]),
            ))
        fig_sty.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sty.update_layout(
            height=500, template="plotly_white",
            title="Style Active Weight Over Time",
            yaxis_title="Active Weight", yaxis=dict(tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sty, use_container_width=True)

        # 최근 스냅샷 바 차트
        latest_s = ss_filt_s.iloc[-1]
        sty_vals = {name: latest_s[col] for col, name in zip(style_cols, style_names)}
        sty_sorted = sorted(sty_vals.items(), key=lambda x: x[1], reverse=True)

        fig_bar_s = go.Figure(go.Bar(
            x=[v for _, v in sty_sorted],
            y=[n for n, _ in sty_sorted],
            orientation="h",
            marker_color=["#3498db" if v >= 0 else "#e67e22" for _, v in sty_sorted],
            text=[f"{v:+.1%}" for _, v in sty_sorted],
            textposition="outside",
        ))
        fig_bar_s.update_layout(
            height=350, template="plotly_white",
            title=f"Style Tilt Snapshot @ {latest_s['date'].strftime('%Y-%m-%d')}",
            xaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_bar_s, use_container_width=True)

        # 지배적 스타일 변화
        if "dominant_ow_style" in ss.columns:
            st.subheader("Dominant OW / UW Style History")
            dom_sty_df = ss_filt_s[["date", "dominant_ow_style", "dominant_uw_style"]].copy()
            dom_sty_df = dom_sty_df.set_index("date")
            st.dataframe(dom_sty_df.tail(30), use_container_width=True)

    # ── Sector/Style Allocation Breakdown ──
    with tab_detail:
        st.subheader("Sector Allocation: Portfolio vs Benchmark")

        # 최근 리밸런싱
        latest_d = ss.iloc[-1]
        port_sec = {name: latest_d.get(f"port_sector_{name}", 0) for name in sector_names}
        bm_sec = {name: latest_d.get(f"bm_sector_{name}", 0) for name in sector_names}

        sec_comp = pd.DataFrame({
            "Sector": sector_names,
            "Portfolio": [port_sec[n] for n in sector_names],
            "Benchmark": [bm_sec[n] for n in sector_names],
            "Active": [port_sec[n] - bm_sec[n] for n in sector_names],
        }).sort_values("Active", ascending=False)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=sec_comp["Sector"], y=sec_comp["Portfolio"],
            name="Portfolio", marker_color="#3498db",
        ))
        fig_comp.add_trace(go.Bar(
            x=sec_comp["Sector"], y=sec_comp["Benchmark"],
            name="Benchmark", marker_color="#95a5a6",
        ))
        fig_comp.update_layout(
            barmode="group", height=450, template="plotly_white",
            title=f"Sector Allocation: Portfolio vs BM @ {latest_d['date'].strftime('%Y-%m-%d')}",
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Style Allocation: Portfolio vs Benchmark")
        port_sty = {name: latest_d.get(f"port_style_{name}", 0) for name in style_names}
        bm_sty = {name: latest_d.get(f"bm_style_{name}", 0) for name in style_names}

        sty_comp = pd.DataFrame({
            "Style": style_names,
            "Portfolio": [port_sty[n] for n in style_names],
            "Benchmark": [bm_sty[n] for n in style_names],
            "Active": [port_sty[n] - bm_sty[n] for n in style_names],
        }).sort_values("Active", ascending=False)

        fig_comp_s = go.Figure()
        fig_comp_s.add_trace(go.Bar(
            x=sty_comp["Style"], y=sty_comp["Portfolio"],
            name="Portfolio", marker_color="#9b59b6",
        ))
        fig_comp_s.add_trace(go.Bar(
            x=sty_comp["Style"], y=sty_comp["Benchmark"],
            name="Benchmark", marker_color="#95a5a6",
        ))
        fig_comp_s.update_layout(
            barmode="group", height=400, template="plotly_white",
            title=f"Style Allocation: Portfolio vs BM @ {latest_d['date'].strftime('%Y-%m-%d')}",
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_comp_s, use_container_width=True)

        # Full tilt data table
        st.subheader("Full Tilt Data")
        st.dataframe(ss, use_container_width=True, height=400)


# ─── Page: Model Structure ─────────────────────────────────

elif page == "🌲 Model Structure":
    st.title("LightGBM Model Structure")

    model_df = data["model"]
    if model_df.empty:
        st.warning("model_structure.csv not found.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Retrain Count", len(model_df))
    c2.metric("Avg Trees", f"{model_df['n_trees'].mean():.0f}")
    c3.metric("Avg Depth", f"{model_df['avg_tree_depth'].mean():.1f}")

    if "retrain_date" in model_df.columns:
        model_df["retrain_date"] = pd.to_datetime(model_df["retrain_date"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=["Number of Trees (Best Iteration)", "Unique Features Used"])

        fig.add_trace(go.Scatter(
            x=model_df["retrain_date"],
            y=model_df["best_iteration"] if "best_iteration" in model_df.columns else model_df["n_trees"],
            mode="lines+markers", name="Best Iteration",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=model_df["retrain_date"],
            y=model_df["n_unique_features_used"],
            mode="lines+markers", name="Unique Features",
        ), row=2, col=1)

        fig.update_layout(height=600, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    imp_df = data["importance"]
    if not imp_df.empty:
        st.subheader("Feature Importance (Top 30)")
        top30 = imp_df.head(30)
        fig_imp = px.bar(
            top30, x="importance", y="feature", color="group",
            orientation="h", height=700,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            title="Top 30 Feature Importance (LightGBM Gain)",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Model Detail Table")
    st.dataframe(model_df, use_container_width=True)


# ─── Page: Attribution ─────────────────────────────────────

elif page == "🔬 Attribution":
    st.title("Attribution Analysis")

    tab1, tab2, tab3 = st.tabs(["Group Contribution", "Li et al. 3-Component", "Feature Importance"])

    with tab1:
        ga = data["group_attr"]
        if not ga.empty:
            ga["date"] = pd.to_datetime(ga["date"])
            groups = [c for c in ga.columns if c != "date"]

            fig = go.Figure()
            for g in groups:
                fig.add_trace(go.Scatter(
                    x=ga["date"], y=ga[g], stackgroup="one", name=g,
                ))
            fig.update_layout(
                height=500, template="plotly_white",
                title="Feature Group Contribution Over Time (Stacked)",
                yaxis_title="Contribution Share", yaxis=dict(tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)

            avg = ga[groups].mean()
            fig_avg = px.bar(
                x=avg.index, y=avg.values,
                labels={"x": "Group", "y": "Avg Contribution"},
                color=avg.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_avg.update_layout(
                height=350, template="plotly_white",
                title="Average Group Contribution",
                yaxis=dict(tickformat=".0%"), showlegend=False,
            )
            st.plotly_chart(fig_avg, use_container_width=True)

    with tab2:
        li = data["li_attr"]
        if not li.empty:
            li["date"] = pd.to_datetime(li["date"])

            fig_li = go.Figure()
            fig_li.add_trace(go.Bar(x=li["date"], y=li["linear_ratio"], name="Linear", marker_color="#2ecc71"))
            fig_li.add_trace(go.Bar(x=li["date"], y=li["marginal_nl_ratio"], name="Marginal NL", marker_color="#3498db"))
            fig_li.add_trace(go.Bar(x=li["date"], y=li["interaction_ratio"], name="Interaction", marker_color="#e74c3c"))
            fig_li.update_layout(
                barmode="stack", height=450, template="plotly_white",
                title="Li et al. 3-Component Attribution Over Time",
                yaxis_title="Ratio", yaxis=dict(tickformat=".0%"),
            )
            st.plotly_chart(fig_li, use_container_width=True)

            group_cols_linear = [c for c in li.columns if c.startswith("linear_") and c != "linear_ratio"]
            group_cols_mnl = [c for c in li.columns if c.startswith("marginal_nl_") and c != "marginal_nl_ratio"]
            group_cols_int = [c for c in li.columns if c.startswith("interaction_") and c != "interaction_ratio"]

            if group_cols_linear:
                st.subheader("Group-Level Li Decomposition (Average)")
                group_names = [c.replace("linear_", "") for c in group_cols_linear]
                avg_lin = [li[c].mean() for c in group_cols_linear]
                avg_mnl = [li[c].mean() for c in group_cols_mnl] if group_cols_mnl else [0] * len(group_names)
                avg_int = [li[c].mean() for c in group_cols_int] if group_cols_int else [0] * len(group_names)

                fig_grp = go.Figure()
                fig_grp.add_trace(go.Bar(x=group_names, y=avg_lin, name="Linear", marker_color="#2ecc71"))
                fig_grp.add_trace(go.Bar(x=group_names, y=avg_mnl, name="Marginal NL", marker_color="#3498db"))
                fig_grp.add_trace(go.Bar(x=group_names, y=avg_int, name="Interaction", marker_color="#e74c3c"))
                fig_grp.update_layout(
                    barmode="stack", height=400, template="plotly_white",
                    yaxis=dict(tickformat=".0%"),
                )
                st.plotly_chart(fig_grp, use_container_width=True)

    with tab3:
        imp = data["importance"]
        if not imp.empty:
            n_show = st.slider("Number of features to show", 10, 100, 30)
            top_n = imp.head(n_show)
            fig_imp = px.bar(
                top_n, x="importance", y="feature", color="group",
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                height=max(400, n_show * 22), template="plotly_white",
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ─── Page: Monthly Regime & OW Report ─────────────────────

elif page == "📅 Monthly Regime & OW Report":
    st.title("Monthly Regime & OW/UW Explanation")

    regime = data["regime"]
    ow_explain = data["ow_explain"]

    if regime.empty and ow_explain.empty:
        st.warning("monthly_regime.csv or ow_explanations.csv not found.")
        st.stop()

    tab_regime, tab_ow_report = st.tabs(["Monthly Regime", "Detailed OW Report"])

    # ── Tab 1: Monthly Regime (existing) ──
    with tab_regime:
        if not regime.empty:
            for _, row in regime.iterrows():
                month = row["year_month"]
                with st.expander(f"📅 {month} — {row['market_direction']} / {row['volatility_regime']} / {row['sector_rotation']}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Market", row["market_direction"])
                    c2.metric("Volatility", row["volatility_regime"])
                    c3.metric("Rotation", row["sector_rotation"])
                    c4.metric("Active Share", f"{row['total_active_share']:.1%}")

                    st.markdown(f"**21d EW Return**: {row['ew_return_21d']:.2%} | **Vol (Ann)**: {row['vol_21d_ann']:.2%}")
                    st.markdown(f"**OW Stocks**: {row['n_ow_stocks']} | **UW Stocks**: {row['n_uw_stocks']}")

                    st.markdown("##### 🟢 Top Overweight Positions")
                    ow_parts = str(row["top_ow_stocks"]).split(" | ")
                    for part in ow_parts:
                        if part and part != "nan":
                            st.markdown(f"- {part}")

                    st.markdown("##### 🔴 Top Underweight Positions")
                    uw_parts = str(row["top_uw_stocks"]).split(" | ")
                    for part in uw_parts:
                        if part and part != "nan":
                            st.markdown(f"- {part}")

            # Regime Timeline
            st.subheader("Regime Timeline")
            regime_df = regime.copy()
            color_map = {"Bullish": "#2ecc71", "Bearish": "#e74c3c", "Sideways": "#95a5a6"}
            colors = [color_map.get(d, "#95a5a6") for d in regime_df["market_direction"]]

            fig_regime = go.Figure()
            fig_regime.add_trace(go.Bar(
                x=regime_df["year_month"],
                y=regime_df["ew_return_21d"],
                marker_color=colors,
                text=[f"{r['market_direction']}" for _, r in regime_df.iterrows()],
                textposition="outside",
            ))
            fig_regime.update_layout(
                height=350, template="plotly_white",
                title="Monthly 21d EW Return (Color = Market Direction)",
                yaxis=dict(tickformat=".1%"),
            )
            st.plotly_chart(fig_regime, use_container_width=True)

    # ── Tab 2: Detailed OW Report (NEW) ──
    with tab_ow_report:
        if ow_explain.empty:
            st.warning("lightgbm_monthly_ow_explanations.csv not found.")
        else:
            st.subheader("Detailed Monthly OW/UW Analysis with Sector & Style Context")

            for _, row in ow_explain.iterrows():
                month = row["year_month"]
                regime_label = row.get("regime_label", "N/A")

                with st.expander(f"📅 {month} — {regime_label}", expanded=True):
                    # Regime 정보
                    st.markdown(f"**Regime**: {regime_label}")
                    st.markdown(f"**Regime Reason**: {row.get('regime_reason', 'N/A')}")
                    st.markdown(f"**Dominant Category Effects**: {row.get('dominant_category_effects', 'N/A')}")

                    st.markdown("---")

                    # 섹터/스타일 Tilt
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("##### 🏢 Sector Tilt")
                        sector_tilt = str(row.get("sector_tilt", ""))
                        if sector_tilt and sector_tilt != "nan":
                            for part in sector_tilt.split(" | "):
                                if part:
                                    # 양수는 녹색, 음수는 빨간색
                                    if "+" in part:
                                        st.markdown(f"- :green[{part}]")
                                    elif "-" in part:
                                        st.markdown(f"- :red[{part}]")
                                    else:
                                        st.markdown(f"- {part}")
                        else:
                            st.markdown("_No significant sector tilt_")

                    with col_b:
                        st.markdown("##### 🎨 Style Tilt")
                        style_tilt = str(row.get("style_tilt", ""))
                        if style_tilt and style_tilt != "nan":
                            for part in style_tilt.split(" | "):
                                if part:
                                    if "+" in part:
                                        st.markdown(f"- :green[{part}]")
                                    elif "-" in part:
                                        st.markdown(f"- :red[{part}]")
                                    else:
                                        st.markdown(f"- {part}")
                        else:
                            st.markdown("_No significant style tilt_")

                    st.markdown("---")
                    st.markdown(f"**OW Stocks**: {row.get('n_ow_stocks', 0)} | **UW Stocks**: {row.get('n_uw_stocks', 0)}")

                    # OW 종목 상세
                    st.markdown("##### 🟢 Top OW Positions (with Sector/Style/Signal)")
                    ow_details = str(row.get("top_ow_details", ""))
                    if ow_details and ow_details != "nan":
                        for part in ow_details.split(" | "):
                            if part:
                                st.markdown(f"- `{part}`")

                    # UW 종목 상세
                    st.markdown("##### 🔴 Top UW Positions (with Sector/Style/Signal)")
                    uw_details = str(row.get("top_uw_details", ""))
                    if uw_details and uw_details != "nan":
                        for part in uw_details.split(" | "):
                            if part:
                                st.markdown(f"- `{part}`")

            # Summary table
            st.subheader("Full OW Explanations Data")
            st.dataframe(ow_explain, use_container_width=True, height=400)


# ─── Footer ───────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**: `outputs/csv/` + `outputs/reports/`")
st.sidebar.markdown("Run `python export_csv.py` to generate data.")
