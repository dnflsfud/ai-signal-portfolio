"""Mobile-optimized Streamlit dashboard for the cc2_harness portfolio.

Designed for phone-first viewing of the baseline_v4 backtest result.

Local run:
    streamlit run streamlit_mobile.py

Phone access on the same Wi-Fi:
    streamlit run streamlit_mobile.py --server.address 0.0.0.0 --server.port 8501
    Then open  http://<your-laptop-IP>:8501  on the phone.

Streamlit Cloud (https://streamlit.io/cloud):
    1. Push this file + outputs/baseline_v4/dashboard_data.pkl to a GitHub repo.
    2. Connect the repo on Streamlit Cloud.
    3. Set Main file path = streamlit_mobile.py.
    4. Optionally set a password via Settings → Secrets:
           password = "your-strong-password"

Data source priority:
    1. outputs/baseline_v4/dashboard_data.pkl  (precomputed, ~3MB — recommended)
    2. outputs/baseline_v4/backtest_result.pkl (~65MB raw, fallback for local dev)

Generate the precomputed file via:
    python scripts/build_dashboard_data.py
"""
from __future__ import annotations

import hmac
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (mobile-first)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="cc2 Portfolio",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 0.6rem; padding-right: 0.6rem;}
      [data-testid="stMetricValue"] {font-size: 1.4rem;}
      [data-testid="stMetricLabel"] {font-size: 0.8rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.25rem; flex-wrap: wrap;}
      .stTabs [data-baseweb="tab"] {padding: 0.4rem 0.6rem; font-size: 0.85rem;}
      .stDataFrame {font-size: 0.85rem;}
      h1 {font-size: 1.5rem !important;}
      h2 {font-size: 1.2rem !important;}
      h3 {font-size: 1.05rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Optional password gate (Streamlit Cloud / public exposure)
# ---------------------------------------------------------------------------
def check_password() -> bool:
    """If `password` is set in st.secrets, require it; else allow free access.

    Use Streamlit Cloud → Settings → Secrets to set:
        password = "your-strong-password"
    """
    expected = None
    try:
        expected = st.secrets.get("password")  # type: ignore[union-attr]
    except FileNotFoundError:
        expected = None

    if not expected:
        return True  # no password configured → open access

    if st.session_state.get("auth_ok"):
        return True

    st.title("🔒 cc2 Portfolio")
    st.caption("Password required.")
    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw, str(expected)):
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("틀린 비밀번호.")
    return False


if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RUN_DIR = Path("outputs/baseline_v4")
DASHBOARD_FILE = RUN_DIR / "dashboard_data.pkl"
RAW_FILE = RUN_DIR / "backtest_result.pkl"
DATA_FILE = Path(r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx")


@st.cache_resource(show_spinner="대시보드 데이터 로딩 중...")
def load_data() -> dict:
    """Load precomputed dashboard package, or build it on the fly from raw."""
    if DASHBOARD_FILE.exists():
        with DASHBOARD_FILE.open("rb") as f:
            return pickle.load(f)

    # Fallback for local dev only — re-build from raw on the fly.
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Neither {DASHBOARD_FILE} nor {RAW_FILE} exists. "
            "Run scripts/build_dashboard_data.py first."
        )

    st.warning(
        f"`{DASHBOARD_FILE.name}` not found, falling back to raw "
        f"`{RAW_FILE.name}` ({RAW_FILE.stat().st_size/1024**2:.0f}MB). "
        "For Streamlit Cloud, run `python scripts/build_dashboard_data.py` "
        "and commit the result."
    )

    # Lazy import to avoid forcing src/ deps when only the precomputed file is shipped
    sys.path.insert(0, str(Path(__file__).parent))
    from scripts.build_dashboard_data import build  # type: ignore

    build(RUN_DIR, DATA_FILE, DASHBOARD_FILE)
    with DASHBOARD_FILE.open("rb") as f:
        return pickle.load(f)


PKG = load_data()
port: pd.Series = PKG["portfolio_returns"].dropna()
bm: pd.Series = PKG["benchmark_returns"].dropna()
common = port.index.intersection(bm.index)
port = port.loc[common]
bm = bm.loc[common]
W_daily: pd.DataFrame = PKG["daily_weights"]
PW: pd.DataFrame = PKG["portfolio_weights"]
bm_full: pd.DataFrame = PKG["bm_weights_daily"]
bm_at_reb: pd.DataFrame = PKG["bm_weights_at_rebalances"]
metrics: dict = PKG["metrics"]
fi_pct: pd.Series = PKG["feature_importance_pct"]
ic_df: pd.DataFrame = PKG["ic_table"]
group_pnl: pd.DataFrame = PKG["group_pnl"]
score_breakdowns: dict[pd.Timestamp, pd.DataFrame] = PKG["score_breakdowns"]
rebal_predictions: dict[pd.Timestamp, pd.Series] = PKG["rebal_predictions"]
GROUPS_DICT: dict[str, list[str]] = PKG["feature_groups"]


def feature_to_bucket(f: str) -> str:
    for name, feats in GROUPS_DICT.items():
        if f in feats:
            return name
    return "Other"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 cc2 Portfolio")
st.caption(
    f"`{PKG['label']}` · {PKG['period']['start']} → {PKG['period']['end']} · "
    f"{len(port):,} days"
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ov, tab_yr, tab_pr, tab_rb, tab_ft, tab_ow = st.tabs(
    ["개요", "연도별", "기간수익", "리밸런싱", "피처분석", "OW점수"]
)

# ---- TAB 1: Overview ------------------------------------------------------
with tab_ov:
    c1, c2 = st.columns(2)
    c1.metric("Annual Return", f"{metrics['annual_return']*100:.2f}%")
    c2.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")

    c1, c2 = st.columns(2)
    c1.metric("Active Return", f"{metrics['active_return']*100:+.2f}%")
    c2.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")

    c1, c2 = st.columns(2)
    c1.metric("Tracking Error", f"{metrics['tracking_error']*100:.2f}%")
    c2.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")

    c1, c2 = st.columns(2)
    c1.metric("Annual Vol", f"{metrics['annual_vol']*100:.2f}%")
    c2.metric("Turnover (2-way)", f"{metrics['avg_annual_turnover']*100:.0f}%")

    st.markdown("### 누적 수익률")
    cum_p = (1 + port).cumprod() - 1
    cum_b = (1 + bm).cumprod() - 1
    cum_df = pd.DataFrame({"Portfolio": cum_p * 100, "Benchmark": cum_b * 100})
    fig = px.line(cum_df, height=320)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        yaxis_title="cumulative %", xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Drawdown")
    rolling_max = cum_p.cummax()
    drawdown = (cum_p - rolling_max) / (1 + rolling_max) * 100
    fig_dd = go.Figure(go.Scatter(
        x=drawdown.index, y=drawdown.values, fill="tozeroy",
        line=dict(color="#FF6B6B"), name="DD",
    ))
    fig_dd.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="%", xaxis_title="",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ---- TAB 2: Yearly --------------------------------------------------------
with tab_yr:
    st.markdown("### 연도별 성과")
    rows = []
    for y in sorted(port.index.year.unique()):
        p = port[port.index.year == y]
        b = bm[bm.index.year == y]
        if len(p) < 5:
            continue
        cum_p = (1 + p).prod() - 1
        cum_b = (1 + b).prod() - 1
        vol = p.std() * np.sqrt(252)
        sh = (p.mean() * 252) / vol if vol > 0 else np.nan
        te = (p - b).std() * np.sqrt(252)
        ar = (p.mean() - b.mean()) * 252
        ir = ar / te if te > 0 else np.nan
        rows.append({
            "Year": int(y), "Port%": cum_p * 100, "BM%": cum_b * 100,
            "Active%": (cum_p - cum_b) * 100, "Vol%": vol * 100,
            "Sharpe": sh, "TE%": te * 100, "IR": ir,
        })
    yr_df = pd.DataFrame(rows).set_index("Year")
    st.dataframe(
        yr_df.style.format("{:+.2f}", subset=["Port%","BM%","Active%","Sharpe","IR"])
                    .format("{:.2f}", subset=["Vol%","TE%"])
                    .background_gradient(subset=["Active%","IR"], cmap="RdYlGn", vmin=-15, vmax=15),
        use_container_width=True,
        height=min(400, 40 + 35 * len(yr_df)),
    )

    st.markdown("### Active Return by Year")
    fig = px.bar(yr_df.reset_index(), x="Year", y="Active%",
                 color="Active%", color_continuous_scale="RdYlGn",
                 color_continuous_midpoint=0, height=280)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---- TAB 3: Period Returns ------------------------------------------------
with tab_pr:
    end = port.index[-1]
    st.markdown(f"### 기간 수익률 — as of {end.strftime('%Y-%m-%d')}")

    specs = [
        ("1d", 1, False), ("1w", 5, False), ("1m", 21, False), ("3m", 63, False),
        ("6m", 126, True), ("1y", 252, True), ("3y", 756, True), ("5y", 1260, True),
        ("MTD", None, False), ("YTD", None, True), ("ITD", None, True),
    ]
    rows = []
    for label, n, with_ann in specs:
        if label == "MTD":
            start = end.replace(day=1); p = port[port.index >= start]; b = bm[bm.index >= start]
        elif label == "YTD":
            start = pd.Timestamp(end.year, 1, 1); p = port[port.index >= start]; b = bm[bm.index >= start]
        elif label == "ITD":
            p = port; b = bm
        else:
            p = port.tail(n); b = bm.tail(n)
        if len(p) < 1:
            continue
        cum_p = (1 + p).prod() - 1
        cum_b = (1 + b).prod() - 1
        yrs = len(p) / 252
        if with_ann and yrs >= 0.4:
            ann = (1 + cum_p) ** (1 / yrs) - 1
            vol = p.std() * np.sqrt(252)
            sh = (p.mean() * 252) / vol if vol > 0 else np.nan
            rows.append({"Period": label, "Days": len(p), "Port%": cum_p*100, "BM%": cum_b*100,
                         "Active%": (cum_p-cum_b)*100, "AnnVol%": vol*100, "Sharpe": sh, "AnnRet%": ann*100})
        else:
            rows.append({"Period": label, "Days": len(p), "Port%": cum_p*100, "BM%": cum_b*100,
                         "Active%": (cum_p-cum_b)*100, "AnnVol%": np.nan, "Sharpe": np.nan, "AnnRet%": np.nan})
    p_df = pd.DataFrame(rows)

    short_df = p_df.iloc[:4].set_index("Period")[["Days", "Port%", "BM%", "Active%"]]
    st.markdown("#### 단기 (vol/sharpe 미산출)")
    st.dataframe(
        short_df.style.format("{:+.2f}", subset=["Port%","BM%","Active%"])
                       .background_gradient(subset=["Active%"], cmap="RdYlGn", vmin=-3, vmax=3),
        use_container_width=True, height=180,
    )

    long_df = p_df.iloc[4:8].set_index("Period")[["Days","Port%","BM%","Active%","AnnVol%","Sharpe","AnnRet%"]]
    st.markdown("#### 장기 (≥6m, 변동성·샤프)")
    st.dataframe(
        long_df.style.format("{:+.2f}", subset=["Port%","BM%","Active%","Sharpe","AnnRet%"])
                      .format("{:.2f}", subset=["AnnVol%"])
                      .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-1, vmax=3),
        use_container_width=True, height=210,
    )

    cal_df = p_df.iloc[8:].set_index("Period")
    show_cols = ["Days","Port%","BM%","Active%"] + (["AnnVol%","Sharpe"] if cal_df["AnnVol%"].notna().any() else [])
    st.markdown("#### MTD / YTD / ITD")
    st.dataframe(
        cal_df[show_cols].style.format("{:+.2f}", subset=[c for c in show_cols if c not in ["Days","AnnVol%"]])
                                .format("{:.2f}", subset=["AnnVol%"] if "AnnVol%" in show_cols else []),
        use_container_width=True, height=160,
    )


# ---- TAB 4: Rebalances (newest first) ------------------------------------
with tab_rb:
    st.markdown("### 리밸런싱별 OW / UW (최신순)")
    n_show = st.slider("표시할 리밸런싱 개수", min_value=4, max_value=20, value=8, step=1)
    top_n = st.slider("OW/UW 종목 수", min_value=5, max_value=15, value=7, step=1)

    reb_dates = PW.index.sort_values(ascending=False)[:n_show]
    for d in reb_dates:
        w_p = PW.loc[d]
        b_p = bm_at_reb.loc[d]
        active = (w_p - b_p).sort_values(ascending=False)
        ow = active.head(top_n)
        uw = active.tail(top_n)[::-1]
        with st.expander(f"📅 {d.strftime('%Y-%m-%d')}", expanded=(d == reb_dates[0])):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔺 Overweight**")
                ow_df = pd.DataFrame({
                    "Tkr": ow.index,
                    "Port%": (w_p.reindex(ow.index) * 100).round(2).values,
                    "BM%": (b_p.reindex(ow.index) * 100).round(2).values,
                    "Act%": (ow * 100).round(2).values,
                })
                st.dataframe(ow_df, hide_index=True, use_container_width=True,
                             height=min(280, 40 + 35 * len(ow_df)))
            with c2:
                st.markdown("**🔻 Underweight**")
                uw_df = pd.DataFrame({
                    "Tkr": uw.index,
                    "Port%": (w_p.reindex(uw.index) * 100).round(2).values,
                    "BM%": (b_p.reindex(uw.index) * 100).round(2).values,
                    "Act%": (uw * 100).round(2).values,
                })
                st.dataframe(uw_df, hide_index=True, use_container_width=True,
                             height=min(280, 40 + 35 * len(uw_df)))


# ---- TAB 5: Feature Analysis ---------------------------------------------
with tab_ft:
    st.markdown("### 피처 그룹 영향력 (LightGBM Gain)")

    # Aggregate by bucket
    feat_names = list(fi_pct.index)
    bucket_rows = []
    covered = set()
    for gname, feats in GROUPS_DICT.items():
        present = [f for f in feats if f in feat_names]
        gain = fi_pct.reindex(present).sum()
        bucket_rows.append({"Bucket": gname, "#Feat": len(present), "Gain%": gain})
        covered.update(present)
    other = [f for f in feat_names if f not in covered]
    if other:
        bucket_rows.append({"Bucket": "Other", "#Feat": len(other), "Gain%": fi_pct.reindex(other).sum()})
    bucket_df = pd.DataFrame(bucket_rows).set_index("Bucket").sort_values("Gain%", ascending=False)

    fig = px.bar(bucket_df.reset_index(), x="Bucket", y="Gain%",
                 color="Gain%", color_continuous_scale="Viridis", height=280)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        bucket_df.style.format("{:.2f}", subset=["Gain%"]),
        use_container_width=True, height=min(340, 40 + 35 * len(bucket_df)),
    )

    st.markdown("### Top 15 개별 피처")
    top15 = fi_pct.sort_values(ascending=False).head(15)
    top15_df = pd.DataFrame({
        "Feature": top15.index,
        "Bucket": [feature_to_bucket(f) for f in top15.index],
        "Gain%": top15.values,
    })
    fig2 = px.bar(top15_df.iloc[::-1], x="Gain%", y="Feature", color="Bucket",
                  orientation="h", height=420)
    fig2.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### IC (Spearman)")
    show_top_ic = ic_df.head(15)[["feature", "bucket", "IC_mean", "IC_IR"]].set_index("feature")
    show_bot_ic = ic_df.tail(8)[["feature", "bucket", "IC_mean", "IC_IR"]].set_index("feature")

    st.markdown("**🟢 Top 15 (양의 예측력)**")
    st.dataframe(
        show_top_ic.style.format("{:.4f}", subset=["IC_mean","IC_IR"])
                          .background_gradient(subset=["IC_mean"], cmap="Greens"),
        use_container_width=True, height=560,
    )
    st.markdown("**🔴 Bottom 8 (음의 IC = reversal)**")
    st.dataframe(
        show_bot_ic.style.format("{:.4f}", subset=["IC_mean","IC_IR"])
                          .background_gradient(subset=["IC_mean"], cmap="Reds_r"),
        use_container_width=True, height=320,
    )

    st.markdown("### 그룹별 L/S 시뮬레이션 (연환산)")
    st.dataframe(
        group_pnl.style.format("{:+.2f}", subset=["AnnRet_%","AnnVol_%","Sharpe"])
                        .background_gradient(subset=["Sharpe"], cmap="RdYlGn", vmin=-2, vmax=4),
        use_container_width=True, height=min(340, 40 + 35 * len(group_pnl)),
    )
    st.caption("각 그룹 시그널을 동일가중 합쳐 Top30%-Bottom30% L/S 일간 수익률로 환산")


# ---- TAB 6: OW score breakdown -------------------------------------------
with tab_ow:
    st.markdown("### OW Top10 점수 Breakdown")
    reb_dates_all = PW.index.sort_values(ascending=False)
    pick = st.selectbox("리밸런싱 시점", options=reb_dates_all,
                        format_func=lambda d: d.strftime("%Y-%m-%d"))

    w_p = PW.loc[pick]
    b_p = bm_at_reb.loc[pick]
    active = (w_p - b_p).sort_values(ascending=False)
    ow_top = active.head(10).index.tolist()
    uw_top = active.tail(10).index.tolist()

    G = score_breakdowns.get(pick, pd.DataFrame())
    preds = rebal_predictions.get(pick, pd.Series(dtype=float))

    def build_table(tkrs):
        data = []
        for t in tkrs:
            row = {"Tkr": t, "Act%": active[t] * 100, "Score": preds.get(t, np.nan)}
            for g in G.columns:
                row[g] = G.loc[t, g] if t in G.index else np.nan
            data.append(row)
        return pd.DataFrame(data).set_index("Tkr")

    if not G.empty:
        st.markdown("#### 🔺 OW Top 10")
        ow_table = build_table(ow_top)
        st.dataframe(
            ow_table.style.format("{:+.2f}", subset=["Act%","Score"] + list(G.columns))
                           .background_gradient(subset=list(G.columns), cmap="RdYlGn", vmin=-2, vmax=2),
            use_container_width=True,
            height=min(420, 40 + 35 * len(ow_table)),
        )
        st.markdown("#### 🔻 UW Top 10")
        uw_table = build_table(uw_top[::-1])
        st.dataframe(
            uw_table.style.format("{:+.2f}", subset=["Act%","Score"] + list(G.columns))
                           .background_gradient(subset=list(G.columns), cmap="RdYlGn", vmin=-2, vmax=2),
            use_container_width=True,
            height=min(420, 40 + 35 * len(uw_table)),
        )

        st.markdown("#### 🏆 종합 스코어 Top 30")
        if not preds.dropna().empty:
            top30_tkrs = preds.dropna().nlargest(30).index.tolist()
            top30_table = build_table(top30_tkrs)
            top30_table.insert(0, "Rank", range(1, len(top30_table) + 1))
            st.dataframe(
                top30_table.style
                    .format({"Rank": "{:d}"})
                    .format("{:+.2f}", subset=["Act%", "Score"] + list(G.columns))
                    .background_gradient(subset=["Score"] + list(G.columns),
                                         cmap="RdYlGn", vmin=-2, vmax=2),
                use_container_width=True,
                height=min(1100, 40 + 35 * len(top30_table)),
            )
        else:
            st.info("이 시점의 composite prediction 데이터가 없습니다.")

        st.caption(
            "각 그룹 z-score는 해당 시점 universe 평균 대비 표준편차 단위. "
            "Score는 모델 예측 (bps, 20일 잔차수익률 단위)."
        )
    else:
        st.info("이 리밸런싱 날짜의 score breakdown 데이터가 없습니다.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"📦 `{DASHBOARD_FILE.name if DASHBOARD_FILE.exists() else RAW_FILE.name}` "
    f"({(DASHBOARD_FILE.stat().st_size if DASHBOARD_FILE.exists() else RAW_FILE.stat().st_size)/1024**2:.1f} MB)  ·  "
    f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
)
