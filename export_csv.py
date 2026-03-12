"""
백테스트 결과를 CSV로 내보내기.
outputs/csv/ 디렉토리에 다음 파일 생성:
  1. daily_performance.csv      - 펀드/BM 일별 수익률 및 누적 수익률
  2. portfolio_weights.csv      - 리밸런싱별 종목 투자 비중
  3. benchmark_weights.csv      - 벤치마크(시총) 비중
  4. feature_importance.csv     - 피처별 중요도 + 그룹
  5. group_attribution.csv      - 그룹별 SHAP 기여도 시계열
  6. li_attribution.csv         - Li et al. 3-component 시계열
  7. ic_series.csv              - IC 시계열
  8. model_structure.csv        - LightGBM 트리 구조 요약
  9. monthly_regime.csv         - 월별 국면 + OW 종목 + 설명
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.backtest import run_backtest, BacktestResult
from src.attribution import run_attribution, explain_period


# ─── Ticker Metadata: Sector / Style / Sub-Industry ─────────────────
TICKER_META = {
    # Mega-Cap Tech
    "AAPL":   {"sector": "Technology",       "style": "Quality Growth",   "sub": "Consumer Electronics"},
    "MSFT":   {"sector": "Technology",       "style": "Quality Growth",   "sub": "Enterprise Software"},
    "GOOGL":  {"sector": "Communication",    "style": "Quality Growth",   "sub": "Digital Advertising"},
    "AMZN":   {"sector": "Consumer Disc.",   "style": "Growth",           "sub": "E-Commerce / Cloud"},
    "META":   {"sector": "Communication",    "style": "Quality Growth",   "sub": "Social Media"},
    # Semiconductors
    "NVDA":   {"sector": "Semiconductors",   "style": "Growth",           "sub": "AI / GPU"},
    "AVGO":   {"sector": "Semiconductors",   "style": "GARP",             "sub": "Networking Chips"},
    "MU":     {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Memory"},
    "AMD":    {"sector": "Semiconductors",   "style": "Growth",           "sub": "CPU / GPU"},
    "000660": {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Memory (KR)"},
    "005930": {"sector": "Semiconductors",   "style": "Cyclical",         "sub": "Foundry / Memory (KR)"},
    # Growth / Platform
    "TSLA":   {"sector": "Consumer Disc.",   "style": "Growth",           "sub": "EV / Energy Storage"},
    "PLTR":   {"sector": "Technology",       "style": "Growth",           "sub": "Data Analytics / AI"},
    "CRM":    {"sector": "Technology",       "style": "Growth",           "sub": "CRM / Cloud SaaS"},
    "NFLX":   {"sector": "Communication",    "style": "Growth",           "sub": "Streaming"},
    # Power / Energy Infra
    "GEV":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Power Equipment"},
    "VRT":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Data Center Cooling"},
    "BE":     {"sector": "Industrials",      "style": "Growth",           "sub": "Fuel Cells"},
    "LITE":   {"sector": "Technology",       "style": "Cyclical",         "sub": "Photonics / Fiber"},
    # Healthcare
    "UNH":    {"sector": "Healthcare",       "style": "Quality",          "sub": "Managed Care"},
    "LLY":    {"sector": "Healthcare",       "style": "Growth",           "sub": "Pharma / GLP-1"},
    "ISRG":   {"sector": "Healthcare",       "style": "Growth",           "sub": "Surgical Robotics"},
    "ABBV":   {"sector": "Healthcare",       "style": "Value",            "sub": "Pharma / Biotech"},
    "REGN":   {"sector": "Healthcare",       "style": "GARP",             "sub": "Biotech"},
    # Financials
    "JPM":    {"sector": "Financials",       "style": "Value",            "sub": "Banking"},
    "V":      {"sector": "Financials",       "style": "Quality Growth",   "sub": "Payments"},
    "MA":     {"sector": "Financials",       "style": "Quality Growth",   "sub": "Payments"},
    "BLK":    {"sector": "Financials",       "style": "Quality",          "sub": "Asset Management"},
    "SPGI":   {"sector": "Financials",       "style": "Quality",          "sub": "Data / Rating"},
    "GS":     {"sector": "Financials",       "style": "Value",            "sub": "Investment Banking"},
    # Consumer
    "COST":   {"sector": "Consumer Staples", "style": "Quality Growth",   "sub": "Warehouse Retail"},
    "HD":     {"sector": "Consumer Disc.",   "style": "Quality",          "sub": "Home Improvement"},
    "PG":     {"sector": "Consumer Staples", "style": "Defensive",        "sub": "Household Products"},
    "MCD":    {"sector": "Consumer Disc.",   "style": "Defensive",        "sub": "QSR"},
    "WMT":    {"sector": "Consumer Staples", "style": "Defensive",        "sub": "Discount Retail"},
    # Industrials / Defense
    "CAT":    {"sector": "Industrials",      "style": "Cyclical",         "sub": "Heavy Equipment"},
    "HON":    {"sector": "Industrials",      "style": "Quality",          "sub": "Diversified Industrials"},
    "DE":     {"sector": "Industrials",      "style": "Cyclical",         "sub": "Agriculture Equipment"},
    "UNP":    {"sector": "Industrials",      "style": "Quality",          "sub": "Railroads"},
    "LMT":    {"sector": "Industrials",      "style": "Defensive",        "sub": "Defense / Aerospace"},
    "ETN":    {"sector": "Industrials",      "style": "Quality Growth",   "sub": "Electrical Equipment"},
    # Energy / Materials / Utilities
    "XOM":    {"sector": "Energy",           "style": "Value",            "sub": "Oil Major"},
    "LNG":    {"sector": "Energy",           "style": "Value",            "sub": "LNG / Natural Gas"},
    "FCX":    {"sector": "Materials",        "style": "Cyclical",         "sub": "Copper Mining"},
    "LIN":    {"sector": "Materials",        "style": "Quality",          "sub": "Industrial Gas"},
    "NEE":    {"sector": "Utilities",        "style": "Defensive",        "sub": "Renewables / Utilities"},
    # Real Estate / Infra / Telecom
    "AMT":    {"sector": "Real Estate",      "style": "Defensive",        "sub": "Tower REIT"},
    "EQIX":   {"sector": "Real Estate",      "style": "Growth",           "sub": "Data Center REIT"},
    "TMUS":   {"sector": "Communication",    "style": "GARP",             "sub": "Wireless Telecom"},
    "PLD":    {"sector": "Real Estate",      "style": "Quality",          "sub": "Logistics REIT"},
}


def export_daily_performance(result: BacktestResult, csv_dir: Path):
    """1. 펀드/BM 일별 수익률."""
    df = pd.DataFrame({
        "fund_daily_return": result.portfolio_returns,
        "bm_daily_return": result.benchmark_returns,
        "active_daily_return": result.active_returns,
        "fund_cumulative": result.cumulative_returns,
        "bm_cumulative": result.cumulative_benchmark,
    })
    df.index.name = "date"
    df.to_csv(csv_dir / "daily_performance.csv")
    print(f"  [1] daily_performance.csv ({len(df)} rows)")


def export_portfolio_weights(result: BacktestResult, csv_dir: Path):
    """2. 리밸런싱별 종목 투자 비중."""
    if not result.portfolio_weights:
        return
    weights_df = pd.DataFrame(result.portfolio_weights).T
    weights_df.index.name = "date"
    weights_df = weights_df.round(6)
    weights_df.to_csv(csv_dir / "portfolio_weights.csv")
    print(f"  [2] portfolio_weights.csv ({len(weights_df)} rebalances)")


def export_benchmark_weights(result: BacktestResult, data: UniverseData, csv_dir: Path):
    """3. 벤치마크(시총) 비중."""
    tickers = [t for t in TICKERS if t in data.returns.columns]
    mktcap = data.market_cap

    rebal_dates = sorted(result.portfolio_weights.keys()) if result.portfolio_weights else []
    bm_weights = {}
    for d in rebal_dates:
        mc = mktcap.loc[d, tickers]
        mc_sum = mc.sum()
        if mc_sum > 0:
            bm_weights[d] = mc / mc_sum
        else:
            bm_weights[d] = pd.Series(1.0 / len(tickers), index=tickers)

    bm_df = pd.DataFrame(bm_weights).T
    bm_df.index.name = "date"
    bm_df = bm_df.round(6)
    bm_df.to_csv(csv_dir / "benchmark_weights.csv")
    print(f"  [3] benchmark_weights.csv ({len(bm_df)} rebalances)")


def export_feature_importance(attribution: dict, feature_groups: dict, csv_dir: Path):
    """4. 피처별 중요도 + 그룹."""
    importance = attribution.get("feature_importance")
    if importance is None:
        return
    # 피처 -> 그룹 매핑
    feature_to_group = {}
    for group_name, features in feature_groups.items():
        for f in features:
            feature_to_group[f] = group_name

    df = pd.DataFrame({
        "feature": importance.index,
        "importance": importance.values,
        "group": [feature_to_group.get(f, "Unknown") for f in importance.index],
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(csv_dir / "feature_importance.csv", index=False)
    print(f"  [4] feature_importance.csv ({len(df)} features)")


def export_group_attribution(attribution: dict, csv_dir: Path):
    """5. 그룹별 SHAP 기여도 시계열."""
    gc_data = attribution.get("group_contributions", {})
    if not gc_data:
        return
    df = pd.DataFrame(gc_data).T
    df.index.name = "date"
    df = df.round(4)
    df.to_csv(csv_dir / "group_attribution.csv")
    print(f"  [5] group_attribution.csv ({len(df)} dates)")


def export_li_attribution(attribution: dict, csv_dir: Path):
    """6. Li et al. 3-component 시계열."""
    detail = attribution.get("linear_nonlinear_detail", {})
    if not detail:
        return
    rows = []
    for d, v in sorted(detail.items()):
        row = {
            "date": d,
            "linear_ratio": v["linear_ratio"],
            "marginal_nl_ratio": v["marginal_nl_ratio"],
            "interaction_ratio": v["interaction_ratio"],
            "nonlinear_ratio": v["nonlinear_ratio"],
        }
        # 그룹별
        for g, val in v.get("group_linear", {}).items():
            row[f"linear_{g}"] = val
        for g, val in v.get("group_marginal_nl", {}).items():
            row[f"marginal_nl_{g}"] = val
        for g, val in v.get("group_interaction", {}).items():
            row[f"interaction_{g}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.round(4)
    df.to_csv(csv_dir / "li_attribution.csv", index=False)
    print(f"  [6] li_attribution.csv ({len(df)} dates)")


def export_ic_series(result: BacktestResult, csv_dir: Path):
    """7. IC 시계열."""
    if len(result.ic_series) == 0:
        return
    df = pd.DataFrame({"date": result.ic_series.index, "IC": result.ic_series.values})
    df = df.round(4)
    df.to_csv(csv_dir / "ic_series.csv", index=False)
    print(f"  [7] ic_series.csv ({len(df)} dates)")


def export_model_structure(result: BacktestResult, csv_dir: Path):
    """8. LightGBM 트리 구조 요약."""
    rows = []
    for m_date, model in sorted(result.models.items()):
        booster = model.booster_
        n_trees = booster.num_trees()
        model_df = booster.trees_to_dataframe()

        # 트리별 리프 수
        leaf_counts = model_df[model_df["node_depth"] == model_df.groupby("tree_index")["node_depth"].transform("max")]

        # 사용된 피처
        split_features = model_df.loc[model_df["split_feature"].notna(), "split_feature"]
        unique_features = split_features.nunique()

        # 평균 깊이
        avg_depth = model_df.groupby("tree_index")["node_depth"].max().mean()

        # 피처별 split 횟수 (상위 10개)
        top_split_features = split_features.value_counts().head(10)

        rows.append({
            "retrain_date": m_date,
            "n_trees": n_trees,
            "n_unique_features_used": unique_features,
            "avg_tree_depth": round(avg_depth, 1),
            "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else n_trees,
            "top_split_features": "; ".join([f"{f}({c})" for f, c in top_split_features.items()]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_dir / "model_structure.csv", index=False)
    print(f"  [8] model_structure.csv ({len(df)} models)")


def export_monthly_regime(
    result: BacktestResult,
    data: UniverseData,
    csv_dir: Path,
    n_months: int = 6,
):
    """9. 최근 n_months개월 월별 국면 + OW 종목 + 설명."""
    tickers = [t for t in TICKERS if t in data.returns.columns]
    returns = data.returns[tickers]
    prices = data.prices[tickers]
    mktcap = data.market_cap[tickers]

    # 리밸런싱 날짜 역순
    rebal_dates = sorted(result.portfolio_weights.keys(), reverse=True)

    # 최근 n_months 개월분 추출
    if not rebal_dates:
        return

    last_date = rebal_dates[0]
    cutoff = last_date - pd.DateOffset(months=n_months)
    recent_rebal = [d for d in rebal_dates if d >= cutoff]

    # 월별 그룹핑
    monthly = {}
    for d in sorted(recent_rebal):
        month_key = d.strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = d  # 해당 월의 첫 리밸런싱 날짜

    rows = []
    for month_key, rebal_date in sorted(monthly.items()):
        w = result.portfolio_weights[rebal_date]
        # 벤치마크 비중
        mc = mktcap.loc[rebal_date, tickers]
        mc_sum = mc.sum()
        bm_w = mc / mc_sum if mc_sum > 0 else pd.Series(1.0 / len(tickers), index=tickers)

        # Active weights = 포트폴리오 - 벤치마크
        active_w = w - bm_w
        active_w = active_w.sort_values(ascending=False)

        # OW/UW 종목 (상위 5 / 하위 5)
        ow_stocks = active_w.head(5)
        uw_stocks = active_w.tail(5)

        # 국면 분석 (해당 월 기준)
        month_start = pd.Timestamp(month_key + "-01")
        month_end = rebal_date
        # 21일 수익률로 시장 방향 추정
        idx = data.dates.get_loc(rebal_date)
        lookback_start = max(0, idx - 21)
        recent_ret = returns.iloc[lookback_start:idx + 1]
        ew_ret = recent_ret.mean(axis=1).sum()

        if ew_ret > 0.02:
            market_dir = "Bullish"
        elif ew_ret < -0.02:
            market_dir = "Bearish"
        else:
            market_dir = "Sideways"

        # 변동성 레짐
        vol_21d = recent_ret.mean(axis=1).std() * np.sqrt(252)
        if vol_21d > 0.25:
            vol_regime = "High Volatility"
        elif vol_21d > 0.15:
            vol_regime = "Normal Volatility"
        else:
            vol_regime = "Low Volatility"

        # 예측값 기반 OW 이유
        pred_row = result.predictions.loc[rebal_date, tickers] if rebal_date in result.predictions.index else pd.Series()

        # OW 종목별 설명 생성
        ow_explanations = []
        for ticker, aw in ow_stocks.items():
            pred_val = pred_row.get(ticker, None)
            port_w = w.get(ticker, 0)
            bm_wt = bm_w.get(ticker, 0)
            reason_parts = []

            if pred_val is not None and not np.isnan(pred_val):
                if pred_val > 0.5:
                    reason_parts.append(f"Strong positive signal (z={pred_val:.2f})")
                elif pred_val > 0:
                    reason_parts.append(f"Mild positive signal (z={pred_val:.2f})")
                else:
                    reason_parts.append(f"Negative signal but optimizer kept OW (z={pred_val:.2f})")

            reason = "; ".join(reason_parts) if reason_parts else "Optimizer allocation"
            ow_explanations.append(f"{ticker}(+{aw:.1%}, {reason})")

        uw_explanations = []
        for ticker, aw in uw_stocks.items():
            pred_val = pred_row.get(ticker, None)
            reason_parts = []
            if pred_val is not None and not np.isnan(pred_val):
                if pred_val < -0.5:
                    reason_parts.append(f"Strong negative signal (z={pred_val:.2f})")
                elif pred_val < 0:
                    reason_parts.append(f"Mild negative signal (z={pred_val:.2f})")
                else:
                    reason_parts.append(f"Positive signal but optimizer UW (z={pred_val:.2f})")
            reason = "; ".join(reason_parts) if reason_parts else "Optimizer allocation"
            uw_explanations.append(f"{ticker}({aw:.1%}, {reason})")

        # 섹터 로테이션
        asset_light = [c for c in ["MSFT", "GOOGL", "META", "PLTR", "CRM", "NFLX"] if c in tickers]
        asset_heavy = [c for c in ["NVDA", "AVGO", "MU", "GEV", "VRT", "BE", "LITE", "000660"] if c in tickers]

        al_ret = recent_ret[asset_light].mean(axis=1).sum() if asset_light else 0
        ah_ret = recent_ret[asset_heavy].mean(axis=1).sum() if asset_heavy else 0
        rotation = "Asset-Heavy" if ah_ret > al_ret + 0.01 else ("Asset-Light" if al_ret > ah_ret + 0.01 else "Neutral")

        rows.append({
            "year_month": month_key,
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "market_direction": market_dir,
            "volatility_regime": vol_regime,
            "sector_rotation": rotation,
            "ew_return_21d": round(ew_ret, 4),
            "vol_21d_ann": round(vol_21d, 4),
            "top_ow_stocks": " | ".join(ow_explanations),
            "top_uw_stocks": " | ".join(uw_explanations),
            "n_ow_stocks": int((active_w > 0.002).sum()),
            "n_uw_stocks": int((active_w < -0.002).sum()),
            "total_active_share": round(active_w.abs().sum() / 2, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_dir / "monthly_regime.csv", index=False)
    print(f"  [9] monthly_regime.csv ({len(df)} months)")


def export_style_sector_tilt(
    result: BacktestResult,
    data: UniverseData,
    csv_dir: Path,
):
    """10. 리밸런싱별 섹터/스타일 Active Weight 분석."""
    tickers = [t for t in TICKERS if t in data.returns.columns]
    mktcap = data.market_cap[tickers]

    # 섹터/스타일 매핑
    ticker_sectors = {t: TICKER_META.get(t, {}).get("sector", "Other") for t in tickers}
    ticker_styles = {t: TICKER_META.get(t, {}).get("style", "Other") for t in tickers}
    all_sectors = sorted(set(ticker_sectors.values()))
    all_styles = sorted(set(ticker_styles.values()))

    rebal_dates = sorted(result.portfolio_weights.keys())
    rows = []

    for d in rebal_dates:
        w = result.portfolio_weights[d]
        mc = mktcap.loc[d, tickers]
        mc_sum = mc.sum()
        bm_w = mc / mc_sum if mc_sum > 0 else pd.Series(1.0 / len(tickers), index=tickers)

        active_w = w - bm_w

        row = {"date": d.strftime("%Y-%m-%d")}

        # 섹터별 active weight 합산
        for sec in all_sectors:
            sec_tickers = [t for t in tickers if ticker_sectors[t] == sec]
            row[f"sector_{sec}"] = round(active_w[sec_tickers].sum(), 6) if sec_tickers else 0.0
            row[f"port_sector_{sec}"] = round(w[sec_tickers].sum(), 6) if sec_tickers else 0.0
            row[f"bm_sector_{sec}"] = round(bm_w[sec_tickers].sum(), 6) if sec_tickers else 0.0

        # 스타일별 active weight 합산
        for sty in all_styles:
            sty_tickers = [t for t in tickers if ticker_styles[t] == sty]
            row[f"style_{sty}"] = round(active_w[sty_tickers].sum(), 6) if sty_tickers else 0.0
            row[f"port_style_{sty}"] = round(w[sty_tickers].sum(), 6) if sty_tickers else 0.0
            row[f"bm_style_{sty}"] = round(bm_w[sty_tickers].sum(), 6) if sty_tickers else 0.0

        # 지배적 섹터/스타일
        sec_active = {sec: row[f"sector_{sec}"] for sec in all_sectors}
        sty_active = {sty: row[f"style_{sty}"] for sty in all_styles}
        row["dominant_ow_sector"] = max(sec_active, key=sec_active.get)
        row["dominant_uw_sector"] = min(sec_active, key=sec_active.get)
        row["dominant_ow_style"] = max(sty_active, key=sty_active.get)
        row["dominant_uw_style"] = min(sty_active, key=sty_active.get)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_dir / "style_sector_tilt.csv", index=False)
    print(f"  [10] style_sector_tilt.csv ({len(df)} rebalances)")


def export_monthly_ow_explanations(
    result: BacktestResult,
    data: UniverseData,
    attribution: dict,
    feature_groups: dict,
    report_dir: Path,
    n_months: int = 6,
):
    """11. 월별 OW 종목 상세 설명 리포트 (outputs/reports/)."""
    tickers = [t for t in TICKERS if t in data.returns.columns]
    returns = data.returns[tickers]
    mktcap = data.market_cap[tickers]

    ticker_sectors = {t: TICKER_META.get(t, {}).get("sector", "Other") for t in tickers}
    ticker_styles = {t: TICKER_META.get(t, {}).get("style", "Other") for t in tickers}
    ticker_subs = {t: TICKER_META.get(t, {}).get("sub", "N/A") for t in tickers}

    rebal_dates = sorted(result.portfolio_weights.keys(), reverse=True)
    if not rebal_dates:
        return

    last_date = rebal_dates[0]
    cutoff = last_date - pd.DateOffset(months=n_months)
    recent_rebal = [d for d in rebal_dates if d >= cutoff]

    # 월별 첫 리밸런싱
    monthly = {}
    for d in sorted(recent_rebal):
        mk = d.strftime("%Y-%m")
        if mk not in monthly:
            monthly[mk] = d

    # Feature group 기여도 (attribution 결과에서)
    group_contribs = attribution.get("group_contributions", {})

    rows = []
    for month_key, rebal_date in sorted(monthly.items()):
        w = result.portfolio_weights[rebal_date]
        mc = mktcap.loc[rebal_date, tickers]
        mc_sum = mc.sum()
        bm_w = mc / mc_sum if mc_sum > 0 else pd.Series(1.0 / len(tickers), index=tickers)
        active_w = (w - bm_w).sort_values(ascending=False)

        # ── Regime 분석 ──
        idx = data.dates.get_loc(rebal_date)
        lookback_start = max(0, idx - 21)
        recent_ret = returns.iloc[lookback_start:idx + 1]
        ew_ret = recent_ret.mean(axis=1).sum()
        vol_21d = recent_ret.mean(axis=1).std() * np.sqrt(252)

        if ew_ret > 0.02:
            market_dir = "Bullish"
        elif ew_ret < -0.02:
            market_dir = "Bearish"
        else:
            market_dir = "Sideways"

        if vol_21d > 0.25:
            vol_regime = "High Volatility"
        elif vol_21d > 0.15:
            vol_regime = "Normal Volatility"
        else:
            vol_regime = "Low Volatility"

        regime_label = f"{market_dir} / {vol_regime}"

        # Regime 이유
        regime_reasons = []
        regime_reasons.append(f"21d EW Ret={ew_ret:+.2%} → {market_dir}")
        regime_reasons.append(f"21d Vol(ann)={vol_21d:.1%} → {vol_regime}")

        # 섹터별 21d 수익률로 어떤 섹터가 주도하는지
        sec_rets = {}
        for sec in set(ticker_sectors.values()):
            sec_t = [t for t in tickers if ticker_sectors[t] == sec]
            if sec_t:
                sec_rets[sec] = recent_ret[sec_t].mean(axis=1).sum()

        if sec_rets:
            top_sec = max(sec_rets, key=sec_rets.get)
            bot_sec = min(sec_rets, key=sec_rets.get)
            regime_reasons.append(f"Leading: {top_sec}({sec_rets[top_sec]:+.2%}), Lagging: {bot_sec}({sec_rets[bot_sec]:+.2%})")
        regime_reason = " | ".join(regime_reasons)

        # ── Dominant Category Effects ──
        # 가장 가까운 attribution 날짜 찾기
        closest_attr_date = None
        if group_contribs:
            attr_dates = sorted(group_contribs.keys())
            diffs = [abs((pd.Timestamp(ad) - rebal_date).days) for ad in attr_dates]
            if diffs:
                min_idx = diffs.index(min(diffs))
                closest_attr_date = attr_dates[min_idx]

        if closest_attr_date and closest_attr_date in group_contribs:
            gc = group_contribs[closest_attr_date]
            gc_sorted = sorted(gc.items(), key=lambda x: x[1], reverse=True)
            dom_effects = " > ".join([f"{g}({v:.1%})" for g, v in gc_sorted])
        else:
            dom_effects = "N/A"

        # ── 섹터/스타일 Tilt 요약 ──
        sec_active = {}
        for sec in set(ticker_sectors.values()):
            sec_t = [t for t in tickers if ticker_sectors[t] == sec]
            if sec_t:
                sec_active[sec] = active_w[sec_t].sum()

        sty_active = {}
        for sty in set(ticker_styles.values()):
            sty_t = [t for t in tickers if ticker_styles[t] == sty]
            if sty_t:
                sty_active[sty] = active_w[sty_t].sum()

        sec_sorted = sorted(sec_active.items(), key=lambda x: x[1], reverse=True)
        sty_sorted = sorted(sty_active.items(), key=lambda x: x[1], reverse=True)

        sector_tilt_str = " | ".join([f"{s}({v:+.1%})" for s, v in sec_sorted if abs(v) > 0.005])
        style_tilt_str = " | ".join([f"{s}({v:+.1%})" for s, v in sty_sorted if abs(v) > 0.005])

        # ── 종목별 OW 상세 설명 ──
        pred_row = result.predictions.loc[rebal_date, tickers] if rebal_date in result.predictions.index else pd.Series()

        ow_details = []
        for ticker in active_w.index:
            aw = active_w[ticker]
            if aw < 0.002:
                continue  # OW 종목만

            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            sec = ticker_sectors.get(ticker, "N/A")
            sty = ticker_styles.get(ticker, "N/A")
            sub = ticker_subs.get(ticker, "N/A")

            # 시그널 해석
            if not np.isnan(pred_val):
                if pred_val > 1.0:
                    signal_desc = f"Very strong positive signal (z={pred_val:.2f})"
                elif pred_val > 0.5:
                    signal_desc = f"Strong positive signal (z={pred_val:.2f})"
                elif pred_val > 0:
                    signal_desc = f"Mild positive signal (z={pred_val:.2f})"
                else:
                    signal_desc = f"Optimizer risk-adjusted OW despite negative signal (z={pred_val:.2f})"
            else:
                signal_desc = "Signal not available"

            ow_details.append(
                f"{ticker}[{sec}/{sty}/{sub}](AW={aw:+.1%}, {signal_desc})"
            )

        uw_details = []
        for ticker in reversed(active_w.index):
            aw = active_w[ticker]
            if aw > -0.002:
                continue

            pred_val = pred_row.get(ticker, np.nan) if not pred_row.empty else np.nan
            sec = ticker_sectors.get(ticker, "N/A")
            sty = ticker_styles.get(ticker, "N/A")

            if not np.isnan(pred_val):
                if pred_val < -0.5:
                    signal_desc = f"Strong negative signal (z={pred_val:.2f})"
                elif pred_val < 0:
                    signal_desc = f"Mild negative signal (z={pred_val:.2f})"
                else:
                    signal_desc = f"Positive signal but BM cap-weight dominates (z={pred_val:.2f})"
            else:
                signal_desc = "Signal not available"

            uw_details.append(
                f"{ticker}[{sec}/{sty}](AW={aw:+.1%}, {signal_desc})"
            )

        rows.append({
            "year_month": month_key,
            "rebal_date": rebal_date.strftime("%Y-%m-%d"),
            "regime_label": regime_label,
            "regime_reason": regime_reason,
            "dominant_category_effects": dom_effects,
            "sector_tilt": sector_tilt_str,
            "style_tilt": style_tilt_str,
            "n_ow_stocks": int((active_w > 0.002).sum()),
            "n_uw_stocks": int((active_w < -0.002).sum()),
            "top_ow_details": " | ".join(ow_details[:10]),
            "top_uw_details": " | ".join(uw_details[:10]),
            "all_ow_details": " | ".join(ow_details),
            "all_uw_details": " | ".join(uw_details),
        })

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "lightgbm_monthly_ow_explanations.csv", index=False)
    print(f"  [11] lightgbm_monthly_ow_explanations.csv ({len(df)} months) → {report_dir}")


def main():
    print("=" * 60)
    print("CSV Export Pipeline")
    print("=" * 60)

    # Setup
    data_path = "./data/ai_signal_data.xlsx"
    csv_dir = Path("./outputs/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load data
    print("\n[Phase 1] Loading data...")
    data = UniverseData(data_path)

    # Phase 2-6: Backtest
    print("\n[Phase 2-6] Running backtest...")
    result = run_backtest(data)

    # Phase 7: Attribution
    print("\n[Phase 7] Running attribution...")
    gc.collect()
    panel = result.panel if result.panel is not None else build_all_features(data)[0]
    feature_names = result.feature_names
    feature_groups = result.feature_groups

    attribution = run_attribution(
        result.models, panel, feature_names, feature_groups,
        weights_history=result.portfolio_weights,
    )

    # Report directory
    report_dir = Path("./outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    print("\n" + "=" * 60)
    print("Exporting CSVs...")
    print("=" * 60)

    export_daily_performance(result, csv_dir)
    export_portfolio_weights(result, csv_dir)
    export_benchmark_weights(result, data, csv_dir)
    export_feature_importance(attribution, feature_groups, csv_dir)
    export_group_attribution(attribution, csv_dir)
    export_li_attribution(attribution, csv_dir)
    export_ic_series(result, csv_dir)
    export_model_structure(result, csv_dir)
    export_monthly_regime(result, data, csv_dir)
    export_style_sector_tilt(result, data, csv_dir)
    export_monthly_ow_explanations(result, data, attribution, feature_groups, report_dir)

    print(f"\nAll CSVs saved to {csv_dir.resolve()}")
    print(f"Reports saved to {report_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
