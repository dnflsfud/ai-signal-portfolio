"""
Phase 2: Extended Feature Engineering (~350 features)

Categories:
  1. Accounting/Fundamental   ~100  (변화율, 가속도, 변동성, 랭크, 교차비율)
  2. Price/Market              ~50  (모멘텀, 리버설, 변동성, 베타, 분포통계)
  3. Sellside/Sentiment        ~60  (리비전 상대강도, 모멘텀, 타겟가, 뉴스)
  4. Conditioning              ~50  (캘린더, 섹터, 사이즈, 레짐 - 바이너리 포함)
  5. Factor/Macro              ~90  (38개 팩터 + 스프레드 + 레짐 바이너리)

핵심 설계 원칙:
  - ~90% 경제적 근거 시그널 + ~10% 단독 IR=0인 conditioning
  - Conditioning = 바이너리/이산 레짐 변수 → GBT interaction 유도
  - 모든 stock-level 피처는 CS Z-score 정규화
  - Conditioning/Factor(broadcast)는 Z-score 제외
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from src.data_loader import UniverseData, TICKERS, ALL_FACTOR_COLUMNS, FACTOR_CATEGORIES


# ===========================================================================
# Utilities
# ===========================================================================

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional Z-score 정규화 (Round 4 방식).
    각 날짜에서 종목 간 평균/표준편차로 표준화.
    Scale 정보를 보존하여 실제 수익률 예측에 유리.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def safe_pct_change(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    shifted = df.shift(periods)
    denom = shifted.replace(0, np.nan)
    return (df - shifted) / denom.abs()


def clip_outliers(df: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
    return df.clip(lower=-n_std, upper=n_std)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank."""
    return df.rank(axis=1, pct=True)


# ===========================================================================
# Revision 전처리 유틸리티
# ===========================================================================

def clean_revision_spikes(rev: pd.DataFrame, threshold: float = 30) -> pd.DataFrame:
    """실적발표 전후 컨센서스 기간 전환으로 인한 급락 스무딩.

    1일 급락 > threshold인 경우 해당일을 전일값(ffill)으로 대체하여
    diff 계산 시 인위적 절벽이 발생하지 않도록 함.

    - 종목당 연 3~4회, 65%가 실적시즌(1,4,7,10월)에 집중
    - 30~90pt 급락이 일반적
    """
    cleaned = rev.copy()
    daily_drop = cleaned.diff()
    spike_mask = daily_drop < -threshold
    cleaned[spike_mask] = np.nan
    cleaned = cleaned.ffill()
    n_spikes = spike_mask.sum().sum()
    print(f"[RevisionClean] {n_spikes}개 스파이크 스무딩 (threshold={threshold})")
    return cleaned


def build_bounded_revision_features(
    rev: pd.DataFrame, prefix: str
) -> Dict[str, pd.DataFrame]:
    """Bounded 지표(-100~100)에 특화된 Revision feature 생성.

    문제: 단순 diff 사용 시, 100 유지(diff=0) → CS z-score 중립/하위
         반면 50→80(diff=+30) → z-score 상위. 100 유지가 더 좋은데 저평가.

    해결:
      1. 기존 diff는 정제된 데이터에서 계산 (스파이크 제거)
      2. level_persist: rolling mean / 100 → 100 유지 시 +1.0 (GBT interaction)
      3. time_at_extreme: 최근 N일 중 극단값 비율 (persistence 포착)
      4. level_dir: level × 변화방향 복합 (상단 유지 = 강한 양수)
    """
    features: Dict[str, pd.DataFrame] = {}

    # ── 기존 feature (정제 데이터 기반) ──
    features[f"{prefix}"] = rev
    features[f"{prefix}_diff_5d"] = rev - rev.shift(5)
    features[f"{prefix}_diff_21d"] = rev - rev.shift(21)
    features[f"{prefix}_diff_63d"] = rev - rev.shift(63)
    features[f"{prefix}_ma_63d"] = rev.rolling(63, min_periods=21).mean()
    features[f"{prefix}_accel"] = (rev - rev.shift(5)) - (rev - rev.shift(21))
    features[f"{prefix}_rank"] = cs_rank(rev)
    features[f"{prefix}_rel_strength"] = cs_rank(rev - rev.shift(21))
    features[f"{prefix}_vol"] = rev.rolling(63, min_periods=21).std()
    features[f"{prefix}_trend"] = (
        rev.rolling(21, min_periods=10).mean()
        - rev.rolling(63, min_periods=21).mean()
    )
    features[f"{prefix}_momentum"] = cs_rank(rev - rev.shift(5))
    features[f"{prefix}_vs_median"] = rev - rev.rolling(252, min_periods=126).median()

    # ── NEW: Bounded 보정 feature ──

    # 1. Level persistence: rolling mean 정규화 → 100 유지 = +1.0
    features[f"{prefix}_level_persist_21d"] = (
        rev.rolling(21, min_periods=5).mean() / 100
    )
    features[f"{prefix}_level_persist_63d"] = (
        rev.rolling(63, min_periods=21).mean() / 100
    )

    # 2. Time at extreme: 극단 유지 비율 (GBT가 "100유지" 패턴 학습)
    features[f"{prefix}_time_high"] = (
        (rev > 70).astype(float).rolling(21, min_periods=5).mean()
    )
    features[f"{prefix}_time_low"] = (
        (rev < -70).astype(float).rolling(21, min_periods=5).mean()
    )

    # 3. Level-direction composite: 레벨 × 변화방향
    #    100유지(dir=0) → 1.0, 100↑(dir=+1) → 1.3, 50↑(dir=+1) → 0.65
    diff_5d = rev - rev.shift(5)
    direction = np.sign(diff_5d)
    features[f"{prefix}_level_dir"] = (rev / 100) * (1 + direction * 0.3)

    return features


# ===========================================================================
# Category 1: Accounting / Fundamental (~100 features)
# ===========================================================================

ACCOUNTING_BASE = [
    "BEST_EPS", "BEST_SALES", "BEST_CALCULATED_FCF",
    "BEST_GROSS_MARGIN", "OPER_MARGIN", "BEST_CAPEX", "BEST_ROE",
]

VALUATION_SHEETS = [
    "BEST_PE_RATIO", "BEST_PEG_RATIO", "BEST_PX_BPS_RATIO",
    "BEST_EV_TO_BEST_EBITDA",
]


    # EPS/Sales/Margin: 변화율·가속도만 중요, level 자체는 제거
LEVEL_SKIP_SHEETS = {"BEST_EPS", "BEST_SALES", "BEST_GROSS_MARGIN", "OPER_MARGIN"}


def build_accounting_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}

    # -- Base fundamentals --
    for sheet in ACCOUNTING_BASE:
        try:
            raw = data.get_sheet(sheet)
        except KeyError:
            continue
        p = sheet.lower()
        # 변화율 (4 windows)
        for w in [21, 63, 126, 252]:
            features[f"{p}_chg_{w}d"] = safe_pct_change(raw, w)
        # 가속도 (변화율의 변화율)
        features[f"{p}_accel"] = safe_pct_change(raw, 21) - safe_pct_change(raw, 63)
        # 변화의 변동성 (fundamental stability)
        chg21 = safe_pct_change(raw, 21)
        features[f"{p}_chg_vol"] = chg21.rolling(63, min_periods=21).std()

        # Level features: EPS/Sales/Margin은 제외 (변화율이 더 중요)
        if sheet not in LEVEL_SKIP_SHEETS:
            features[f"{p}_level_z"] = cross_sectional_zscore(raw)
            features[f"{p}_rank"] = cs_rank(raw)
            med = raw.rolling(252, min_periods=126).median()
            features[f"{p}_vs_median"] = (raw / med.replace(0, np.nan)) - 1

    # -- Valuation: 4 sheets × 7 features = 28 --
    for sheet in VALUATION_SHEETS:
        try:
            raw = data.get_sheet(sheet)
        except KeyError:
            continue
        p = sheet.lower()
        features[f"{p}_level_z"] = cross_sectional_zscore(raw)
        features[f"{p}_chg_21d"] = safe_pct_change(raw, 21)
        features[f"{p}_chg_63d"] = safe_pct_change(raw, 63)
        features[f"{p}_accel"] = safe_pct_change(raw, 21) - safe_pct_change(raw, 63)
        med = raw.rolling(252, min_periods=126).median()
        features[f"{p}_vs_median"] = (raw / med.replace(0, np.nan)) - 1
        features[f"{p}_vol"] = safe_pct_change(raw, 21).rolling(63, min_periods=21).std()
        features[f"{p}_rank"] = cs_rank(raw)

    # -- Cross-ratios (~10) --
    _add_cross_ratios(data, features)

    return features


def _add_cross_ratios(data: UniverseData, features: Dict[str, pd.DataFrame]):
    """Accounting 교차비율 피처."""
    def _safe_get(name):
        try:
            return data.get_sheet(name)
        except KeyError:
            return None

    eps = _safe_get("BEST_EPS")
    sales = _safe_get("BEST_SALES")
    gm = _safe_get("BEST_GROSS_MARGIN")
    om = _safe_get("OPER_MARGIN")
    fcf = _safe_get("BEST_CALCULATED_FCF")
    capex = _safe_get("BEST_CAPEX")
    roe = _safe_get("BEST_ROE")
    pe = _safe_get("BEST_PE_RATIO")
    mc = data.market_cap

    if eps is not None and sales is not None:
        # Earnings quality: EPS growth > Sales growth = quality
        features["earnings_quality_63d"] = safe_pct_change(eps, 63) - safe_pct_change(sales, 63)
        features["earnings_quality_252d"] = safe_pct_change(eps, 252) - safe_pct_change(sales, 252)

    if gm is not None and om is not None:
        features["op_leverage_63d"] = safe_pct_change(om, 63) - safe_pct_change(gm, 63)

    if fcf is not None and eps is not None:
        features["cash_conversion_z"] = cross_sectional_zscore(fcf / eps.replace(0, np.nan).abs())

    if capex is not None and sales is not None:
        ratio = capex / sales.replace(0, np.nan).abs()
        features["capex_intensity_z"] = cross_sectional_zscore(ratio)
        features["capex_intensity_chg"] = safe_pct_change(ratio, 63)

    if roe is not None and pe is not None:
        features["roe_pe_z"] = cross_sectional_zscore(roe / pe.replace(0, np.nan).abs())

    if eps is not None:
        features["mkcap_eps_divg"] = safe_pct_change(mc, 63) - safe_pct_change(eps, 63)


# ===========================================================================
# Category 2: Price / Market (~50 features)
# ===========================================================================

def build_price_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}
    returns = data.returns
    prices = data.prices
    mktcap = data.market_cap

    # --- Reversal (3) ---
    for w in [5, 10, 21]:
        features[f"reversal_{w}d"] = -1 * returns.rolling(w, min_periods=w).sum()

    # --- Momentum (3) ---
    for w in [63, 126, 252]:
        features[f"momentum_{w}d"] = returns.rolling(w, min_periods=w).sum()

    # --- Risk-adjusted momentum (3) ---
    for w in [63, 126, 252]:
        mom = returns.rolling(w, min_periods=w).sum()
        vol = returns.rolling(w, min_periods=w).std().replace(0, np.nan)
        features[f"risk_adj_mom_{w}d"] = mom / vol

    # --- Realized volatility (3) ---
    for w in [21, 63, 126]:
        features[f"realized_vol_{w}d"] = returns.rolling(w, min_periods=w).std() * np.sqrt(252)

    # --- Volatility ratio & change (2) ---
    v21 = returns.rolling(21).std()
    v126 = returns.rolling(126).std().replace(0, np.nan)
    features["vol_ratio_21_126"] = v21 / v126
    v21_lag = returns.shift(21).rolling(21).std().replace(0, np.nan)
    features["vol_change_21d"] = v21 / v21_lag - 1

    # --- MA distance (3) ---
    for w in [21, 50, 200]:
        ma = prices.rolling(w, min_periods=w).mean().replace(0, np.nan)
        features[f"price_dist_ma_{w}d"] = (prices / ma) - 1

    # --- MA crossover (2) ---
    ma21 = prices.rolling(21).mean()
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    features["ma_cross_21_50"] = (ma21 / ma50.replace(0, np.nan)) - 1
    features["ma_cross_50_200"] = (ma50 / ma200.replace(0, np.nan)) - 1

    # --- Drawdown & 52-week range (4) ---
    rmax63 = prices.rolling(63, min_periods=1).max().replace(0, np.nan)
    features["drawdown_63d"] = (prices / rmax63) - 1
    rmax252 = prices.rolling(252, min_periods=126).max().replace(0, np.nan)
    rmin252 = prices.rolling(252, min_periods=126).min().replace(0, np.nan)
    features["dist_52w_high"] = (prices / rmax252) - 1
    features["dist_52w_low"] = (prices / rmin252) - 1
    range252 = (rmax252 - rmin252).replace(0, np.nan)
    features["range_position_52w"] = (prices - rmin252) / range252

    # --- Mktcap rank (1) ---
    features["mktcap_rank"] = mktcap.rank(axis=1, pct=True)

    # --- Relative momentum (stock vs market) (3) ---
    ew_ret = returns.mean(axis=1)
    for w in [21, 63, 126]:
        stock_mom = returns.rolling(w, min_periods=w).sum()
        mkt_mom = ew_ret.rolling(w, min_periods=w).sum()
        features[f"rel_mom_{w}d"] = stock_mom.sub(mkt_mom, axis=0)

    # --- Momentum rank (2) ---
    for w in [21, 63]:
        features[f"mom_rank_{w}d"] = cs_rank(returns.rolling(w, min_periods=w).sum())

    # --- Return distribution (4) ---
    for w in [21, 63]:
        features[f"ret_skew_{w}d"] = returns.rolling(w, min_periods=w).skew()
        features[f"ret_kurt_{w}d"] = returns.rolling(w, min_periods=w).kurt()

    # --- Max/Min return (4) ---
    for w in [21, 63]:
        features[f"max_ret_{w}d"] = returns.rolling(w, min_periods=w).max()
        features[f"min_ret_{w}d"] = returns.rolling(w, min_periods=w).min()

    # --- Positive return ratio (2) ---
    for w in [21, 63]:
        features[f"pos_ret_ratio_{w}d"] = (returns > 0).astype(float).rolling(w, min_periods=w).mean()

    # --- Downside deviation & Up/Down ratio (3) ---
    neg_ret = returns.clip(upper=0)
    for w in [21, 63]:
        features[f"downside_vol_{w}d"] = neg_ret.rolling(w, min_periods=w).std() * np.sqrt(252)
    pos_vol21 = returns.clip(lower=0).rolling(21).std()
    neg_vol21 = neg_ret.rolling(21).std().replace(0, np.nan)
    features["up_down_vol_ratio"] = pos_vol21 / neg_vol21

    # --- Rolling beta to EW market (2) ---
    mkt = returns.mean(axis=1)
    for w in [63]:
        xy = returns.mul(mkt, axis=0)
        e_xy = xy.rolling(w, min_periods=w).mean()
        e_x = returns.rolling(w, min_periods=w).mean()
        e_y = mkt.rolling(w, min_periods=w).mean()
        cov_xy = e_xy - e_x.mul(e_y, axis=0)
        var_y = mkt.rolling(w, min_periods=w).var().replace(0, np.nan)
        beta = cov_xy.div(var_y, axis=0)
        features[f"beta_{w}d"] = beta
        # Idiosyncratic vol
        resid = returns - beta.mul(mkt, axis=0)
        features[f"idio_vol_{w}d"] = resid.rolling(w, min_periods=w).std() * np.sqrt(252)

    # --- Trend consistency (1) ---
    ret5d = returns.rolling(5, min_periods=5).sum()
    features["trend_consist_63d"] = (ret5d > 0).astype(float).rolling(63, min_periods=21).mean()

    return features


# ===========================================================================
# Category 3: Sellside / Sentiment (~60 features)
# ===========================================================================

def build_sellside_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}

    # --- Analyst Recommendation (~8) ---
    try:
        rec = data.get_sheet("EQY_REC_CONS")
        features["analyst_rec_level"] = rec
        features["analyst_rec_diff_5d"] = rec - rec.shift(5)
        features["analyst_rec_diff_21d"] = rec - rec.shift(21)
        features["analyst_rec_diff_63d"] = rec - rec.shift(63)
        features["analyst_rec_accel"] = (rec - rec.shift(21)) - (rec - rec.shift(63))
        features["analyst_rec_rank"] = cs_rank(rec)
        features["analyst_rec_stability"] = rec.rolling(63, min_periods=21).std()
        med = rec.rolling(252, min_periods=126).median()
        features["analyst_rec_vs_median"] = rec - med
    except KeyError:
        pass

    # --- Target Price (~10) ---
    try:
        tg = data.get_sheet("Factset_TG_Price")
        px = data.prices.replace(0, np.nan)
        upside = (tg / px) - 1
        features["tg_upside"] = upside
        features["tg_upside_diff_5d"] = upside - upside.shift(5)
        features["tg_upside_diff_21d"] = upside - upside.shift(21)
        features["tg_upside_rank"] = cs_rank(upside)
        features["tg_upside_z"] = cross_sectional_zscore(upside)
        med = upside.rolling(126, min_periods=63).median()
        features["tg_upside_vs_median"] = upside - med
        features["tg_upside_vol"] = upside.rolling(63, min_periods=21).std()
        features["tg_mom_21d"] = safe_pct_change(tg, 21)
        features["tg_mom_63d"] = safe_pct_change(tg, 63)
        features["tg_conviction"] = safe_pct_change(tg, 21).rolling(63, min_periods=21).std()
    except KeyError:
        pass

    # --- EPS Revision (~17, spike-cleaned + bounded-adjusted) ---
    try:
        eps_rev_raw = data.get_sheet("Factset_EPS_Revision")
        eps_rev = clean_revision_spikes(eps_rev_raw, threshold=30)
        features.update(build_bounded_revision_features(eps_rev, "eps_rev"))
    except KeyError:
        pass

    # --- Sales Revision (~17, spike-cleaned + bounded-adjusted) ---
    try:
        sales_rev_raw = data.get_sheet("Factset_Sales_Revision")
        sales_rev = clean_revision_spikes(sales_rev_raw, threshold=30)
        features.update(build_bounded_revision_features(sales_rev, "sales_rev"))
    except KeyError:
        pass

    # --- Cross-revision (~4, on cleaned data) ---
    try:
        eps_rev_raw = data.get_sheet("Factset_EPS_Revision")
        sales_rev_raw = data.get_sheet("Factset_Sales_Revision")
        eps_rev = clean_revision_spikes(eps_rev_raw, threshold=30)
        sales_rev = clean_revision_spikes(sales_rev_raw, threshold=30)
        features["rev_divergence"] = eps_rev - sales_rev
        features["rev_rank_divergence"] = cs_rank(eps_rev) - cs_rank(sales_rev)
        features["rev_combined"] = (eps_rev + sales_rev) / 2
        features["rev_breadth"] = ((eps_rev > 0) & (sales_rev > 0)).astype(float)
    except KeyError:
        pass

    # --- News Sentiment (~10) ---
    try:
        news = data.get_sheet("NEWS_SENTIMENT_DAILY_AVG")
        features["news_raw"] = news
        features["news_ma5"] = news.rolling(5, min_periods=1).mean()
        features["news_ma21"] = news.rolling(21, min_periods=5).mean()
        features["news_ma63"] = news.rolling(63, min_periods=21).mean()
        features["news_trend"] = news.rolling(5, min_periods=1).mean() - news.rolling(21, min_periods=5).mean()
        features["news_accel"] = (news - news.shift(5)) - (news - news.shift(21))
        features["news_vol"] = news.rolling(21, min_periods=10).std()
        features["news_surprise"] = news - news.rolling(21, min_periods=5).mean()
        features["news_rank"] = cs_rank(news)
        rng = news.rolling(126, min_periods=63)
        features["news_range_pos"] = (news - rng.min()) / (rng.max() - rng.min()).replace(0, np.nan)
    except KeyError:
        pass

    # --- Sent Trend (~5) ---
    try:
        sent_mom = data.get_sheet("Sent_Trend_Momentum_Timeseries")
        features["sent_momentum"] = sent_mom
        features["sent_momentum_diff"] = sent_mom - sent_mom.shift(21)
        features["sent_momentum_rank"] = cs_rank(sent_mom)
    except KeyError:
        pass
    try:
        sent_21 = data.get_sheet("Sent_Trend_21d_Timeseries")
        features["sent_21d"] = sent_21
        features["sent_21d_accel"] = sent_21 - sent_21.shift(21)
    except KeyError:
        pass

    return features


# ===========================================================================
# Category 4: Conditioning (~50 features)
# 단독 IR=0이지만 GBT interaction에서 강력한 변수들
# ===========================================================================

def build_conditioning_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}
    dates = data.dates
    tickers = [t for t in TICKERS if t in data.returns.columns]
    n = len(tickers)
    returns = data.returns

    def bcast(vals_1d):
        """1D array/series → broadcast to (dates × tickers)."""
        v = np.asarray(vals_1d).reshape(-1, 1)
        return pd.DataFrame(np.tile(v, (1, n)), index=dates, columns=tickers)

    # ===== Calendar (~15) =====
    month = dates.month
    features["cal_is_Q1"] = bcast((month <= 3).astype(float))
    features["cal_is_Q2"] = bcast(((month >= 4) & (month <= 6)).astype(float))
    features["cal_is_Q3"] = bcast(((month >= 7) & (month <= 9)).astype(float))
    features["cal_is_Q4"] = bcast((month >= 10).astype(float))
    features["cal_is_jan"] = bcast((month == 1).astype(float))
    features["cal_is_qtr_end"] = bcast(month.isin([3, 6, 9, 12]).astype(float))
    features["cal_is_yr_end"] = bcast((month == 12).astype(float))
    features["cal_is_earnings"] = bcast(month.isin([1, 2, 4, 5, 7, 8, 10, 11]).astype(float))
    # month_in_quarter: 1=first month, 2=mid, 3=last (proxy for days to earnings)
    miq = ((month - 1) % 3 + 1).astype(float)
    features["cal_month_in_qtr"] = bcast(miq)
    features["cal_is_mid_qtr"] = bcast((miq == 2).astype(float))
    features["cal_is_first_half"] = bcast((month <= 6).astype(float))
    dow = dates.dayofweek
    features["cal_is_monday"] = bcast((dow == 0).astype(float))
    features["cal_is_friday"] = bcast((dow == 4).astype(float))

    # ===== Sector one-hot (~10) =====
    meta = data.meta
    if isinstance(meta, pd.DataFrame) and "sector" in meta.columns:
        sector_map = meta["sector"]
    elif isinstance(meta, pd.DataFrame) and len(meta.columns) > 0:
        sector_map = meta.iloc[:, 0]
    else:
        sector_map = pd.Series("Unknown", index=tickers)

    for sec in sector_map.unique():
        if str(sec) in ("nan", "Unknown"):
            continue
        vals = np.zeros((len(dates), n))
        for i, t in enumerate(tickers):
            if sector_map.get(t, "") == sec:
                vals[:, i] = 1.0
        features[f"sector_{sec}"] = pd.DataFrame(vals, index=dates, columns=tickers)

    # ===== Size buckets (~5) =====
    mcr = data.market_cap.rank(axis=1, pct=True)
    features["is_mega_cap"] = (mcr > 0.8).astype(float)
    features["is_large_cap"] = ((mcr > 0.6) & (mcr <= 0.8)).astype(float)
    features["is_mid_cap"] = ((mcr > 0.3) & (mcr <= 0.6)).astype(float)
    features["is_small_cap"] = (mcr <= 0.3).astype(float)
    features["size_rank"] = mcr

    # ===== Market regime (continuous + binary) (~15) =====
    ew_ret = returns.mean(axis=1)
    for w in [21, 63]:
        r = ew_ret.rolling(w, min_periods=w).sum()
        features[f"regime_mkt_ret_{w}d"] = bcast(r)
    vol_21 = returns.rolling(21, min_periods=21).std().mean(axis=1) * np.sqrt(252)
    vol_63 = returns.rolling(63, min_periods=63).std().mean(axis=1) * np.sqrt(252)
    features["regime_avg_vol_21d"] = bcast(vol_21)
    features["regime_avg_vol_63d"] = bcast(vol_63)

    # Cross-sectional dispersion
    cs_disp_21 = returns.rolling(21, min_periods=21).mean().std(axis=1)
    features["regime_dispersion_21d"] = bcast(cs_disp_21)

    # Market breadth
    ma50 = data.prices.rolling(50, min_periods=50).mean()
    breadth = (data.prices > ma50).sum(axis=1) / data.prices.shape[1]
    features["regime_breadth_50d"] = bcast(breadth)

    # --- BINARY regime flags (핵심: standalone IR≈0, interaction에서 강력) ---
    vol_med = vol_21.rolling(252, min_periods=126).median()
    features["is_high_vol"] = bcast((vol_21 > vol_med * 1.2).astype(float))
    features["is_low_vol"] = bcast((vol_21 < vol_med * 0.8).astype(float))
    ret_63 = ew_ret.rolling(63, min_periods=63).sum()
    features["is_bull_market"] = bcast((ret_63 > 0.05).astype(float))
    features["is_bear_market"] = bcast((ret_63 < -0.05).astype(float))
    features["is_trending"] = bcast((ret_63.abs() > 0.08).astype(float))
    disp_med = cs_disp_21.rolling(252, min_periods=126).median()
    features["is_high_dispersion"] = bcast((cs_disp_21 > disp_med * 1.2).astype(float))

    # ===== Fundamental regime (~7, spike-cleaned) =====
    try:
        eps_rev = clean_revision_spikes(data.get_sheet("Factset_EPS_Revision"), 30)
        rev_pos_pct = (eps_rev > 0).sum(axis=1) / eps_rev.shape[1]
        features["regime_rev_breadth_eps"] = bcast(rev_pos_pct)
        features["is_rev_expansion"] = bcast((rev_pos_pct > 0.6).astype(float))
    except KeyError:
        pass
    try:
        sales_rev = clean_revision_spikes(data.get_sheet("Factset_Sales_Revision"), 30)
        rev_pos_s = (sales_rev > 0).sum(axis=1) / sales_rev.shape[1]
        features["regime_rev_breadth_sales"] = bcast(rev_pos_s)
    except KeyError:
        pass
    try:
        news = data.get_sheet("NEWS_SENTIMENT_DAILY_AVG")
        sent_pos = (news > 0).sum(axis=1) / news.shape[1]
        features["regime_sent_breadth"] = bcast(sent_pos)
    except KeyError:
        pass
    try:
        rec = data.get_sheet("EQY_REC_CONS")
        rec_mean = rec.mean(axis=1)
        features["regime_avg_rec"] = bcast(rec_mean)
    except KeyError:
        pass

    return features


# ===========================================================================
# Category 5: Factor / Macro Conditioning (~90 features)
# ===========================================================================

def build_factor_features(data: UniverseData) -> Dict[str, pd.DataFrame]:
    features: Dict[str, pd.DataFrame] = {}

    if not data.has_factor_data():
        return features

    factor_ret = data.factor_returns
    factor_px = data.factor_prices
    tickers = [t for t in TICKERS if t in data.returns.columns]
    n = len(tickers)
    common_dates = data.dates.intersection(factor_ret.index)

    def bcast(series: pd.Series) -> pd.DataFrame:
        vals = series.reindex(common_dates).values.reshape(-1, 1)
        return pd.DataFrame(np.tile(vals, (1, n)), index=common_dates, columns=tickers)

    # ===== 1. Factor ETF Momentum/Vol/Accel (7 × 5 = 35) =====
    etf_factors = FACTOR_CATEGORIES.get("Factor_ETF", [])
    for f in etf_factors:
        if f not in factor_ret.columns:
            continue
        for w in [21, 63]:
            features[f"fac_{f}_mom_{w}d"] = bcast(factor_ret[f].rolling(w, min_periods=w).sum())
            features[f"fac_{f}_vol_{w}d"] = bcast(factor_ret[f].rolling(w, min_periods=w).std() * np.sqrt(252))
        m21 = factor_ret[f].rolling(21, min_periods=21).sum()
        m63 = factor_ret[f].rolling(63, min_periods=63).sum()
        features[f"fac_{f}_accel"] = bcast(m21 - m63)

    # ===== 2. Factor spreads (7) =====
    def _spread(a, b, w):
        if a in factor_ret.columns and b in factor_ret.columns:
            return factor_ret[a].rolling(w).sum() - factor_ret[b].rolling(w).sum()
        return None

    for w in [21, 63]:
        s = _spread("F_Value", "F_Growth", w)
        if s is not None:
            features[f"fac_value_growth_{w}d"] = bcast(s)

    for name, a, b in [
        ("risk_on_off", "F_HiBeta", "F_MinVol"),
        ("qual_growth", "F_Quality", "F_Growth"),
        ("hidiv_growth", "F_HiDiv", "F_Growth"),
    ]:
        s = _spread(a, b, 21)
        if s is not None:
            features[f"fac_{name}_21d"] = bcast(s)

    # ===== 3. Market index features (~8) =====
    for idx in ["SPX", "NDX"]:
        if idx not in factor_ret.columns:
            continue
        for w in [21, 63]:
            features[f"fac_{idx}_mom_{w}d"] = bcast(factor_ret[idx].rolling(w, min_periods=w).sum())

    if "MXEF" in factor_ret.columns and "MXWD" in factor_ret.columns:
        s = factor_ret["MXEF"].rolling(21).sum() - factor_ret["MXWD"].rolling(21).sum()
        features["fac_em_dev_spread"] = bcast(s)
    if "SPX" in factor_ret.columns and "SX5E" in factor_ret.columns:
        s = factor_ret["SPX"].rolling(21).sum() - factor_ret["SX5E"].rolling(21).sum()
        features["fac_us_eu_spread"] = bcast(s)

    # ===== 4. VIX / SKEW (~8) =====
    if factor_px is not None and "VIX" in factor_px.columns:
        vix = factor_px["VIX"]
        features["fac_VIX_level"] = bcast(vix)
        features["fac_VIX_chg_5d"] = bcast(vix - vix.shift(5))
        rm = vix.rolling(63).mean()
        rs = vix.rolling(63).std().replace(0, np.nan)
        features["fac_VIX_zscore"] = bcast((vix - rm) / rs)
        # Binary VIX regime
        features["fac_VIX_elevated"] = bcast((vix > 20).astype(float))
        features["fac_VIX_panic"] = bcast((vix > 30).astype(float))
        # VRP proxy: VIX vs realized vol of SPX
        if "SPX" in factor_ret.columns:
            rv = factor_ret["SPX"].rolling(21).std() * np.sqrt(252) * 100
            features["fac_vrp_proxy"] = bcast(vix - rv)

    if factor_px is not None and "SKEW" in factor_px.columns:
        features["fac_SKEW_level"] = bcast(factor_px["SKEW"])
        features["fac_SKEW_chg"] = bcast(factor_px["SKEW"] - factor_px["SKEW"].shift(21))

    # ===== 5. Rates (~8) =====
    if factor_px is not None:
        if "UST_10Y" in factor_px.columns and "UST_2Y" in factor_px.columns:
            slope = factor_px["UST_10Y"] - factor_px["UST_2Y"]
            features["fac_yield_slope"] = bcast(slope)
            features["fac_yield_slope_chg"] = bcast(slope - slope.shift(21))
            # Curve regime
            features["fac_curve_inverted"] = bcast((slope < 0).astype(float))
            features["fac_curve_steep"] = bcast((slope > 1.0).astype(float))

        if "UST_10Y" in factor_px.columns and "US_BEI10" in factor_px.columns:
            real = factor_px["UST_10Y"] - factor_px["US_BEI10"]
            features["fac_real_rate"] = bcast(real)
            features["fac_real_rate_chg"] = bcast(real - real.shift(21))

        if "UST_10Y" in factor_px.columns:
            r10 = factor_px["UST_10Y"]
            features["fac_rate_vol"] = bcast(r10.diff().rolling(21).std())

        if "UST_10Y" in factor_px.columns and "GER_10Y" in factor_px.columns:
            features["fac_us_ger_spread"] = bcast(factor_px["UST_10Y"] - factor_px["GER_10Y"])

    # ===== 6. FX (~6) =====
    if factor_px is not None and "DXY" in factor_px.columns:
        dxy = factor_px["DXY"]
        rm = dxy.rolling(63).mean()
        rs = dxy.rolling(63).std().replace(0, np.nan)
        features["fac_dxy_zscore"] = bcast((dxy - rm) / rs)
        features["fac_dxy_mom_21d"] = bcast(dxy.pct_change(21))
        features["fac_usd_strong"] = bcast(((dxy - rm) / rs > 1).astype(float))

    if factor_px is not None and "USDKRW" in factor_px.columns:
        features["fac_usdkrw_chg"] = bcast(factor_px["USDKRW"].pct_change(21))

    if factor_ret is not None:
        em_cols = [c for c in ["USDKRW", "USDCNH"] if c in factor_ret.columns]
        if em_cols:
            em_avg = factor_ret[em_cols].mean(axis=1)
            features["fac_em_fx_mom"] = bcast(em_avg.rolling(21).sum())

    # ===== 7. Commodities (~6) =====
    for c in ["WTI", "GOLD"]:
        if c in factor_ret.columns:
            for w in [21, 63]:
                features[f"fac_{c}_mom_{w}d"] = bcast(factor_ret[c].rolling(w).sum())
    if "COPPER" in factor_ret.columns and "GOLD" in factor_ret.columns:
        # Copper/Gold ratio = risk appetite proxy
        if factor_px is not None and "COPPER" in factor_px.columns and "GOLD" in factor_px.columns:
            cg = factor_px["COPPER"] / factor_px["GOLD"].replace(0, np.nan)
            features["fac_copper_gold"] = bcast(cg.pct_change(21))
    if "BCOM" in factor_ret.columns:
        features["fac_cmd_mom_21d"] = bcast(factor_ret["BCOM"].rolling(21).sum())

    # ===== 8. GS Thematic (~3) =====
    for t in ["GS_AI", "GS_Nuclear", "GS_SemiHW"]:
        if t in factor_ret.columns:
            features[f"fac_{t}_mom"] = bcast(factor_ret[t].rolling(21).sum())

    # ===== 9. Macro Sentiment (~5) =====
    if factor_px is not None:
        if "CESI_US" in factor_px.columns:
            cesi = factor_px["CESI_US"]
            features["fac_cesi_level"] = bcast(cesi)
            features["fac_cesi_chg"] = bcast(cesi - cesi.shift(21))
        if "AAII_Bull" in factor_px.columns and "AAII_Bear" in factor_px.columns:
            spread = factor_px["AAII_Bull"] - factor_px["AAII_Bear"]
            features["fac_aaii_spread"] = bcast(spread)
            features["fac_aaii_chg"] = bcast(spread - spread.shift(21))

    return features


# ===========================================================================
# Assembly
# ===========================================================================

def build_all_features(
    data: UniverseData,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    accounting = build_accounting_features(data)
    price = build_price_features(data)
    sellside = build_sellside_features(data)
    conditioning = build_conditioning_features(data)
    factor = build_factor_features(data)

    feature_groups = {
        "Accounting": list(accounting.keys()),
        "Price": list(price.keys()),
        "Sellside": list(sellside.keys()),
        "Conditioning": list(conditioning.keys()),
        "Factor": list(factor.keys()),
    }

    all_features: Dict[str, pd.DataFrame] = {}
    all_features.update(accounting)
    all_features.update(price)
    all_features.update(sellside)
    all_features.update(conditioning)
    all_features.update(factor)

    tickers = [t for t in TICKERS if t in data.returns.columns]

    # CS Z-score: conditioning / factor(broadcast)는 제외
    skip_zscore = set(conditioning.keys()) | set(factor.keys())
    for name, df in all_features.items():
        if name not in skip_zscore:
            all_features[name] = cross_sectional_zscore(df)

    # 극단값 클리핑
    for name, df in all_features.items():
        all_features[name] = clip_outliers(df)

    # 3D → 2D panel
    records = []
    for feat_name, df in all_features.items():
        stacked = df.stack()
        stacked.name = feat_name
        records.append(stacked)

    panel = pd.concat(records, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.fillna(0)

    feature_names = list(all_features.keys())

    print(f"[FeatureEngine] 총 피처 수: {len(feature_names)}")
    for group, names in feature_groups.items():
        print(f"  {group:15s}: {len(names)}개")

    return panel, feature_names, feature_groups
