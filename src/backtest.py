"""
Phase 5: Walk-Forward 백테스트
- train_window: 756일(3년)
- retrain_freq: 63일(3개월)
- prediction_horizon: 20일
- rebalance_freq: 5일(주간)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train, TRAIN_WINDOW
from src.portfolio_optimizer import optimize_portfolio, estimate_covariance

REBALANCE_FREQ = 10  # 격주 리밸런싱
ONE_WAY_TC = 0.0010  # 편도 거래비용 10bps (대형주 기준)


class BacktestResult:
    """백테스트 결과 컨테이너."""

    def __init__(self):
        self.portfolio_weights: Dict[pd.Timestamp, pd.Series] = {}
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        self.turnover: pd.Series = pd.Series(dtype=float)
        self.predictions: Optional[pd.DataFrame] = None
        self.targets: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.ic_series: pd.Series = pd.Series(dtype=float)
        self.panel: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_groups: Optional[Dict] = None

    @property
    def cumulative_returns(self) -> pd.Series:
        return (1 + self.portfolio_returns).cumprod()

    @property
    def cumulative_benchmark(self) -> pd.Series:
        return (1 + self.benchmark_returns).cumprod()

    @property
    def active_returns(self) -> pd.Series:
        return self.portfolio_returns - self.benchmark_returns

    def compute_metrics(self) -> Dict[str, float]:
        """주요 성과 지표 계산."""
        port = self.portfolio_returns.dropna()
        bm = self.benchmark_returns.dropna()
        active = self.active_returns.dropna()

        ann_factor = 252

        # Portfolio metrics
        ann_ret = port.mean() * ann_factor
        ann_vol = port.std() * np.sqrt(ann_factor)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # Active metrics
        active_ret = active.mean() * ann_factor
        active_vol = active.std() * np.sqrt(ann_factor)
        ir = active_ret / active_vol if active_vol > 0 else 0

        # Drawdown
        cum = self.cumulative_returns
        rolling_max = cum.cummax()
        drawdown = (cum / rolling_max) - 1
        max_dd = drawdown.min()

        # Turnover: 리밸런싱당 평균 turnover * 연간 리밸런싱 횟수
        if len(self.turnover) > 0:
            n_years = len(port) / ann_factor
            total_turnover = self.turnover.sum()
            avg_turnover = total_turnover / n_years if n_years > 0 else 0
        else:
            avg_turnover = 0

        # IC
        avg_ic = self.ic_series.mean() if len(self.ic_series) > 0 else 0

        # 연간 거래비용
        annual_tc = avg_turnover * ONE_WAY_TC

        return {
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "active_return": active_ret,
            "tracking_error": active_vol,
            "information_ratio": ir,
            "max_drawdown": max_dd,
            "avg_annual_turnover": avg_turnover,
            "avg_ic": avg_ic,
            "annual_tc": annual_tc,
        }

    def summary(self) -> str:
        m = self.compute_metrics()
        lines = [
            "=" * 50,
            "백테스트 결과 요약",
            "=" * 50,
            f"  연간 수익률:      {m['annual_return']:.2%}",
            f"  연간 변동성:      {m['annual_vol']:.2%}",
            f"  Sharpe Ratio:     {m['sharpe_ratio']:.2f}",
            f"  Active Return:    {m['active_return']:.2%}",
            f"  Tracking Error:   {m['tracking_error']:.2%}",
            f"  Information Ratio:{m['information_ratio']:.2f}",
            f"  Max Drawdown:     {m['max_drawdown']:.2%}",
            f"  연간 Turnover:    {m['avg_annual_turnover']:.0%}",
            f"  연간 거래비용:    {m['annual_tc']:.2%} (편도 {ONE_WAY_TC*10000:.0f}bps)",
            f"  평균 IC:          {m['avg_ic']:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


def _get_sector_map(data: UniverseData) -> Dict[str, str]:
    """종목별 섹터 매핑 추출."""
    meta = data.meta
    sector_map = {}

    if isinstance(meta, pd.DataFrame):
        if "sector" in meta.columns:
            for ticker in TICKERS:
                if ticker in meta.index:
                    sector_map[ticker] = str(meta.loc[ticker, "sector"])
        elif len(meta.columns) > 0:
            col = meta.columns[0]
            for ticker in TICKERS:
                if ticker in meta.index:
                    sector_map[ticker] = str(meta.loc[ticker, col])

    return sector_map


def compute_ic(predictions: pd.Series, realized: pd.Series) -> float:
    """Information Coefficient (rank correlation)."""
    valid = predictions.notna() & realized.notna()
    if valid.sum() < 3:
        return np.nan
    return predictions[valid].corr(realized[valid], method="spearman")


def run_backtest(
    data: UniverseData,
    rebalance_freq: int = REBALANCE_FREQ,
) -> BacktestResult:
    """전체 백테스트 실행."""
    result = BacktestResult()

    print("[Backtest] Phase 2: 피처 생성 중...")
    panel, feature_names, feature_groups = build_all_features(data)

    print("[Backtest] Phase 3: 타겟 생성 중...")
    targets = build_targets(data)

    print("[Backtest] Phase 4: 모델 학습 및 예측 중...")
    all_dates = data.dates
    models, predictions = walk_forward_train(
        panel, targets, feature_names, all_dates,
    )
    result.models = models
    result.predictions = predictions
    result.targets = targets  # IC 계산에 사용
    result.panel = panel
    result.feature_names = feature_names
    result.feature_groups = feature_groups

    print("[Backtest] Phase 5-6: 포트폴리오 구축 중...")
    returns = data.returns
    mktcap = data.market_cap
    tickers = [t for t in TICKERS if t in returns.columns]
    n_tickers = len(tickers)
    sector_map = _get_sector_map(data)

    prev_weights = np.ones(n_tickers) / n_tickers
    port_rets = []
    bm_rets = []
    turnovers = []
    ic_values = []
    weight_history = {}

    # 예측이 시작되는 날짜부터
    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        print("[Backtest] 유효한 예측이 없습니다.")
        return result

    start_idx = all_dates.get_loc(pred_valid.index[0])

    first_rebal = True

    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]

        # 리밸런싱 시점: 10일(격주) 간격
        is_rebal = ((t_idx - start_idx) % rebalance_freq == 0) or first_rebal
        if is_rebal:
            pred_row = predictions.loc[t_date, tickers]

            if pred_row.notna().sum() >= 10:
                # 시가총액 가중(MCW) 벤치마크
                mc_row = mktcap.loc[t_date, tickers]
                mc_sum = mc_row.sum()
                bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n_tickers) / n_tickers

                # 공분산 추정 (mega-cap 변동성 조정 포함)
                hist_start = max(0, t_idx - 126)
                hist_returns = returns[tickers].iloc[hist_start:t_idx]
                cov_matrix = estimate_covariance(hist_returns, bm_weights=bm_w)

                # 최적화
                new_weights = optimize_portfolio(
                    expected_returns=pred_row,
                    cov_matrix=cov_matrix,
                    prev_weights=prev_weights,
                    sector_map=sector_map if sector_map else None,
                    bm_weights=bm_w,
                )

                turnover = np.abs(new_weights - prev_weights).sum()
                turnovers.append((t_date, turnover))
                weight_history[t_date] = pd.Series(new_weights, index=tickers)
                prev_weights = new_weights
                first_rebal = False

        # 일간 수익률
        daily_ret = returns.loc[t_date, tickers].values
        if np.any(np.isnan(daily_ret)):
            daily_ret = np.nan_to_num(daily_ret, 0)

        port_ret = np.dot(prev_weights, daily_ret)

        # 거래비용 차감 (리밸런싱 당일)
        if turnovers and turnovers[-1][0] == t_date:
            tc_cost = turnovers[-1][1] * ONE_WAY_TC
            port_ret -= tc_cost
        # 시가총액 가중(MCW) 벤치마크
        mc_row_daily = mktcap.loc[t_date, tickers]
        mc_sum_daily = mc_row_daily.sum()
        bm_w_daily = (mc_row_daily / mc_sum_daily).values if mc_sum_daily > 0 else np.ones(n_tickers) / n_tickers
        bm_ret = np.dot(bm_w_daily, daily_ret)

        port_rets.append((t_date, port_ret))
        bm_rets.append((t_date, bm_ret))

        # IC 계산 (리밸런싱 시점): 예측 vs 실현 specific return
        if turnovers and turnovers[-1][0] == t_date:
            pred_row = predictions.loc[t_date, tickers]
            # specific return(타겟)이 있으면 사용, 없으면 forward return
            if t_date in targets.index:
                realized = targets.loc[t_date, tickers]
            elif t_idx + 20 < len(all_dates):
                realized = returns[tickers].iloc[t_idx + 1: t_idx + 21].sum()
            else:
                realized = None
            if realized is not None:
                ic = compute_ic(pred_row, realized)
                if not np.isnan(ic):
                    ic_values.append((t_date, ic))

    # 결과 정리
    result.portfolio_returns = pd.Series(
        dict(port_rets), name="portfolio"
    ).sort_index()
    result.benchmark_returns = pd.Series(
        dict(bm_rets), name="benchmark"
    ).sort_index()
    result.turnover = pd.Series(
        dict(turnovers), name="turnover"
    ).sort_index()
    result.portfolio_weights = weight_history
    result.ic_series = pd.Series(
        dict(ic_values), name="IC"
    ).sort_index()

    print(result.summary())
    return result
