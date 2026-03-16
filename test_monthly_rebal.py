"""
Quick Test: 월말 1회 리밸런싱 + 일간 weight drift
현재 설정(TE≤2.5%, 거래비용 10bps)과 비교.
"""

import sys
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train
from src.portfolio_optimizer import optimize_portfolio, estimate_covariance
from src.backtest import _get_sector_map, compute_ic, ONE_WAY_TC


def is_month_end(date, all_dates):
    """영업일 기준 월말인지 확인."""
    idx = all_dates.get_loc(date)
    if idx + 1 >= len(all_dates):
        return True
    next_date = all_dates[idx + 1]
    return date.month != next_date.month


def run_monthly_backtest(predictions, data, tickers, sector_map, all_dates):
    """월말 1회 리밸런싱 + 일간 weight drift."""
    returns = data.returns
    mktcap = data.market_cap
    n = len(tickers)

    weights = np.ones(n) / n  # 초기 EW
    port_rets = []
    bm_rets = []
    turnovers = []
    rebal_count = 0

    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        return None

    start_idx = all_dates.get_loc(pred_valid.index[0])
    last_rebal_month = None

    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]
        cur_month = (t_date.year, t_date.month)

        # 월말 리밸런싱 체크
        do_rebal = False
        if is_month_end(t_date, all_dates) and cur_month != last_rebal_month:
            pred_row = predictions.loc[t_date, tickers]
            if pred_row.notna().sum() >= 10:
                do_rebal = True

        if do_rebal:
            hist_start = max(0, t_idx - 126)
            hist_returns = returns[tickers].iloc[hist_start:t_idx]
            cov_matrix = estimate_covariance(hist_returns)

            mc_row = mktcap.loc[t_date, tickers]
            mc_sum = mc_row.sum()
            bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n) / n

            new_weights = optimize_portfolio(
                expected_returns=pred_row,
                cov_matrix=cov_matrix,
                prev_weights=weights,
                sector_map=sector_map if sector_map else None,
                bm_weights=bm_w,
            )

            turnover = np.abs(new_weights - weights).sum()
            turnovers.append(turnover)
            weights = new_weights
            last_rebal_month = cur_month
            rebal_count += 1

        # 일간 수익률
        daily_ret = returns.loc[t_date, tickers].values
        if np.any(np.isnan(daily_ret)):
            daily_ret = np.nan_to_num(daily_ret, 0)

        port_ret = np.dot(weights, daily_ret)

        # 거래비용 (리밸런싱 당일)
        if do_rebal and turnovers:
            tc_cost = turnovers[-1] * ONE_WAY_TC
            port_ret -= tc_cost

        # BM
        mc_row = mktcap.loc[t_date, tickers]
        mc_sum = mc_row.sum()
        bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n) / n
        bm_ret = np.dot(bm_w, daily_ret)

        port_rets.append(port_ret)
        bm_rets.append(bm_ret)

        # Weight drift: 수익률에 따라 비중 변동
        weights = weights * (1 + daily_ret)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum

    # 결과 계산
    port_rets = np.array(port_rets)
    bm_rets = np.array(bm_rets)
    active = port_rets - bm_rets

    ann_ret = np.mean(port_rets) * 252
    bm_ann_ret = np.mean(bm_rets) * 252
    active_ret = np.mean(active) * 252
    te = np.std(active) * np.sqrt(252)
    ir = active_ret / te if te > 0 else 0
    sharpe = ann_ret / (np.std(port_rets) * np.sqrt(252)) if np.std(port_rets) > 0 else 0

    cum = np.cumprod(1 + port_rets)
    rolling_max = np.maximum.accumulate(cum)
    max_dd = np.min(cum / rolling_max - 1)

    n_years = len(port_rets) / 252
    total_to = sum(turnovers)
    avg_turnover = total_to / n_years if n_years > 0 else 0
    annual_tc = avg_turnover * ONE_WAY_TC

    return {
        "annual_return": ann_ret,
        "bm_return": bm_ann_ret,
        "active_return": active_ret,
        "tracking_error": te,
        "information_ratio": ir,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "annual_tc": annual_tc,
        "rebal_count": rebal_count,
        "n_days": len(port_rets),
    }


def run_current_backtest(predictions, data, tickers, sector_map, all_dates):
    """현재 설정: 10일 주기 리밸런싱 (drift 없음)."""
    from src.backtest import REBALANCE_FREQ
    returns = data.returns
    mktcap = data.market_cap
    n = len(tickers)

    weights = np.ones(n) / n
    port_rets = []
    bm_rets = []
    turnovers = []

    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        return None

    start_idx = all_dates.get_loc(pred_valid.index[0])
    last_rebal_idx = start_idx - REBALANCE_FREQ

    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]

        if t_idx - last_rebal_idx >= REBALANCE_FREQ:
            pred_row = predictions.loc[t_date, tickers]
            if pred_row.notna().sum() >= 10:
                hist_start = max(0, t_idx - 126)
                hist_returns = returns[tickers].iloc[hist_start:t_idx]
                cov_matrix = estimate_covariance(hist_returns)

                mc_row = mktcap.loc[t_date, tickers]
                mc_sum = mc_row.sum()
                bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n) / n

                new_weights = optimize_portfolio(
                    expected_returns=pred_row,
                    cov_matrix=cov_matrix,
                    prev_weights=weights,
                    sector_map=sector_map if sector_map else None,
                    bm_weights=bm_w,
                )

                turnover = np.abs(new_weights - weights).sum()
                turnovers.append(turnover)
                weights = new_weights
                last_rebal_idx = t_idx

        daily_ret = returns.loc[t_date, tickers].values
        if np.any(np.isnan(daily_ret)):
            daily_ret = np.nan_to_num(daily_ret, 0)

        port_ret = np.dot(weights, daily_ret)
        if turnovers and last_rebal_idx == t_idx:
            tc_cost = turnovers[-1] * ONE_WAY_TC
            port_ret -= tc_cost

        mc_row = mktcap.loc[t_date, tickers]
        mc_sum = mc_row.sum()
        bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n) / n
        bm_ret = np.dot(bm_w, daily_ret)

        port_rets.append(port_ret)
        bm_rets.append(bm_ret)

    port_rets = np.array(port_rets)
    bm_rets = np.array(bm_rets)
    active = port_rets - bm_rets

    ann_ret = np.mean(port_rets) * 252
    active_ret = np.mean(active) * 252
    te = np.std(active) * np.sqrt(252)
    ir = active_ret / te if te > 0 else 0
    sharpe = ann_ret / (np.std(port_rets) * np.sqrt(252)) if np.std(port_rets) > 0 else 0

    cum = np.cumprod(1 + port_rets)
    rolling_max = np.maximum.accumulate(cum)
    max_dd = np.min(cum / rolling_max - 1)

    n_years = len(port_rets) / 252
    avg_turnover = sum(turnovers) / n_years if n_years > 0 else 0
    annual_tc = avg_turnover * ONE_WAY_TC

    return {
        "annual_return": ann_ret,
        "active_return": active_ret,
        "tracking_error": te,
        "information_ratio": ir,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "annual_tc": annual_tc,
        "rebal_count": len(turnovers),
        "n_days": len(port_rets),
    }


def main():
    print("=" * 70)
    print("  월말 1회 리밸런싱 vs 현재(10일 주기) 비교 테스트")
    print("  TE ≤ 2.5%, 거래비용 편도 10bps")
    print("=" * 70)

    t0 = time.time()
    print("\n데이터 로드 & 모델 학습...")
    data = UniverseData("./data/ai_signal_data.xlsx")
    panel, feature_names, _ = build_all_features(data)
    targets = build_targets(data)
    all_dates = data.dates
    _, predictions = walk_forward_train(panel, targets, feature_names, all_dates)

    tickers = [t for t in TICKERS if t in data.returns.columns]
    sector_map = _get_sector_map(data)
    print(f"  학습 완료: {time.time()-t0:.0f}초\n")

    # 현재 설정 (10일 주기)
    print("=" * 70)
    print("[A] 현재: 10일 주기 리밸런싱 (no drift)")
    print("=" * 70)
    t1 = time.time()
    result_current = run_current_backtest(predictions, data, tickers, sector_map, all_dates)
    print(f"  완료: {time.time()-t1:.0f}초")

    # 월말 1회 리밸런싱
    print(f"\n{'='*70}")
    print("[B] 월말 1회 리밸런싱 (with daily drift)")
    print("=" * 70)
    t2 = time.time()
    result_monthly = run_monthly_backtest(predictions, data, tickers, sector_map, all_dates)
    print(f"  완료: {time.time()-t2:.0f}초")

    # 비교 출력
    print(f"\n{'='*70}")
    print("  비교 결과")
    print("=" * 70)

    header = f"  {'지표':<20} | {'[A] 10일 주기':>15} | {'[B] 월말 1회':>15} | {'차이':>10}"
    print(header)
    print("  " + "-" * 68)

    metrics = [
        ("연간 수익률", "annual_return", True),
        ("Active Return", "active_return", True),
        ("Tracking Error", "tracking_error", True),
        ("Information Ratio", "information_ratio", False),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("연간 Turnover", "avg_turnover", True),
        ("연간 거래비용", "annual_tc", True),
        ("리밸런싱 횟수", "rebal_count", False),
    ]

    for label, key, is_pct in metrics:
        a = result_current[key]
        b = result_monthly[key]
        diff = b - a
        if is_pct:
            print(f"  {label:<20} | {a:>14.2%} | {b:>14.2%} | {diff:>+9.2%}")
        else:
            print(f"  {label:<20} | {a:>15.2f} | {b:>15.2f} | {diff:>+10.2f}")

    print("=" * 70)

    # 승자 판정
    ar_a = result_current["active_return"]
    ar_b = result_monthly["active_return"]
    te_a = result_current["tracking_error"]
    te_b = result_monthly["tracking_error"]
    tc_a = result_current["annual_tc"]
    tc_b = result_monthly["annual_tc"]

    print(f"\n  ★ Active Return:  {'월말' if ar_b > ar_a else '10일주기'} 승 ({ar_b:.2%} vs {ar_a:.2%})")
    print(f"  ★ TE:             {'월말' if te_b < te_a else '10일주기'} 승 ({te_b:.2%} vs {te_a:.2%})")
    print(f"  ★ 거래비용:       {'월말' if tc_b < tc_a else '10일주기'} 승 ({tc_b:.2%} vs {tc_a:.2%})")
    print(f"  ★ Sharpe:         {'월말' if result_monthly['sharpe_ratio'] > result_current['sharpe_ratio'] else '10일주기'} 승")

    print(f"\n  총 소요시간: {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
