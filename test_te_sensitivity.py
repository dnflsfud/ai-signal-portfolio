"""
TE Sensitivity Test: 12.5% vs 15% vs 17.5% vs 20%
구조 변경 없이 MAX_TE_ANNUAL만 변경하여 알파 비교
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train
from src.portfolio_optimizer import optimize_portfolio, estimate_covariance
from src.backtest import BacktestResult, compute_ic, _get_sector_map, REBALANCE_FREQ, ONE_WAY_TC

import src.portfolio_optimizer as po


def run_backtest_with_te(data, panel, feature_names, feature_groups, targets, models, predictions, te_annual):
    """특정 TE로 백테스트 실행 (모델 재학습 없이 포트폴리오만 재구축)."""
    result = BacktestResult()
    result.models = models
    result.predictions = predictions
    result.targets = targets
    result.panel = panel
    result.feature_names = feature_names
    result.feature_groups = feature_groups

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

    all_dates = data.dates
    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        return result

    start_idx = all_dates.get_loc(pred_valid.index[0])
    first_rebal = True
    rebalance_freq = REBALANCE_FREQ

    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]

        is_rebal = ((t_idx - start_idx) % rebalance_freq == 0) or first_rebal
        if is_rebal:
            pred_row = predictions.loc[t_date, tickers]

            if pred_row.notna().sum() >= 10:
                mc_row = mktcap.loc[t_date, tickers]
                mc_sum = mc_row.sum()
                bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n_tickers) / n_tickers

                hist_start = max(0, t_idx - 126)
                hist_returns = returns[tickers].iloc[hist_start:t_idx]
                cov_matrix = estimate_covariance(hist_returns, bm_weights=bm_w)

                new_weights = optimize_portfolio(
                    expected_returns=pred_row,
                    cov_matrix=cov_matrix,
                    prev_weights=prev_weights,
                    sector_map=sector_map if sector_map else None,
                    bm_weights=bm_w,
                    max_te_annual=te_annual,  # TE 변경!
                )

                turnover = np.abs(new_weights - prev_weights).sum()
                turnovers.append((t_date, turnover))
                weight_history[t_date] = pd.Series(new_weights, index=tickers)
                prev_weights = new_weights
                first_rebal = False

        daily_ret = returns.loc[t_date, tickers].values
        if np.any(np.isnan(daily_ret)):
            daily_ret = np.nan_to_num(daily_ret, 0)

        port_ret = np.dot(prev_weights, daily_ret)

        if turnovers and turnovers[-1][0] == t_date:
            tc_cost = turnovers[-1][1] * ONE_WAY_TC
            port_ret -= tc_cost

        mc_row_daily = mktcap.loc[t_date, tickers]
        mc_sum_daily = mc_row_daily.sum()
        bm_w_daily = (mc_row_daily / mc_sum_daily).values if mc_sum_daily > 0 else np.ones(n_tickers) / n_tickers
        bm_ret = np.dot(bm_w_daily, daily_ret)

        port_rets.append((t_date, port_ret))
        bm_rets.append((t_date, bm_ret))

        if turnovers and turnovers[-1][0] == t_date:
            pred_row = predictions.loc[t_date, tickers]
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

    result.portfolio_returns = pd.Series(dict(port_rets), name="portfolio").sort_index()
    result.benchmark_returns = pd.Series(dict(bm_rets), name="benchmark").sort_index()
    result.turnover = pd.Series(dict(turnovers), name="turnover").sort_index()
    result.portfolio_weights = weight_history
    result.ic_series = pd.Series(dict(ic_values), name="IC").sort_index()

    return result


def main():
    data_path = str(PROJECT_ROOT / "data" / "ai_signal_data.xlsx")

    print("=" * 70)
    print("  TE Sensitivity Analysis: 12.5% vs 15% vs 17.5% vs 20%")
    print("=" * 70)

    # Phase 1: 공통 데이터/모델 (1번만 실행)
    print("\n[1/2] 데이터 로드 & 모델 학습 (공통)...")
    data = UniverseData(data_path)
    panel, feature_names, feature_groups = build_all_features(data)
    targets = build_targets(data)
    all_dates = data.dates
    models, predictions = walk_forward_train(panel, targets, feature_names, all_dates)

    # Phase 2: 각 TE로 포트폴리오 구축
    te_values = [0.125, 0.15, 0.175, 0.20]
    results = {}

    for te in te_values:
        label = f"TE={te:.1%}"
        print(f"\n[2/2] {label} 포트폴리오 구축 중...")
        r = run_backtest_with_te(data, panel, feature_names, feature_groups, targets, models, predictions, te)
        results[label] = r.compute_metrics()
        print(f"  -> Active Return: {results[label]['active_return']:.2%}, "
              f"TE: {results[label]['tracking_error']:.2%}, "
              f"IR: {results[label]['information_ratio']:.2f}")

    # 결과 비교표
    print("\n")
    print("=" * 90)
    print("  TE Sensitivity 결과 비교")
    print("=" * 90)

    header = f"{'Metric':<25}"
    for te in te_values:
        header += f"{'TE=' + f'{te:.1%}':>15}"
    print(header)
    print("-" * 90)

    metrics_display = [
        ("Annual Return", "annual_return", True),
        ("Annual Vol", "annual_vol", True),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Active Return (Alpha)", "active_return", True),
        ("Tracking Error (실현)", "tracking_error", True),
        ("Information Ratio", "information_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Annual Turnover", "avg_annual_turnover", True),
        ("Annual TC", "annual_tc", True),
        ("Avg IC", "avg_ic", False),
    ]

    for label, key, is_pct in metrics_display:
        row = f"{label:<25}"
        for te in te_values:
            val = results[f"TE={te:.1%}"][key]
            if is_pct:
                if key == "avg_annual_turnover":
                    row += f"{val:>14.0%} "
                else:
                    row += f"{val:>14.2%} "
            else:
                row += f"{val:>14.4f} "
        print(row)

    print("-" * 90)

    # Alpha 변화 분석
    base_alpha = results["TE=12.5%"]["active_return"]
    print(f"\n  기준 Alpha (TE=12.5%): {base_alpha:.2%}")
    for te in te_values[1:]:
        label = f"TE={te:.1%}"
        new_alpha = results[label]["active_return"]
        diff = new_alpha - base_alpha
        pct_change = (diff / abs(base_alpha) * 100) if base_alpha != 0 else 0
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {label}: Alpha {new_alpha:.2%} ({arrow} {abs(diff):.2%}, {pct_change:+.1f}%)")

    # IR 변화
    base_ir = results["TE=12.5%"]["information_ratio"]
    print(f"\n  기준 IR (TE=12.5%): {base_ir:.2f}")
    for te in te_values[1:]:
        label = f"TE={te:.1%}"
        new_ir = results[label]["information_ratio"]
        diff = new_ir - base_ir
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {label}: IR {new_ir:.2f} ({arrow} {abs(diff):.2f})")

    print("\n" + "=" * 90)
    print("  결론")
    print("=" * 90)
    best_alpha_te = max(te_values, key=lambda t: results[f"TE={t:.1%}"]["active_return"])
    best_ir_te = max(te_values, key=lambda t: results[f"TE={t:.1%}"]["information_ratio"])
    print(f"  최고 Alpha: TE={best_alpha_te:.1%} -> {results[f'TE={best_alpha_te:.1%}']['active_return']:.2%}")
    print(f"  최고 IR:    TE={best_ir_te:.1%} -> {results[f'TE={best_ir_te:.1%}']['information_ratio']:.2f}")

    if best_alpha_te > 0.125:
        print(f"\n  -> TE를 {best_alpha_te:.1%}로 확대하면 Alpha가 상승합니다.")
    else:
        print(f"\n  -> 현재 TE=12.5%가 최적이며, TE 확대 시 Alpha가 오히려 감소합니다.")

    print("=" * 90)


if __name__ == "__main__":
    main()