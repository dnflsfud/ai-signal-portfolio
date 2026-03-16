"""
Grid Search v2: 옵티마이저 파라미터 최적화 (경량 버전)
TE 하드캡 15% 고정, 핵심 파라미터만 집중 탐색.
전략: 2단계 — Phase 1 coarse grid → Phase 2 fine grid around best.
"""

import sys
import warnings
import itertools
import time
import numpy as np
import pandas as pd
from pathlib import Path

# suppress solver warnings to reduce I/O
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train
from src.portfolio_optimizer import optimize_portfolio, estimate_covariance
from src.backtest import _get_sector_map, REBALANCE_FREQ


def run_optimization_only(
    predictions, data, tickers, sector_map, all_dates, rebalance_freq,
    risk_aversion, turnover_penalty, max_te_annual, max_single_turnover, sector_deviation,
):
    """모델 예측값 → 옵티마이저 파라미터로 포트폴리오 구축 + 성과."""
    returns = data.returns
    mktcap = data.market_cap
    n_tickers = len(tickers)

    prev_weights = np.ones(n_tickers) / n_tickers
    port_rets = []
    bm_rets = []
    turnovers = []

    pred_valid = predictions.dropna(how="all")
    if len(pred_valid) == 0:
        return None

    start_idx = all_dates.get_loc(pred_valid.index[0])
    last_rebal_idx = start_idx - rebalance_freq

    for t_idx in range(start_idx, len(all_dates)):
        t_date = all_dates[t_idx]

        if t_idx - last_rebal_idx >= rebalance_freq:
            pred_row = predictions.loc[t_date, tickers]
            if pred_row.notna().sum() >= 10:
                hist_start = max(0, t_idx - 126)
                hist_returns = returns[tickers].iloc[hist_start:t_idx]
                cov_matrix = estimate_covariance(hist_returns)

                mc_row = mktcap.loc[t_date, tickers]
                mc_sum = mc_row.sum()
                bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n_tickers) / n_tickers

                new_weights = optimize_portfolio(
                    expected_returns=pred_row,
                    cov_matrix=cov_matrix,
                    prev_weights=prev_weights,
                    sector_map=sector_map if sector_map else None,
                    bm_weights=bm_w,
                    risk_aversion=risk_aversion,
                    turnover_penalty=turnover_penalty,
                    max_te_annual=max_te_annual,
                    sector_deviation=sector_deviation,
                )

                turnover = np.abs(new_weights - prev_weights).sum()
                turnovers.append(turnover)
                prev_weights = new_weights
                last_rebal_idx = t_idx

        daily_ret = returns.loc[t_date, tickers].values
        if np.any(np.isnan(daily_ret)):
            daily_ret = np.nan_to_num(daily_ret, 0)

        port_ret = np.dot(prev_weights, daily_ret)
        mc_row = mktcap.loc[t_date, tickers]
        mc_sum = mc_row.sum()
        bm_w = (mc_row / mc_sum).values if mc_sum > 0 else np.ones(n_tickers) / n_tickers
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

    return {
        "annual_return": ann_ret,
        "active_return": active_ret,
        "tracking_error": te,
        "information_ratio": ir,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
    }


def grid_search(predictions, data, tickers, sector_map, all_dates, param_grid):
    """파라미터 그리드 탐색."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)
    print(f"  총 {total}개 조합 탐색")

    results = []
    t0 = time.time()

    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        try:
            metrics = run_optimization_only(
                predictions, data, tickers, sector_map, all_dates,
                REBALANCE_FREQ, **params,
            )
        except Exception:
            metrics = None

        if metrics is not None:
            results.append({**params, **metrics})

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("  Grid Search v2: TE <= 15% Active Return 최대화")
    print("=" * 70)

    # ============================================================
    # Phase 1: 데이터 + 모델 (1회)
    # ============================================================
    print("\n[Phase 0] 데이터 로드 & 모델 학습...")
    t0 = time.time()

    data = UniverseData("./data/ai_signal_data.xlsx")
    panel, feature_names, feature_groups = build_all_features(data)
    targets = build_targets(data)
    all_dates = data.dates
    models, predictions = walk_forward_train(panel, targets, feature_names, all_dates)

    tickers = [t for t in TICKERS if t in data.returns.columns]
    sector_map = _get_sector_map(data)

    print(f"  모델 학습 완료: {time.time()-t0:.0f}초\n")

    # ============================================================
    # Phase 1: Coarse Grid (핵심 파라미터 위주)
    # ============================================================
    print("=" * 70)
    print("[Phase 1] Coarse Grid Search")
    print("=" * 70)

    coarse_grid = {
        "risk_aversion":       [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "turnover_penalty":    [0.05, 0.1, 0.3, 0.5, 1.0],
        "max_te_annual":       [0.15],           # 15% 하드캡
        "max_single_turnover": [0.15, 0.30],
        "sector_deviation":    [0.10, 0.20],
    }
    # 7*5*1*2*2 = 140 combos

    df1 = grid_search(predictions, data, tickers, sector_map, all_dates, coarse_grid)

    # TE <= 15% 필터
    valid1 = df1[df1["tracking_error"] <= 0.15].copy()
    print(f"\n  Coarse: TE<=15% 유효 {len(valid1)}/{len(df1)}개")

    if len(valid1) > 0:
        top1 = valid1.nlargest(5, "active_return")
        print("\n  Top 5 (Coarse):")
        for rank, (_, r) in enumerate(top1.iterrows(), 1):
            print(f"    #{rank}: RA={r['risk_aversion']:.1f}, TP={r['turnover_penalty']:.2f}, "
                  f"MST={r['max_single_turnover']:.2f}, SD={r['sector_deviation']:.2f} | "
                  f"AR={r['active_return']:.2%}, TE={r['tracking_error']:.2%}, "
                  f"IR={r['information_ratio']:.2f}, Sharpe={r['sharpe_ratio']:.2f}")

        best1 = valid1.loc[valid1["active_return"].idxmax()]
    else:
        best1 = df1.loc[df1["tracking_error"].idxmin()]

    # ============================================================
    # Phase 2: Fine Grid (Best 주변 정밀 탐색)
    # ============================================================
    print(f"\n{'='*70}")
    print("[Phase 2] Fine Grid Search (Best 주변)")
    print("=" * 70)

    best_ra = best1["risk_aversion"]
    best_tp = best1["turnover_penalty"]
    best_mst = best1["max_single_turnover"]
    best_sd = best1["sector_deviation"]

    # Best 주변으로 정밀 탐색
    ra_fine = sorted(set([
        max(0.01, best_ra - 0.2), max(0.01, best_ra - 0.1),
        best_ra,
        best_ra + 0.1, best_ra + 0.2,
    ]))
    tp_fine = sorted(set([
        max(0.01, best_tp - 0.1), max(0.01, best_tp - 0.05),
        best_tp,
        best_tp + 0.05, best_tp + 0.1,
    ]))
    mst_fine = sorted(set([
        max(0.05, best_mst - 0.05), best_mst, best_mst + 0.05,
    ]))
    sd_fine = sorted(set([
        max(0.05, best_sd - 0.05), best_sd, best_sd + 0.05,
    ]))

    # TE 캡: 15% + 약간 여유 (실제 TE가 15% 이내인지 확인)
    fine_grid = {
        "risk_aversion":       ra_fine,
        "turnover_penalty":    tp_fine,
        "max_te_annual":       [0.15],
        "max_single_turnover": mst_fine,
        "sector_deviation":    sd_fine,
    }

    total_fine = 1
    for v in fine_grid.values():
        total_fine *= len(v)
    print(f"  Fine grid: {total_fine}개 조합")

    df2 = grid_search(predictions, data, tickers, sector_map, all_dates, fine_grid)

    # 합산
    df_all = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    valid_all = df_all[df_all["tracking_error"] <= 0.15].copy()

    # ============================================================
    # Phase 3: 결과 분석
    # ============================================================
    print(f"\n{'='*70}")
    print("[결과] TE <= 15% 최적 파라미터")
    print("=" * 70)

    print(f"\n  전체 탐색: {len(df_all)}개, TE<=15% 유효: {len(valid_all)}개\n")

    if len(valid_all) > 0:
        top20 = valid_all.nlargest(20, "active_return")
        print(f"  {'Rank':>4} | {'RA':>5} | {'TP':>5} | {'MST':>5} | {'SD':>5} || "
              f"{'AR':>7} | {'TE':>7} | {'IR':>6} | {'Sharpe':>6} | {'MaxDD':>7} | {'TO':>6} | {'AnnRet':>7}")
        print("  " + "-" * 100)

        for rank, (_, r) in enumerate(top20.iterrows(), 1):
            print(f"  {rank:>4} | {r['risk_aversion']:>5.2f} | {r['turnover_penalty']:>5.2f} | "
                  f"{r['max_single_turnover']:>5.2f} | {r['sector_deviation']:>5.2f} || "
                  f"{r['active_return']:>6.2%} | {r['tracking_error']:>6.2%} | "
                  f"{r['information_ratio']:>6.2f} | {r['sharpe_ratio']:>6.2f} | "
                  f"{r['max_drawdown']:>6.2%} | {r['avg_turnover']:>5.0%} | "
                  f"{r['annual_return']:>6.2%}")

        best = valid_all.loc[valid_all["active_return"].idxmax()]
    else:
        print("  TE<=15% 조합 없음. 가장 낮은 TE:")
        best = df_all.loc[df_all["tracking_error"].idxmin()]

    print(f"\n{'='*70}")
    print("  ★ BEST: TE <= 15% 내 Active Return 최대 조합")
    print("=" * 70)
    print(f"  RISK_AVERSION       = {best['risk_aversion']}")
    print(f"  TURNOVER_PENALTY    = {best['turnover_penalty']}")
    print(f"  MAX_TE_ANNUAL       = 0.15")
    print(f"  MAX_SINGLE_TURNOVER = {best['max_single_turnover']}")
    print(f"  SECTOR_DEVIATION    = {best['sector_deviation']}")
    print(f"  ---")
    print(f"  Active Return  = {best['active_return']:.4%}")
    print(f"  Tracking Error = {best['tracking_error']:.4%}")
    print(f"  IR             = {best['information_ratio']:.4f}")
    print(f"  Sharpe         = {best['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown   = {best['max_drawdown']:.4%}")
    print(f"  Annual Return  = {best['annual_return']:.4%}")
    print(f"  Avg Turnover   = {best['avg_turnover']:.0%}")
    print("=" * 70)

    # CSV 저장
    out_path = ROOT / "outputs" / "grid_search_results.csv"
    df_all.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  전체 결과 저장: {out_path}")

    best_path = ROOT / "outputs" / "best_params.txt"
    with open(best_path, "w") as f:
        f.write(f"RISK_AVERSION = {best['risk_aversion']}\n")
        f.write(f"TURNOVER_PENALTY = {best['turnover_penalty']}\n")
        f.write(f"MAX_TE_ANNUAL = 0.15\n")
        f.write(f"MAX_SINGLE_TURNOVER = {best['max_single_turnover']}\n")
        f.write(f"SECTOR_DEVIATION = {best['sector_deviation']}\n")
        f.write(f"---\n")
        f.write(f"Active Return  = {best['active_return']:.4%}\n")
        f.write(f"Tracking Error = {best['tracking_error']:.4%}\n")
        f.write(f"IR             = {best['information_ratio']:.4f}\n")
        f.write(f"Sharpe         = {best['sharpe_ratio']:.4f}\n")
        f.write(f"Max Drawdown   = {best['max_drawdown']:.4%}\n")
        f.write(f"Annual Return  = {best['annual_return']:.4%}\n")
        f.write(f"Avg Turnover   = {best['avg_turnover']:.0%}\n")
    print(f"  최적 파라미터 저장: {best_path}")

    total_time = time.time() - t0
    print(f"\n  총 소요 시간: {total_time/60:.1f}분")


if __name__ == "__main__":
    main()
