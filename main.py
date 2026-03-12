"""
AI Signal Portfolio Construction System
main.py - 전체 파이프라인 실행 및 시각화 + CSV 내보내기 + GitHub 자동 배포

실행: python main.py --data_path ./data/ai_signal_data.xlsx --output_dir ./outputs/
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path

from src.data_loader import UniverseData
from src.feature_engine import build_all_features
from src.backtest import run_backtest, BacktestResult
from src.attribution import run_attribution, explain_period

from export_csv import (
    export_daily_performance,
    export_portfolio_weights,
    export_benchmark_weights,
    export_feature_importance,
    export_group_attribution,
    export_li_attribution,
    export_ic_series,
    export_model_structure,
    export_monthly_regime,
    export_style_sector_tilt,
    export_monthly_ow_explanations,
    git_push_outputs,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 시각화 함수들
# ---------------------------------------------------------------------------

def plot_cumulative_returns(result: BacktestResult, output_dir: Path):
    """1. 누적 수익률 (전략 vs 벤치마크)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result.cumulative_returns.index, result.cumulative_returns.values,
            label="Strategy", linewidth=1.5)
    ax.plot(result.cumulative_benchmark.index, result.cumulative_benchmark.values,
            label="Benchmark (MktCap)", linewidth=1.5, alpha=0.7)
    ax.set_title("Cumulative Returns: Strategy vs Benchmark")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "01_cumulative_returns.png", dpi=150)
    plt.close(fig)


def plot_rolling_ir(result: BacktestResult, output_dir: Path, window: int = 252):
    """2. Rolling IR (252일)."""
    active = result.active_returns
    rolling_mean = active.rolling(window, min_periods=window // 2).mean() * 252
    rolling_std = active.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    rolling_ir = rolling_mean / rolling_std.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_ir.index, rolling_ir.values, linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="IR=1.0")
    ax.set_title(f"Rolling Information Ratio ({window}d)")
    ax.set_ylabel("IR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "02_rolling_ir.png", dpi=150)
    plt.close(fig)


def plot_drawdown(result: BacktestResult, output_dir: Path):
    """3. Drawdown."""
    cum = result.cumulative_returns
    rolling_max = cum.cummax()
    drawdown = (cum / rolling_max) - 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "03_drawdown.png", dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(result: BacktestResult, output_dir: Path):
    """4. 월별 수익률 히트맵."""
    rets = result.portfolio_returns.copy()
    rets.index = pd.to_datetime(rets.index)
    monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    heatmap_data = pivot.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{m}" for m in range(1, 13)])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Monthly Returns Heatmap")

    # 수치 표시
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Return")
    fig.tight_layout()
    fig.savefig(output_dir / "04_monthly_heatmap.png", dpi=150)
    plt.close(fig)


def plot_shap_importance(attribution: dict, output_dir: Path):
    """5. SHAP feature importance."""
    importance = attribution.get("feature_importance")
    if importance is None:
        return

    top_n = min(30, len(importance))
    top = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top.values[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_title(f"Top {top_n} Feature Importance (LightGBM Gain)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "05_feature_importance.png", dpi=150)
    plt.close(fig)


def plot_group_contributions(attribution: dict, output_dir: Path):
    """6. Feature group 기여도 시계열."""
    group_contribs = attribution.get("group_contributions", {})
    if not group_contribs:
        return

    df = pd.DataFrame(group_contribs).T
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(14, 5))
    df.plot.area(ax=ax, alpha=0.7, stacked=True)
    ax.set_title("Feature Group Contribution Over Time")
    ax.set_ylabel("Contribution Share")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "06_group_contributions.png", dpi=150)
    plt.close(fig)


def plot_linear_nonlinear(attribution: dict, output_dir: Path):
    """7. 선형/비선형 비율 추이."""
    ratios = attribution.get("linear_ratios", {})
    if not ratios:
        return

    dates = sorted(ratios.keys())
    linear = [ratios[d][0] for d in dates]
    nonlinear = [ratios[d][1] for d in dates]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(dates)), linear, label="Linear", alpha=0.7)
    ax.bar(range(len(dates)), nonlinear, bottom=linear, label="Nonlinear", alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in dates], rotation=45)
    ax.set_title("Linear vs Nonlinear Decomposition")
    ax.set_ylabel("Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_linear_nonlinear.png", dpi=150)
    plt.close(fig)


def plot_ic_series(result: BacktestResult, output_dir: Path):
    """8. IC 시계열."""
    if len(result.ic_series) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(result.ic_series.index, result.ic_series.values, width=5, alpha=0.6)
    ax.axhline(y=result.ic_series.mean(), color="red", linestyle="--",
               label=f"Mean IC={result.ic_series.mean():.4f}")
    ax.set_title("Information Coefficient (IC) Over Time")
    ax.set_ylabel("IC (Spearman)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "08_ic_series.png", dpi=150)
    plt.close(fig)


def plot_portfolio_weights(result: BacktestResult, output_dir: Path):
    """9. 포트폴리오 비중 변화."""
    if not result.portfolio_weights:
        return

    weights_df = pd.DataFrame(result.portfolio_weights).T
    weights_df.index = pd.to_datetime(weights_df.index)

    fig, ax = plt.subplots(figsize=(14, 6))
    weights_df.plot.area(ax=ax, alpha=0.7, stacked=True)
    ax.set_title("Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "09_portfolio_weights.png", dpi=150)
    plt.close(fig)


def plot_retrain_correlation(result: BacktestResult, output_dir: Path):
    """10. 재훈련 전후 예측 상관 추이."""
    if result.predictions is None:
        return

    model_dates = sorted(result.models.keys())
    if len(model_dates) < 2:
        return

    correlations = []
    for i in range(1, len(model_dates)):
        d_prev = model_dates[i - 1]
        d_curr = model_dates[i]
        p_prev = result.predictions.loc[d_prev].dropna()
        p_curr = result.predictions.loc[d_curr].dropna()
        common = p_prev.index.intersection(p_curr.index)
        if len(common) >= 3:
            corr = p_prev[common].corr(p_curr[common])
            correlations.append((d_curr, corr))

    if not correlations:
        return

    dates, corrs = zip(*correlations)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, corrs, "o-", markersize=4)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="Target: 0.95")
    ax.set_title("Prediction Correlation Between Retrain Windows")
    ax.set_ylabel("Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "10_retrain_correlation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Signal Portfolio Construction")
    parser.add_argument("--data_path", type=str, default="./data/ai_signal_data.xlsx")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: 데이터 로드
    print("=" * 60)
    print("Phase 1: 데이터 로드")
    print("=" * 60)
    data = UniverseData(args.data_path)
    print(data.summary())
    print()

    # Phase 2~6: 백테스트 (피처, 타겟, 모델, 포트폴리오 포함)
    print("=" * 60)
    print("Phase 2~6: 백테스트 실행")
    print("=" * 60)
    result = run_backtest(data)
    print()

    # Phase 7: Attribution
    print("=" * 60)
    print("Phase 7: Attribution 분석")
    print("=" * 60)
    import gc
    gc.collect()
    # 백테스트에서 이미 생성된 panel 재사용 (메모리 절약)
    if result.panel is not None:
        panel = result.panel
        feature_names = result.feature_names
        feature_groups = result.feature_groups
    else:
        panel, feature_names, feature_groups = build_all_features(data)
    attribution = run_attribution(
        result.models, panel, feature_names, feature_groups,
        weights_history=result.portfolio_weights,
    )

    # Group contributions 출력
    if attribution["group_contributions"]:
        print("\nFeature Group 기여도 (마지막 모델):")
        last_date = sorted(attribution["group_contributions"].keys())[-1]
        for group, ratio in attribution["group_contributions"][last_date].items():
            print(f"  {group:15s}: {ratio:.1%}")

    # Li et al. 3-Component Attribution 출력
    if attribution["linear_nonlinear_detail"]:
        last_date = sorted(attribution["linear_nonlinear_detail"].keys())[-1]
        detail = attribution["linear_nonlinear_detail"][last_date]
        print(f"\nLi et al. 3-Component Attribution (Pictet 논문 방식):")
        print(f"  Linear:              {detail['linear_ratio']:.1%}")
        print(f"  Marginal Non-linear: {detail['marginal_nl_ratio']:.1%}")
        print(f"  Interaction:         {detail['interaction_ratio']:.1%}")
        print(f"  (비선형 합계:         {detail['nonlinear_ratio']:.1%})")
        print(f"  그룹별 Linear 기여:")
        for g, v in detail["group_linear"].items():
            print(f"    {g:15s}: {v:.1%}")
        print(f"  그룹별 Marginal NL 기여:")
        for g, v in detail["group_marginal_nl"].items():
            print(f"    {g:15s}: {v:.1%}")
        print(f"  그룹별 Interaction 기여:")
        for g, v in detail["group_interaction"].items():
            print(f"    {g:15s}: {v:.1%}")

        # 시점별 평균 출력
        all_details = attribution["linear_nonlinear_detail"]
        avg_lin = np.mean([d["linear_ratio"] for d in all_details.values()])
        avg_mnl = np.mean([d["marginal_nl_ratio"] for d in all_details.values()])
        avg_int = np.mean([d["interaction_ratio"] for d in all_details.values()])
        print(f"\n  전체 평균:")
        print(f"    Linear:              {avg_lin:.1%}")
        print(f"    Marginal Non-linear: {avg_mnl:.1%}")
        print(f"    Interaction:         {avg_int:.1%}")
    elif attribution["linear_ratios"]:
        last_date = sorted(attribution["linear_ratios"].keys())[-1]
        lin, nonlin = attribution["linear_ratios"][last_date]
        print(f"\n선형/비선형 비율: {lin:.1%} / {nonlin:.1%}")

    # Portfolio SHAP decomposition 출력
    if attribution["portfolio_decomposition"]:
        last_date = sorted(attribution["portfolio_decomposition"].keys())[-1]
        port_decomp = attribution["portfolio_decomposition"][last_date]
        print(f"\n포트폴리오 SHAP 분해 ({last_date.strftime('%Y-%m-%d')}):")
        for g, v in port_decomp["group_contrib"].items():
            print(f"  {g:15s}: {v:+.6f}")

    # Market Explainer: 최근 기간 분석
    print("\n" + "=" * 60)
    print("Market Explainer: 최근 시장 분석")
    print("=" * 60)
    dates = data.dates
    recent_end = dates[-1].strftime("%Y-%m-%d")
    recent_start = dates[-63].strftime("%Y-%m-%d")
    mkt_analysis = explain_period(data.returns, data.prices, recent_start, recent_end)

    print(f"  기간: {mkt_analysis['period']} ({mkt_analysis['trading_days']}일)")
    print(f"\n  그룹별 수익률:")
    for g, r in mkt_analysis["group_returns"].items():
        print(f"    {g}: {r:.2%}")
    print(f"\n  Rotation: {mkt_analysis['rotation_analysis']['description']}")
    print(f"    Spread (Heavy-Light): {mkt_analysis['rotation_analysis']['spread (Heavy - Light)']:.2%}")
    print(f"\n  Regime:")
    for k, v in mkt_analysis["regime"].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print()

    # Phase 8: 시각화
    print("=" * 60)
    print("Phase 8: 시각화 생성")
    print("=" * 60)

    plot_cumulative_returns(result, output_dir)
    print("  [1/10] 누적 수익률")

    plot_rolling_ir(result, output_dir)
    print("  [2/10] Rolling IR")

    plot_drawdown(result, output_dir)
    print("  [3/10] Drawdown")

    plot_monthly_heatmap(result, output_dir)
    print("  [4/10] 월별 수익률 히트맵")

    plot_shap_importance(attribution, output_dir)
    print("  [5/10] SHAP feature importance")

    plot_group_contributions(attribution, output_dir)
    print("  [6/10] Feature group 기여도")

    plot_linear_nonlinear(attribution, output_dir)
    print("  [7/10] 선형/비선형 비율")

    plot_ic_series(result, output_dir)
    print("  [8/10] IC 시계열")

    plot_portfolio_weights(result, output_dir)
    print("  [9/10] 포트폴리오 비중")

    plot_retrain_correlation(result, output_dir)
    print("  [10/10] 재훈련 전후 상관")

    print(f"\n모든 차트가 {output_dir}에 저장되었습니다.")

    # Phase 9: CSV 내보내기 + GitHub 자동 배포
    print("\n" + "=" * 60)
    print("Phase 9: 대시보드 CSV 내보내기")
    print("=" * 60)

    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"\nCSVs → {csv_dir.resolve()}")
    print(f"Reports → {report_dir.resolve()}")

    # Phase 10: GitHub 자동 배포
    print("\n" + "=" * 60)
    print("Phase 10: GitHub 자동 배포")
    print("=" * 60)
    git_push_outputs()

    print("\n" + "=" * 60)
    print("전체 파이프라인 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
