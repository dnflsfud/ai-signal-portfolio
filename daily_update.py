"""
일간 증분 업데이트 스크립트

새 가격 데이터가 추가되면 전체 백테스트를 다시 돌리지 않고,
마지막 상태에서 이어서 수익률만 업데이트한다.

- 리밸런싱이 아닌 날: 가격 drift만 반영 (모델 실행 없음)
- 리밸런싱 날: 피처 → 예측 → MVO 최적화 → 새 비중

사용법:
  python daily_update.py                         # 기본: 저장된 상태에서 이어서 업데이트
  python daily_update.py --full-init             # 처음 한 번: 전체 백테스트 후 상태 저장
  python daily_update.py --data_path ./data/ai_signal_data.xlsx
"""

import argparse
import pickle
import warnings
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import UniverseData, TICKERS
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train, train_model, predict_cross_sectional, TRAIN_WINDOW, VAL_WINDOW
from src.portfolio_optimizer import (
    optimize_portfolio, estimate_covariance,
    project_portfolio_weights, project_capped_weights,
)
from src.backtest import (
    REBALANCE_FREQ, ONE_WAY_TC, get_sector_map, compute_ic, BacktestResult,
    get_benchmark_fn,
    compute_signal_confidence, apply_dynamic_execution,
    apply_pead_boost, apply_growth_tilt,
    apply_value_trap_gate, apply_signal_stability_shrinkage,
)
from src.config import DEFAULT_CONFIG
from src.utils import annualise_return, compute_performance_metrics

# State schema version. Bump when DailyState fields change so old pickles
# can't be silently loaded with mismatched assumptions (e.g. EW-vs-cap BM).
STATE_SCHEMA_VERSION = 2  # v2: cap-weighted BM (was EW in v1)

STATE_FILE = ROOT / "outputs" / "daily_state.pkl"
CSV_DIR = ROOT / "outputs" / "csv"


# ---------------------------------------------------------------------------
# 상태 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class DailyState:
    """증분 업데이트에 필요한 최소 상태."""
    # 현재 비중 (drift 반영)
    weights: np.ndarray
    tickers: List[str]
    # 리밸런싱 카운터
    days_since_rebal: int
    rebalance_freq: int
    # 누적 성과
    port_rets: List[tuple]       # [(date, ret), ...]
    bm_rets: List[tuple]
    spx_rets: List[tuple]
    turnovers: List[tuple]       # [(date, turnover), ...]
    # 비중 이력
    rebal_weights: Dict          # 리밸런싱일 목표비중
    daily_weights: Dict          # 매일 drift 비중
    # 마지막 처리일
    last_date: pd.Timestamp
    # 모델 & 예측 (리밸런싱 시 필요)
    models: Dict                 # {retrain_date: model}
    predictions: pd.DataFrame    # 전체 예측 DataFrame
    # IC
    ic_values: List[tuple]
    # 기타
    sector_map: Dict
    first_rebal_done: bool = True
    # v2: 벤치마크 비중 이력 (리밸런싱일별 cap-weighted)
    bm_rebal_weights: Dict = field(default_factory=dict)
    schema_version: int = STATE_SCHEMA_VERSION


def save_state(state: DailyState):
    """상태를 pickle로 저장."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)
    print(f"  상태 저장: {STATE_FILE}")
    print(f"  마지막 처리일: {state.last_date.strftime('%Y-%m-%d')}")
    print(f"  누적 {len(state.port_rets)}일 수익률")


def prune_state(state: DailyState, max_daily_weights_days: int = 504):
    """상태 크기 관리: 오래된 daily weights 정리."""
    # daily_weights는 최근 2년분만 유지
    if len(state.daily_weights) > max_daily_weights_days:
        sorted_dates = sorted(state.daily_weights.keys())
        cutoff = sorted_dates[-max_daily_weights_days]
        state.daily_weights = {d: w for d, w in state.daily_weights.items() if d >= cutoff}
        print(f"  [Prune] daily_weights: 최근 {max_daily_weights_days}일만 유지")

    # 오래된 모델 정리 (최근 3개만 유지)
    if len(state.models) > 3:
        sorted_dates = sorted(state.models.keys())
        keep = sorted_dates[-3:]
        state.models = {d: m for d, m in state.models.items() if d in keep}
        print(f"  [Prune] models: 최근 3개만 유지")


def load_state() -> DailyState:
    """저장된 상태 로드 (type + schema 버전 검증)."""
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"저장된 상태가 없습니다: {STATE_FILE}\n"
            "먼저 --full-init 으로 초기화하세요."
        )
    with open(STATE_FILE, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, DailyState):
        raise TypeError(
            f"Loaded object is {type(state).__name__}, expected DailyState. "
            "State file may be corrupted or tampered."
        )
    # v1 → v2 마이그레이션: v1 상태는 EW BM으로 누적되어 있어 cap-weighted
    # 코드와 섞이면 IR이 깨집니다. 강제로 --full-init 재실행을 요구합니다.
    state_version = getattr(state, "schema_version", 1)
    if state_version < STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"daily_state.pkl schema v{state_version} < current v{STATE_SCHEMA_VERSION}.\n"
            "이전 상태는 equal-weight 벤치마크로 누적되었으므로 cap-weighted "
            "코드와 호환되지 않습니다.\n"
            f"  → '{STATE_FILE}' 를 삭제하고 --full-init 으로 재초기화하세요."
        )
    print(f"  상태 로드: 마지막 처리일 {state.last_date.strftime('%Y-%m-%d')} (schema v{state_version})")
    return state


def backup_state():
    """리밸런싱 전 상태 백업."""
    import shutil
    backup_path = STATE_FILE.parent / "state_backup.pkl"
    if STATE_FILE.exists():
        shutil.copy2(STATE_FILE, backup_path)
        print(f"  [Backup] 상태 백업 → {backup_path}")
    return backup_path


def restore_state():
    """백업에서 상태 복구."""
    import shutil
    backup_path = STATE_FILE.parent / "state_backup.pkl"
    if backup_path.exists():
        shutil.copy2(backup_path, STATE_FILE)
        print(f"  [Restore] 백업에서 복구 ← {backup_path}")
        return True
    print(f"  [Restore] 백업 파일 없음")
    return False


def validate_new_weights(weights: np.ndarray, tickers: list) -> bool:
    """새 비중이 제약 조건 만족하는지 검증."""
    checks = []

    # 합계 = 1
    checks.append(("sum=1", abs(weights.sum() - 1.0) < 1e-4))
    # 음수 없음
    checks.append(("non-negative", (weights >= -1e-6).all()))
    # 최대 비중
    checks.append(("max_weight<=15%", weights.max() <= 0.15 + 1e-4))
    # NaN 없음
    checks.append(("no_nan", not np.any(np.isnan(weights))))

    all_pass = all(p for _, p in checks)
    if not all_pass:
        failed = [name for name, p in checks if not p]
        print(f"  [Validation FAIL] 비중 검증 실패: {failed}")
    return all_pass


# ---------------------------------------------------------------------------
# 초기화: 전체 백테스트 후 상태 저장
# ---------------------------------------------------------------------------
def full_init(data: UniverseData) -> DailyState:
    """전체 백테스트를 돌리고 마지막 상태를 저장.

    Production run_backtest를 위임 호출하므로 다음을 자동 보장:
      - look-ahead 수정 (drift→rebal→TC 순, 같은 날 신규 비중으로 PnL 계산 X)
      - VTG / growth_tilt / PEAD / signal_stability score 조정
      - dynamic execution / partial rebalance / no-trade band
      - cap-weighted BM, score-gate, mega-cap protection 등 모든 production 룰

    walk_forward_train만 직접 호출하는 이유는 EWMA feature tracker 객체를
    외부 export로 빼기 위해서 (run_backtest는 트래커를 노출하지 않음).
    """
    print("\n[Full Init] 피처 생성...")
    panel, feature_names, feature_groups = build_all_features(data)

    print("[Full Init] 타겟 생성...")
    targets = build_targets(data)

    print("[Full Init] 모델 학습 & 예측 (EWMA 트래커 보존)...")
    all_dates = data.dates
    models, predictions, raw_predictions, ewma_tracker = walk_forward_train(
        panel, targets, feature_names, all_dates,
    )

    print("[Full Init] production run_backtest 위임 실행...")
    from src.backtest import run_backtest
    result = run_backtest(
        data,
        config=DEFAULT_CONFIG,
        precomputed_panel=panel,
        precomputed_feature_names=feature_names,
        precomputed_feature_groups=feature_groups,
        precomputed_targets=targets,
        precomputed_models=models,
        precomputed_predictions=predictions,
        precomputed_raw_predictions=raw_predictions,
    )

    # ------------------------------------------------------------------
    # BacktestResult → DailyState 변환
    # ------------------------------------------------------------------
    returns = data.returns
    tickers = [t for t in data.tickers if t in returns.columns]
    n_tickers = len(tickers)
    sector_map = get_sector_map(data)
    bm_weights_fn = get_benchmark_fn(data, tickers, config=DEFAULT_CONFIG)

    last_date = (
        result.portfolio_returns.index[-1]
        if len(result.portfolio_returns)
        else all_dates[-1]
    )

    if result.daily_weights and last_date in result.daily_weights:
        prev_weights = (
            result.daily_weights[last_date]
            .reindex(tickers).fillna(0.0).values
        )
    else:
        prev_weights = np.ones(n_tickers) / n_tickers

    bm_rebal_weights = {
        d: pd.Series(bm_weights_fn(d, tickers, n_tickers), index=tickers)
        for d in result.portfolio_weights.keys()
    }

    port_rets = [(d, float(v)) for d, v in result.portfolio_returns.items()]
    bm_rets = [(d, float(v)) for d, v in result.benchmark_returns.items()]
    spx_rets = [(d, float(v)) for d, v in result.spx_returns.items()]
    turnovers = [(d, float(v)) for d, v in result.turnover.items()]
    ic_values = [(d, float(v)) for d, v in result.ic_series.items()]

    rebal_dates = sorted(result.portfolio_weights.keys())
    if rebal_dates:
        try:
            last_rebal_idx = all_dates.get_loc(rebal_dates[-1])
            last_idx = all_dates.get_loc(last_date)
            days_since_rebal = max(0, last_idx - last_rebal_idx)
        except KeyError:
            days_since_rebal = 0
    else:
        days_since_rebal = 0

    state = DailyState(
        weights=prev_weights,
        tickers=tickers,
        days_since_rebal=days_since_rebal,
        rebalance_freq=DEFAULT_CONFIG.rebalance_freq,
        port_rets=port_rets,
        bm_rets=bm_rets,
        spx_rets=spx_rets,
        turnovers=turnovers,
        rebal_weights=dict(result.portfolio_weights),
        daily_weights=dict(result.daily_weights),
        last_date=last_date,
        models=dict(result.models),
        predictions=result.predictions,
        ic_values=ic_values,
        sector_map=sector_map,
        first_rebal_done=True,
        bm_rebal_weights=bm_rebal_weights,
        schema_version=STATE_SCHEMA_VERSION,
    )

    prune_state(state)
    save_state(state)
    _export_csvs(state)
    _export_ewma(ewma_tracker)
    _print_summary(state)
    return state


# ---------------------------------------------------------------------------
# 증분 업데이트: 새 날짜만 처리
# ---------------------------------------------------------------------------
def incremental_update(data: UniverseData) -> DailyState:
    """새로운 가격 데이터만 처리하여 수익률 업데이트.

    리밸런싱이 아닌 날: drift만 (연산 거의 없음)
    리밸런싱 날: 피처 → 예측 → MVO 최적화
    """
    state = load_state()
    all_dates = data.dates
    returns = data.returns
    tickers = state.tickers

    has_spx = data.has_factor_data() and "SPX" in data.factor_returns.columns
    spx_factor = data.factor_returns["SPX"] if has_spx else None

    # Cap-weighted BM closure (production parity)
    bm_weights_fn = get_benchmark_fn(data, tickers, config=DEFAULT_CONFIG)

    # 새로운 날짜 찾기
    last_loc = all_dates.get_loc(state.last_date)
    new_start = last_loc + 1
    if new_start >= len(all_dates):
        print("  새로운 데이터가 없습니다.")
        return state

    new_dates = all_dates[new_start:]
    print(f"\n  새로운 영업일: {len(new_dates)}일")
    print(f"  기간: {new_dates[0].strftime('%Y-%m-%d')} ~ {new_dates[-1].strftime('%Y-%m-%d')}")

    # 리밸런싱 시 모델 재학습이 필요한지 체크
    # (마지막 모델 학습일로부터 63일 이상 경과 시)
    last_model_date = max(state.models.keys()) if state.models else None
    needs_retrain = False
    retrain_threshold = 63  # 3개월

    prev_weights = state.weights.copy()
    n_new_rebal = 0
    n_new_drift = 0

    backup_state()

    try:
        # Production execution timing (mirrors src/backtest.py:1015-1029):
        #   1. Book today's PnL with weights ENTERING the day (pre-drift)
        #   2. Drift weights with today's return
        #   3. Rebalance at close → new weights take effect NEXT bar
        #   4. TC charged to today's PnL (cash leaves the book today)
        # The previous order rebalanced before booking and charged today's
        # return to the new weights — that is look-ahead.
        n_tickers = len(tickers)
        for t_idx_offset, t_date in enumerate(new_dates):
            t_idx = new_start + t_idx_offset

            # --- Step 1: Today's PnL with weights ENTERING the day -----------
            daily_ret = returns.loc[t_date, tickers].values
            if np.any(np.isnan(daily_ret)):
                daily_ret = np.nan_to_num(daily_ret, 0)
            port_ret = float(np.dot(prev_weights, daily_ret))
            bm_w_today = bm_weights_fn(t_date, tickers, n_tickers)
            bm_ret = float(np.dot(bm_w_today, daily_ret))

            # --- Step 2: Drift weights from today's return -------------------
            prev_weights = prev_weights * (1 + daily_ret)
            w_sum = prev_weights.sum()
            if w_sum > 0:
                prev_weights = prev_weights / w_sum

            # --- Step 3: Rebalance at close (effective NEXT bar) -------------
            state.days_since_rebal += 1
            is_rebal = state.days_since_rebal >= state.rebalance_freq
            rebalanced_today = False
            if is_rebal:
                if last_model_date is not None:
                    days_since_train = (t_date - last_model_date).days
                    if days_since_train > retrain_threshold * 1.5:
                        needs_retrain = True

                if needs_retrain:
                    print(f"  [{t_date.strftime('%Y-%m-%d')}] 모델 재학습 실행...")
                    panel, feature_names, _ = build_all_features(data)
                    targets = build_targets(data)
                    train_start = max(0, t_idx - TRAIN_WINDOW)
                    train_end = t_idx - VAL_WINDOW
                    val_start = t_idx - VAL_WINDOW
                    val_end = t_idx
                    train_dates = all_dates[train_start:train_end]
                    val_dates = all_dates[val_start:val_end]
                    new_model = train_model(panel, targets, feature_names, train_dates, val_dates)
                    state.models[t_date] = new_model
                    pred = predict_cross_sectional(new_model, panel, feature_names, t_date)
                    for ticker in pred.index:
                        if ticker in state.predictions.columns:
                            state.predictions.loc[t_date, ticker] = pred[ticker]

                    # Post-prediction overlays (production parity, mirrors
                    # run_backtest:1329~1361). Apply on a working copy of the
                    # FULL frame so PEAD's earnings searchsorted and
                    # growth-tilt's 63d revision MA can see history; then write
                    # back ONLY the t_date row — older rows already had
                    # overlays applied during full_init / earlier retrains.
                    adjusted = state.predictions.copy()
                    if (getattr(DEFAULT_CONFIG, "pead_boost_enabled", False)
                            and getattr(data, "earnings_timeline", None) is not None):
                        adjusted = apply_pead_boost(adjusted, data, DEFAULT_CONFIG)
                    if getattr(DEFAULT_CONFIG, "growth_tilt_enabled", False):
                        adjusted = apply_growth_tilt(adjusted, data, DEFAULT_CONFIG)
                    if getattr(DEFAULT_CONFIG, "value_trap_gate_enabled", False):
                        adjusted = apply_value_trap_gate(adjusted, panel, DEFAULT_CONFIG)
                    if getattr(DEFAULT_CONFIG, "signal_stability_lambda", 0.0) > 0.0:
                        # raw_predictions not stored in DailyState — fall back
                        # to predictions itself (the function handles None).
                        # Users who enable signal-stability should run
                        # --full-init periodically to re-anchor properly.
                        adjusted = apply_signal_stability_shrinkage(
                            adjusted, None, DEFAULT_CONFIG,
                            retrain_freq=getattr(DEFAULT_CONFIG, "retrain_freq", 63),
                        )
                    state.predictions.loc[t_date] = adjusted.loc[t_date]

                    needs_retrain = False
                    last_model_date = t_date
                    print(f"    학습 완료: train={len(train_dates)}d, val={len(val_dates)}d")

                if t_date in state.predictions.index:
                    pred_row = state.predictions.loc[t_date, tickers]
                else:
                    valid_pred_dates = state.predictions.dropna(how="all").index
                    pred_date = valid_pred_dates[valid_pred_dates <= t_date][-1]
                    pred_row = state.predictions.loc[pred_date, tickers]

                if pred_row.notna().sum() >= 10:
                    # Production execution path (mirrors src/backtest.py:1077-1162):
                    #   (a) target_weights ← MVO (with cov shrinkage)
                    #   (b) confidence    ← spread × trailing-IC
                    #   (c) candidate     ← apply_dynamic_execution
                    #                       (no-trade band + partial η, conf-scaled)
                    #   (d) new_weights   ← project_portfolio_weights
                    #                       (hard TE + sector + score-gate constraints)
                    # Skipping any of (b)/(c)/(d) makes incremental output diverge
                    # from production backtest — that was the prior bug.
                    cov_lookback = int(getattr(DEFAULT_CONFIG, "cov_lookback", 126))
                    bm_w = bm_weights_fn(t_date, tickers, n_tickers)
                    hist_start = max(0, t_idx - cov_lookback)
                    hist_returns = returns[tickers].iloc[hist_start:t_idx]
                    cov_matrix = estimate_covariance(
                        hist_returns, bm_weights=bm_w, config=DEFAULT_CONFIG,
                    )

                    # (a) Target weights from MVO
                    target_weights = optimize_portfolio(
                        expected_returns=pred_row,
                        cov_matrix=cov_matrix,
                        prev_weights=prev_weights,
                        sector_map=state.sector_map if state.sector_map else None,
                        bm_weights=bm_w,
                    )

                    # (b) Confidence from spread + trailing IC
                    trailing_ic_window = int(
                        getattr(DEFAULT_CONFIG, "trailing_ic_window", 6)
                    )
                    if len(state.ic_values) >= 2:
                        recent_ics = [v for _, v in state.ic_values[-trailing_ic_window:]]
                        trailing_ic_mean = float(np.nanmean(recent_ics))
                    else:
                        trailing_ic_mean = 0.0
                    # raw_pred_row not tracked in DailyState — function falls back
                    # to pred_row when None (see compute_signal_confidence:792).
                    confidence = compute_signal_confidence(
                        pred_row, None, trailing_ic_mean,
                    )

                    # (c) Smoothed candidate (no-trade band + partial execution)
                    candidate_weights = apply_dynamic_execution(
                        prev_weights, target_weights, confidence, DEFAULT_CONFIG,
                    )

                    # (d) Hard-constraint projection
                    if getattr(DEFAULT_CONFIG, "use_score_based", False):
                        new_weights = project_capped_weights(
                            candidate_weights=candidate_weights,
                            max_weight=DEFAULT_CONFIG.max_weight,
                            fallback_weights=target_weights,
                            config=DEFAULT_CONFIG,
                        )
                    else:
                        new_weights = project_portfolio_weights(
                            candidate_weights=candidate_weights,
                            expected_returns=pred_row,
                            cov_matrix=cov_matrix,
                            prev_weights=prev_weights,
                            sector_map=state.sector_map if state.sector_map else None,
                            bm_weights=bm_w,
                            max_te_annual=DEFAULT_CONFIG.max_te_annual,
                            sector_deviation=DEFAULT_CONFIG.sector_deviation,
                            config=DEFAULT_CONFIG,
                            fallback_weights=target_weights,
                        )

                    if not validate_new_weights(new_weights, tickers):
                        print(f"  [{t_date.strftime('%Y-%m-%d')}] 비중 검증 실패 → 이전 비중 유지")
                    else:
                        turnover = float(np.abs(new_weights - prev_weights).sum())
                        # --- Step 4: TC charged to today's PnL ----------------
                        tc_cost = turnover * ONE_WAY_TC
                        port_ret -= tc_cost
                        state.turnovers.append((t_date, turnover))
                        state.rebal_weights[t_date] = pd.Series(new_weights, index=tickers)
                        state.bm_rebal_weights[t_date] = pd.Series(bm_w, index=tickers)
                        prev_weights = new_weights  # effective NEXT bar
                        state.days_since_rebal = 0
                        rebalanced_today = True
                        n_new_rebal += 1
                        print(f"  [{t_date.strftime('%Y-%m-%d')}] 리밸런싱 "
                              f"(turnover: {turnover:.1%}, conf: {confidence:.2f})")

            # --- Record (after all PnL/TC/drift/rebal adjustments) -----------
            state.port_rets.append((t_date, port_ret))
            state.bm_rets.append((t_date, bm_ret))
            state.daily_weights[t_date] = pd.Series(prev_weights.copy(), index=tickers)

            if not rebalanced_today:
                n_new_drift += 1
            if spx_factor is not None and t_date in spx_factor.index:
                state.spx_rets.append((t_date, spx_factor.loc[t_date]))

    except Exception as e:
        print(f"\n  [ERROR] 증분 업데이트 실패: {e}")
        print(f"  백업에서 복구 시도...")
        if restore_state():
            state = load_state()
            print(f"  복구 완료: {state.last_date.strftime('%Y-%m-%d')}")
        else:
            print(f"  복구 실패!")
        return state

    # 상태 업데이트
    state.weights = prev_weights
    state.last_date = new_dates[-1]

    print(f"\n  처리 완료:")
    print(f"    Drift only: {n_new_drift}일")
    print(f"    리밸런싱:   {n_new_rebal}회")

    prune_state(state)
    save_state(state)
    _export_csvs(state)
    _print_summary(state)
    return state


# ---------------------------------------------------------------------------
# CSV 내보내기 & 성과 요약
# ---------------------------------------------------------------------------
def _export_csvs(state: DailyState):
    """현재 상태를 CSV로 내보내기."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. daily_performance.csv
    port_s = pd.Series(dict(state.port_rets), name="fund_daily_return").sort_index()
    bm_s = pd.Series(dict(state.bm_rets), name="bm_daily_return").sort_index()
    df = pd.DataFrame({
        "fund_daily_return": port_s,
        "bm_daily_return": bm_s,
        "active_daily_return": port_s - bm_s,
        "fund_cumulative": (1 + port_s).cumprod(),
        "bm_cumulative": (1 + bm_s).cumprod(),
    })
    if state.spx_rets:
        spx_s = pd.Series(dict(state.spx_rets)).sort_index()
        df["spx_daily_return"] = spx_s
        df["spx_cumulative"] = (1 + spx_s).cumprod()
    df.index.name = "date"
    df.to_csv(CSV_DIR / "daily_performance.csv")
    print(f"  CSV 저장: daily_performance.csv ({len(df)} rows)")

    # 2. portfolio_weights.csv (리밸런싱일 목표비중)
    if state.rebal_weights:
        w_df = pd.DataFrame(state.rebal_weights).T.round(6)
        w_df.index.name = "date"
        w_df.to_csv(CSV_DIR / "portfolio_weights.csv")
        print(f"  CSV 저장: portfolio_weights.csv ({len(w_df)} rebalances)")

    # 3. daily_weights.csv (매일 drift 비중)
    if state.daily_weights:
        dw_df = pd.DataFrame(state.daily_weights).T.round(6)
        dw_df.index.name = "date"
        dw_df.to_csv(CSV_DIR / "daily_weights.csv")
        print(f"  CSV 저장: daily_weights.csv ({len(dw_df)} days)")

    # 4. benchmark_weights.csv (리밸런싱일 cap-weighted BM 비중)
    bm_rebal = getattr(state, "bm_rebal_weights", None) or {}
    if bm_rebal:
        bm_df = pd.DataFrame(bm_rebal).T.round(6)
        bm_df.index.name = "date"
        bm_df.to_csv(CSV_DIR / "benchmark_weights.csv")
        print(f"  CSV 저장: benchmark_weights.csv ({len(bm_df)} rebalances, cap-weighted)")

    # 5. ic_series.csv (리밸런싱일 IC)
    if state.ic_values:
        ic_df = pd.DataFrame(state.ic_values, columns=["date", "IC"]).round(4)
        ic_df.to_csv(CSV_DIR / "ic_series.csv", index=False)
        print(f"  CSV 저장: ic_series.csv ({len(ic_df)} dates)")


def _export_ewma(ewma_tracker) -> None:
    """EWMA feature importance 이력을 CSV로 내보내기."""
    from src.model_trainer import EWMAFeatureTracker
    if not isinstance(ewma_tracker, EWMAFeatureTracker):
        return
    df = ewma_tracker.export_history()
    if len(df) > 0:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        df.round(6).to_csv(CSV_DIR / "ewma_feature_importance.csv")
        print(f"  CSV 저장: ewma_feature_importance.csv ({len(df)} retrains x {len(df.columns)} features)")


def _print_summary(state: DailyState):
    """현재 성과 요약 출력."""
    port_s = pd.Series(dict(state.port_rets)).sort_index()
    bm_s = pd.Series(dict(state.bm_rets)).sort_index()

    # Canonical geometric metrics (ddof=1)
    metrics = compute_performance_metrics(port_s, bm_s, 252)
    ann_ret = metrics.get("annual_return", 0)
    ann_vol = metrics.get("annual_vol", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    active_ret = metrics.get("active_return", 0)
    te = metrics.get("tracking_error", 0)
    ir = metrics.get("information_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)

    cum = (1 + port_s).cumprod()

    # 최근 수익률
    if len(port_s) >= 5:
        last_5d = (1 + port_s.iloc[-5:]).prod() - 1
    else:
        last_5d = port_s.sum()

    if len(port_s) >= 21:
        last_1m = (1 + port_s.iloc[-21:]).prod() - 1
    else:
        last_1m = (1 + port_s).prod() - 1

    print(f"\n{'='*50}")
    print(f"  성과 요약 (~{state.last_date.strftime('%Y-%m-%d')})")
    print(f"{'='*50}")
    print(f"  연간 수익률:    {ann_ret:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Active Return:  {active_ret:.2%}")
    print(f"  Info Ratio:     {ir:.2f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"  최근 5일:       {last_5d:+.2%}")
    print(f"  최근 1개월:     {last_1m:+.2%}")
    print(f"  현재 누적:      {cum.iloc[-1]:.4f}")

    # 현재 비중 상위 5
    w = pd.Series(state.weights, index=state.tickers).sort_values(ascending=False)
    print(f"\n  현재 비중 Top 5:")
    for t, v in w.head(5).items():
        print(f"    {t:>8s}: {v:.1%}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Signal 일간 증분 업데이트")
    parser.add_argument("--data_path", type=str, default=r"C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\ai_signal_data.xlsx")
    parser.add_argument("--full-init", action="store_true",
                        help="전체 백테스트 후 상태 초기화")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  AI Signal CC2 - Daily Update")
    print("=" * 60)

    data = UniverseData(args.data_path)
    del data.raw
    import gc; gc.collect()

    if args.full_init:
        print("\n[모드] 전체 초기화 (Full Init)")
        full_init(data)
    else:
        print("\n[모드] 증분 업데이트 (Incremental)")
        incremental_update(data)

    print(f"\n  소요시간: {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
