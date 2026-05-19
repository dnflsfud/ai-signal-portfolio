"""
Phase 4: LightGBM 모델 학습
- objective: regression (연속값)
- 출력: cross-sectional Z-score -> expected_return 변환
- 훈련: 3년(756일) rolling window
- 재훈련: 3개월(63일)마다
- Validation: 훈련 마지막 6개월(126일)
- EWMA Feature Importance: 재훈련마다 feature importance를 EWMA로 추적,
  시간 감쇠 평균으로 feature selection/weighting에 활용 (selection bias 감소)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional

from src.config import PipelineConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# EWMA Feature Importance Tracker
# ---------------------------------------------------------------------------
class EWMAFeatureTracker:
    """재훈련마다 feature importance를 EWMA로 누적 추적.

    - alpha=0.3: 새 importance 30%, 기존 history 70%
    - cold start: ewma_min_retrains 미만이면 uniform 반환
    - drop_pct: 하위 N% feature를 제거 (최소 min_features 유지)
    - weighting: sqrt(ewma / mean) 로 feature scaling
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.ewma_importance: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.n_updates: int = 0
        self.history: List[Dict] = []  # [{date, importances}] for export

    def init_full_features(self, all_feature_names: List[str]) -> None:
        """전체 feature 목록으로 초기화. walk_forward_train 시작 시 1회 호출."""
        self.feature_names = list(all_feature_names)
        self.ewma_importance = np.ones(len(all_feature_names)) / len(all_feature_names)

    def update(self, model: lgb.LGBMRegressor, active_features: List[str],
               retrain_date: pd.Timestamp) -> None:
        """새 모델의 feature importance로 EWMA 업데이트.

        active_features가 전체 feature_names의 부분집합일 수 있으므로,
        전체 배열에서 해당 위치만 업데이트한다.
        """
        raw_imp = model.feature_importances_.astype(float)

        # Normalize to sum=1
        total = raw_imp.sum()
        if total > 0:
            norm_imp = raw_imp / total
        else:
            norm_imp = np.ones(len(raw_imp)) / len(raw_imp)

        # 전체 feature 배열에 매핑
        full_imp = np.zeros(len(self.feature_names))
        for i, fname in enumerate(active_features):
            full_idx = self.feature_names.index(fname)
            full_imp[full_idx] = norm_imp[i]

        alpha = self.config.ewma_alpha
        self.ewma_importance = alpha * full_imp + (1 - alpha) * self.ewma_importance

        # Re-normalize
        total = self.ewma_importance.sum()
        if total > 0:
            self.ewma_importance = self.ewma_importance / total

        self.n_updates += 1
        self.history.append({
            "date": retrain_date,
            "importances": self.ewma_importance.copy(),
        })

    def is_ready(self) -> bool:
        """Cold start 기간이 지났는지 확인."""
        return self.n_updates >= self.config.ewma_min_retrains

    def get_active_features(self, feature_names: List[str]) -> List[str]:
        """EWMA importance 기반으로 활성 feature 목록 반환.

        하위 drop_pct feature를 제거하되 min_features 이상 유지.
        Cold start 기간에는 전체 feature 반환.
        """
        if not self.config.ewma_enabled or not self.is_ready():
            return list(feature_names)

        n_total = len(feature_names)
        n_drop = int(n_total * self.config.ewma_drop_pct)
        n_keep = max(n_total - n_drop, self.config.ewma_min_features)
        n_keep = min(n_keep, n_total)

        # EWMA importance 기준 상위 n_keep개 선택
        sorted_idx = np.argsort(-self.ewma_importance)[:n_keep]
        active = [feature_names[i] for i in sorted(sorted_idx)]
        return active

    def get_feature_weights(self, feature_names: List[str]) -> Optional[np.ndarray]:
        """EWMA importance 기반 feature scaling weights 반환.

        sqrt(ewma / mean) 으로 scaling → 중요 feature 증폭, 약한 feature 감쇠.
        Cold start 기간에는 None 반환 (uniform).
        """
        if not self.config.ewma_enabled or not self.is_ready():
            return None

        mean_imp = self.ewma_importance.mean()
        if mean_imp <= 0:
            return None

        weights = np.sqrt(self.ewma_importance / mean_imp)
        # Clip to [0.5, 2.0] to prevent extreme scaling
        weights = np.clip(weights, 0.5, 2.0)
        return weights

    def export_history(self) -> pd.DataFrame:
        """EWMA importance 이력을 DataFrame으로 반환."""
        if not self.history or self.feature_names is None:
            return pd.DataFrame()

        rows = []
        for h in self.history:
            row = {"date": h["date"]}
            for i, fname in enumerate(self.feature_names):
                row[fname] = h["importances"][i]
            rows.append(row)
        return pd.DataFrame(rows).set_index("date")

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
LGBM_PARAMS = DEFAULT_CONFIG.lgbm_params

# 예측 스무딩: 새 예측 = alpha * new + (1-alpha) * old
# 낮을수록 예측이 천천히 변함 → turnover 감소, 재훈련 상관 증가
PREDICTION_EMA_ALPHA = DEFAULT_CONFIG.prediction_ema_alpha

TRAIN_WINDOW = DEFAULT_CONFIG.train_window       # 3년
RETRAIN_FREQ = DEFAULT_CONFIG.retrain_freq       # 3개월
VAL_WINDOW = DEFAULT_CONFIG.val_window           # 6개월


def _prepare_train_data(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    train_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """훈련 데이터(X, y) 준비. NaN 행 제거."""
    # panel은 MultiIndex (date, ticker)
    mask = panel.index.get_level_values("date").isin(train_dates)
    X_panel = panel.loc[mask, feature_names]

    # targets를 동일한 MultiIndex로 변환
    target_stacked = targets.stack()
    target_stacked.index.names = ["date", "ticker"]
    y_panel = target_stacked.reindex(X_panel.index)

    # NaN 제거
    valid = y_panel.notna() & X_panel.notna().all(axis=1)
    X = X_panel.loc[valid].values
    y = y_panel.loc[valid].values

    return X, y


def train_model(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    config: PipelineConfig = None,
    feature_scale: Optional[np.ndarray] = None,
) -> lgb.LGBMRegressor:
    """단일 모델 훈련 (early stopping with validation)."""
    config = config or DEFAULT_CONFIG
    X_train, y_train = _prepare_train_data(panel, targets, feature_names, train_dates)
    X_val, y_val = _prepare_train_data(panel, targets, feature_names, val_dates)

    # EWMA feature scaling (numpy 레벨 — panel 복사 불필요)
    if feature_scale is not None:
        X_train = X_train * feature_scale[np.newaxis, :]
        if len(X_val) > 0:
            X_val = X_val * feature_scale[np.newaxis, :]

    model = lgb.LGBMRegressor(**config.lgbm_params)

    # REDESIGN D: looser early stopping (default 100 rounds, configurable)
    # to avoid degenerate 10-20 tree models when val loss plateaus briefly.
    es_rounds = getattr(config, "early_stopping_rounds", 100)

    if len(X_val) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    return model


MIN_TREES = 10  # 최소 트리 수 (이하면 degenerate로 판단)


def _compute_window_bounds(
    t_idx: int,
    train_window: int,
    val_window: int,
    embargo: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute (train_start, train_end, val_start, val_end) with embargo gap.

    Layout (indices, train_window includes val_window+embargo internally is NOT
    assumed — embargo is carved out of the train window's right edge):

        train:   [t_idx - train_window,         t_idx - val_window - embargo)
        embargo: [t_idx - val_window - embargo, t_idx - val_window)            (dropped)
        val:     [t_idx - val_window,           t_idx - embargo)
        embargo: [t_idx - embargo,              t_idx)                          (dropped)
        predict: t_idx

    With embargo=forward_horizon=20, the last train label's forward window ends
    exactly at val_start, and the last val label's forward window ends exactly
    at the predict bar — no peek.

    Returns None if any window becomes empty (caller should skip the retrain
    and reuse the previous model).
    """
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    train_start = max(0, t_idx - train_window)
    train_end = t_idx - val_window - embargo
    val_start = t_idx - val_window
    val_end = t_idx - embargo

    if train_end <= train_start:
        return None
    if val_end <= val_start:
        return None

    return train_start, train_end, val_start, val_end


def predict_cross_sectional(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    feature_names: List[str],
    pred_date: pd.Timestamp,
    feature_scale: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    단일 날짜의 cross-sectional 예측.
    raw 예측값을 cross-sectional Z-score로 변환.
    """
    mask = panel.index.get_level_values("date") == pred_date
    X = panel.loc[mask, feature_names]

    if len(X) == 0:
        return pd.Series(dtype=float)

    X_vals = X.values
    if feature_scale is not None:
        X_vals = X_vals * feature_scale[np.newaxis, :]

    raw_pred = model.predict(X_vals)
    tickers = X.index.get_level_values("ticker")

    pred = pd.Series(raw_pred, index=tickers, name="expected_return")

    # Cross-sectional Z-score 변환
    mean = pred.mean()
    std = pred.std()
    if std > 0:
        pred = (pred - mean) / std

    return pred


def walk_forward_train(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    feature_names: List[str],
    all_dates: pd.DatetimeIndex,
    train_window: int = TRAIN_WINDOW,
    retrain_freq: int = RETRAIN_FREQ,
    val_window: int = VAL_WINDOW,
    config: PipelineConfig = None,
) -> Tuple[Dict[pd.Timestamp, lgb.LGBMRegressor], pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward 방식으로 모델 학습 및 예측 생성.

    Returns:
        models: {재훈련 시점: 모델}
        predictions: DataFrame (date x ticker) EMA 블렌딩된 예측값
        raw_predictions: DataFrame (date x ticker) 블렌딩 전 순수 모델 예측값 (IC 계산용)
    """
    config = config or DEFAULT_CONFIG
    # When explicit window args differ from module defaults, honour them;
    # otherwise fall back to config values.
    if train_window == TRAIN_WINDOW:
        train_window = config.train_window
    if retrain_freq == RETRAIN_FREQ:
        retrain_freq = config.retrain_freq
    if val_window == VAL_WINDOW:
        val_window = config.val_window

    # OOS hold-out enforcement: during tuning, prevent predictions beyond
    # the reserved OOS window so accidental peeking can't contaminate the
    # selection-bias accounting. See config.enforce_oos_holdout for rationale.
    oos_cutoff_idx: Optional[int] = None
    if getattr(config, "enforce_oos_holdout", False):
        cutoff_str = getattr(config, "train_cutoff_date", None)
        if cutoff_str:
            cutoff_ts = pd.Timestamp(cutoff_str)
            # Find last index position where date <= cutoff.
            mask_le_cutoff = all_dates <= cutoff_ts
            if mask_le_cutoff.any():
                oos_cutoff_idx = int(np.where(mask_le_cutoff)[0][-1]) + 1
                print(f"[WalkForward] OOS hold-out ACTIVE — predicting only "
                      f"through {cutoff_str} (idx {oos_cutoff_idx}/{len(all_dates)}, "
                      f"reserving {len(all_dates) - oos_cutoff_idx} days)")
            else:
                raise ValueError(
                    f"train_cutoff_date={cutoff_str} is before all_dates start "
                    f"({all_dates[0]}). Cannot enforce OOS hold-out."
                )

    predictions = pd.DataFrame(index=all_dates, columns=targets.columns, dtype=float)
    raw_predictions = pd.DataFrame(index=all_dates, columns=targets.columns, dtype=float)
    models = {}
    current_model = None
    prev_model = None
    # Track the (active_features, active_fw) tuple that each retained model
    # was actually trained on, so that when a degenerate model triggers a
    # "reuse prev_model" fallback we also restore the matching feature set.
    # Without this, EWMA keeps dropping features independently and the next
    # predict() call feeds a narrower X into a wider model -> ValueError
    # ("X has 328 features, LGBMRegressor expects 345").
    current_features: Optional[List[str]] = None
    current_fw: Optional[np.ndarray] = None
    prev_features: Optional[List[str]] = None
    prev_fw: Optional[np.ndarray] = None
    last_train_idx = -retrain_freq  # 첫 루프에서 바로 훈련
    prev_pred = None  # EMA 스무딩용

    # EWMA Feature Importance Tracker
    ewma_tracker = EWMAFeatureTracker(config)
    ewma_tracker.init_full_features(feature_names)
    active_features = list(feature_names)  # 초기: 전체 feature 사용
    active_fw = None  # 초기: uniform

    # OOS hold-out: if cutoff is set, iterate only up to the cutoff index.
    last_idx_exclusive = oos_cutoff_idx if oos_cutoff_idx is not None else len(all_dates)

    for t_idx in range(train_window, last_idx_exclusive):
        t_date = all_dates[t_idx]

        # 재훈련 시점인지 확인
        do_retrain = (t_idx - last_train_idx >= retrain_freq or current_model is None)

        if do_retrain:
            embargo = getattr(config, "embargo_days", 0)
            bounds = _compute_window_bounds(t_idx, train_window, val_window, embargo)

            if bounds is None and current_model is not None:
                # Window too narrow under the current embargo — reuse previous
                # model rather than training on (near-)empty data. Discipline
                # > completeness.
                print(f"[ModelTrainer] embargo skip @ {t_date.strftime('%Y-%m-%d')} "
                      f"(t_idx={t_idx}, embargo={embargo}, val_window={val_window}, "
                      f"train_window={train_window}) — reusing prev model.")
                last_train_idx = t_idx  # 다음 retrain 시점 갱신
                do_retrain = False

        if do_retrain:
            if bounds is None:
                # No prev model AND bounds collapsed — emergency bootstrap on
                # legacy (leaky) windows so the loop can start. Only hit when
                # embargo + val_window > train_window at the first iteration.
                print(f"[ModelTrainer] WARNING: embargo window collapse at "
                      f"t_idx={t_idx} with no prev model — bootstrap on full window.")
                train_start = max(0, t_idx - train_window)
                train_end = t_idx - val_window
                val_start = t_idx - val_window
                val_end = t_idx
            else:
                train_start, train_end, val_start, val_end = bounds

            train_dates = all_dates[train_start:train_end]
            val_dates = all_dates[val_start:val_end]

            # EWMA: 이전 재훈련 결과 기반으로 active feature 선택 (candidate)
            candidate_features = ewma_tracker.get_active_features(feature_names)
            n_dropped = len(feature_names) - len(candidate_features)

            # EWMA: candidate feature 의 weight 벡터 (numpy 레벨 적용)
            fw = ewma_tracker.get_feature_weights(feature_names)
            if fw is not None:
                candidate_fw = np.array(
                    [fw[feature_names.index(f)] for f in candidate_features]
                )
            else:
                candidate_fw = None

            # Snapshot current (feature_set, weights) before swapping models.
            # If the new model turns out to be degenerate we will rewind
            # BOTH the model AND the feature set to this snapshot.
            prev_model = current_model
            prev_features = current_features
            prev_fw = current_fw

            new_model = train_model(panel, targets, candidate_features, train_dates, val_dates,
                                    config=config, feature_scale=candidate_fw)

            # Degenerate 모델 fallback: n_trees < MIN_TREES 이면 이전 (model+features+fw) 재사용
            n_trees = new_model.n_estimators_
            if n_trees < MIN_TREES and prev_model is not None:
                print(f"[ModelTrainer] WARNING: Degenerate model ({n_trees} trees) "
                      f"-> reuse prev model AND prev feature set ({len(prev_features)} features)")
                current_model = prev_model
                active_features = prev_features
                active_fw = prev_fw
                current_features = prev_features
                current_fw = prev_fw
            else:
                current_model = new_model
                active_features = candidate_features
                active_fw = candidate_fw
                current_features = candidate_features
                current_fw = candidate_fw
                # EWMA: 새 모델의 importance로 tracker 업데이트
                ewma_tracker.update(current_model, active_features, t_date)

            # Attach the active feature set (and weights) to the model so
            # downstream code (attribution.py SHAP, re-prediction) can slice
            # the panel to the exact columns this model was trained on.
            # Without this, any subsequent LGBM predict() with the full panel
            # crashes with "number of features in data != training data".
            try:
                current_model._active_features = list(active_features)
                current_model._active_fw = (
                    None if active_fw is None else np.array(active_fw, copy=True)
                )
            except Exception:
                pass

            models[t_date] = current_model
            last_train_idx = t_idx

            ewma_status = ""
            if ewma_tracker.is_ready():
                ewma_status = f", EWMA: {len(active_features)}/{len(feature_names)} features"
                if n_dropped > 0:
                    ewma_status += f" (-{n_dropped})"

            print(f"[ModelTrainer] 재훈련 @ {t_date.strftime('%Y-%m-%d')} "
                  f"(train: {len(train_dates)}d, val: {len(val_dates)}d, trees: {n_trees}{ewma_status})")

        # 예측: EWMA feature weight를 numpy 레벨에서 적용
        pred = predict_cross_sectional(current_model, panel, active_features, t_date,
                                       feature_scale=active_fw)

        # raw 예측 저장 (EMA 블렌딩 전, IC 계산용) — vectorized
        common_raw = pred.index.intersection(raw_predictions.columns)
        if len(common_raw) > 0:
            raw_predictions.loc[t_date, common_raw] = pred[common_raw].values

        # EMA 스무딩: α=0.5로 완화
        if prev_pred is not None and len(pred) > 0:
            alpha = config.prediction_ema_alpha
            common = pred.index.intersection(prev_pred.index)
            blended = alpha * pred[common] + (1 - alpha) * prev_pred[common]
            pred[common] = blended

        if len(pred) > 0:
            prev_pred = pred.copy()
            # vectorized 저장
            common_pred = pred.index.intersection(predictions.columns)
            if len(common_pred) > 0:
                predictions.loc[t_date, common_pred] = pred[common_pred].values

    valid_count = predictions.notna().sum().sum()
    print(f"[ModelTrainer] 예측 완료: {valid_count}개 관측치")

    if ewma_tracker.is_ready():
        print(f"[ModelTrainer] EWMA Feature Tracker: {ewma_tracker.n_updates} updates, "
              f"final active features: {len(active_features)}/{len(feature_names)}")

    return models, predictions, raw_predictions, ewma_tracker
