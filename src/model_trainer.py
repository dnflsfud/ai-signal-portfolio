"""
Phase 4: LightGBM 모델 학습
- objective: regression (연속값)
- 출력: cross-sectional Z-score -> expected_return 변환
- 훈련: 3년(756일) rolling window
- 재훈련: 3개월(63일)마다
- Validation: 훈련 마지막 6개월(126일)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple, Optional

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.008,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 40,
    "subsample": 0.7,
    "colsample_bytree": 0.4,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "n_estimators": 1500,
    "verbose": -1,
    "random_state": 42,
}

# 예측 스무딩: 새 예측 = alpha * new + (1-alpha) * old
# 낮을수록 예측이 천천히 변함 → turnover 감소, 재훈련 상관 증가
PREDICTION_EMA_ALPHA = 0.5

TRAIN_WINDOW = 756      # 3년
RETRAIN_FREQ = 63       # 3개월
VAL_WINDOW = 126         # 6개월


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
) -> lgb.LGBMRegressor:
    """단일 모델 훈련 (early stopping with validation)."""
    X_train, y_train = _prepare_train_data(panel, targets, feature_names, train_dates)
    X_val, y_val = _prepare_train_data(panel, targets, feature_names, val_dates)

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    if len(X_val) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)

    return model


MIN_TREES = 10  # 최소 트리 수 (이하면 degenerate로 판단)


def predict_cross_sectional(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    feature_names: List[str],
    pred_date: pd.Timestamp,
) -> pd.Series:
    """
    단일 날짜의 cross-sectional 예측.
    raw 예측값을 cross-sectional Z-score로 변환.
    """
    mask = panel.index.get_level_values("date") == pred_date
    X = panel.loc[mask, feature_names]

    if len(X) == 0:
        return pd.Series(dtype=float)

    raw_pred = model.predict(X.values)
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
) -> Tuple[Dict[pd.Timestamp, lgb.LGBMRegressor], pd.DataFrame]:
    """
    Walk-forward 방식으로 모델 학습 및 예측 생성.

    Returns:
        models: {재훈련 시점: 모델}
        predictions: DataFrame (date x ticker) 예측값
    """
    predictions = pd.DataFrame(index=all_dates, columns=targets.columns, dtype=float)
    models = {}
    current_model = None
    prev_model = None
    last_train_idx = -retrain_freq  # 첫 루프에서 바로 훈련
    prev_pred = None  # EMA 스무딩용

    for t_idx in range(train_window, len(all_dates)):
        t_date = all_dates[t_idx]

        # 재훈련 시점인지 확인
        if t_idx - last_train_idx >= retrain_freq or current_model is None:
            train_start = max(0, t_idx - train_window)
            train_end = t_idx - val_window
            val_start = t_idx - val_window
            val_end = t_idx

            if train_end <= train_start:
                train_end = t_idx
                val_start = t_idx
                val_end = t_idx

            train_dates = all_dates[train_start:train_end]
            val_dates = all_dates[val_start:val_end]

            prev_model = current_model
            new_model = train_model(panel, targets, feature_names, train_dates, val_dates)

            # Degenerate 모델 fallback: n_trees < MIN_TREES이면 이전 모델 재사용
            n_trees = new_model.n_estimators_
            if n_trees < MIN_TREES and prev_model is not None:
                print(f"[ModelTrainer] WARNING: Degenerate model ({n_trees} trees) -> reuse prev model")
                current_model = prev_model
            else:
                current_model = new_model
            models[t_date] = current_model
            last_train_idx = t_idx

            print(f"[ModelTrainer] 재훈련 @ {t_date.strftime('%Y-%m-%d')} "
                  f"(train: {len(train_dates)}d, val: {len(val_dates)}d, trees: {n_trees})")

        # 예측: 현재 모델만 사용 (앙상블 제거 — 시그널 즉시 반영)
        pred = predict_cross_sectional(current_model, panel, feature_names, t_date)

        # EMA 스무딩: α=0.5로 완화 (이전 0.2 → 새 정보 50% 반영)
        if prev_pred is not None and len(pred) > 0:
            alpha = PREDICTION_EMA_ALPHA
            common = pred.index.intersection(prev_pred.index)
            blended = alpha * pred[common] + (1 - alpha) * prev_pred[common]
            # 블렌딩 후 다시 Z-score 정규화 (scale 유지)
            mean_b = blended.mean()
            std_b = blended.std()
            if std_b > 0:
                blended = (blended - mean_b) / std_b
            pred[common] = blended

        if len(pred) > 0:
            prev_pred = pred.copy()
            for ticker in pred.index:
                if ticker in predictions.columns:
                    predictions.loc[t_date, ticker] = pred[ticker]

    valid_count = predictions.notna().sum().sum()
    print(f"[ModelTrainer] 예측 완료: {valid_count}개 관측치")

    return models, predictions
