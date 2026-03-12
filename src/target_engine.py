"""
Phase 3: 타겟 변수 생성
20일 Specific Return = PCA 잔차 수익률

각 시점 t에서:
1. 과거 252일 일간 수익률로 PCA fitting (n_components=5)
2. t~t+20 영업일 forward cumulative return
3. PCA common component 제거
4. 잔차 = Specific Return = 타겟

look-ahead bias 방지: PCA fitting은 반드시 과거 데이터만 사용.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from src.data_loader import UniverseData, TICKERS

PCA_COMPONENTS = 5
PCA_LOOKBACK = 252
FORWARD_HORIZON = 20


def compute_forward_returns(returns: pd.DataFrame, horizon: int = FORWARD_HORIZON) -> pd.DataFrame:
    """t~t+horizon 영업일 forward cumulative return 계산."""
    fwd = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for i in range(len(returns) - horizon):
        fwd.iloc[i] = returns.iloc[i + 1: i + 1 + horizon].sum()
    return fwd


def compute_specific_returns(
    returns: pd.DataFrame,
    n_components: int = PCA_COMPONENTS,
    lookback: int = PCA_LOOKBACK,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    """
    각 시점에서 PCA 잔차 기반 20일 Specific Return 계산.

    Returns:
        specific_ret: DataFrame (dates x tickers), 값 = 20일 specific return
    """
    dates = returns.index
    tickers = returns.columns
    n_dates = len(dates)

    # Forward cumulative returns
    fwd_ret = compute_forward_returns(returns, horizon)

    specific_ret = pd.DataFrame(np.nan, index=dates, columns=tickers)

    for t in range(lookback, n_dates - horizon):
        # 과거 lookback일 일간 수익률로 PCA fitting
        hist_returns = returns.iloc[t - lookback: t].copy()

        # 결측치가 너무 많으면 스킵
        valid_mask = hist_returns.notna().all(axis=1)
        hist_clean = hist_returns.loc[valid_mask]

        if len(hist_clean) < lookback // 2:
            continue

        # PCA fitting
        try:
            pca = PCA(n_components=min(n_components, len(tickers) - 1))
            pca.fit(hist_clean.values)
        except Exception:
            continue

        # 시점 t의 forward return
        fwd_t = fwd_ret.iloc[t].values.reshape(1, -1)

        if np.any(np.isnan(fwd_t)):
            continue

        # Common component = PCA reconstruction
        factors = pca.transform(fwd_t)
        common = pca.inverse_transform(factors)

        # Specific return = forward return - common component
        spec = fwd_t - common
        specific_ret.iloc[t] = spec.flatten()

    return specific_ret


def build_targets(data: UniverseData) -> pd.DataFrame:
    """
    UniverseData에서 타겟 변수(20일 Specific Return) 생성.

    Returns:
        targets: DataFrame (dates x tickers)
    """
    returns = data.returns
    targets = compute_specific_returns(returns)

    valid_count = targets.notna().sum().sum()
    total = targets.size
    print(f"[TargetEngine] 타겟 생성 완료 (Round4 방식: Simple PCA Residual)")
    print(f"  기간: {targets.index[0].strftime('%Y-%m-%d')} ~ {targets.index[-1].strftime('%Y-%m-%d')}")
    print(f"  유효 관측치: {valid_count} / {total} ({valid_count / total * 100:.1f}%)")

    return targets
