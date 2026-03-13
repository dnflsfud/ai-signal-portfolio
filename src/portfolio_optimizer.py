"""
Phase 6: 포트폴리오 최적화
cvxpy Mean-Variance Optimization.

목적함수: Maximize(E[r] @ w - lambda * risk - tc * turnover)
- risk = quad_form(w - bm_weights, cov_matrix)
- turnover = norm1(w - prev_weights)

제약: sum(w)=1, w>=0, TE<=5%(ann), 섹터 +-10% vs BM
Cov matrix: 126일 Ledoit-Wolf shrinkage
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from typing import Optional, Dict, List

RISK_AVERSION = 0.5       # Grid Search 최적
TURNOVER_PENALTY = 0.05   # Grid Search 최적
MAX_TE_ANNUAL = 0.025     # 연율화 TE 상한 2.5%
MAX_SINGLE_TURNOVER = 0.15
SECTOR_DEVIATION = 0.20
COV_LOOKBACK = 126


def estimate_covariance(returns: pd.DataFrame, lookback: int = COV_LOOKBACK) -> np.ndarray:
    """Ledoit-Wolf shrinkage로 공분산 행렬 추정."""
    recent = returns.iloc[-lookback:].dropna()
    if len(recent) < 30:
        return np.eye(returns.shape[1]) * 0.04 / 252

    lw = LedoitWolf()
    lw.fit(recent.values)
    return lw.covariance_


def build_sector_constraints(
    tickers: List[str],
    sector_map: Dict[str, str],
    bm_weights: np.ndarray,
) -> List:
    """섹터 제약 조건 생성: EW 대비 +-10%."""
    constraints = []

    # 섹터별 종목 인덱스
    sector_groups: Dict[str, List[int]] = {}
    for i, t in enumerate(tickers):
        sec = sector_map.get(t, "Unknown")
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(i)

    return sector_groups


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    sector_map: Optional[Dict[str, str]] = None,
    bm_weights: Optional[np.ndarray] = None,
    risk_aversion: float = RISK_AVERSION,
    turnover_penalty: float = TURNOVER_PENALTY,
    max_te_annual: float = MAX_TE_ANNUAL,
    sector_deviation: float = SECTOR_DEVIATION,
) -> np.ndarray:
    """
    MVO 포트폴리오 최적화 (TE 제약).

    Args:
        expected_returns: 종목별 기대수익률 (Z-score)
        cov_matrix: 공분산 행렬 (일간)
        prev_weights: 이전 비중 (없으면 BM)
        sector_map: {ticker: sector}
        risk_aversion: 위험회피계수
        turnover_penalty: 회전율 페널티
        max_te_annual: 연율화 TE 상한 (기본 5%)
        sector_deviation: 섹터 편차 허용 범위

    Returns:
        최적 비중 배열
    """
    n = len(expected_returns)
    tickers = list(expected_returns.index)
    if bm_weights is None:
        bm_weights = np.ones(n) / n  # Default to EW
    bm_weights = np.asarray(bm_weights, dtype=float)

    if prev_weights is None:
        prev_weights = bm_weights.copy()

    # 변수
    w = cp.Variable(n)

    # 기대수익
    mu = expected_returns.values
    ret = mu @ w

    # 리스크: quad_form(w - bm, Sigma)
    active = w - bm_weights
    risk = cp.quad_form(active, cp.psd_wrap(cov_matrix))

    # 턴오버
    turnover = cp.norm1(w - prev_weights)

    # 목적함수
    objective = cp.Maximize(ret - risk_aversion * risk - turnover_penalty * turnover)

    # TE 제약: (w-bm)' Σ (w-bm) * 252 <= max_te^2
    # → daily variance <= max_te^2 / 252
    max_daily_te_var = max_te_annual ** 2 / 252.0

    # 제약
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        risk <= max_daily_te_var,              # TE ≤ 5% (연율화) 하드 제약
        turnover <= MAX_SINGLE_TURNOVER,       # 1회 리밸런싱 턴오버 하드캡
    ]

    # 섹터 제약
    if sector_map is not None:
        sector_groups = build_sector_constraints(tickers, sector_map, bm_weights)
        for sec, indices in sector_groups.items():
            if len(indices) == 0:
                continue
            sector_bm = sum(bm_weights[i] for i in indices)
            sector_w = cp.sum([w[i] for i in indices])
            constraints.append(sector_w >= sector_bm - sector_deviation)
            constraints.append(sector_w <= sector_bm + sector_deviation)

    # 풀기
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, max_iters=500)
    except cp.SolverError:
        try:
            prob.solve(solver=cp.SCS, max_iters=5000)
        except cp.SolverError:
            return bm_weights.copy()

    if prob.status in ("optimal", "optimal_inaccurate"):
        opt_w = np.array(w.value).flatten()
        # 음수 클리핑 및 정규화
        opt_w = np.maximum(opt_w, 0)
        opt_w = opt_w / opt_w.sum()
        return opt_w

    return bm_weights.copy()
