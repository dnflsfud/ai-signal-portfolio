"""
Phase 6: portfolio optimisation.

The core optimizer is long-only benchmark-aware MVO:
    maximize(mu @ w - lambda * risk - tc * turnover)

Hard constraints are reused by both:
- the target optimizer
- the post-smoothing execution projection

That keeps the realised book inside the same feasible region as the target
book without changing any configured parameter values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger(__name__)

# M3: condition-number threshold. Covariance matrices above this are ill-
# conditioned enough that cvxpy quad_form solves start producing garbage or
# NaN weights. ~1e8 ≈ 8 digits of precision loss on a 64-bit solve.
_COV_COND_WARN_THRESHOLD = 1e8

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases (read from DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
RISK_AVERSION = DEFAULT_CONFIG.risk_aversion
TURNOVER_PENALTY = DEFAULT_CONFIG.turnover_penalty
MAX_TE_ANNUAL = DEFAULT_CONFIG.max_te_annual
MAX_SINGLE_TURNOVER = DEFAULT_CONFIG.max_single_turnover
SECTOR_DEVIATION = DEFAULT_CONFIG.sector_deviation
COV_LOOKBACK = DEFAULT_CONFIG.cov_lookback
BM_WEIGHT_FLOOR = DEFAULT_CONFIG.bm_weight_floor
MAX_ACTIVE_SHARE = DEFAULT_CONFIG.max_active_share
MAX_WEIGHT = DEFAULT_CONFIG.max_weight
MAX_ACTIVE_PER_STOCK = DEFAULT_CONFIG.max_active_per_stock
USE_SCORE_BASED = DEFAULT_CONFIG.use_score_based


def print_optimizer_config(n_tickers: int = 15, config: PipelineConfig = None):
    """Print the effective optimiser constraints."""
    config = config or DEFAULT_CONFIG
    ew = 1.0 / n_tickers
    lines = [
        "+" + "-" * 48 + "+",
        "|       Portfolio Optimizer Constraints        |",
        "+" + "-" * 48 + "+",
        f"| MAX_WEIGHT           : {config.max_weight:>6.1%}            |",
        f"| MAX_ACTIVE_PER_STOCK : +/-{config.max_active_per_stock:>4.1%}         |",
        f"| BM_WEIGHT_FLOOR      : {config.bm_weight_floor:.0%} of BM ({ew * config.bm_weight_floor:.1%}) |",
        f"| MAX_ACTIVE_SHARE     : {config.max_active_share:.0%}               |",
        f"| MAX_TE_ANNUAL        : {config.max_te_annual:.1%}            |",
        f"| SECTOR_DEVIATION     : +/-{config.sector_deviation:.0%}            |",
        f"| RISK_AVERSION        : {config.risk_aversion:<6}           |",
        f"| TURNOVER_PENALTY     : {config.turnover_penalty:<6}           |",
        f"| MAX_SINGLE_TURNOVER  : {config.max_single_turnover:.0%}               |",
        f"| MODE                 : {'SCORE-BASED' if config.use_score_based else 'MVO':<12} |",
        "+" + "-" * 48 + "+",
    ]
    print("\n".join(lines))


def estimate_covariance(
    returns: pd.DataFrame,
    lookback: int = COV_LOOKBACK,
    bm_weights: Optional[np.ndarray] = None,
    config: PipelineConfig = None,
) -> np.ndarray:
    """Estimate covariance via Ledoit-Wolf shrinkage."""
    config = config or DEFAULT_CONFIG
    if lookback == COV_LOOKBACK:
        lookback = config.cov_lookback

    recent = returns.iloc[-lookback:].dropna()
    if len(recent) < 30:
        return np.eye(returns.shape[1]) * 0.04 / 252.0

    lw = LedoitWolf()
    lw.fit(recent.values)
    cov = lw.covariance_.copy()

    # Mild mega-cap volatility shrinkage while preserving PSD via D @ S @ D.
    if bm_weights is not None:
        n = len(bm_weights)
        mean_bm = 1.0 / n
        vols = np.sqrt(np.diag(cov))
        avg_vol = vols.mean()
        scale = np.ones(n)
        for i in range(n):
            if bm_weights[i] > mean_bm * 2:
                if vols[i] > 0:
                    shrink_factor = (0.5 * avg_vol + 0.5 * vols[i]) / vols[i]
                else:
                    shrink_factor = 1.0
                scale[i] = np.sqrt(shrink_factor)
        cov = np.diag(scale) @ cov @ np.diag(scale)

    # M3: warn on ill-conditioned covariance before it reaches cvxpy.
    # Using np.linalg.cond is O(n^3) but covs are 50x50 at most — negligible.
    try:
        cond = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if not np.isfinite(cond) or cond > _COV_COND_WARN_THRESHOLD:
        logger.warning(
            "estimate_covariance: ill-conditioned covariance (cond=%.2e, threshold=%.0e). "
            "MVO may produce unstable or fallback weights.",
            cond, _COV_COND_WARN_THRESHOLD,
        )

    return cov


def build_sector_constraints(
    tickers: List[str],
    sector_map: Dict[str, str],
) -> Dict[str, List[int]]:
    """Return sector -> position-index mapping."""
    sector_groups: Dict[str, List[int]] = {}
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "Unknown")
        sector_groups.setdefault(sector, []).append(i)
    return sector_groups


def score_based_weights(
    expected_returns: pd.Series,
    max_weight: float = MAX_WEIGHT,
    min_weight: float = 0.002,
) -> np.ndarray:
    """Simple softmax-based weighting for score-only mode."""
    scores = expected_returns.values.copy()
    scores_shifted = scores - scores.max()
    exp_scores = np.exp(scores_shifted)
    raw_weights = exp_scores / exp_scores.sum()

    for _ in range(10):
        raw_weights = np.maximum(raw_weights, min_weight)
        raw_weights = np.minimum(raw_weights, max_weight)
        raw_weights = raw_weights / raw_weights.sum()
        if raw_weights.max() <= max_weight + 1e-6 and raw_weights.min() >= min_weight - 1e-6:
            break

    return raw_weights


def _init_diagnostics(
    diagnostics: Optional[Dict[str, Any]],
    *,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """Initialise standard diagnostics fields while preserving caller metadata."""
    if diagnostics is None:
        return None
    diagnostics.update({
        "mode": mode,
        "solver": None,
        "status": None,
        "used_fallback": False,
        "fallback_reason": None,
    })
    return diagnostics


def _solve_problem(
    prob: cp.Problem,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> bool:
    """Solve a CVXPY problem with ECOS fallback to SCS."""
    last_error = None
    for solver, solver_name, max_iters in (
        (cp.ECOS, "ECOS", 500),
        (cp.SCS, "SCS", 5000),
    ):
        try:
            prob.solve(solver=solver, max_iters=max_iters)
            if diagnostics is not None:
                diagnostics["solver"] = solver_name
                diagnostics["status"] = prob.status
            return True
        except cp.SolverError as exc:
            last_error = str(exc)
        except ValueError as exc:
            # cvxpy raises ValueError("Problem data contains NaN or Inf.")
            # when mu/cov has non-finite entries. Treat as solver failure so
            # the caller falls back to bm_weights — matches the pre-2026-04-21
            # behaviour where this case was implicitly handled by the fact
            # that cvxpy used to propagate the same condition as SolverError.
            last_error = f"invalid-input: {exc}"

    if diagnostics is not None:
        diagnostics["status"] = "solver_error"
        diagnostics["fallback_reason"] = last_error or "solver_error"
    return False


def compute_bm_proportional_active_cap(
    bm_weights: np.ndarray,
    cov_matrix: Optional[np.ndarray],
    config: PipelineConfig,
) -> np.ndarray:
    """Return per-name active-cap multipliers derived from BM weight + vol.

    Multiplier semantics:
      - 1.0 → no change vs symmetric base cap.
      - >1.0 → more active room (mega caps).
      - <1.0 → less active room (high-vol names).

    Returns a flat array of length n with the final multiplicative scale.
    Caller applies: cap_i = base_cap × multiplier_i.
    """
    n = len(bm_weights)
    mult = np.ones(n, dtype=float)

    if not getattr(config, "bm_proportional_cap_enabled", False):
        return mult

    # BM-proportional term
    bm_top = float(np.max(bm_weights)) if n > 0 else 1.0
    if bm_top <= 0:
        bm_term = np.ones(n)
    else:
        top_scale = float(getattr(config, "bm_proportional_cap_bm_scale_at_top", 1.5))
        bm_term = 1.0 + (top_scale - 1.0) * (bm_weights / bm_top)

    # Vol-proportional term (inverse): high-vol → smaller cap
    if cov_matrix is not None and cov_matrix.shape[0] == n:
        vols = np.sqrt(np.clip(np.diag(cov_matrix), 1e-12, None))
        med_vol = float(np.median(vols))
        floor = float(getattr(config, "bm_proportional_cap_vol_scale_floor", 0.5))
        vol_term = np.clip(med_vol / vols, floor, 1.0)
    else:
        vol_term = np.ones(n)

    mult = bm_term * vol_term
    return mult


def _build_mvo_constraints(
    w: cp.Variable,
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: np.ndarray,
    sector_map: Optional[Dict[str, str]],
    bm_weights: np.ndarray,
    max_te_annual: float,
    sector_deviation: float,
    config: PipelineConfig,
) -> Tuple[cp.Expression, cp.Expression, List[cp.Constraint]]:
    """Build the shared hard constraints used by optimisation and projection."""
    n = len(expected_returns)
    tickers = list(expected_returns.index)
    mu = expected_returns.values

    active = w - bm_weights
    risk = cp.quad_form(active, cp.psd_wrap(cov_matrix))
    turnover = cp.norm1(w - prev_weights)

    max_daily_te_var = max_te_annual ** 2 / 252.0
    single_turnover_limit = config.max_single_turnover

    # Per-name max_weight (default uniform).
    max_weight_per = np.full(n, config.max_weight, dtype=float)

    # Per-name active bounds (default: symmetric at config.max_active_per_stock).
    base_active_cap = config.max_active_per_stock
    max_ow_per = np.full(n, base_active_cap, dtype=float)
    max_uw_per = np.full(n, base_active_cap, dtype=float)

    # BM-proportional cap infrastructure (OFF by default; see config).
    bm_mult = compute_bm_proportional_active_cap(bm_weights, cov_matrix, config)
    max_ow_per = max_ow_per * bm_mult
    max_uw_per = max_uw_per * bm_mult

    megacap_enabled = getattr(config, "mega_cap_protection_enabled", False)

    if megacap_enabled:
        mega_bm_thr = config.mega_cap_bm_threshold
        wide_uw_cap = config.mega_cap_wide_uw_cap
        funding_mode = getattr(config, "mega_cap_funding_mode", False)
        funding_k = int(getattr(config, "mega_cap_funding_k", 0))
        funding_score_max = float(getattr(config, "mega_cap_funding_score_max", 0.0))

        mega_indices = [
            i for i in range(n)
            if bm_weights[i] >= mega_bm_thr
        ]

        if funding_mode and funding_k > 0 and mega_indices:
            scored = [
                (i, mu[i] if np.isfinite(mu[i]) else 0.0)
                for i in mega_indices
            ]
            eligible = [(i, s) for i, s in scored if s < funding_score_max]
            eligible.sort(key=lambda x: x[1])
            funding_set = {i for i, _ in eligible[:funding_k]}

            for i in mega_indices:
                if i in funding_set:
                    max_uw_per[i] = max(max_uw_per[i], wide_uw_cap)
                    max_ow_per[i] = 0.0
                else:
                    max_uw_per[i] = 0.0

    constraints: List[cp.Constraint] = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight_per,
        risk <= max_daily_te_var,
        turnover <= single_turnover_limit,
    ]

    weight_floor = bm_weights * config.bm_weight_floor
    for i in range(n):
        constraints.append(w[i] >= weight_floor[i])

    for i in range(n):
        constraints.append(w[i] - bm_weights[i] <= max_ow_per[i])
        constraints.append(bm_weights[i] - w[i] <= max_uw_per[i])

    if getattr(config, "enforce_score_gated_ow", False):
        score_threshold = getattr(config, "score_threshold_for_ow", 0.0)
        for i in range(n):
            score_i = mu[i]
            # Use <= so that score == threshold (e.g. value-trap-gated names
            # zeroed by vtg_scale=0.0) cannot be overweighted. Previously
            # `<` allowed score==0.0 to slip through and MVO could still OW
            # them via the diversification term — defeating the gate's intent.
            if not np.isfinite(score_i) or score_i <= score_threshold:
                constraints.append(w[i] <= bm_weights[i])

    constraints.append(cp.norm1(w - bm_weights) <= config.max_active_share)

    sec_dev = sector_deviation
    if sector_map is not None:
        sector_groups = build_sector_constraints(tickers, sector_map)
        for indices in sector_groups.values():
            if not indices:
                continue
            sector_bm = float(np.sum(bm_weights[indices]))
            sector_w = cp.sum(w[indices])
            constraints.append(sector_w >= sector_bm - sec_dev)
            constraints.append(sector_w <= sector_bm + sec_dev)

    return risk, turnover, constraints


def project_capped_weights(
    candidate_weights: np.ndarray,
    max_weight: float = MAX_WEIGHT,
    fallback_weights: Optional[np.ndarray] = None,
    config: PipelineConfig = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project weights onto the long-only capped simplex."""
    config = config or DEFAULT_CONFIG
    if max_weight == MAX_WEIGHT:
        max_weight = config.max_weight

    candidate = np.asarray(candidate_weights, dtype=float)
    fallback = np.asarray(
        fallback_weights if fallback_weights is not None else candidate_weights,
        dtype=float,
    ).copy()

    diag = _init_diagnostics(diagnostics, mode="projection_capped")
    w = cp.Variable(len(candidate))
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w - candidate)),
        [cp.sum(w) == 1, w >= 0, w <= max_weight],
    )
    if not _solve_problem(prob, diag) or prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = diag.get("fallback_reason") or prob.status or "projection_failed"
        return fallback

    projected = np.asarray(w.value, dtype=float).flatten()
    if not np.all(np.isfinite(projected)):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = "non_finite_projection"
        return fallback

    return projected


def project_portfolio_weights(
    candidate_weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
    sector_map: Optional[Dict[str, str]] = None,
    bm_weights: Optional[np.ndarray] = None,
    max_te_annual: float = MAX_TE_ANNUAL,
    sector_deviation: float = SECTOR_DEVIATION,
    config: PipelineConfig = None,
    fallback_weights: Optional[np.ndarray] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project candidate weights back into the existing MVO hard constraints."""
    config = config or DEFAULT_CONFIG
    if max_te_annual == MAX_TE_ANNUAL:
        max_te_annual = config.max_te_annual
    if sector_deviation == SECTOR_DEVIATION:
        sector_deviation = config.sector_deviation

    candidate = np.asarray(candidate_weights, dtype=float)
    n = len(candidate)
    if bm_weights is None:
        bm_weights = np.ones(n) / n
    bm_weights = np.asarray(bm_weights, dtype=float)
    if prev_weights is None:
        prev_weights = bm_weights.copy()
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)

    fallback = np.asarray(
        fallback_weights if fallback_weights is not None else bm_weights,
        dtype=float,
    ).copy()

    diag = _init_diagnostics(diagnostics, mode="projection_mvo")
    w = cp.Variable(n)
    _, _, constraints = _build_mvo_constraints(
        w=w,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        sector_map=sector_map,
        bm_weights=bm_weights,
        max_te_annual=max_te_annual,
        sector_deviation=sector_deviation,
        config=config,
    )
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - candidate)), constraints)
    if not _solve_problem(prob, diag) or prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = diag.get("fallback_reason") or prob.status or "projection_failed"
        return _safe_step_toward_benchmark(prev_weights, fallback, config, diag)

    projected = np.asarray(w.value, dtype=float).flatten()
    if not np.all(np.isfinite(projected)):
        if diag is not None:
            diag["used_fallback"] = True
            diag["fallback_reason"] = "non_finite_projection"
        return _safe_step_toward_benchmark(prev_weights, fallback, config, diag)

    return projected


def _safe_step_toward_benchmark(
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    config: PipelineConfig,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Largest feasible step from ``prev_weights`` toward ``target_weights``
    under the ESSENTIAL hard caps only (MVO-1 fix).

    The old optimiser/projection fallback returned ``bm_weights`` verbatim on
    infeasibility, which can BREACH ``max_single_turnover`` (norm1(bm - prev)
    may exceed the cap) and ``max_weight``, and reverts the whole book to
    passive (zero active alpha). Instead we solve::

        minimise ||w - target||^2
        s.t.  sum(w)=1,  w>=0,  w<=max_weight,
              norm1(w - prev) <= max_single_turnover

    ``prev_weights`` itself satisfies this set (norm1(0)=0), so it is always
    feasible and the recorded turnover / max_weight caps can never be violated
    by the fallback. If even this degenerate solve fails (e.g. ``prev`` already
    breaches max_weight after drift), hold ``prev``.
    """
    prev = np.asarray(prev_weights, dtype=float).flatten()
    target = np.asarray(target_weights, dtype=float).flatten()
    n = len(prev)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w - target)),
        [
            cp.sum(w) == 1,
            w >= 0,
            w <= config.max_weight,
            cp.norm1(w - prev) <= config.max_single_turnover,
        ],
    )
    if (_solve_problem(prob)
            and prob.status in ("optimal", "optimal_inaccurate")
            and w.value is not None):
        out = np.asarray(w.value, dtype=float).flatten()
        if np.all(np.isfinite(out)):
            out = np.clip(out, 0.0, None)
            s = out.sum()
            if s > 0:
                out = out / s
            # Enforce the turnover cap exactly. ECOS satisfies the constraint
            # only to solver tolerance (~1e-5) and the clip+renorm above can
            # nudge it a hair over; scale the step back toward prev so
            # norm1(out - prev) <= cap holds to float precision. A convex move
            # toward prev preserves sum=1, w>=0 and w<=max_weight.
            cap = config.max_single_turnover
            step = out - prev
            to = float(np.abs(step).sum())
            if to > cap and to > 0:
                out = prev + (cap / to) * step
            if diagnostics is not None:
                diagnostics["fallback_book"] = "safe_step"
            return out
    if diagnostics is not None:
        diagnostics["fallback_book"] = "hold_prev"
    return prev.copy()


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
    config: PipelineConfig = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Optimise the target portfolio under the configured hard constraints."""
    config = config or DEFAULT_CONFIG
    if risk_aversion == RISK_AVERSION:
        risk_aversion = config.risk_aversion
    if turnover_penalty == TURNOVER_PENALTY:
        turnover_penalty = config.turnover_penalty
    if max_te_annual == MAX_TE_ANNUAL:
        max_te_annual = config.max_te_annual
    if sector_deviation == SECTOR_DEVIATION:
        sector_deviation = config.sector_deviation

    n = len(expected_returns)
    if bm_weights is None:
        bm_weights = np.ones(n) / n
    bm_weights = np.asarray(bm_weights, dtype=float)
    if prev_weights is None:
        prev_weights = bm_weights.copy()
    else:
        prev_weights = np.asarray(prev_weights, dtype=float)

    diag = _init_diagnostics(
        diagnostics,
        mode="score_based" if config.use_score_based else "mvo",
    )

    if config.use_score_based:
        if diag is not None:
            diag["status"] = "score_based"
        return score_based_weights(expected_returns, max_weight=config.max_weight)

    w = cp.Variable(n)
    mu = expected_returns.values
    ret = mu @ w
    risk, turnover, constraints = _build_mvo_constraints(
        w=w,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        sector_map=sector_map,
        bm_weights=bm_weights,
        max_te_annual=max_te_annual,
        sector_deviation=sector_deviation,
        config=config,
    )
    objective = cp.Maximize(ret - risk_aversion * risk - turnover_penalty * turnover)

    prob = cp.Problem(objective, constraints)
    if not _solve_problem(prob, diag):
        if diag is not None:
            diag["used_fallback"] = True
        return _safe_step_toward_benchmark(prev_weights, bm_weights, config, diag)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        opt_w = np.asarray(w.value, dtype=float).flatten()
        if not np.all(np.isfinite(opt_w)):
            if diag is not None:
                diag["used_fallback"] = True
                diag["fallback_reason"] = "non_finite_solution"
            return _safe_step_toward_benchmark(prev_weights, bm_weights, config, diag)
        return opt_w

    if diag is not None:
        diag["used_fallback"] = True
        diag["fallback_reason"] = prob.status or "non_optimal_status"
    return _safe_step_toward_benchmark(prev_weights, bm_weights, config, diag)
