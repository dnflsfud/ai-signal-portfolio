"""MVO-1 + DYNEXEC-1 audit-fix regression tests.

MVO-1: an infeasible MVO must NOT silently book the raw benchmark (which can
breach max_single_turnover / max_weight). The constraint-preserving fallback
takes the largest feasible step toward the target within the essential caps.

DYNEXEC-1: compute_signal_confidence's spread term must vary with z-score
sharpness instead of saturating at 1.0 for every realistic input.
"""
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.portfolio_optimizer import _safe_step_toward_benchmark, optimize_portfolio
from src.backtest import compute_signal_confidence


def _feasible_prev():
    # All <= max_weight (0.15), sums to 1, but L1-far from the uniform bm.
    return np.array([0.15, 0.15, 0.15, 0.15, 0.15,
                     0.13, 0.04, 0.04, 0.02, 0.02])


def test_safe_step_respects_turnover_and_weight_caps():
    prev = _feasible_prev()
    bm = np.ones(10) / 10.0
    cap = DEFAULT_CONFIG.max_single_turnover
    mw = DEFAULT_CONFIG.max_weight
    assert np.abs(bm - prev).sum() > cap  # scenario is genuinely infeasible-in-one-step

    out = _safe_step_toward_benchmark(prev, bm, DEFAULT_CONFIG)

    assert abs(out.sum() - 1.0) < 1e-6
    assert (out >= -1e-9).all()
    assert (out <= mw + 1e-6).all()
    # turnover cap honoured (the bug: raw bm would have breached it)
    assert np.abs(out - prev).sum() <= cap + 1e-6
    # did not overshoot all the way to the benchmark
    assert np.abs(out - bm).sum() > 1e-6


def test_optimize_portfolio_fallback_is_constraint_preserving():
    n = 10
    mu = pd.Series(np.linspace(-1.0, 1.0, n), index=[f"T{i}" for i in range(n)])
    cov = np.full((n, n), np.nan)  # forces solver failure -> fallback path
    prev = _feasible_prev()
    bm = np.ones(n) / n

    out = optimize_portfolio(
        expected_returns=mu, cov_matrix=cov, prev_weights=prev,
        bm_weights=bm, config=DEFAULT_CONFIG,
    )

    # Old behaviour returned bm verbatim (turnover ~0.56 >> 0.15 cap).
    assert np.abs(out - prev).sum() <= DEFAULT_CONFIG.max_single_turnover + 1e-6
    assert abs(out.sum() - 1.0) < 1e-6
    assert np.abs(out - bm).sum() > 1e-6  # not the raw benchmark


def test_safe_step_holds_when_prev_breaches_max_weight():
    # prev already over max_weight by more than the turnover budget allows to fix
    prev = np.array([0.5, 0.5] + [0.0] * 8)
    bm = np.ones(10) / 10.0
    out = _safe_step_toward_benchmark(prev, bm, DEFAULT_CONFIG)
    # degenerate -> holds prev (never returns a wilder book)
    assert abs(out.sum() - 1.0) < 1e-6
    assert np.abs(out - prev).sum() <= DEFAULT_CONFIG.max_single_turnover + 1e-6


def test_signal_confidence_spread_not_saturated():
    # Flat cross-section (small z spread) vs sharp cross-section (wide z spread)
    # must yield different confidence at a FIXED trailing IC. Pre-fix the
    # spread term clipped to 1.0 for both, making them identical.
    idx = [f"T{i}" for i in range(40)]
    flat = pd.Series(np.linspace(-0.4, 0.4, 40), index=idx)   # spread ~0.8
    sharp = pd.Series(np.linspace(-2.5, 2.5, 40), index=idx)  # spread ~5

    ic = 0.03
    c_flat = compute_signal_confidence(flat, flat, ic)
    c_sharp = compute_signal_confidence(sharp, sharp, ic)

    assert c_sharp > c_flat + 1e-3, (c_flat, c_sharp)
    # sharp end should be (near) saturated, flat end clearly below it
    assert c_flat < 1.0
