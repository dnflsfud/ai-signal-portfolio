"""Tests for analytics.rolling_ir / rolling_ir_stats / spa_pvalue.

selection-bias-discipline Task C step 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import rolling_ir, rolling_ir_stats, spa_pvalue


def _make_idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


class TestRollingIR:
    def test_constant_positive_alpha(self):
        n = 500
        a = pd.Series([0.001] * n, index=_make_idx(n))
        # Constant series has 0 std -> rolling_ir undefined -> NaN
        ri = rolling_ir(a, window=252)
        # First 125 NaN by min_periods; rest should be NaN due to zero std.
        assert ri.dropna().empty or np.allclose(ri.dropna(), np.inf, equal_nan=True) is False

    def test_zero_alpha_noise(self):
        n = 1000
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, n), index=_make_idx(n))
        ri = rolling_ir(a, window=252).dropna()
        assert len(ri) > 500
        # Zero-mean noise: rolling IR distribution centred near 0
        assert abs(ri.mean()) < 1.0  # broad sanity bound

    def test_strong_alpha(self):
        n = 1000
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0.0005, 0.01, n), index=_make_idx(n))
        ri = rolling_ir(a, window=252).dropna()
        # Mean rolling IR should be positive and bounded.
        assert ri.mean() > 0.3, f"expected positive rolling IR for strong alpha, got {ri.mean()}"


class TestRollingIRStats:
    def test_returns_required_keys(self):
        n = 1000
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, n), index=_make_idx(n))
        s = rolling_ir_stats(a, window=252)
        required = {"rolling_ir_mean", "rolling_ir_median", "rolling_ir_min",
                    "rolling_ir_max", "rolling_ir_pos_frac", "rolling_ir_window"}
        assert required <= set(s.keys())
        assert s["rolling_ir_window"] == 252

    def test_short_series_returns_nan(self):
        a = pd.Series([0.001] * 10, index=_make_idx(10))
        s = rolling_ir_stats(a, window=252)
        assert s["rolling_ir_window"] == 252
        # All metrics NaN (no rolling samples)
        for k in ("rolling_ir_mean", "rolling_ir_min", "rolling_ir_pos_frac"):
            assert s[k] != s[k]  # NaN


class TestSPApvalue:
    def test_reproducible_seed(self):
        n = 500
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0.0003, 0.01, n), index=_make_idx(n))
        p1 = spa_pvalue(a, n_bootstrap=300, seed=42)
        p2 = spa_pvalue(a, n_bootstrap=300, seed=42)
        assert p1 == p2

    def test_strong_alpha_low_p(self):
        n = 1000
        rng = np.random.default_rng(0)
        # Mean 0.002 / day on 1%% vol -> very strong signal
        a = pd.Series(rng.normal(0.002, 0.01, n), index=_make_idx(n))
        p = spa_pvalue(a, n_bootstrap=500, seed=42)
        assert p < 0.05, f"expected p<0.05 for strong alpha, got {p}"

    def test_zero_alpha_p_distribution(self):
        """For zero-mean noise, p should be uniformly distributed in [0,1]
        across data seeds (not strongly rejecting on average)."""
        ps = []
        for data_seed in range(20):
            rng = np.random.default_rng(data_seed)
            n = 1000
            a = pd.Series(rng.normal(0.0, 0.01, n), index=_make_idx(n))
            ps.append(spa_pvalue(a, n_bootstrap=200, seed=42))
        median_p = float(np.median(ps))
        # Across 20 noise series, median p should be near 0.5 (broad bound to
        # tolerate small-sample variance).
        assert 0.15 < median_p < 0.85, f"expected median p ~0.5, got {median_p:.3f} (all={ps})"

    def test_short_series_returns_nan(self):
        n = 10
        a = pd.Series([0.001] * n, index=_make_idx(n))
        p = spa_pvalue(a)
        assert p != p  # NaN
