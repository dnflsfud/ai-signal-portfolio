"""Unit tests for fx_surcharge_per_ticker FX cost layer (fx-cost-modeling).

Validates:
- DEFAULT_CONFIG registration of KRX names
- __post_init__ validation (negative raises, >100bp warns)
- Vector math correctness on synthetic walk-step
- compute_metrics annual_tc uses accumulated TC (not scalar fallback)
- Backward-compat: empty fx_surcharge_per_ticker matches legacy scalar path
"""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult
from src.config import DEFAULT_CONFIG, PipelineConfig


# ---------------------------------------------------------------------------
# Config layer (Step 0)
# ---------------------------------------------------------------------------
def test_default_config_has_krw_tickers():
    """DEFAULT_CONFIG should register 000660 + 005930 at 3bp each."""
    fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
    assert fx == {"000660": 0.0003, "005930": 0.0003}, fx


def test_negative_surcharge_raises():
    """__post_init__ rejects negative values."""
    with pytest.raises(ValueError, match="must be >= 0"):
        replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={"X": -0.001})


def test_excessive_surcharge_warns():
    """> 100bp emits UserWarning but does not raise (operator override allowed)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={"X": 0.02})
        assert any("unusually high" in str(w.message) for w in caught), \
            [str(w.message) for w in caught]


def test_empty_dict_allowed():
    """Empty dict is a valid override (disables FX layer entirely)."""
    cfg = replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={})
    assert cfg.fx_surcharge_per_ticker == {}


# ---------------------------------------------------------------------------
# Vector math (Step 1)
# ---------------------------------------------------------------------------
def test_tc_cost_vector_math_krw():
    """Per-ticker TC vector: USD tickers get only one_way_tc, KRW gets +3bp."""
    fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
    one_way_tc = DEFAULT_CONFIG.one_way_tc
    tickers = ["AAPL", "000660", "MSFT", "005930"]
    delta_w = np.array([0.05, 0.05, 0.05, 0.05])
    fx_vec = np.array([fx.get(t, 0.0) for t in tickers])
    tc_per_ticker = one_way_tc + fx_vec
    tc_cost = float(np.sum(delta_w * tc_per_ticker))
    # AAPL:   0.05 * 0.0010  = 5.0e-5
    # 000660: 0.05 * 0.0013  = 6.5e-5
    # MSFT:   0.05 * 0.0010  = 5.0e-5
    # 005930: 0.05 * 0.0013  = 6.5e-5
    # total                  = 2.3e-4
    assert abs(tc_cost - 2.3e-4) < 1e-9, tc_cost


def test_tc_cost_usd_only_matches_scalar():
    """When portfolio has no KRX names, per-ticker == scalar tc * turnover."""
    fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
    one_way_tc = DEFAULT_CONFIG.one_way_tc
    tickers = ["AAPL", "MSFT", "GOOGL"]
    delta_w = np.array([0.03, 0.05, 0.02])
    fx_vec = np.array([fx.get(t, 0.0) for t in tickers])
    tc_per_ticker = one_way_tc + fx_vec
    tc_cost_vec = float(np.sum(delta_w * tc_per_ticker))
    tc_cost_scalar = float(delta_w.sum() * one_way_tc)
    assert abs(tc_cost_vec - tc_cost_scalar) < 1e-12, (tc_cost_vec, tc_cost_scalar)


# ---------------------------------------------------------------------------
# compute_metrics (Step 1)
# ---------------------------------------------------------------------------
def test_compute_metrics_uses_accumulated_tc():
    """annual_tc reads from self.tc_costs when populated."""
    br = BacktestResult()
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    br.portfolio_returns = pd.Series(0.0005, index=idx, name="portfolio")
    br.benchmark_returns = pd.Series(0.0004, index=idx, name="benchmark")
    # 12 rebal/yr, each 1bp = 12bp/yr
    rebal_idx = idx[::21]
    br.turnover = pd.Series(0.01, index=rebal_idx, name="turnover")
    br.tc_costs = pd.Series(1e-4, index=rebal_idx, name="tc_cost")
    m = br.compute_metrics()
    expected = len(rebal_idx) * 1e-4  # ~12 * 1e-4 = 12bp
    assert abs(m["annual_tc"] - expected) < 1e-7, m["annual_tc"]


def test_compute_metrics_legacy_fallback():
    """Without tc_costs (empty Series), falls back to turnover * ONE_WAY_TC."""
    from src.backtest import ONE_WAY_TC
    br = BacktestResult()
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    br.portfolio_returns = pd.Series(0.0005, index=idx, name="portfolio")
    br.benchmark_returns = pd.Series(0.0004, index=idx, name="benchmark")
    rebal_idx = idx[::21]
    br.turnover = pd.Series(0.10, index=rebal_idx, name="turnover")
    # tc_costs intentionally left as empty default Series
    assert len(br.tc_costs) == 0, "tc_costs should default empty"
    m = br.compute_metrics()
    # avg_annual_turnover ~ 0.10 * 12 = 1.20 (two-way)
    # annual_tc = 1.20 * 0.001 = 12bp
    expected = m["avg_annual_turnover"] * ONE_WAY_TC
    assert abs(m["annual_tc"] - expected) < 1e-9, (m["annual_tc"], expected)
