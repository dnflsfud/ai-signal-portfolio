"""Tests for walk-forward embargo logic (data-leakage-fix Task A step 0).

Tests the `_compute_window_bounds` helper that carves an embargo gap between
train/val and val/predict windows so the 20d forward target's label window
cannot peek into the next window. See López de Prado (2018) Ch. 7.
"""
from __future__ import annotations

import warnings

import pytest

from src.config import PipelineConfig
from src.model_trainer import _compute_window_bounds


class TestEmbargoBounds:
    def test_embargo_drops_last_train_samples_within_horizon(self):
        """train_end == t_idx - val_window - embargo (not t_idx - val_window)."""
        bounds = _compute_window_bounds(
            t_idx=1400, train_window=1260, val_window=126, embargo=20,
        )
        assert bounds is not None
        train_start, train_end, val_start, val_end = bounds
        assert train_end == 1400 - 126 - 20  # = 1254
        assert val_start == 1400 - 126        # = 1274
        # Critical: last train label's forward window (length=embargo=20)
        # ends exactly at val_start (no peek).
        assert train_end + 20 == val_start
        # val_end leaves another embargo before the predict bar t_idx.
        assert val_end == 1400 - 20           # = 1380
        assert val_end + 20 == 1400

    def test_embargo_zero_matches_legacy(self):
        """embargo_days=0 reproduces the legacy (leaky) layout."""
        bounds = _compute_window_bounds(
            t_idx=1400, train_window=1260, val_window=126, embargo=0,
        )
        assert bounds is not None
        train_start, train_end, val_start, val_end = bounds
        # Legacy: train_end == val_start == t_idx - val_window, val_end == t_idx.
        assert train_end == val_start
        assert val_end == 1400

    def test_embargo_skip_when_window_too_narrow(self):
        """Returns None when embargo + val_window >= effective train window."""
        # t_idx=200, train_window=1260 (clipped to 200 by max), val_window=126,
        # embargo=200 → train_end = 200 - 126 - 200 = -126 < train_start=0.
        bounds = _compute_window_bounds(
            t_idx=200, train_window=1260, val_window=126, embargo=200,
        )
        assert bounds is None

    def test_embargo_at_data_start_clips_train_start(self):
        """train_start clipped to 0 near the beginning of the date range."""
        bounds = _compute_window_bounds(
            t_idx=200, train_window=1260, val_window=126, embargo=20,
        )
        assert bounds is not None
        train_start, train_end, val_start, val_end = bounds
        assert train_start == 0
        assert train_end == 200 - 126 - 20    # = 54
        assert val_start == 200 - 126         # = 74
        assert val_end == 200 - 20            # = 180

    def test_embargo_negative_rejected(self):
        """Negative embargo is a programming error and must raise."""
        with pytest.raises(ValueError):
            _compute_window_bounds(1400, 1260, 126, -1)


class TestConfigEmbargoField:
    def test_default_embargo_equals_forward_horizon(self):
        """Out-of-the-box config carries the recommended embargo."""
        c = PipelineConfig()
        assert c.embargo_days == c.forward_horizon == 20

    def test_negative_embargo_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig(embargo_days=-1)

    def test_embargo_smaller_than_horizon_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PipelineConfig(embargo_days=0)
        assert any("label leak" in str(w.message).lower() for w in caught)
