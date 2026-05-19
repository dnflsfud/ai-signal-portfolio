"""Tests for compute_config_fingerprint + save/load_checkpoint fingerprint guard.

selection-bias-discipline Task C step 0 (2026-05-19).
"""
from __future__ import annotations

import dataclasses
import pickle
import pytest

from src.config import PipelineConfig
from src.backtest import (
    FINGERPRINT_KEYS,
    compute_config_fingerprint,
    save_checkpoint,
    load_checkpoint,
)


class TestFingerprintFunction:
    def test_stable_under_no_change(self):
        c = PipelineConfig()
        assert compute_config_fingerprint(c) == compute_config_fingerprint(c)

    def test_changes_on_fingerprint_key(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, train_window=999)
        assert compute_config_fingerprint(c2) != fp

    def test_changes_on_lgbm_params(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        new_params = dict(c.lgbm_params)
        new_params["num_leaves"] = 7
        c2 = dataclasses.replace(c, lgbm_params=new_params)
        assert compute_config_fingerprint(c2) != fp

    def test_stable_under_safe_key_change(self):
        """rebalance_freq and MVO knobs are in SAFE_FOR_CACHE_REUSE, NOT in FINGERPRINT_KEYS."""
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, rebalance_freq=42, turnover_penalty=0.99,
                                  max_te_annual=0.99)
        assert compute_config_fingerprint(c2) == fp

    def test_stable_under_post_pred_overlay_change(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, value_trap_gate_enabled=False,
                                  growth_tilt_enabled=False,
                                  pead_boost_enabled=False)
        assert compute_config_fingerprint(c2) == fp

    def test_changes_on_embargo(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, embargo_days=0)
        assert compute_config_fingerprint(c2) != fp

    def test_changes_on_cutoff(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, train_cutoff_date="2023-12-31")
        assert compute_config_fingerprint(c2) != fp

    def test_changes_on_feature_mode(self):
        c = PipelineConfig()
        fp = compute_config_fingerprint(c)
        c2 = dataclasses.replace(c, feature_mode="lean")
        assert compute_config_fingerprint(c2) != fp

    def test_fingerprint_is_short_hex(self):
        fp = compute_config_fingerprint(PipelineConfig())
        assert isinstance(fp, str) and len(fp) == 16
        int(fp, 16)  # raises if not hex


class TestSaveLoadRoundtrip:
    def test_load_accepts_matching_fingerprint(self, tmp_path):
        c = PipelineConfig()
        save_checkpoint("phase1", {"hello": 123}, output_dir=str(tmp_path), config=c)
        out = load_checkpoint("phase1", output_dir=str(tmp_path), config=c)
        assert out is not None
        assert out["hello"] == 123
        assert "_fingerprint" in out

    def test_load_rejects_mismatched_fingerprint(self, tmp_path, caplog):
        c = PipelineConfig()
        save_checkpoint("phase1", {"hello": 123}, output_dir=str(tmp_path), config=c)
        c2 = dataclasses.replace(c, train_window=999)  # FINGERPRINT key change
        out = load_checkpoint("phase1", output_dir=str(tmp_path), config=c2)
        assert out is None

    def test_load_accepts_legacy_payload_without_fingerprint(self, tmp_path, caplog):
        """Backwards compat: payload saved without config arg should still load."""
        save_checkpoint("phase1", {"hello": 123}, output_dir=str(tmp_path))  # no config
        c = PipelineConfig()
        out = load_checkpoint("phase1", output_dir=str(tmp_path), config=c)
        assert out is not None  # accepted with warning
        assert "_fingerprint" not in out

    def test_load_without_config_skips_fingerprint_check(self, tmp_path):
        """Backwards compat: load_checkpoint(phase) without config still works."""
        c = PipelineConfig()
        save_checkpoint("phase1", {"hello": 123}, output_dir=str(tmp_path), config=c)
        c2 = dataclasses.replace(c, train_window=999)  # would mismatch, but no config arg
        out = load_checkpoint("phase1", output_dir=str(tmp_path))  # no config
        assert out is not None
        assert out["hello"] == 123
