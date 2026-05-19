"""AC check for Task A step 1 (oos-holdout-default).

Verifies:
  1. PipelineConfig defaults enforce_oos_holdout=True, cutoff=2024-12-31
  2. compose_config research -> holdout True
  3. compose_config deploy -> holdout False
  4. compose_config production -> DeprecationWarning + research fallback
  5. compose_config oos_verify -> peek logged to experiment_inventory.json
"""
from __future__ import annotations

import json
import pathlib
import warnings

from src.config import PipelineConfig
from run_variant import compose_config


def test_defaults_holdout_on_with_cutoff():
    c = PipelineConfig()
    assert c.enforce_oos_holdout is True
    assert c.train_cutoff_date == "2024-12-31"


def test_research_enforces_holdout():
    cfg = compose_config({"label": "t", "overrides": {}, "tuning_mode": "research"})
    assert cfg.enforce_oos_holdout is True


def test_deploy_disables_holdout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = compose_config({"label": "t_deploy", "overrides": {}, "tuning_mode": "deploy"})
    assert cfg.enforce_oos_holdout is False
    log = pathlib.Path("outputs/deploy_log.txt")
    assert log.exists()
    assert "t_deploy" in log.read_text(encoding="utf-8")


def test_production_deprecated_and_research_semantics():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = compose_config({"label": "t_dep", "overrides": {}, "tuning_mode": "production"})
    assert cfg.enforce_oos_holdout is True
    dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("DEPRECATED" in m for m in dep_msgs), dep_msgs


def test_oos_verify_records_peek(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {}
    compose_config({"label": "test_peek", "overrides": {}, "tuning_mode": "oos_verify"})
    inv_path = pathlib.Path("experiment_inventory.json")
    assert inv_path.exists()
    after = json.loads(inv_path.read_text(encoding="utf-8"))
    assert after.get("n_oos_peeks") == 1
    assert after["oos_peeks"][0]["label"] == "test_peek"


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown tuning_mode"):
        compose_config({"label": "t", "overrides": {}, "tuning_mode": "bogus_mode"})
