"""Iter21 FULL rebuild: iter15 config + REDESIGN W + REDESIGN X.

Rebuilds Phase 1 (data), Phase 2 (panel with feature_mode='core' = 56
features), Phase 3 (targets), Phase 4 (walk-forward LightGBM), then
runs a 3-way Phase 5-6 comparison:

  iter15_baseline  — W off, X off (iter15 production baseline)
  iter15_W_only    — decile funding on, regime gating off
  iter15_W_and_X   — F winner: decile + regime_off=0.30 + shrink=0.40

All three variants SHARE the same trained models/predictions — only
the portfolio construction layer changes — so differences are purely
from REDESIGN W/X, not from model re-training noise.

Output: outputs/iter21_full/{metrics,summary,FINAL_REPORT}
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from src.data_loader import UniverseData
from src.feature_engine import build_all_features
from src.target_engine import build_targets
from src.model_trainer import walk_forward_train
from src.backtest import run_backtest
from src.config import DEFAULT_CONFIG, PipelineConfig
from src.harness import (
    run_variant as _harness_run_variant,
    sub_ir as _sub_ir,  # re-export for callers that reference it
)


DATA_PATH = "./data/ai_signal_data.xlsx"
OUT_ROOT = Path("./outputs/iter21_full")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUT_ROOT / "prepared_iter21.pkl"


def run_variant(label: str, overrides: dict, data, panel, feature_names,
                feature_groups, targets, models, predictions, raw_predictions):
    """Thin wrapper around src.harness.run_variant.

    iter21_full's historical metrics schema stores P1_ir/P2_ir/P3_ir at
    the top level (not nested under sub_periods), and does not emit the
    annual_turnover_2way alias — `sub_period_mode="flat"` matches that.
    """
    return _harness_run_variant(
        label, overrides,
        data=data, panel=panel, feature_names=feature_names,
        feature_groups=feature_groups, targets=targets, models=models,
        predictions=predictions, raw_predictions=raw_predictions,
        out_dir=OUT_ROOT / label,
        sub_period_mode="flat",
        attach_turnover_alias=False,
    )


def main():
    t0 = time.time()

    # ---------- Stage 1-4: prepare or load cached pipeline ----------
    if CACHE_PATH.exists():
        print(f"[iter21] loading cached pipeline -> {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        data = cache["data"]
        panel = cache["panel"]
        feature_names = cache["feature_names"]
        feature_groups = cache["feature_groups"]
        targets = cache["targets"]
        models = cache["models"]
        predictions = cache["predictions"]
        raw_predictions = cache.get("raw_predictions")
        print(f"  features={len(feature_names)}, panel={panel.shape}, "
              f"models={len(models)}, predictions={predictions.shape}")
    else:
        print("[iter21] Phase 1: loading UniverseData …")
        data = UniverseData(DATA_PATH)
        print(f"  tickers={len(data.tickers)}, dates={len(data.dates)}")
        del data.raw
        gc.collect()

        print(f"[iter21] Phase 2: build_all_features "
              f"(feature_mode={DEFAULT_CONFIG.feature_mode}) …")
        panel, feature_names, feature_groups = build_all_features(data)
        print(f"  features={len(feature_names)}, panel={panel.shape}")

        print("[iter21] Phase 3: build_targets …")
        targets = build_targets(data)
        print(f"  targets shape={targets.shape}, "
              f"valid={targets.notna().sum().sum()}")

        print("[iter21] Phase 4: walk_forward_train …")
        ret = walk_forward_train(panel, targets, feature_names, data.dates)
        # walk_forward_train returns (models, predictions, raw_predictions, ewma)
        models, predictions, raw_predictions = ret[0], ret[1], ret[2]
        print(f"  models={len(models)}, predictions={predictions.shape}")

        print(f"[iter21] caching pipeline -> {CACHE_PATH}")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "data": data,
                "panel": panel,
                "feature_names": feature_names,
                "feature_groups": feature_groups,
                "targets": targets,
                "models": models,
                "predictions": predictions,
                "raw_predictions": raw_predictions,
            }, f)

    print(f"[iter21] prep done in {time.time()-t0:.0f}s")

    kwargs = dict(data=data, panel=panel, feature_names=feature_names,
                  feature_groups=feature_groups, targets=targets, models=models,
                  predictions=predictions, raw_predictions=raw_predictions)

    # ---------- Stage 5: 3-way variant comparison ----------
    variants = [
        ("iter15_baseline",  # W off, X off — reproduces iter15 production
         {"decile_funding_enabled": False, "regime_gate_enabled": False}),
        ("iter15_W_only",    # Decile funding only (no regime gate)
         {"decile_funding_enabled": True,  "regime_gate_enabled": False}),
        ("iter15_W_and_X",   # F winner: decile + regime (off=0.30, shrink=0.40)
         {"decile_funding_enabled": True,  "regime_gate_enabled": True,
          "decile_funding_regime_off": 0.30,
          "regime_active_shrink_max": 0.40,
          "regime_no_trade_scale_max": 2.0,
          "regime_eta_shrink_max": 0.50}),
        # --- Recovery experiments: can ANY W+X variant beat baseline? ---
        ("G_decile_strict",  # ultra-selective decile (only very negative)
         {"decile_funding_enabled": True,  "regime_gate_enabled": False,
          "decile_funding_score_max": -0.30,
          "decile_funding_per_group_k": 1,
          "decile_funding_wide_uw_cap": 0.08}),
        ("H_regime_very_mild",  # regime gate with tiny effect
         {"decile_funding_enabled": False, "regime_gate_enabled": True,
          "regime_active_shrink_max": 0.10,
          "regime_no_trade_scale_max": 1.2,
          "regime_eta_shrink_max": 0.15,
          "regime_stress_floor": 0.50}),
        ("I_exec_only",  # regime gate affects execution only (eta/NTB), not bounds
         {"decile_funding_enabled": False, "regime_gate_enabled": True,
          "regime_active_shrink_max": 0.0,
          "regime_no_trade_scale_max": 2.0,
          "regime_eta_shrink_max": 0.50,
          "regime_stress_floor": 0.30}),
    ]

    rows = []
    for label, ov in variants:
        r, m = run_variant(label, ov, **kwargs)
        rows.append({
            "label": label,
            "IR": round(float(m.get("information_ratio") or 0.0), 3),
            "active": round(float(m.get("active_return") or 0.0) * 100, 2),
            "TE": round(float(m.get("tracking_error") or 0.0) * 100, 2),
            "sharpe": round(float(m.get("sharpe_ratio") or 0.0), 3),
            "ann_ret": round(float(m.get("annual_return") or 0.0) * 100, 2),
            "turn": round(float(m.get("avg_annual_turnover") or 0.0) * 100, 1),
            "P1_IR": round(float(m.get("P1_ir") or 0.0), 3),
            "P2_IR": round(float(m.get("P2_ir") or 0.0), 3),
            "P3_IR": round(float(m.get("P3_ir") or 0.0), 3),
            "maxDD": round(float(m.get("max_drawdown") or 0.0) * 100, 2),
        })

    # ---------- Stage 6: print + save summary ----------
    hdr = ["label", "IR", "active%", "TE%", "sharpe", "ann_ret%",
           "turn%", "P1_IR", "P2_IR", "P3_IR", "maxDD%"]
    print("\n================ ITER21 FULL COMPARISON ================")
    print("{:<22} {:>6} {:>8} {:>6} {:>7} {:>9} {:>7} {:>7} {:>7} {:>7} {:>8}".format(*hdr))
    for r in rows:
        print("{label:<22} {IR:>6} {active:>8} {TE:>6} {sharpe:>7} "
              "{ann_ret:>9} {turn:>7} {P1_IR:>7} {P2_IR:>7} {P3_IR:>7} {maxDD:>8}".format(**r))

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps({"variants": rows,
                    "elapsed_sec": round(time.time() - t0, 1)},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[iter21] summary saved -> {summary_path}")
    print(f"[iter21] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
