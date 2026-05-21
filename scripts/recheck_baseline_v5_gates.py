"""Evaluate baseline_v5 against the 5 primary promotion gates (2026-05-19 v2).

Gates (see docs/BASELINE.md):
  1. IR (cutoff-trimmed) >= research baseline IR (trimmed)
  2. rolling_ir_pos_frac >= research baseline - 0.05
  3. rolling_ir_min     >= research baseline rolling_ir_min - 0.20
  4. spa_pvalue         <= 0.10
  5. annual_turnover_2way <= max(research, legacy_deploy) + 0.05

Reads metrics from:
  outputs/baseline_v5/metrics.json              (candidate)
  outputs/iter15_FINAL_postfix/metrics.json     (research baseline)
  outputs/baseline_v4/metrics.json              (legacy deploy)

Cutoff-trimmed IR is computed from backtest_result.pkl (trim portfolio_returns
to <= 2024-12-31).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUTOFF = pd.Timestamp("2024-12-31")


def _trimmed_ir(label: str) -> float:
    pkl = ROOT / "outputs" / label / "backtest_result.pkl"
    if not pkl.exists():
        return float("nan")
    with pkl.open("rb") as fh:
        r = pickle.load(fh)
    port = r.portfolio_returns.dropna()
    bench = r.benchmark_returns.dropna()
    common = port.index.intersection(bench.index)
    port = port.loc[common].loc[common <= CUTOFF]
    bench = bench.loc[common].loc[common <= CUTOFF]
    if len(port) < 60:
        return float("nan")
    active = port - bench
    if active.std(ddof=0) == 0:
        return float("nan")
    return float(active.mean() / active.std(ddof=0) * np.sqrt(252))


def _metric(label: str, key: str) -> float:
    p = ROOT / "outputs" / label / "metrics.json"
    if not p.exists():
        return float("nan")
    d = json.loads(p.read_text(encoding="utf-8"))["metrics"]
    v = d.get(key)
    return float(v) if v is not None else float("nan")


def main() -> int:
    cand = "baseline_v5"
    base = "iter15_FINAL_postfix"
    legacy = "baseline_v4"

    ir_c = _trimmed_ir(cand)
    ir_b = _trimmed_ir(base)
    rf_c = _metric(cand, "rolling_ir_pos_frac")
    rf_b = _metric(base, "rolling_ir_pos_frac")
    rm_c = _metric(cand, "rolling_ir_min")
    rm_b = _metric(base, "rolling_ir_min")
    spa_c = _metric(cand, "spa_pvalue")
    to_c = _metric(cand, "avg_annual_turnover")
    to_b = _metric(base, "avg_annual_turnover")
    to_l = _metric(legacy, "avg_annual_turnover")

    # Gate computations
    g1_thr = ir_b
    g1_pass = ir_c >= g1_thr
    g2_thr = rf_b - 0.05
    g2_pass = rf_c >= g2_thr
    g3_thr = rm_b - 0.20
    g3_pass = rm_c >= g3_thr
    g4_thr = 0.10
    g4_pass = spa_c <= g4_thr
    g5_thr = max(to_b, to_l) + 0.05
    g5_pass = to_c <= g5_thr

    def fmt(b):
        return "PASS" if b else "FAIL"

    print("=" * 76)
    print(f"  baseline_v5 vs primary gates (2026-05-19 v2)")
    print(f"  Research baseline: {base}    Legacy deploy: {legacy}")
    print("=" * 76)
    print(f"  1. IR (trimmed)            : {ir_c:7.4f} vs >= {g1_thr:7.4f}  -> {fmt(g1_pass)}")
    print(f"  2. rolling_ir_pos_frac     : {rf_c:7.4f} vs >= {g2_thr:7.4f}  -> {fmt(g2_pass)}")
    print(f"  3. rolling_ir_min          : {rm_c:7.4f} vs >= {g3_thr:7.4f}  -> {fmt(g3_pass)}")
    print(f"  4. spa_pvalue              : {spa_c:7.4f} vs <= {g4_thr:7.4f}  -> {fmt(g4_pass)}")
    print(f"  5. annual_turnover_2way    : {to_c:7.4f} vs <= {g5_thr:7.4f}  -> {fmt(g5_pass)}")
    print(f"     (max(research={to_b:.4f}, legacy={to_l:.4f}) + 0.05)")
    print("-" * 76)
    all_pass = all([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    verdict = "PROMOTION-ELIGIBLE" if all_pass else "NOT-ELIGIBLE"
    print(f"  OVERALL: {verdict}")
    print("=" * 76)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
