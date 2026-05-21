"""PCA target scale forensics - diagnostic on existing backtest pkls.

Compares prediction magnitude (cross-sectional std) between the two PCA scale
modes (raw vs daily_eq) using the saved pkls. The pkl saves two prediction
fields:
  - raw_predictions : output of walk_forward_train BEFORE post-process layers
  - predictions     : AFTER PEAD/growth_tilt/VTG/signal-stability stack

By comparing
  ratio_raw      = std(raw_predictions_daily_eq) / std(raw_predictions_raw)
  ratio_post     = std(predictions_daily_eq)     / std(predictions_raw)
the ratio_post / ratio_raw delta tells us how much each post-process layer
amplifies or dampens the magnitude change introduced by PCA scaling.

Output: outputs/diagnostics/pca_scale_layer_ratio.csv + stdout summary.

This is a MINIMAL forensic - full per-layer breakdown would require running
with instrumented backtest. This script gives the headline (pre vs post
post-process) decomposition.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PKL_RAW = ROOT / "outputs" / "iter15_FINAL_postfix" / "backtest_result.pkl"
PKL_DEQ = ROOT / "outputs" / "exp_pca_scale" / "daily_eq" / "backtest_result.pkl"

OUT_CSV = ROOT / "outputs" / "diagnostics" / "pca_scale_layer_ratio.csv"


def _cross_sectional_std(df: pd.DataFrame) -> pd.Series:
    """std across tickers for each date (axis=1)."""
    return df.std(axis=1, ddof=0)


def _mean_std(df: pd.DataFrame) -> float:
    s = _cross_sectional_std(df).dropna()
    return float(s.mean()) if len(s) else float("nan")


def main() -> int:
    if not PKL_RAW.exists() or not PKL_DEQ.exists():
        print(f"missing pkls: raw={PKL_RAW.exists()}, deq={PKL_DEQ.exists()}")
        return 1
    with PKL_RAW.open("rb") as fh:
        r_raw = pickle.load(fh)
    with PKL_DEQ.open("rb") as fh:
        r_deq = pickle.load(fh)

    pre_raw = getattr(r_raw, "raw_predictions", None)
    post_raw = getattr(r_raw, "predictions", None)
    pre_deq = getattr(r_deq, "raw_predictions", None)
    post_deq = getattr(r_deq, "predictions", None)

    if any(x is None for x in (pre_raw, post_raw, pre_deq, post_deq)):
        print(f"missing prediction fields: pre_raw={pre_raw is not None}, "
              f"post_raw={post_raw is not None}, pre_deq={pre_deq is not None}, "
              f"post_deq={post_deq is not None}")
        return 2

    # Align dates between the two runs
    common = pre_raw.index.intersection(pre_deq.index)
    pre_raw = pre_raw.loc[common]
    post_raw = post_raw.loc[common]
    pre_deq = pre_deq.loc[common]
    post_deq = post_deq.loc[common]

    mu_pre_raw = _mean_std(pre_raw)
    mu_post_raw = _mean_std(post_raw)
    mu_pre_deq = _mean_std(pre_deq)
    mu_post_deq = _mean_std(post_deq)

    ratio_pre = mu_pre_deq / mu_pre_raw if mu_pre_raw > 0 else float("nan")
    ratio_post = mu_post_deq / mu_post_raw if mu_post_raw > 0 else float("nan")
    amplification = ratio_post / ratio_pre if ratio_pre and ratio_pre == ratio_pre else float("nan")

    expected_theoretical = 1.0 / np.sqrt(20)  # daily_eq divides by sqrt(forward_horizon)

    print("=" * 72)
    print("  PCA target scale forensics - raw vs daily_eq")
    print("=" * 72)
    print(f"  Common dates             : {len(common)}")
    print(f"  Theoretical pre ratio    : {expected_theoretical:.4f}  (= 1/sqrt(20))")
    print(f"  Observed pre ratio       : {ratio_pre:.4f}  "
          f"(stdev raw_predictions: {mu_pre_raw:.4f} -> {mu_pre_deq:.4f})")
    print(f"  Observed post ratio      : {ratio_post:.4f}  "
          f"(stdev final predictions: {mu_post_raw:.4f} -> {mu_post_deq:.4f})")
    print(f"  Post/Pre amplification   : {amplification:.4f}")
    print()
    print("  Interpretation:")
    if amplification > 1.05:
        print(f"  -> Post-process layers AMPLIFY the magnitude effect "
              f"({amplification:.2f}x). The downstream pipeline is producing")
        print("     a smaller-than-theoretical contraction in the daily_eq run,")
        print("     i.e. some additive boost (PEAD/growth_tilt) is contributing a")
        print("     fixed magnitude that dominates the contracted raw signal.")
    elif amplification < 0.95:
        print(f"  -> Post-process layers DAMPEN the magnitude effect "
              f"({amplification:.2f}x). Unexpected; investigate.")
    else:
        print(f"  -> Post-process magnitude effect is ~theoretical "
              f"({amplification:.2f}). Post-process layers are nearly magnitude-")
        print("     invariant, so IR sensitivity comes from elsewhere "
              "(e.g. portfolio optimizer's relative weighting of signal vs risk).")

    # Per-layer breakdown isn't available without instrumentation; the pre/post
    # ratio is the headline. Save what we have.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {"layer": "raw_predictions (pre post-process)", "mean_std_raw": mu_pre_raw,
             "mean_std_daily_eq": mu_pre_deq, "ratio_deq_over_raw": ratio_pre},
            {"layer": "predictions (post post-process)", "mean_std_raw": mu_post_raw,
             "mean_std_daily_eq": mu_post_deq, "ratio_deq_over_raw": ratio_post},
        ]
    )
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print()
    print(f"  Wrote: {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
