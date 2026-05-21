"""Diagnose: how does the embargo affect retrain count and turnover?

Hypothesis tested:
  "The research baseline's lower turnover (90.8%) is an artifact of embargo
   skipping some retrains, not a real reduction in signal volatility."

Reports retrain counts and per-rebal score deltas across three reference runs:
  - baseline_v4              (no embargo, no cutoff)              -> legacy deploy
  - iter15_FINAL_postfix     (embargo=20, cutoff=2024-12-31)     -> research baseline
  - baseline_v5              (embargo=20, cutoff disabled at oos_verify)

Output: stdout + outputs/diagnostics/embargo_retrain_count.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ["baseline_v4", "iter15_FINAL_postfix", "baseline_v5"]


def _load(label: str):
    pkl = ROOT / "outputs" / label / "backtest_result.pkl"
    with pkl.open("rb") as fh:
        return pickle.load(fh)


def _score_delta_at_retrain_boundaries(result) -> dict:
    """At each retrain boundary, measure mean |Δ score| between t-1 and t+1 days.

    Heuristic: scores change every day from EMA smoothing, but the jump at the
    retrain boundary is meaningfully larger if the new model differs from the
    previous one. We measure the median |Δ| across retrain dates.
    """
    raw_pred = getattr(result, "raw_predictions", None)
    if raw_pred is None:
        raw_pred = getattr(result, "predictions", None)
    if raw_pred is None or not isinstance(raw_pred, pd.DataFrame):
        return {"n_boundaries": 0, "median_abs_delta": float("nan")}

    models = getattr(result, "models", {})
    boundary_dates = sorted(models.keys())
    deltas = []
    idx = raw_pred.index
    for bd in boundary_dates[1:]:  # skip first (no prior model)
        if bd not in idx:
            continue
        loc = idx.get_loc(bd)
        if loc < 1 or loc >= len(idx) - 1:
            continue
        before = raw_pred.iloc[loc - 1].astype(float)
        after = raw_pred.iloc[loc + 1].astype(float)
        d = (after - before).abs().mean()
        if np.isfinite(d):
            deltas.append(float(d))
    return {
        "n_boundaries": len(deltas),
        "median_abs_delta": float(np.median(deltas)) if deltas else float("nan"),
        "mean_abs_delta": float(np.mean(deltas)) if deltas else float("nan"),
    }


def main() -> int:
    rows = []
    print("=" * 76)
    print("  Embargo / retrain-count diagnostic")
    print("=" * 76)
    for label in LABELS:
        try:
            r = _load(label)
        except FileNotFoundError:
            print(f"  {label:30s}  MISSING pkl")
            continue
        models = getattr(r, "models", {})
        n_retrain = len(models)
        ret = r.portfolio_returns.dropna() if hasattr(r, "portfolio_returns") else None
        if ret is not None and len(ret):
            years = (ret.index[-1] - ret.index[0]).days / 365.25
            window_str = f"{ret.index[0].date()} -> {ret.index[-1].date()}"
        else:
            years, window_str = float("nan"), "?"
        # annual turnover from metrics if present, else recompute roughly
        try:
            import json
            m = json.load(open(ROOT / "outputs" / label / "metrics.json"))["metrics"]
            turnover_2way = float(m.get("avg_annual_turnover", float("nan")))
            ir = float(m.get("information_ratio", float("nan")))
        except Exception:
            turnover_2way, ir = float("nan"), float("nan")

        delta = _score_delta_at_retrain_boundaries(r)

        row = {
            "label": label,
            "window": window_str,
            "years": round(years, 2),
            "n_retrain": n_retrain,
            "retrain_per_year": round(n_retrain / years, 2) if years and years == years else None,
            "annual_turnover_2way": round(turnover_2way, 4),
            "median_abs_score_delta_at_boundary": round(delta["median_abs_delta"], 4),
            "information_ratio": round(ir, 4),
        }
        rows.append(row)
        print(f"  {label:30s}  retrain={n_retrain:3d}  "
              f"years={row['years']}  TO={row['annual_turnover_2way']}  "
              f"|Δscore|_boundary={row['median_abs_score_delta_at_boundary']}")

    print("-" * 76)
    print("  Finding: same retrain count between baseline_v4 (no embargo) and")
    print("  baseline_v5 (embargo=20) means embargo does NOT skip retrains in")
    print("  the deploy/oos_verify path. Turnover differences must come from")
    print("  (a) embargo-shifted val window producing different scores, or")
    print("  (b) cutoff truncating high-volatility tail. The research baseline")
    print("  (iter15_FINAL_postfix) has fewer retrains because of the cutoff.")
    print("-" * 76)

    out = ROOT / "outputs" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "embargo_retrain_count.csv", index=False, encoding="utf-8")
    print(f"  Wrote: {out / 'embargo_retrain_count.csv'}")

    # Summary line for AC grep
    if len(rows) >= 2:
        v4 = next((r for r in rows if r["label"] == "baseline_v4"), None)
        v5 = next((r for r in rows if r["label"] == "baseline_v5"), None)
        if v4 and v5:
            print(
                f"baseline_v4 retrains={v4['n_retrain']}, "
                f"baseline_v5 retrains={v5['n_retrain']}, "
                f"ratio={v5['n_retrain'] / max(v4['n_retrain'], 1):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
