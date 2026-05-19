"""Re-compute rolling_ir / spa_pvalue on existing baselines' metrics.json.

selection-bias-discipline Task C step 2: extends compute_performance_metrics
with new keys. Existing artifacts (iter15_FINAL_postfix, baseline_v5,
iter15_65tkr_reb21_vtg, baseline_v4, ablation_*) miss these keys.

This script re-computes metrics from each backtest_result.pkl and merges the
new keys into metrics.json. Does NOT touch existing keys to avoid surprising
diff. Idempotent.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import rolling_ir_stats, spa_pvalue  # noqa: E402

NEW_KEYS = {
    "rolling_ir_mean", "rolling_ir_median", "rolling_ir_min", "rolling_ir_max",
    "rolling_ir_pos_frac", "rolling_ir_window", "spa_pvalue",
}


def backfill(d: Path) -> bool:
    metrics_path = d / "metrics.json"
    pkl_path = d / "backtest_result.pkl"
    if not metrics_path.exists() or not pkl_path.exists():
        return False
    body = json.loads(metrics_path.read_text(encoding="utf-8"))
    m = body.get("metrics", {})
    if NEW_KEYS <= set(m.keys()):
        print(f"[backfill] SKIP {d.name} (already has rolling/SPA keys)")
        return False

    try:
        with pkl_path.open("rb") as fh:
            r = pickle.load(fh)
    except Exception as exc:
        print(f"[backfill] WARN {d.name}: pkl unreadable ({exc})")
        return False

    port = r.portfolio_returns.dropna()
    bm = r.benchmark_returns.dropna()
    active = (port - bm.reindex(port.index).fillna(0.0)).dropna()
    m.update(rolling_ir_stats(active, window=252))
    m["spa_pvalue"] = spa_pvalue(active, n_bootstrap=1000, block_size=10, seed=42)
    body["metrics"] = m
    metrics_path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    print(f"[backfill] OK   {d.name}: rolling_ir_mean={m['rolling_ir_mean']:.3f}, "
          f"spa_p={m['spa_pvalue']:.4f}")
    return True


def main():
    candidates = [
        ROOT / "outputs" / "iter15_FINAL_postfix",
        ROOT / "outputs" / "iter15_65tkr_reb21_vtg",
        ROOT / "outputs" / "baseline_v5",
        ROOT / "outputs" / "baseline_v4",
    ]
    # Also all outputs/ablation/ablation_*
    abl_dir = ROOT / "outputs" / "ablation"
    if abl_dir.exists():
        candidates.extend(sorted(d for d in abl_dir.iterdir()
                                 if d.is_dir() and d.name.startswith("ablation_")))
    # PCA scale variant
    pca_dir = ROOT / "outputs" / "exp_pca_scale" / "daily_eq"
    if pca_dir.exists():
        candidates.append(pca_dir)

    n_updated = 0
    for d in candidates:
        if backfill(d):
            n_updated += 1
    print(f"\n[backfill] updated {n_updated} metrics.json files")


if __name__ == "__main__":
    main()
