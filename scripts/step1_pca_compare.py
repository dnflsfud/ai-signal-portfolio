"""Quick raw vs daily_eq comparison for Task C step 1 finding doc."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.step2_comparison import core_metrics, sub_ir  # noqa: E402

CUTOFF = pd.Timestamp("2024-12-31")


def load_summary(label: str, path: str) -> dict:
    with (ROOT / path / "backtest_result.pkl").open("rb") as fh:
        r = pickle.load(fh)
    port = r.portfolio_returns.dropna()
    bm = r.benchmark_returns.dropna()
    m_full = json.loads((ROOT / path / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    trim = core_metrics(port, bm, trim_end=CUTOFF)
    return {
        "label": label,
        "full_ir": m_full["information_ratio"],
        "trim_ir": trim["information_ratio"],
        "trim_active_ret": trim["active_return"],
        "trim_te": trim["tracking_error"],
        "trim_P1": sub_ir(port, bm, "2018-11-23", "2021-05-11"),
        "trim_P2": sub_ir(port, bm, "2021-05-12", "2023-10-27"),
        "trim_P3": sub_ir(port, bm, "2023-10-30", "2024-12-31"),
        "rolling_ir_mean": m_full.get("rolling_ir_mean", float("nan")),
        "rolling_ir_min": m_full.get("rolling_ir_min", float("nan")),
        "rolling_ir_pos_frac": m_full.get("rolling_ir_pos_frac", float("nan")),
        "spa_p": m_full.get("spa_pvalue", float("nan")),
        "turnover_2way": m_full.get("avg_annual_turnover", float("nan")),
        "avg_ic": m_full.get("avg_ic", float("nan")),
    }


def main():
    raw = load_summary("raw (iter15_FINAL_postfix)", "outputs/iter15_FINAL_postfix")
    daily_eq = load_summary("daily_eq (exp_pca_scale/daily_eq)", "outputs/exp_pca_scale/daily_eq")

    delta_ir_trim = daily_eq["trim_ir"] - raw["trim_ir"]
    delta_ir_full = daily_eq["full_ir"] - raw["full_ir"]
    promote = abs(delta_ir_trim) >= 0.02

    print(f"\n=== PCA target scale A/B ===")
    print(f"raw      trim_IR={raw['trim_ir']:.4f}  full_IR={raw['full_ir']:.4f}")
    print(f"daily_eq trim_IR={daily_eq['trim_ir']:.4f}  full_IR={daily_eq['full_ir']:.4f}")
    print(f"Δ trim_IR = {delta_ir_trim:+.4f}")
    print(f"Δ full_IR = {delta_ir_full:+.4f}")

    print(f"\nrolling/SPA:")
    print(f"  raw:      rolling_ir_mean={raw['rolling_ir_mean']:.3f}, spa_p={raw['spa_p']:.4f}")
    print(f"  daily_eq: rolling_ir_mean={daily_eq['rolling_ir_mean']:.3f}, spa_p={daily_eq['spa_p']:.4f}")

    out_dir = ROOT / "outputs" / "exp_pca_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PCA target scale A/B (Task C step 1)",
        "",
        "**Question**: Does dividing the PCA residual by sqrt(forward_horizon)",
        "change downstream behavior? Cross-sectional Z-score normalization in",
        "model_trainer should make this invariant — if IR changes, some module",
        "downstream depends on absolute residual magnitude (signal stability",
        "shrinkage, PEAD composition, EMA blending).",
        "",
        f"**Verdict**: |Δ trim_IR| = {abs(delta_ir_trim):.4f} "
        f"{'≥' if promote else '<'} 0.02 → "
        f"{'downstream IS sensitive — investigate' if promote else 'downstream invariant — keep raw default'}.",
        "",
        "## Headline comparison (cutoff-trimmed, 1592 obs)",
        "",
        "| Metric | raw (baseline) | daily_eq | Δ |",
        "|---|---:|---:|---:|",
        f"| trim IR | {raw['trim_ir']:.4f} | {daily_eq['trim_ir']:.4f} | {delta_ir_trim:+.4f} |",
        f"| trim active return | {raw['trim_active_ret']*100:.2f}% | {daily_eq['trim_active_ret']*100:.2f}% | {(daily_eq['trim_active_ret']-raw['trim_active_ret'])*100:+.2f}p |",
        f"| trim TE | {raw['trim_te']*100:.2f}% | {daily_eq['trim_te']*100:.2f}% | {(daily_eq['trim_te']-raw['trim_te'])*100:+.2f}p |",
        f"| trim P1 IR | {raw['trim_P1']:+.3f} | {daily_eq['trim_P1']:+.3f} | {daily_eq['trim_P1']-raw['trim_P1']:+.3f} |",
        f"| trim P2 IR | {raw['trim_P2']:+.3f} | {daily_eq['trim_P2']:+.3f} | {daily_eq['trim_P2']-raw['trim_P2']:+.3f} |",
        f"| trim P3 IR | {raw['trim_P3']:+.3f} | {daily_eq['trim_P3']:+.3f} | {daily_eq['trim_P3']-raw['trim_P3']:+.3f} |",
        f"| Avg IC | {raw['avg_ic']:.4f} | {daily_eq['avg_ic']:.4f} | {daily_eq['avg_ic']-raw['avg_ic']:+.4f} |",
        f"| Turnover 2-way | {raw['turnover_2way']*100:.1f}% | {daily_eq['turnover_2way']*100:.1f}% | {(daily_eq['turnover_2way']-raw['turnover_2way'])*100:+.1f}p |",
        f"| rolling_ir_mean (full) | {raw['rolling_ir_mean']:.3f} | {daily_eq['rolling_ir_mean']:.3f} | {daily_eq['rolling_ir_mean']-raw['rolling_ir_mean']:+.3f} |",
        f"| SPA p-value (full) | {raw['spa_p']:.4f} | {daily_eq['spa_p']:.4f} | — |",
        "",
        "## Decision",
        "",
    ]
    if promote:
        lines.append("|Δ trim_IR| ≥ 0.02. Downstream modules ARE sensitive to absolute")
        lines.append("residual magnitude — likely culprits: signal_stability_lambda,")
        lines.append("PEAD boost composition, or prediction EMA blending. Promotion of")
        lines.append("`daily_eq` is OUT OF SCOPE for this step (Task C step 1 is verification,")
        lines.append("not promotion). Filed as future-work to identify and isolate the")
        lines.append("sensitive module.")
    else:
        lines.append("|Δ trim_IR| < 0.02 → downstream is invariant to absolute residual")
        lines.append("magnitude (cross-sectional Z-score downstream as expected). The")
        lines.append("`pca_target_scale_mode` config field stays in place as a future")
        lines.append("lever; default remains `\"raw\"`. No production change needed.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python run_variant.py --variant variants/exp_pca_scale_daily_eq.yaml --no-cache")
    lines.append("python scripts/step1_pca_compare.py")
    lines.append("```")
    lines.append("")

    out = out_dir / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if not promote else 2


if __name__ == "__main__":
    sys.exit(main())
