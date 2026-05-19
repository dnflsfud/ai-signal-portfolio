"""Build outputs/baseline_v5/oos_report.md and check promote gate."""
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
B5 = ROOT / "outputs" / "baseline_v5"
RESEARCH = ROOT / "outputs" / "iter15_FINAL_postfix"


def load_pp(d: Path):
    with (d / "backtest_result.pkl").open("rb") as fh:
        r = pickle.load(fh)
    return r.portfolio_returns.dropna(), r.benchmark_returns.dropna()


def fmt(v, p=3):
    return f"{v:.{p}f}" if isinstance(v, (int, float)) and v == v else "—"


def main():
    b5_port, b5_bm = load_pp(B5)
    res_port, res_bm = load_pp(RESEARCH)

    # Cutoff-trimmed comparison (apples-to-apples vs research baseline)
    b5_trim = core_metrics(b5_port, b5_bm, trim_end=CUTOFF)
    res_trim = core_metrics(res_port, res_bm, trim_end=CUTOFF)
    # Full-window (the peek)
    b5_full = core_metrics(b5_port, b5_bm)

    # Sub-period IRs
    sp = {
        "P1 (2018-11~2021-05)": ("2018-11-23", "2021-05-11"),
        "P2 (2021-05~2023-10)": ("2021-05-12", "2023-10-27"),
        "P3 trim (2023-10~2024-12)": ("2023-10-30", "2024-12-31"),
        "P3 full + post-cutoff (~2026-05)": ("2023-10-30", "2026-05-15"),
    }
    b5_sub = {k: sub_ir(b5_port, b5_bm, s, e) for k, (s, e) in sp.items()}
    res_sub = {k: sub_ir(res_port, res_bm, s, e) for k, (s, e) in sp.items()}

    inv = json.loads((ROOT / "experiment_inventory.json").read_text(encoding="utf-8"))
    n_peeks = inv.get("n_oos_peeks", 0)
    peeks_b5 = [p for p in inv.get("oos_peeks", []) if p["label"] == "baseline_v5"]

    # Gate criteria from BASELINE.md
    gate_ir       = b5_trim["information_ratio"] >= res_trim["information_ratio"]
    gate_p2       = b5_sub["P2 (2021-05~2023-10)"] >= res_sub["P2 (2021-05~2023-10)"] - 0.10
    # turnover gate uses metrics.json (avg_annual_turnover, two-way)
    b5_to  = json.loads((B5 / "metrics.json").read_text(encoding="utf-8"))["metrics"]["avg_annual_turnover"]
    res_to = json.loads((RESEARCH / "metrics.json").read_text(encoding="utf-8"))["metrics"]["avg_annual_turnover"]
    gate_to = b5_to <= res_to + 0.05
    all_pass = bool(gate_ir and gate_p2 and gate_to)

    last_date_b5 = b5_port.index[-1].date()
    last_date_res = res_port.index[-1].date()

    lines = []
    lines.append("# baseline_v5 OOS Verification")
    lines.append("")
    lines.append("**Generated**: 2026-05-19")
    lines.append(f"**Peek label**: `baseline_v5`")
    lines.append(f"**experiment_inventory.json `n_oos_peeks`**: {n_peeks} (this run = peek #{len(peeks_b5)} for baseline_v5)")
    lines.append(f"**baseline_v5 last date**: {last_date_b5}")
    lines.append(f"**research baseline last date**: {last_date_res}")
    lines.append("")
    lines.append("## Configuration delta vs iter15_FINAL_postfix")
    lines.append("")
    lines.append("Single knob: `feature_mode: \"core\" → \"lean\"`. All other production overlays unchanged.")
    lines.append("")

    lines.append("## Metrics — cutoff-trimmed (fair comparison, 1592 obs)")
    lines.append("")
    lines.append("| Metric | research baseline | baseline_v5 | Δ |")
    lines.append("|---|---:|---:|---:|")
    for key, label in [
        ("information_ratio", "Information Ratio"),
        ("active_return",     "Active Return"),
        ("tracking_error",    "Tracking Error"),
        ("sharpe",            "Sharpe"),
        ("annual_return",     "Annual Return"),
        ("max_drawdown",      "Max Drawdown"),
    ]:
        r_v, b_v = res_trim[key], b5_trim[key]
        d = b_v - r_v
        sign = "+" if d >= 0 else ""
        if key in ("active_return", "tracking_error", "annual_return", "max_drawdown"):
            lines.append(f"| {label} | {r_v*100:+.2f}% | {b_v*100:+.2f}% | {sign}{d*100:.2f}p |")
        else:
            lines.append(f"| {label} | {fmt(r_v)} | {fmt(b_v)} | {sign}{d:.3f} |")
    lines.append("")

    lines.append("## Sub-period IR")
    lines.append("")
    lines.append("| Window | research baseline | baseline_v5 | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in sp:
        d = b5_sub[k] - res_sub[k]
        sign = "+" if d >= 0 else ""
        lines.append(f"| {k} | {fmt(res_sub[k])} | {fmt(b5_sub[k])} | {sign}{d:.3f} |")
    lines.append("")
    lines.append("> The 'P3 full + post-cutoff' row is the OOS peek payoff — that's the segment")
    lines.append("> the research baseline could not see and the embargo + cutoff reserved.")
    lines.append("")

    lines.append("## Turnover")
    lines.append("")
    lines.append("| Metric | research baseline | baseline_v5 | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Annual Turnover 2-way | {res_to*100:.1f}% | {b5_to*100:.1f}% | {(b5_to-res_to)*100:+.1f}p |")
    lines.append("")

    lines.append("## Gate checks (docs/BASELINE.md criteria)")
    lines.append("")
    def mark(b): return "✅" if b else "❌"
    lines.append(f"- {mark(gate_ir)} **IR ≥ baseline IR** "
                 f"(baseline_v5 trimmed IR {fmt(b5_trim['information_ratio'])} "
                 f"vs research baseline {fmt(res_trim['information_ratio'])})")
    lines.append(f"- {mark(gate_p2)} **P2 IR ≥ baseline P2 IR − 0.10** "
                 f"(baseline_v5 P2 {fmt(b5_sub['P2 (2021-05~2023-10)'])} "
                 f"vs research baseline P2 {fmt(res_sub['P2 (2021-05~2023-10)'])}, "
                 f"floor {fmt(res_sub['P2 (2021-05~2023-10)']-0.10)})")
    lines.append(f"- {mark(gate_to)} **Turnover ≤ baseline + 5 p.p.** "
                 f"(baseline_v5 {b5_to*100:.1f}% vs research baseline {res_to*100:.1f}%, "
                 f"ceiling {(res_to+0.05)*100:.1f}%)")
    lines.append("")

    verdict = "**PROMOTE**" if all_pass else "**DO NOT PROMOTE**"
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    if all_pass:
        lines.append("All gate checks pass on the cutoff-trimmed window. The post-cutoff "
                     "extension (2025-01 onward) preserves the lift (P3-full IR "
                     f"= {fmt(b5_sub['P3 full + post-cutoff (~2026-05)'])} "
                     f"vs research baseline {fmt(res_sub['P3 full + post-cutoff (~2026-05)'])}).")
        lines.append("")
        lines.append("Next steps: BASELINE.md / CLAUDE.md updated to point canonical "
                     "research baseline at `baseline_v5`; rollback_log.md gains entries "
                     "for the 6 overlays/swap decisions; `outputs/baseline_v4/` is "
                     "untouched (deploy path cutover deferred to a separate task).")
    else:
        lines.append("At least one gate failed. baseline_v5 is NOT promoted; "
                     "research baseline remains `iter15_FINAL_postfix`. Per Task B "
                     "discipline, no further peek is permitted for this candidate. "
                     "A new candidate would require a new ablation round.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Headline numbers for downstream docs")
    lines.append("")
    lines.append("Cutoff-trimmed:")
    lines.append(f"- IR {fmt(b5_trim['information_ratio'])} "
                 f"(research baseline {fmt(res_trim['information_ratio'])})")
    lines.append(f"- P1 {fmt(b5_sub['P1 (2018-11~2021-05)'])} / "
                 f"P2 {fmt(b5_sub['P2 (2021-05~2023-10)'])} / "
                 f"P3 trim {fmt(b5_sub['P3 trim (2023-10~2024-12)'])}")
    lines.append("")
    lines.append("Full window (the peek):")
    lines.append(f"- IR {fmt(b5_full['information_ratio'])}")
    lines.append(f"- P3 full {fmt(b5_sub['P3 full + post-cutoff (~2026-05)'])}")
    lines.append(f"- Annual return {b5_full['annual_return']*100:.2f}%")
    lines.append(f"- Last date {last_date_b5}")
    lines.append("")

    out = B5 / "oos_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print()
    print(f"VERDICT: {'PROMOTE' if all_pass else 'DO NOT PROMOTE'}")
    print(f"  gate_ir={gate_ir}, gate_p2={gate_p2}, gate_turnover={gate_to}")
    print(f"  trimmed IR: {b5_trim['information_ratio']:.3f} vs {res_trim['information_ratio']:.3f}")
    print(f"  trimmed P2: {b5_sub['P2 (2021-05~2023-10)']:.3f} vs {res_sub['P2 (2021-05~2023-10)']:.3f}")
    print(f"  n_oos_peeks: {n_peeks}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
