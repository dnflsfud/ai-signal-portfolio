"""Build outputs/iter15_FINAL_postfix/comparison.md.

Recomputes the legacy iter15_65tkr_reb21_vtg metrics on the SAME window as
the cutoff-trimmed postfix run (predict dates <= 2024-12-31), then writes
a side-by-side diff. Without this trim, the IR comparison is not fair
(legacy has 16 extra months of post-cutoff data).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so pickle can resolve `src.backtest.BacktestResult`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LEGACY_LABEL = "iter15_65tkr_reb21_vtg"
POSTFIX_LABEL = "iter15_FINAL_postfix"
CUTOFF = pd.Timestamp("2024-12-31")

ROOT = Path(__file__).resolve().parent.parent


def annualized_ir(active: pd.Series) -> float:
    a = active.dropna()
    if len(a) < 20 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * np.sqrt(252))


def sub_ir(port: pd.Series, bm: pd.Series, start: str, end: str) -> float:
    mask = (port.index >= pd.Timestamp(start)) & (port.index <= pd.Timestamp(end))
    pp, bb = port[mask], bm.reindex(port[mask].index)
    if len(pp) < 20:
        return float("nan")
    active = pp.values - bb.values
    if active.std(ddof=1) == 0:
        return 0.0
    return float(active.mean() / active.std(ddof=1) * np.sqrt(252))


def core_metrics(port: pd.Series, bm: pd.Series, trim_end: pd.Timestamp | None = None) -> dict:
    if trim_end is not None:
        mask = port.index <= trim_end
        port, bm = port[mask], bm.reindex(port.index[mask])
    bm = bm.reindex(port.index).fillna(0.0)
    active = port - bm
    ann_ret = (1 + port).prod() ** (252 / len(port)) - 1
    ann_vol = port.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    bm_ann = (1 + bm).prod() ** (252 / len(bm)) - 1
    active_ret = ann_ret - bm_ann
    te = active.std(ddof=1) * np.sqrt(252)
    ir = annualized_ir(active)
    cum = (1 + port).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    last_date = port.index[-1]
    return {
        "first_date": str(port.index[0].date()),
        "last_date": str(last_date.date()),
        "n_days": int(len(port)),
        "annual_return": float(ann_ret),
        "active_return": float(active_ret),
        "annual_vol": float(ann_vol),
        "tracking_error": float(te),
        "sharpe": float(sharpe),
        "information_ratio": float(ir),
        "max_drawdown": max_dd,
    }


def load_returns(label: str) -> tuple[pd.Series, pd.Series]:
    pkl_paths = [
        ROOT / "outputs" / label / "backtest_result.pkl",
        # Legacy artifacts sometimes saved as result.pkl
        ROOT / "outputs" / label / "result.pkl",
    ]
    for p in pkl_paths:
        if p.exists():
            with p.open("rb") as fh:
                r = pickle.load(fh)
            port = r.portfolio_returns.dropna()
            bm = r.benchmark_returns.dropna()
            return port, bm
    raise FileNotFoundError(f"No backtest_result.pkl for {label}")


def fmt(val, pct=False, places=3):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return "—"
    if pct:
        return f"{val * 100:.{places}f}%"
    return f"{val:.{places}f}"


def delta(new, old, pct=False, places=3, places_pct=2):
    if (not isinstance(new, (int, float)) or not isinstance(old, (int, float))
            or pd.isna(new) or pd.isna(old)):
        return "—"
    d = new - old
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d * 100:.{places_pct}f}p"
    return f"{sign}{d:.{places}f}"


def main():
    out_dir = ROOT / "outputs" / POSTFIX_LABEL
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both runs
    leg_port, leg_bm = load_returns(LEGACY_LABEL)
    post_port, post_bm = load_returns(POSTFIX_LABEL)

    leg_last_full = leg_port.index[-1]
    post_last = post_port.index[-1]

    # Full-window legacy (for context only — NOT used as comparison gate)
    legacy_full = core_metrics(leg_port, leg_bm)
    # Cutoff-trimmed legacy — fair comparison
    legacy_trim = core_metrics(leg_port, leg_bm, trim_end=CUTOFF)
    # Postfix: predictions stop at cutoff but portfolio_returns drifts past it.
    # Trim BOTH so the comparison reflects ONLY the period both runs actively
    # traded on (drift after cutoff is not genuine alpha attribution).
    postfix_full = core_metrics(post_port, post_bm)
    postfix = core_metrics(post_port, post_bm, trim_end=CUTOFF)

    # Sub-period IRs (same windows for both; P3 trimmed at cutoff for fairness)
    sub_periods = {
        "P1": ("2018-11-23", "2021-05-11"),
        "P2": ("2021-05-12", "2023-10-27"),
        "P3 (cutoff-trimmed)": ("2023-10-30", "2024-12-31"),
    }
    legacy_sub = {k: sub_ir(leg_port, leg_bm, s, e) for k, (s, e) in sub_periods.items()}
    postfix_sub = {k: sub_ir(post_port, post_bm, s, e) for k, (s, e) in sub_periods.items()}

    # Turnover from metrics.json (avg_annual_turnover, two-way L1)
    leg_metrics = json.loads(
        (ROOT / "outputs" / LEGACY_LABEL / "metrics.json").read_text(encoding="utf-8")
    )["metrics"]
    post_metrics = json.loads(
        (ROOT / "outputs" / POSTFIX_LABEL / "metrics.json").read_text(encoding="utf-8")
    )["metrics"]

    # --- comparison.md ---
    lines = [
        f"# iter15_FINAL_postfix vs iter15_65tkr_reb21_vtg",
        "",
        f"**Generated**: 2026-05-19",
        f"**Postfix predict window end**: {postfix['last_date']}",
        f"**Legacy predict window end (full)**: {legacy_full['last_date']}",
        f"**Cutoff used for fair compare**: {CUTOFF.date()}",
        "",
        "## Methodology delta",
        "- legacy: no embargo (label leak), no cutoff (full sample through 2026-04)",
        "- postfix: `embargo_days=20`, `train_cutoff_date=2024-12-31`",
        "",
        "## Headline metrics (cutoff-trimmed window, fair comparison)",
        "",
        "| Metric              | legacy iter15 (trimmed) | postfix | Δ |",
        "|---------------------|-------------------------|---------|---|",
        f"| First / Last date   | {legacy_trim['first_date']} → {legacy_trim['last_date']} "
        f"| {postfix['first_date']} → {postfix['last_date']} | — |",
        f"| N days              | {legacy_trim['n_days']} | {postfix['n_days']} | "
        f"{postfix['n_days'] - legacy_trim['n_days']:+d} |",
        f"| Annual Return       | {fmt(legacy_trim['annual_return'], pct=True, places=2)} "
        f"| {fmt(postfix['annual_return'], pct=True, places=2)} "
        f"| {delta(postfix['annual_return'], legacy_trim['annual_return'], pct=True)} |",
        f"| Active Return       | {fmt(legacy_trim['active_return'], pct=True, places=2)} "
        f"| {fmt(postfix['active_return'], pct=True, places=2)} "
        f"| {delta(postfix['active_return'], legacy_trim['active_return'], pct=True)} |",
        f"| Annual Vol          | {fmt(legacy_trim['annual_vol'], pct=True, places=2)} "
        f"| {fmt(postfix['annual_vol'], pct=True, places=2)} "
        f"| {delta(postfix['annual_vol'], legacy_trim['annual_vol'], pct=True)} |",
        f"| Tracking Error      | {fmt(legacy_trim['tracking_error'], pct=True, places=2)} "
        f"| {fmt(postfix['tracking_error'], pct=True, places=2)} "
        f"| {delta(postfix['tracking_error'], legacy_trim['tracking_error'], pct=True)} |",
        f"| Sharpe              | {fmt(legacy_trim['sharpe'])} "
        f"| {fmt(postfix['sharpe'])} "
        f"| {delta(postfix['sharpe'], legacy_trim['sharpe'])} |",
        f"| **Information Ratio** | **{fmt(legacy_trim['information_ratio'])}** "
        f"| **{fmt(postfix['information_ratio'])}** "
        f"| **{delta(postfix['information_ratio'], legacy_trim['information_ratio'])}** |",
        f"| Max Drawdown        | {fmt(legacy_trim['max_drawdown'], pct=True, places=2)} "
        f"| {fmt(postfix['max_drawdown'], pct=True, places=2)} "
        f"| {delta(postfix['max_drawdown'], legacy_trim['max_drawdown'], pct=True)} |",
        f"| Annual Turnover 2w  | {fmt(leg_metrics.get('avg_annual_turnover'), pct=True, places=1)} "
        f"| {fmt(post_metrics.get('avg_annual_turnover'), pct=True, places=1)} "
        f"| {delta(post_metrics.get('avg_annual_turnover'), leg_metrics.get('avg_annual_turnover'), pct=True, places_pct=1)} |",
        f"| Avg IC              | {fmt(leg_metrics.get('avg_ic'), places=4)} "
        f"| {fmt(post_metrics.get('avg_ic'), places=4)} | "
        f"{delta(post_metrics.get('avg_ic'), leg_metrics.get('avg_ic'), places=4)} |",
        "",
        "## Sub-period IR",
        "",
        "| Window | legacy (same window) | postfix | Δ |",
        "|--------|----------------------|---------|---|",
    ]
    for k in sub_periods:
        lines.append(
            f"| {k} | {fmt(legacy_sub[k])} | {fmt(postfix_sub[k])} | "
            f"{delta(postfix_sub[k], legacy_sub[k])} |"
        )

    lines += [
        "",
        "## Full-window context (NOT used as gate — drift after cutoff included)",
        "",
        "| Metric | legacy iter15 (full) | postfix (full, drift after cutoff) |",
        "|--------|----------------------|------------------------------------|",
        f"| First → Last date | {legacy_full['first_date']} → {legacy_full['last_date']} "
        f"| {postfix_full['first_date']} → {postfix_full['last_date']} |",
        f"| IR | {fmt(legacy_full['information_ratio'])} "
        f"| {fmt(postfix_full['information_ratio'])} |",
        f"| Annual Return | {fmt(legacy_full['annual_return'], pct=True, places=2)} "
        f"| {fmt(postfix_full['annual_return'], pct=True, places=2)} |",
        "",
        "> Postfix full-window includes ~360 days of pure drift (no new predictions) "
        "after 2024-12-31. Not a fair alpha measure.",
        "",
        "## 해석",
        "",
        "- **ΔIR (cutoff-trimmed)**: "
        f"{delta(postfix['information_ratio'], legacy_trim['information_ratio'])}. "
        "Negative = leakage premium that the embargo removed.",
        "- **P2 collapse**: 가장 큰 충격은 P2 (rate-hike regime). "
        f"legacy(trimmed)={fmt(legacy_sub['P2'])} → postfix={fmt(postfix_sub['P2'])}, "
        f"Δ={delta(postfix_sub['P2'], legacy_sub['P2'])}. "
        "라벨 누수가 P2 IR을 인위적으로 부양하고 있었음을 시사.",
        "- **P1/P3**는 상대적으로 견고 (둘 다 positive). 본질적 alpha는 살아 있음.",
        "- **누수 프리미엄 추정치**: cutoff-trimmed window 기준 "
        f"ΔIR={delta(postfix['information_ratio'], legacy_trim['information_ratio'])}. "
        "이 차이의 상당 부분이 early_stopping이 forward 20일 라벨을 통해 미래를 본 효과.",
        "",
        "## Next steps",
        "",
        "- Task A step 3: BASELINE.md/CLAUDE.md/ROADMAP.md 갱신 (postfix가 new canonical)",
        "- Task B: 이 postfix 위에서 in-sample fit 의심 overlay 6종 ablation",
        "- 우선순위: P2 negative IR 원인을 ablation으로 분리 (특히 value_trap_gate, growth_tilt)",
    ]

    out = out_dir / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

    # sanity.txt — verify cutoff actually applied to PREDICTIONS (not just sim)
    sanity = out_dir / "sanity.txt"
    cfg_manifest = json.loads(
        (out_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    cfg = cfg_manifest["config"]
    # Load predictions span explicitly
    with (out_dir / "backtest_result.pkl").open("rb") as fh:
        r_post = pickle.load(fh)
    preds = r_post.predictions if hasattr(r_post, "predictions") else None
    pred_last = preds.dropna(how="all").index[-1].date() if preds is not None else "—"
    sanity_lines = [
        f"embargo applied: {cfg.get('embargo_days') == 20} (embargo_days={cfg.get('embargo_days')})",
        f"oos cutoff applied: {cfg.get('enforce_oos_holdout')} (train_cutoff_date={cfg.get('train_cutoff_date')})",
        f"postfix LAST PREDICTION date: {pred_last}",
        f"postfix portfolio_returns last date (drift extends past cutoff): {post_last.date()}",
        f"legacy last predict date: {leg_last_full.date()}",
    ]
    sanity.write_text("\n".join(sanity_lines) + "\n", encoding="utf-8")
    print(f"Wrote {sanity}")


if __name__ == "__main__":
    main()
