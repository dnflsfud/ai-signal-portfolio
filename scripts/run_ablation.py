"""Run all ablation_*.yaml variants and summarise.

Sequential subprocess execution (no parallelism — LightGBM/cvxpy already
saturate one CPU). Each variant is run with --no-cache (Phase 1/2/4 deltas
across variants would otherwise silently reuse stale artifacts).

Output: outputs/ablation/summary.csv with deltas vs iter15_FINAL_postfix.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_PER_VARIANT = 900  # 15 min — generous; typical run ~3-4 min

# Cutoff matching Task A step 2's fair-comparison methodology.
import pandas as pd
CUTOFF = pd.Timestamp("2024-12-31")


METRIC_KEYS = [
    "ir", "active_return", "tracking_error", "sharpe",
    "max_drawdown", "annual_turnover_2way", "avg_ic",
    "P1_ir", "P2_ir", "P3_ir",
]
# Cutoff-trimmed metric keys (computed by recomputing from backtest_result.pkl).
TRIMMED_KEYS = [
    "trimmed_ir", "trimmed_active_return", "trimmed_tracking_error",
    "trimmed_sharpe", "trimmed_P1_ir", "trimmed_P2_ir", "trimmed_P3_ir",
]


def _ensure_root_on_path() -> None:
    """Make `scripts.step2_comparison` importable when invoked as a script."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _pkl_is_usable(pkl: Path, min_bytes: int = 50_000_000) -> bool:
    """Quick heuristic: a valid backtest_result.pkl is ~200 MB.

    Files materially smaller (e.g. < 50 MB) almost certainly come from a
    truncated write (interrupted subprocess). Returning False forces a re-run.
    """
    if not pkl.exists():
        return False
    try:
        return pkl.stat().st_size >= min_bytes
    except OSError:
        return False


def collect_trimmed_metrics(variant_dir: Path) -> dict:
    """Recompute IR/sub-period IRs over the cutoff-trimmed window.

    Loads <variant_dir>/backtest_result.pkl, trims portfolio_returns and
    benchmark_returns to <= CUTOFF, then reuses scripts.step2_comparison
    helpers for parity with Task A step 2's methodology.

    Returns dict of Nones if the pkl is missing or unloadable (truncated).
    """
    _ensure_root_on_path()
    from scripts.step2_comparison import core_metrics, sub_ir  # type: ignore

    pkl = variant_dir / "backtest_result.pkl"
    if not pkl.exists():
        return {k: None for k in TRIMMED_KEYS}
    try:
        with pkl.open("rb") as fh:
            r = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, ValueError) as exc:
        print(f"[run_ablation] WARN: pkl unloadable at {pkl} ({exc}); skipping trimmed.",
              flush=True)
        return {k: None for k in TRIMMED_KEYS}
    port = r.portfolio_returns.dropna()
    bm = r.benchmark_returns.dropna()
    m = core_metrics(port, bm, trim_end=CUTOFF)
    return {
        "trimmed_ir": m["information_ratio"],
        "trimmed_active_return": m["active_return"],
        "trimmed_tracking_error": m["tracking_error"],
        "trimmed_sharpe": m["sharpe"],
        "trimmed_P1_ir": sub_ir(port, bm, "2018-11-23", "2021-05-11"),
        "trimmed_P2_ir": sub_ir(port, bm, "2021-05-12", "2023-10-27"),
        "trimmed_P3_ir": sub_ir(port, bm, "2023-10-30", "2024-12-31"),
    }


def collect_metrics(metrics_path: Path) -> dict:
    """Flatten metrics.json -> single row dict for the summary CSV."""
    body = json.loads(metrics_path.read_text(encoding="utf-8"))
    m = body.get("metrics", {})
    sp = m.get("sub_periods", {})
    return {
        "label": body.get("label", metrics_path.parent.name),
        "ir": m.get("information_ratio"),
        "active_return": m.get("active_return"),
        "tracking_error": m.get("tracking_error"),
        "sharpe": m.get("sharpe_ratio"),
        "max_drawdown": m.get("max_drawdown"),
        "annual_turnover_2way": m.get("avg_annual_turnover"),
        "avg_ic": m.get("avg_ic"),
        "P1_ir": sp.get("P1_ir"),
        "P2_ir": sp.get("P2_ir"),
        "P3_ir": sp.get("P3_ir"),
    }


def run_one(manifest_path: Path, skip_if_exists: bool = False) -> dict:
    """Run a single variant via run_variant.py subprocess; return summary row.

    When ``skip_if_exists`` is True and the variant's metrics.json already
    exists, the subprocess is skipped and only the on-disk artifacts are read.
    Idempotent for safe re-runs after partial failures.
    """
    label = manifest_path.stem
    variant_dir = ROOT / "outputs" / "ablation" / label
    metrics_path = variant_dir / "metrics.json"
    pkl_path = variant_dir / "backtest_result.pkl"

    if skip_if_exists and metrics_path.exists() and _pkl_is_usable(pkl_path):
        row = collect_metrics(metrics_path)
        row.update(collect_trimmed_metrics(variant_dir))
        row["error"] = ""
        ti = row.get("trimmed_ir")
        ti_str = f"{ti:.3f}" if isinstance(ti, (int, float)) and ti == ti else "NaN"
        print(f"[run_ablation] SKIP {label} (cached): IR={row['ir']:.3f}, "
              f"trimmed_IR={ti_str}", flush=True)
        return row
    if skip_if_exists and metrics_path.exists() and not _pkl_is_usable(pkl_path):
        sz = pkl_path.stat().st_size if pkl_path.exists() else 0
        print(f"[run_ablation] REDO {label}: metrics.json exists but pkl truncated "
              f"({sz:,} bytes < 50 MB threshold). Re-running.", flush=True)

    print(f"[run_ablation] -> {label} ...", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "run_variant.py",
        "--variant", str(manifest_path),
        "--no-cache",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            # errors="replace" prevents UnicodeDecodeError in _readerthread
            # when subprocess emits cp949 Korean bytes (Windows console default).
            encoding="utf-8", errors="replace",
            timeout=TIMEOUT_PER_VARIANT,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"[run_ablation] TIMEOUT {label}: {exc}", flush=True)
        return {"label": label, "error": "TIMEOUT",
                **{k: None for k in METRIC_KEYS},
                **{k: None for k in TRIMMED_KEYS}}
    elapsed = int(time.time() - t0)

    # rc != 0 OR missing metrics.json both signal real failure.
    if proc.returncode != 0 and not metrics_path.exists():
        err_tail = (proc.stderr or "")[-500:]
        print(f"[run_ablation] FAIL {label} ({elapsed}s rc={proc.returncode}): {err_tail!r}",
              flush=True)
        return {"label": label, "error": f"rc={proc.returncode}: {err_tail}",
                **{k: None for k in METRIC_KEYS},
                **{k: None for k in TRIMMED_KEYS}}

    if not metrics_path.exists():
        return {"label": label, "error": f"missing {metrics_path}",
                **{k: None for k in METRIC_KEYS},
                **{k: None for k in TRIMMED_KEYS}}

    row = collect_metrics(metrics_path)
    row.update(collect_trimmed_metrics(variant_dir))
    row["error"] = ""
    print(f"[run_ablation] OK   {label} ({elapsed}s): IR={row['ir']:.3f}, "
          f"trimmed_IR={row['trimmed_ir']:.3f}", flush=True)
    return row


def delta_vs_baseline(rows: list[dict], baseline_label: str) -> list[dict]:
    """Append delta_* columns. Baseline row gets all deltas = 0."""
    base = next((r for r in rows if r["label"] == baseline_label), None)
    if base is None:
        raise RuntimeError(f"baseline row '{baseline_label}' not in rows")
    for r in rows:
        for k in METRIC_KEYS + TRIMMED_KEYS:
            bv, vv = base.get(k), r.get(k)
            r[f"delta_{k}"] = (vv - bv) if (isinstance(bv, (int, float))
                                            and isinstance(vv, (int, float))) else None
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stable column order: full-window metrics first, then cutoff-trimmed.
    cols = ["label", "error"]
    for k in METRIC_KEYS:
        cols.append(k)
        cols.append(f"delta_{k}")
    for k in TRIMMED_KEYS:
        cols.append(k)
        cols.append(f"delta_{k}")
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # round floats to 6 dp for stability
            rclean = {}
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    rclean[c] = round(v, 6)
                else:
                    rclean[c] = v
            writer.writerow(rclean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-metrics",
        default="outputs/iter15_FINAL_postfix/metrics.json",
        help="Path to baseline metrics.json (added as first row of summary).",
    )
    parser.add_argument(
        "--pattern", default="variants/ablation_*.yaml",
        help="Glob for ablation variant manifests.",
    )
    parser.add_argument(
        "--out", default="outputs/ablation/summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--skip-successful", action="store_true",
        help="Skip the subprocess for any variant whose metrics.json already "
             "exists; recompute summary row from cached artifacts. Idempotent.",
    )
    args = parser.parse_args()

    baseline_metrics = ROOT / args.baseline_metrics
    if not baseline_metrics.exists():
        print(f"[run_ablation] ERROR: baseline metrics not found: {baseline_metrics}",
              file=sys.stderr)
        return 1

    manifests = sorted((ROOT / "variants").glob("ablation_*.yaml"))
    if not manifests:
        print("[run_ablation] ERROR: no ablation_*.yaml manifests found", file=sys.stderr)
        return 1

    rows: list[dict] = []
    # Baseline row first (preserves intent ordering — see step 1 file)
    base_row = collect_metrics(baseline_metrics)
    # Trimmed metrics for the baseline come from its backtest_result.pkl too.
    base_dir = baseline_metrics.parent
    base_row.update(collect_trimmed_metrics(base_dir))
    base_row["error"] = ""
    rows.append(base_row)
    baseline_label = base_row["label"]

    failures = 0
    for i, mp in enumerate(manifests, 1):
        print(f"[run_ablation] [{i}/{len(manifests)}] {mp.name}", flush=True)
        row = run_one(mp, skip_if_exists=args.skip_successful)
        if row.get("error"):
            failures += 1
        rows.append(row)

    rows = delta_vs_baseline(rows, baseline_label)
    out_path = ROOT / args.out
    write_csv(rows, out_path)
    print(f"[run_ablation] wrote {out_path} ({len(rows)} rows; {failures} failures)",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
