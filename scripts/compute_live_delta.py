"""Compute L1 drift between live target snapshots and the latest backtest's
portfolio_weights for the same dates.

Used by update_and_deploy.py / daily_update.py as the live-vs-backtest drift
monitor (final-v1-promotion step 4, 2026-05-19 v2).

Output: outputs/live_delta_log.csv with columns
  asof, n_tickers, l1_drift, max_drift_ticker, max_drift_value, n_missing_tickers

Reads:
  outputs/live_log/*.csv             (one file per business day)
  outputs/baseline_v4/backtest_result.pkl  (latest production backtest)

Behaviour:
  - Append-only: existing rows in outputs/live_delta_log.csv are preserved.
  - For any asof that has a live snapshot but no log row, append.
  - L1 drift: sum_i |w_live[i] - w_bt[i]| on union of tickers (missing = 0).
  - Threshold warning: when the most recent row's l1_drift > 0.10, print
    "[LIVE-DELTA WARN] asof=... l1_drift=... exceeds 0.10".
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIVE_DIR = ROOT / "outputs" / "live_log"
BT_PKL = ROOT / "outputs" / "baseline_v4" / "backtest_result.pkl"
LOG_PATH = ROOT / "outputs" / "live_delta_log.csv"

DRIFT_THRESHOLD = 0.10
LOG_COLUMNS = ["asof", "n_tickers", "l1_drift",
               "max_drift_ticker", "max_drift_value", "n_missing_tickers"]


def _load_existing_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        try:
            df = pd.read_csv(LOG_PATH)
            # Normalise asof to YYYY-MM-DD string for stable equality.
            df["asof"] = df["asof"].astype(str)
            return df
        except Exception as e:
            print(f"  [warn] could not read existing log ({e}); will overwrite.")
    return pd.DataFrame(columns=LOG_COLUMNS)


def _load_backtest_weights() -> dict:
    if not BT_PKL.exists():
        return {}
    with BT_PKL.open("rb") as fh:
        r = pickle.load(fh)
    pw = getattr(r, "portfolio_weights", {}) or {}
    return {pd.Timestamp(k).strftime("%Y-%m-%d"): pd.Series(v) for k, v in pw.items()}


def _nearest_bt_date(asof_str: str, bt_dates_sorted: list) -> str | None:
    """Find largest bt_date <= asof. Returns None if no such date."""
    asof = pd.Timestamp(asof_str)
    valid = [d for d in bt_dates_sorted if pd.Timestamp(d) <= asof]
    return valid[-1] if valid else None


def main() -> int:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log = _load_existing_log()
    bt_weights = _load_backtest_weights()
    bt_dates_sorted = sorted(bt_weights.keys())

    snapshots = sorted(LIVE_DIR.glob("*.csv"))
    if not snapshots:
        print(f"  [compute_live_delta] no live snapshots in {LIVE_DIR.relative_to(ROOT)}")
        # Still emit empty log so AC can grep.
        if not LOG_PATH.exists():
            log.to_csv(LOG_PATH, index=False, encoding="utf-8")
        return 0

    existing_asofs = set(log["asof"].astype(str)) if len(log) else set()
    new_rows = []

    for snap in snapshots:
        asof = snap.stem  # YYYY-MM-DD
        if asof in existing_asofs:
            continue
        try:
            live_df = pd.read_csv(snap)
            live = pd.Series(live_df["target_weight"].values, index=live_df["ticker"]).astype(float)
        except Exception as e:
            print(f"  [warn] could not parse {snap.name}: {e}")
            continue

        bt_date = _nearest_bt_date(asof, bt_dates_sorted)
        if bt_date is None:
            row = {"asof": asof, "n_tickers": len(live),
                   "l1_drift": float("nan"),
                   "max_drift_ticker": "", "max_drift_value": float("nan"),
                   "n_missing_tickers": len(live)}
        else:
            bt = bt_weights[bt_date].astype(float)
            all_tickers = sorted(set(live.index) | set(bt.index))
            live_a = live.reindex(all_tickers).fillna(0.0)
            bt_a = bt.reindex(all_tickers).fillna(0.0)
            diff = (live_a - bt_a).abs()
            l1 = float(diff.sum())
            max_tkr = diff.idxmax() if len(diff) else ""
            row = {"asof": asof, "n_tickers": len(all_tickers),
                   "l1_drift": l1,
                   "max_drift_ticker": str(max_tkr),
                   "max_drift_value": float(diff.max() if len(diff) else 0.0),
                   "n_missing_tickers": int(
                       (~live.index.isin(bt.index)).sum() + (~bt.index.isin(live.index)).sum()
                   )}
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
        log = pd.concat([log, new_df], ignore_index=True)
        log = log.drop_duplicates(subset=["asof"], keep="last")
        log = log.sort_values("asof").reset_index(drop=True)
        log.to_csv(LOG_PATH, index=False, encoding="utf-8")
        print(f"  [compute_live_delta] appended {len(new_rows)} rows. total={len(log)}")
    else:
        # Ensure file exists for AC.
        if not LOG_PATH.exists():
            log.to_csv(LOG_PATH, index=False, encoding="utf-8")

    # Threshold check on most-recent row
    if len(log) and "l1_drift" in log.columns:
        recent = log.iloc[-1]
        drift = recent["l1_drift"]
        if pd.notna(drift) and float(drift) > DRIFT_THRESHOLD:
            print(f"[LIVE-DELTA WARN] asof={recent['asof']} l1_drift={drift:.4f} "
                  f"exceeds {DRIFT_THRESHOLD:.2f} threshold. Investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
