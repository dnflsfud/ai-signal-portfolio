"""Print baseline_v5 portfolio summary + last-month OW snapshot."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABEL = "baseline_v5"
LOOKBACK_DAYS = 21  # ~1 month of business days


def fmt_pct(v, places=2):
    return f"{v*100:+.{places}f}%" if isinstance(v, (int, float)) and v == v else "-"


def main():
    pkl = ROOT / "outputs" / LABEL / "backtest_result.pkl"
    with pkl.open("rb") as fh:
        r = pickle.load(fh)
    m = json.loads((ROOT / "outputs" / LABEL / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    port_w = r.portfolio_weights      # dict {rebal_date: pd.Series(ticker->weight)}
    # Some BacktestResult variants expose daily_weights instead - check both
    daily_w = getattr(r, "daily_weights", None)
    bm_curr = getattr(r, "bm_weights_history", None)  # benchmark per rebal

    # Reconstruct active weights snapshots at recent rebal dates.
    # Use the rebalance dates we have in portfolio_weights.
    rebal_dates = sorted(port_w.keys())
    last_date = rebal_dates[-1]
    cutoff_start = last_date - pd.Timedelta(days=int(LOOKBACK_DAYS * 1.5))  # ~21 business ~ 30 calendar
    recent_rebals = [d for d in rebal_dates if d >= cutoff_start]

    print("=" * 72)
    print(f"  Portfolio: {LABEL}")
    print(f"  Window: {r.portfolio_returns.dropna().index[0].date()} -> "
          f"{r.portfolio_returns.dropna().index[-1].date()}")
    print(f"  (embargo_days=20, train_cutoff_date=2024-12-31, feature_mode=lean,")
    print(f"   tuning_mode=oos_verify - full-window incl. post-cutoff peek)")
    print("=" * 72)

    print()
    print("  --- Full-window metrics (from metrics.json) ---")
    print(f"  Annual Return : {fmt_pct(m.get('annual_return'))}")
    print(f"  Active Return : {fmt_pct(m.get('active_return'))}")
    print(f"  Annual Vol    : {fmt_pct(m.get('annual_vol'))}")
    print(f"  Tracking Err  : {fmt_pct(m.get('tracking_error'))}")
    print(f"  Sharpe        : {m.get('sharpe_ratio', float('nan')):.3f}")
    print(f"  Information R : {m.get('information_ratio', float('nan')):.3f}")
    print(f"  Max Drawdown  : {fmt_pct(m.get('max_drawdown'))}")
    print(f"  Turnover 2-way: {fmt_pct(m.get('avg_annual_turnover'), 1)}")
    print(f"  Avg IC        : {m.get('avg_ic', float('nan')):.4f}")
    print(f"  rolling_ir mean / min / pos_frac : "
          f"{m.get('rolling_ir_mean', float('nan')):.3f} / "
          f"{m.get('rolling_ir_min', float('nan')):.3f} / "
          f"{m.get('rolling_ir_pos_frac', float('nan')):.2f}")
    print(f"  SPA p-value   : {m.get('spa_pvalue', float('nan')):.4f}")

    sp = m.get("sub_periods") or {}
    if sp:
        print()
        print("  --- Sub-period IR (diagnostic - NOT promotion gate) ---")
        print(f"  P1 (2018-11~2021-05) : {sp.get('P1_ir', float('nan')):.3f}")
        print(f"  P2 (2021-05~2023-10) : {sp.get('P2_ir', float('nan')):.3f}")
        print(f"  P3 (2023-10~2026-04) : {sp.get('P3_ir', float('nan')):.3f}")

    # SPX context
    if "spx_annual_return" in m:
        print()
        print("  --- S&P 500 reference (same window) ---")
        print(f"  SPX Annual Ret: {fmt_pct(m['spx_annual_return'])}")
        print(f"  SPX Sharpe    : {m.get('spx_sharpe', float('nan')):.3f}")

    # Latest rebal snapshot
    print()
    print("=" * 72)
    print(f"  Last rebalance date: {last_date.date()}")
    print(f"  Recent rebalances within ~30 calendar days: {len(recent_rebals)}")
    for d in recent_rebals:
        print(f"    - {d.date()}")
    print("=" * 72)

    # OW snapshot at last rebal date - reconstruct cap-weighted BM from data
    last_port = port_w[last_date]
    from src.data_loader import UniverseData
    from src.config import DEFAULT_CONFIG
    data = UniverseData(DEFAULT_CONFIG.data_path)

    def cap_weighted_bm(t: pd.Timestamp, tickers) -> pd.Series:
        mc = data.market_cap.reindex(columns=list(tickers)).ffill()
        # As-of lookup
        asof = mc.index[mc.index <= t]
        row = mc.loc[asof[-1]] if len(asof) else pd.Series(np.nan, index=tickers)
        row = row.where(np.isfinite(row) & (row > 0))
        s = row.sum()
        if not np.isfinite(s) or s <= 0:
            return pd.Series(1.0 / len(tickers), index=tickers)
        return row / s

    last_bm = cap_weighted_bm(last_date, last_port.index).reindex(last_port.index).fillna(0)
    active = (last_port - last_bm).sort_values(ascending=False)

    print()
    print(f"  --- TOP 10 OW (active = port - bm) on {last_date.date()} ---")
    print(f"  {'Ticker':<8}{'Port w':>10}{'BM w':>10}{'Active':>10}")
    for i, (tkr, val) in enumerate(active.head(10).items()):
        pw = last_port.get(tkr, float("nan"))
        bw = last_bm.get(tkr, float("nan"))
        print(f"  {tkr:<8}{pw*100:>9.2f}%{bw*100:>9.2f}%{val*100:>+9.2f}%")

    print()
    print(f"  --- BOTTOM 10 UW on {last_date.date()} ---")
    for i, (tkr, val) in enumerate(active.tail(10).items()):
        pw = last_port.get(tkr, float("nan"))
        bw = last_bm.get(tkr, float("nan"))
        print(f"  {tkr:<8}{pw*100:>9.2f}%{bw*100:>9.2f}%{val*100:>+9.2f}%")

    # Average OW over the last month across recent_rebals snapshots
    if len(recent_rebals) > 1:
        actives = []
        for d in recent_rebals:
            if d in port_w:
                pv = port_w[d]
                bv = cap_weighted_bm(d, pv.index).reindex(pv.index).fillna(0)
                actives.append(pv - bv)
        if actives:
            avg_active = pd.concat(actives, axis=1).mean(axis=1).sort_values(ascending=False)
            print()
            print(f"  --- AVERAGE TOP 10 OW over last {len(actives)} rebalances ---")
            print(f"  {'Ticker':<8}{'Avg Active':>12}")
            for tkr, val in avg_active.head(10).items():
                print(f"  {tkr:<8}{val*100:>+11.2f}%")
            print()
            print(f"  --- AVERAGE BOTTOM 5 UW over last {len(actives)} rebalances ---")
            for tkr, val in avg_active.tail(5).items():
                print(f"  {tkr:<8}{val*100:>+11.2f}%")


if __name__ == "__main__":
    main()
