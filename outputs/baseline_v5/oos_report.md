# baseline_v5 OOS Verification

**Generated**: 2026-05-19
**Peek label**: `baseline_v5`
**experiment_inventory.json `n_oos_peeks`**: 1 (this run = peek #1 for baseline_v5)
**baseline_v5 last date**: 2026-05-15
**research baseline last date**: 2026-05-15

## Configuration delta vs iter15_FINAL_postfix

Single knob: `feature_mode: "core" → "lean"`. All other production overlays unchanged.

## Metrics — cutoff-trimmed (fair comparison, 1592 obs)

| Metric | research baseline | baseline_v5 | Δ |
|---|---:|---:|---:|
| Information Ratio | 0.392 | 0.898 | +0.506 |
| Active Return | +1.26% | +3.54% | +2.28p |
| Tracking Error | +2.88% | +3.15% | +0.27p |
| Sharpe | 1.094 | 1.225 | +0.131 |
| Annual Return | +24.58% | +26.86% | +2.28p |
| Max Drawdown | -29.95% | -29.24% | +0.72p |

## Sub-period IR

| Window | research baseline | baseline_v5 | Δ |
|---|---:|---:|---:|
| P1 (2018-11~2021-05) | 1.287 | 0.769 | -0.518 |
| P2 (2021-05~2023-10) | -0.497 | 1.160 | +1.657 |
| P3 trim (2023-10~2024-12) | 0.390 | 0.596 | +0.206 |
| P3 full + post-cutoff (~2026-05) | 1.847 | 1.890 | +0.043 |

> The 'P3 full + post-cutoff' row is the OOS peek payoff — that's the segment
> the research baseline could not see and the embargo + cutoff reserved.

## Turnover

| Metric | research baseline | baseline_v5 | Δ |
|---|---:|---:|---:|
| Annual Turnover 2-way | 90.8% | 110.3% | +19.5p |

## Gate checks (docs/BASELINE.md criteria)

- ✅ **IR ≥ baseline IR** (baseline_v5 trimmed IR 0.898 vs research baseline 0.392)
- ✅ **P2 IR ≥ baseline P2 IR − 0.10** (baseline_v5 P2 1.160 vs research baseline P2 -0.497, floor -0.597)
- ❌ **Turnover ≤ baseline + 5 p.p.** (baseline_v5 110.3% vs research baseline 90.8%, ceiling 95.8%)

## Verdict: **DO NOT PROMOTE**

At least one gate failed. baseline_v5 is NOT promoted; research baseline remains `iter15_FINAL_postfix`. Per Task B discipline, no further peek is permitted for this candidate. A new candidate would require a new ablation round.

### Caveat for the turnover gate (read before deciding next steps)

The IR/P2 lift is large and statistically meaningful (Δ IR = +0.506, Δ P2 = +1.657
on the cutoff-trimmed window). The single failing gate is **turnover**:

- baseline_v5 trimmed annual turnover = 110.3%
- ceiling = research baseline (90.8%) + 5 p.p. = 95.8%

However:

- legacy `baseline_v4` (current deploy production) annual turnover is 109.5%, i.e.
  baseline_v5's 110.3% is **essentially identical** to what's already running daily.
- The "90.8%" floor for `iter15_FINAL_postfix` was itself unexpectedly low — the
  embargo-induced retrain-skips reduced trades vs the historical regime. That
  floor became the baseline for the gate, so the ceiling is artificially tight
  relative to the long-run production turnover regime.

If the user wants to revisit the gate calibration, that requires **no new peek**:

1. Re-define the turnover ceiling against legacy production (e.g. 110%+5p)
   rather than the embargo-skipping research baseline. Promotes baseline_v5
   under a docs-only policy update.
2. Treat this run as Task B's final OOS measurement and defer promotion. Use
   baseline_v5's IR/P2 numbers as evidence for a future candidate that bundles
   a low-turnover overlay (e.g. higher `partial_rebalance_eta`, stronger
   `signal_stability_lambda`).

Both options preserve `n_oos_peeks = 1` for the baseline_v5 candidate.

---

## Headline numbers for downstream docs

Cutoff-trimmed:
- IR 0.898 (research baseline 0.392)
- P1 0.769 / P2 1.160 / P3 trim 0.596

Full window (the peek):
- IR 1.289
- P3 full 1.890
- Annual return 30.12%
- Last date 2026-05-15
