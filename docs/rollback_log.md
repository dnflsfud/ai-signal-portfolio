# Rollback Log — Destructive Experiments

This file archives experiments that were tested against the iter15_FINAL
baseline and rolled back because they hurt net IR, P2 IR, or both.

**Purpose**: keep `config.py` lean by removing the dead-but-documented
levers, while preserving the research history so future work does not
blindly re-try the same ideas.

**Authoritative baseline**: `outputs/iter15_FINAL/metrics.json`
(IR 1.156, P1 +1.526, P2 +0.107, P3 +1.711, turnover 115.6% two-way).

Each row: what was tried, when, headline result, root cause diagnosis.

---

## Levers deleted from `src/config.py` on 2026-04-20

All code implementing these behaviours has been deleted from `src/backtest.py`,
`src/portfolio_optimizer.py`, `src/model_trainer.py`, and `src/features/assembly.py`.
They can be reconstructed from git history if ever needed.

### 1. `quality_gate_*` — Composite boost + EPS-cut penalty (REDESIGN S, iter12)

- **Tested**: 2026-04-14, iter12.
- **Intent**: post-process prediction tweak — `+β × cs_zscore(cheap × quality × revision)`
  combined with `−γ × clip(−rev_chg_3m / 20, 0, 1)` to demote stocks with collapsing
  EPS revisions.
- **Result**: P2 IR +0.283 (intended) but P1 −0.491 + P3 −0.313 (collateral damage).
  Net IR 1.183 → 1.005 (−0.178). Went from 6 wins to 3 wins against codex on the
  9-metric comparison.
- **Root cause**: the composite boost also amplified Quality/Value alpha in regimes
  where it was already the dominant factor, over-concentrating the book.
- **Verdict**: deleted. Do not retry at composite scale. A narrower eps-cut penalty
  (no boost side) could be worth re-testing, but only as a separate experiment
  with strict P1/P3 non-degradation gates.

### 2. `mom_tilt_*` — N-month momentum tilt (iter19h, iter19j)

- **Tested**: 2026-04-17.
- **Intent**: post-prediction boost on rank of 126d (6M) risk-adjusted momentum.
- **Result**: iter19h 3M weight=0.20 IR 1.115; iter19j 6M weight=0.20 IR 1.311
  (vs iter15 1.598 at the time of the test). Pure loss on both horizons.
- **Root cause**: model already encodes momentum via `momentum_252d`, `ma_cross_*`,
  `beta_63d`. Adding post-process tilt just doubled the exposure → P2/P3 both down.
- **Verdict**: deleted. If momentum needs more weight, add it at the feature level,
  not as post-process.

### 3. `valuation_dampener_*` — Cheapness-composite subtraction (iter19f, iter19g)

- **Tested**: 2026-04-17.
- **Intent**: subtract `weight × (cheap_composite − 0.5) × 2` from predictions so
  the model's deep-value bets would be partially neutralised.
- **Result**: iter19g (weight=0.08) P1/P3 slight up, P2 −0.32 (collapse).
  Total IR 1.544 vs iter15 1.598. iter19f (weight=0.15) worse: IR 1.463.
- **Root cause**: our value-tilted names cluster in Financials/Staples — the same
  names that work in P2 rate-shock. Dampening them killed P2.
- **Verdict**: deleted.

### 4. `loss_guard_*` — Trailing-loss active dampener (iter18, iter18b)

- **Tested**: 2026-04-17.
- **Intent**: post-optimizer reduce OW weight on stocks that dropped more than
  7% in the trailing 42 days, floor 30%.
- **Result**: destructive across both layers tried (vol-adaptive layer 1 and
  trailing-loss layer 2). Killed good signals during drawdowns that later
  mean-reverted.
- **Root cause**: the "loss" signal aliases with the alpha model's own "out of
  favor → cheap → buy" logic. Cutting active during drawdowns trims exactly
  the names that would deliver the mean-reversion trade.
- **Verdict**: deleted.

### 5. `sample_weight_*` — Regime-aware LightGBM sample weights (iter16b)

- **Tested**: 2026-04-16.
- **Intent**: upweight training samples from high-stress rate environments
  (10Y and VIX z-score elevated) to give the model more "rate shock" coverage.
- **Result**: destructive. 5-year training window dominated by low-vol bull
  regime; the boost did not generalise because the "stressed" samples were
  too few and too non-IID with forward 20d targets in P3.
- **Root cause**: sample-weighting inside a regression-tree model does not
  create regime-specific sub-models; it just nudges the loss surface. Too few
  "P2-like" samples to move the tree splits meaningfully.
- **Verdict**: deleted. Regime-specific modelling needs regime-specific models
  (ensemble of regime-conditional learners), not weighted training.

### 6. `regime_gate_*`, `regime_stress_*`, `regime_active_shrink_*`, `regime_no_trade_*`, `regime_eta_*` — Macro-stress portfolio degrade-to-BM (REDESIGN X, iter17, iter21 H)

- **Tested**: 2026-04-15 (iter21), re-tested 2026-04-17 (iter17 NTB-only variant).
- **Intent**: compute macro-stress indicator from UST_10Y level + DXY, and in
  high-stress regimes shrink active budget / widen no-trade band / reduce eta.
- **Result**: all variants net-destructive on iter15 baseline.
  - iter17 full regime gate: destructive.
  - iter17d NTB-only: destructive.
  - iter21 H_regime_very_mild: −0.025 IR (closest but still negative).
- **Root cause**: by the time the macro stress indicator fires, the damage is
  already priced in. Shrinking active into BM locks in the underperformance
  and gives up the recovery leg. Also: the 10Y-level z-score fires in both
  P1 reflation and P2 hike regimes, so it does not cleanly isolate P2.
- **Verdict**: deleted. P2 fix needs a signal-layer change, not a portfolio-layer
  clamp.

### 7. `decile_funding_*` — Decile-group UW funding set (REDESIGN W, iter20, iter21)

- **Tested**: 2026-04-15.
- **Intent**: generalise mega-cap funding. Take top-50 BM names, split into 10
  BM-decile groups, and within each group pick the worst-K scoring names as the
  "funding set" with enlarged UW cap. Spread UW across Tech/Financials/Staples/
  Healthcare/Industrials instead of concentrating on Big Tech.
- **Result**: iter21 FULL verdict — W alone −0.050 IR, W+X combined −0.141 IR
  on iter15 baseline. Even with regime gating off at 0.30 / 0.45 the best variant
  only reaches the 0.42–0.43 IR range vs iter15 1.16.
- **Root cause**: the decile buckets below the mega-cap layer (rank 6-50) are
  already near-neutral in our book. Enlarging UW there creates negative active
  tilts on stocks the model is indifferent about → no alpha upside, just extra
  turnover and risk concentration in the bottom decile.
- **Verdict**: deleted. The original `mega_cap_funding_mode=True` with 4-worst
  concentration is retained because it targets a specific structural
  over-exposure (mega caps dominate BM and cannot be all simultaneously OW'd).

### 8. `mega_cap_ow_multiplier`, `mega_cap_max_weight_absolute` — BM-proportional OW cap (iter18)

- **Tested**: 2026-04-17.
- **Intent**: let high-scoring mega caps break through the uniform `max_weight=0.15`
  ceiling by scaling headroom with BM weight.
- **Result**: hurt IR. Extra mega-cap OW headroom pulled active budget away from
  small-cap alpha where the model actually had edge.
- **Root cause**: our model's strongest signals are in the tail (rank 20-50), not
  the top. Giving the top more room starves the tail.
- **Verdict**: deleted. Mega caps stay capped at `max_weight=0.15`.

### 9. `zero_weight_bottom_n` — Hard-zero the worst-N names (iter22)

- **Tested**: 2026-04-18.
- **Intent**: force-zero the N worst-scoring stocks to free active budget for OW.
- **Result**: IR 1.598 → 0.702, P2 −1.09. Catastrophic.
- **Root cause**: hard zeros violate soft MVO TE quadratic — the optimizer
  cannot offset the resulting active spike elsewhere without breaching sector
  deviation. Effectively the book lost diversification.
- **Verdict**: deleted. Use soft UW via active bounds, never hard zeros.

---

## Lever retention decisions

Everything not listed above — mega_cap_protection (4-worst funding mode),
PEAD boost, growth tilt (50/50 EPS/Sales, 0.25 weight), score-gated OW —
survives in the live baseline.

---

## Promotions

### 2026-04-22 — `exp_baseline_v2_pp_cleaned` promoted to `baseline_v3`

- **Code change**: `apply_pead_boost` + `apply_growth_tilt` in `src/backtest.py`
  switched from `data.get_sheet("Factset_*_Revision")` (raw) to
  `get_cleaned_revision(data, sheet, config=config)` — shared helper in
  `src/features/sellside.py`. Also applies to `assembly.build_growth_composites`.
- **Result artifacts**: `outputs/baseline_v3/` (snapshot of exp_baseline_v2_pp_cleaned)
- **Why promoted**: strict Pareto dominance over baseline_v2:
  - IR 1.066 vs 1.024 (+0.042)
  - P1 +1.661 vs +1.658 (preserved)
  - P2 −0.388 vs −0.471 (+0.083 recovery)
  - P3 +1.678 vs +1.640 (+0.037)
  - Sharpe 1.245 vs 1.238; TE 3.63% vs 3.64%
- **Gate pass rate**: 4/5 (baseline_v2 was 3/5). Only P2 ≥ −0.30 gate still
  fails (−0.388 is the best seen for this architecture).
- **What the fix is actually doing**: previously the feature panel saw the
  reversion_gated cleaned revision stream, but the PEAD boost (weight 0.30)
  and growth_tilt (weight 0.25) post-process overlays were reading the
  RAW revision sheets. Every rollover UP artifact the cleaner was removing
  at the feature level was being re-admitted through the overlay. The fix
  makes the two layers see the same stream.

### 2026-04-21 — Window-sweep negative results

Four revision-MA window configurations tested. All underperformed baseline_v2
on IR gate. Kept as historical record — do not retry without new evidence:

| Variant | IR | P1 | P2 | P3 | Verdict |
|---|---:|---:|---:|---:|---|
| exp_revision_ma10d (all 10d, cleaned PP) | 1.010 | +0.902 | −0.156 | +1.984 | ❌ P1 collapse (−0.755) |
| exp_revision_ma21d (all 21d, cleaned PP) | 0.906 | +1.114 | −0.552 | +2.091 | ❌ non-monotonic, worst P2 |
| exp_revision_ma_dual (10d + 63d, cleaned PP) | 0.966 | +1.464 | −0.399 | +1.687 | ❌ dual is midpoint not max |
| baseline_v3 (63d, cleaned PP) | **1.066** | +1.661 | −0.388 | +1.678 | ✅ promoted |

Lesson: the model cannot do regime-selective horizon mixing on revision MAs
in this architecture. Further window tuning is rejected. P2 recovery must
come from signal-layer work (ROADMAP Phase 2).

### 2026-04-21 — `reversion_gated` promoted to `baseline_v2`

- **Config change**: `DEFAULT_CONFIG.revision_clean_mode: "down_only" → "reversion_gated"`
- **Result artifacts**: `outputs/baseline_v2/` (snapshot of `exp_revision_reversion_gated` run)
- **Why promoted**: in the apples-to-apples A/B under the post-Phase-1-cleanup code,
  `reversion_gated` delivered IR 1.024 vs `down_only` 0.903 (+0.121 IR,
  +0.488 P3 IR, ≈ 0 turnover change). The earlier `iter15_FINAL` stored baseline
  (IR 1.156) is no longer reproducible with the current codebase (Phase 1
  dead-code removal introduced numerical drift) — so rather than measure
  candidates against an un-reproducible target, we rotated to a reproducible
  one that strictly dominates its OFF-state on the current pipeline.
- **Known cost**: P2 IR declined from iter15_FINAL's +0.107 to −0.471. This is
  a combined effect of (a) the pipeline-regression baseline drift and
  (b) `reversion_gated` removing some genuine bad-signal continuations in the
  2021-05 → 2023-10 rate-shock regime. Phase 2 (signal-layer P2 fix) now
  carries the burden of recovering this.
- **`symmetric` mode rejected**: hit user's feared failure case — masked
  genuine analyst upgrades, producing P1 IR 1.135 vs 1.763 (−0.628). Kept in
  config for sensitivity probes but not a promotion candidate.

## Policy going forward

- New experiment = new directory `outputs/iter{N}_<name>/`, never overwrite the
  top-level `outputs/`.
- A lever only enters `config.py` after it beats the baseline on the gate
  defined in [`BASELINE.md`](BASELINE.md).
- Once beaten, document in a new section of this log (pre-merge), not inline
  in `config.py` comments.
- Failed experiments older than 30 days belong in this file, not in code.
