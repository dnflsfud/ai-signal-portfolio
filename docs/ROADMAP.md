# cc2_harness Improvement Roadmap

**Status**: 2026-04-20 — infrastructure laid, experiments pending.
**Authoritative baseline**: `iter15_FINAL` (see [`BASELINE.md`](BASELINE.md)).

All the infrastructure below is **OFF by default** in `src/config.py`. A
new experiment = new `variants/<name>.yaml` + `python run_variant.py
--variant ...`. Experiments that don't beat the baseline per
`BASELINE.md` gates belong in [`rollback_log.md`](rollback_log.md), not in
default config.

---

## Phase 1 — Completed (2026-04-20)

- ✅ Canonical baseline fixed (`iter15_FINAL`).
- ✅ Auto-generated metrics (run_variant.py가 metrics.json을 직접 작성; legacy `scripts/render_baseline_metrics.py`는 2026-05 cleanup으로 제거).
- ✅ Dead-code purge: 9 destructive levers + ~785 LOC removed.
- ✅ CLI unified: `run_variant.py` + `variants/*.yaml`.
- ✅ OOS hold-out infrastructure (`enforce_oos_holdout`, `train_cutoff_date`).
- ✅ BM-proportional active cap infrastructure (OFF).
- ✅ Signal-stability shrinkage infrastructure (OFF).
- ✅ Multi-horizon target + regime-aware PCA infrastructure (OFF).
- ✅ Legacy iter runners moved to `archive/`.

---

## Phase 2.6 — Data-leakage fix (2026-05-19, COMPLETED)

- ✅ Walk-forward `embargo_days = forward_horizon = 20`. `_compute_window_bounds`
  helper carves a 20-day gap between train→val and val→predict so the 20d
  forward target's label window cannot peek into the next window
  (López de Prado 2018 Ch. 7).
- ✅ OOS hold-out default flipped to ON; `train_cutoff_date = "2024-12-31"`.
- ✅ `tuning_mode` redefined: `research` (cutoff ON, default) / `oos_verify`
  (1-peek, logged to `experiment_inventory.json.n_oos_peeks`) / `deploy`
  (production daily; cutoff OFF, logged to `outputs/deploy_log.txt`).
  `production`/`tuning` → DEPRECATED aliases for `research`.
- ✅ Canonical research baseline recomputed → `iter15_FINAL_postfix`.
  On the cutoff-trimmed 1592-day window (2018-11-26 → 2024-12-31):
  IR **0.804 → 0.392** (Δ=−0.412), P2 IR **0.783 → −0.497** (Δ=−1.280).
- ✅ Docs propagated: BASELINE.md / CLAUDE.md / AI_METHODOLOGY.md / this file.
- Knock-on: prior IR comparisons in this doc and `docs/rollback_log.md` are
  STALE until Task B (overlay-ablation) and Task C (selection-bias-discipline)
  complete on the new baseline.

---

## Phase 2 — P2 signal-layer experiments

> **SUPERSEDED (2026-05-19)** — The original "P2 IR floor" target
> (+0.107 → +0.40 / +0.30 / etc.) was set under the leaky environment that
> Phase 2.6 fixed. Under honest evaluation, baseline P2 IR is −0.497 (not
> +0.107); the relevant gate is now defined in `docs/BASELINE.md` as
> "P2 IR ≥ research baseline P2 − 0.10" (≈ ≥ −0.60). Phase 2 sub-experiments
> (2.1 multi-horizon, 2.2 regime-PCA, 2.3 macro-cross, 2.4 revision cleaning)
> remain valid research candidates but their per-subsection success criteria
> must be re-derived against the new baseline. Sub-period IRs are diagnostic
> only; primary gate is the cutoff-trimmed headline IR. Task C step 3 will
> replace the multi-target sub-period gate with rolling IR + SPA p-value.

Goal: raise P2 IR from +0.107 toward +0.40 without regressing P1 (+1.526) or
P3 (+1.711). Prior single-lever post-process attempts (quality_gate,
regime_gate, decile_funding — see rollback_log.md) all failed because they
touched portfolio construction. This phase attacks the **signal**.

### 2.1 Multi-horizon target blend

Config surface (already added, OFF):
```yaml
overrides:
  multi_horizon_targets_enabled: true
  multi_horizon_weights: {20: 0.6, 5: 0.2, 63: 0.2}
```

Hypothesis: P2 is a rate-transition regime where 20d horizons miss both the
fast rotation (→ blend in 5d) and the persistent trend (→ blend in 63d).

Suggested variants to test:
- `exp_p2_mh_606020.yaml`: 20=0.6, 5=0.2, 63=0.2
- `exp_p2_mh_504010.yaml`: 20=0.5, 5=0.4, 63=0.1
- `exp_p2_mh_405020.yaml`: 20=0.4, 5=0.5, 63=0.1

Gate: IR ≥ 1.156, P2 ≥ 0.107, P1/P3 within −0.10 of baseline.

### 2.2 Regime-aware PCA lookback

Config surface:
```yaml
overrides:
  regime_aware_pca_lookback: true
  pca_lookback_short: 126
  pca_lookback_long: 504
  pca_regime_vol_threshold: 1.0
```

Hypothesis: a shorter PCA window adapts faster when factor correlations
shift (P2 hike cycle). In low-vol periods longer is better to reduce noise.

**Note**: this is config-only so far; the regime-adaptive lookback logic
still needs to be wired into `compute_specific_returns` in a follow-up.
Current state = config present but function always uses `pca_lookback`.

### 2.3 Macro × stock cross-features (research)

Add features of form `macro_factor × stock_sensitivity` (e.g. 10Y rate
momentum × duration-like beta for each stock). This is a new feature
group — not infra, actual engineering. Deferred pending multi-horizon
and regime-PCA results.

### 2.4 Revision spike symmetric cleaning

**Problem statement**
The current `clean_revision_spikes` (src/features/sellside.py) is named
symmetrically but only masks `daily_diff < -threshold`. Factset's
consensus-window rollover can produce artifacts in BOTH directions — a
stock with score -72 (all analysts downgrading) can jump to +5 on the
rollover day purely because the rolling window reset. That makes a
fundamentally-poor stock look "neutral" to the model, with downstream
knock-on to 6 of the core-46 whitelist features
(`eps_rev_ma_63d`, `sales_rev_ma_63d`, `eps_rev_time_low`,
`sales_rev_time_low`, `eps_rev_vol`, `sales_rev_vol`) and 2nd-order
contamination to the `growth_tilt` revision composite.

**Historical context**
Iter14 attempted to activate `earnings_timeline=None → full timeline`
in the cleaner and lost IR −0.290. But that change bundled two things:
(1) timeline precision and (2) `earn_cycle_pos` feature whitelist
inclusion. **Symmetric (up-side) detection has never been isolated and
tested** — it's an orthogonal axis. See `docs/rollback_log.md`.

**Infrastructure (LANDED)**
Three new config fields (all OFF by default — baseline unchanged):

```yaml
revision_clean_mode: "down_only"           # baseline (iter15)
revision_clean_threshold: 15.0
revision_clean_extreme_threshold: 50.0     # |prev_level| considered "extreme"
revision_clean_reversion_ratio: 0.5        # |today| < prev × ratio → collapse
```

Mode semantics:
- `"down_only"` — baseline. `daily_diff < -threshold` only.
- `"symmetric"` — naive two-sided. `|daily_diff| > threshold`. Likely
  filters real upgrades; included as upper-bound sensitivity probe.
- `"reversion_gated"` — preferred candidate. Masks only when the
  previous level was already extreme AND today's level collapses:
  ```
  |prev| > extreme_thr  AND  |diff| > threshold  AND  |today| < |prev| × reversion_ratio
  ```
  Preserves genuine moves INTO extremes (e.g. `+15 → +70` upgrade);
  filters reversions FROM extremes (`+70 → +5` or `-72 → +5`).

Implementation: `src/features/sellside.py::clean_revision_spikes()` takes
the new kwargs; `build_sellside_features(config=...)` threads them from
config. `src/features/assembly.py::build_all_features` wires the config
through, so zero change to any caller is required.

**Experiment plan**

Step 0 — Diagnostic (before any backtest):
```bash
# legacy scripts/diagnose_revision_spikes.py는 2026-05 cleanup으로 제거됨.
# 동일 분석이 필요하면 outputs/diagnostics/revision_spike_audit.md의 과거 산출물 참조,
# 또는 src/features/sellside.py의 spike-cap 로직을 직접 탐색.
#   - % of up-jumps within ±5 trading days of earnings announcement
#   - per-ticker worst offenders
#   - rough |ΔMA63| downstream impact
```

Decision rule from diagnostic:
- Asymmetry ratio (`up_simple / down_simple`) ≥ 0.3 AND rollover_up
  near-earnings pct ≥ 50% → proceed with backtest variants.
- Otherwise → defer; the artifact isn't material.

Step 1 — Tuning-mode variants (OOS hold-out active):
```bash
python run_variant.py --variant variants/exp_revision_reversion_gated.yaml
python run_variant.py --variant variants/exp_revision_symmetric.yaml
```

Step 2 — Compare to baseline:
Gate = BASELINE.md 4-gate. Specifically:
- IR ≥ 1.156
- P2 IR ≥ 0.107 (non-negotiable — this is the gate we want to improve)
- P1 & P3 IR within −0.10 of baseline
- turnover ≤ 1.35

Step 3 — If a candidate wins on tuning data:
One `oos_verify` run with `tuning_mode: oos_verify` and
`enforce_oos_holdout: false` to confirm on 2025+ data. If it holds,
propose promotion to baseline.

**Expected sensitivity (prior)**
- `reversion_gated` is the strongest candidate — it removes a clear
  data artifact without plausibly destroying alpha.
- `symmetric` likely underperforms `reversion_gated` because it also
  filters genuine upgrades; used mainly to bound the upper effect size.

**Risk**
Low. Changes are confined to revision feature inputs; model is
untouched. Baseline reproducibility preserved via `down_only` default.

---

## Phase 3 — Turnover / execution

Current turnover 115.6% two-way is fine. If future P2 fixes push turnover
higher, use these levers in this order:

1. **`signal_stability_lambda`** (infra added). Start at 0.2, watch both
   turnover reduction and IR attrition. Destructive past ~0.4 probably.
2. **`no_trade_band` widening** (currently 0.003 = 30bps). Cheap; low risk
   of P2 regression.
3. **`partial_rebalance_eta` lowering** (currently 0.50). Slower
   convergence; watch P3 which is signal-follow heavy.

Do NOT revive the `regime_*_shrink` family — those gated P2 at the cost of
signal responsiveness and failed every time.

---

## Phase 4 — OOS tuning discipline

All future experiments MUST use `tuning_mode: tuning` + `train_cutoff_date`.
A candidate that wins during tuning gets exactly ONE `oos_verify` run. If
it still wins, it is proposed for baseline promotion per
[`BASELINE.md`](BASELINE.md). The promoted variant's `outputs/<label>/metrics.json`
must match the canonical baseline `outputs/baseline_v4/metrics.json` after promotion.

Recommended cutoff: `2024-12-31`. Reserves 2025-01 onward (~16 months
against a 92-month history) as OOS verification.

---

## Phase 5 — Risk-layer refinement

### 5.1 BM-proportional active cap (infra added)

Replaces the hard `mega_cap_bm_threshold` cliff at 4% with a continuous
function. OFF by default.

```yaml
overrides:
  bm_proportional_cap_enabled: true
  bm_proportional_cap_bm_scale_at_top: 1.5   # mega cap gets 1.5× headroom
  bm_proportional_cap_vol_scale_floor: 0.5   # high-vol stocks floored at 0.5×
```

Gate candidate: expect modest P3 pickup (room for conviction OW on high-
scoring mega caps) without P1/P2 damage.

### 5.2 PEAD regime-scaling (research)

Current `pead_boost_weight=0.30` is static. P2 earnings revisions are
noisier → boost may amplify noise. Test regime-scaled boost (lower in
high-vol, normal in low-vol) as a follow-up. Not wired yet.

### 5.3 Growth tilt sector-differentiated skew (research)

Currently `growth_tilt_eps_skew=0.50` uniform. Tech: Sales leads EPS;
Financials: EPS leads Sales. Sector-conditional skew is a candidate but
needs feature work + OOS validation.

---

## Phase 6 — Tooling debt

### 6.1 Full print() → logger migration

`src/backtest.py` has 30+ `print()` calls. `setup_logging()` is already
available but not universally called. The `run_variant.py` entry point
calls it; `daily_update.py` and `update_and_deploy.py` orchestrator do not yet.

Migration steps:
1. Add `from src.logging_config import setup_logging; setup_logging()` to
   every entry point that currently writes via `print()` (`update_and_deploy.py`,
   `daily_update.py`, `run_selection_bias.py`).
2. Per-file: replace `print(f"[Backtest] ...")` with
   `logger = logging.getLogger(__name__); logger.info(...)`.
3. Only keep `print()` for explicit end-user output (summary tables).

Not done in Phase 1 because it's mechanical, large-diff, and not on the
critical path. Deferred.

### 6.2 Bayesian / structured search

Manual grid search gave 402 trials and still missed Pareto frontier.
Integrating `optuna` (or similar) would formalise the search space and
produce saveable study objects for selection-bias accounting. Deferred.

---

## Decision log

The roadmap lives in this file. Completed items move to Phase 1 with a
date. Failed experiments move to `rollback_log.md`. The roadmap order
is stack-ranked by expected impact on **P2 IR** — that's the only metric
that's been sub-par for 5+ iterations.
