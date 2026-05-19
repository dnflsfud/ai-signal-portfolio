# Canonical Baseline (Research)

> ## 2026-05-19 Update — research baseline rotated due to data-leakage fix
>
> **Canonical research baseline**: `iter15_FINAL_postfix`
> **Artifacts**: `outputs/iter15_FINAL_postfix/`
> **Manifest**: `variants/iter15_FINAL_postfix.yaml` (`tuning_mode: research`)
> **Environment**: `embargo_days=20` (= `forward_horizon`), `train_cutoff_date=2024-12-31`
>
> ### Why rotated
>
> Task A of `data-leakage-fix` phase removed a label-leakage path in
> `walk_forward_train`: train→val and val→predict windows had a 0-day gap
> while the target is 20-day forward return, so early-stopping selection was
> made on labels that overlapped with the validation/prediction windows.
>
> Comparison vs legacy `baseline_v4` (= `iter15_65tkr_reb21_vtg`) on the
> SAME cutoff-trimmed window (1592 days, 2018-11-26 → 2024-12-31):
>
> | Metric | legacy (trimmed) | postfix | Δ |
> |--------|-----------------:|--------:|--:|
> | **IR**                | **0.804** | **0.392** | **−0.412** |
> | Annual Return         | 26.05% | 24.58% | −1.48 p |
> | Active Return         |  2.73% |  1.26% | −1.48 p |
> | Tracking Error        |  2.93% |  2.88% | −0.05 p |
> | Sharpe                |  1.149 |  1.094 | −0.055 |
> | Max Drawdown          | −30.34% | −29.95% | +0.39 p |
> | Annual Turnover 2-way | 113.6% |  90.8% | −22.8 p |
> | Avg IC                | 0.0450 | 0.0463 | +0.0013 |
> | **P1 IR**             |  0.844 |  1.287 | **+0.444** |
> | **P2 IR**             |  0.783 | **−0.497** | **−1.280** |
> | **P3 IR** (Oct23–Dec24) |  0.780 |  0.390 | −0.389 |
>
> The legacy `baseline_v4` headline IR=1.310 was measured on the FULL window
> (through 2026-05). On the same window the *trimmed legacy* IR is 0.804 — the
> remainder (1.310 vs 0.804) is just the post-cutoff drift period. The
> embargo-induced Δ is −0.412 on the trimmed comparison.
>
> **P2 collapses to negative** under honest evaluation. Most of the
> previously-celebrated "P2 alpha" was the label leak. This is the empirical
> answer to the structural review of 2026-05-18.
>
> ### Gate criteria for future variants
>
> A new variant promotes to canonical baseline only when, evaluated under
> `tuning_mode: research` (cutoff enforced):
>
> 1. **IR ≥ current postfix IR (0.392) on the cutoff-trimmed window.**
> 2. **P2 IR ≥ postfix P2 IR − 0.10** (i.e. ≥ −0.60). No regression on
>    the regime where leakage was hiding.
> 3. **Turnover ≤ postfix turnover + 5 percentage points two-way.**
> 4. Process: all tuning under `research`; final OOS verification uses
>    `tuning_mode: oos_verify` **exactly once** per candidate; the peek
>    counter (`experiment_inventory.json.n_oos_peeks`) goes up by 1.
> 5. Statistical primary gate (rolling IR + SPA p-value) is introduced in
>    Task C step 2; until then, items 1-4 govern.
>
> Sub-period IRs (P1/P2/P3) remain **diagnostic only**; they are not
> independently gated to avoid amplifying multiple-comparison cost. Drift in
> any single sub-period is acceptable if the headline + P2 floor hold.
>
> ### deploy vs research separation
>
> - `tuning_mode: deploy` manifests (`iter15_65tkr_reb21_vtg.yaml`,
>   `iter15_FINAL.yaml`) continue to run WITHOUT cutoff for the daily
>   `update_and_deploy.bat` flow. The deployed strategy uses every available
>   day of data — that's the right behaviour for production. Research/
>   promotion decisions however must use `research` so cutoff applies.
> - `outputs/baseline_v4/` remains in place for the deploy/dashboard path.
>   The research baseline is the separate `outputs/iter15_FINAL_postfix/`.

---

# Baseline v4 — Legacy (pre-leak-fix) Reference Run

> **STATUS (2026-05-19)**: LEGACY — preserved for deploy path + historical
> comparison. **NOT** the research baseline for new variant promotion. See
> the "Canonical Baseline (Research)" section above.

**Status**: LOCKED (2026-04-24)
**Baseline label**: `baseline_v4` (originally `iter15_65tkr_reb21_vtg`)
**Artifacts root**: `outputs/baseline_v4/`
**Replaces**: `baseline_v3` (2026-04-22, `exp_baseline_v2_pp_cleaned`)
**Ancestry**: `iter15_FINAL` → `baseline_v2` → `baseline_v3` → **`baseline_v4`** → `iter15_FINAL_postfix`

---

## Why baseline was rotated to v4 on 2026-04-24

Two structural changes landed together after investigating a user concern
that "the model OWs cheap-industry-declining names (e.g. CRM) just because
margin accel is positive":

1. **Universe expansion 60 → 65 tickers** — added BAC, CSCO, INTC, ORCL, TSM
   per data refresh 2026-04-23. `data_loader.TICKERS` and
   `COMPANY_TO_TICKER` updated accordingly.
2. **Rebalance frequency 10 → 21 days** — restored DEFAULT_CONFIG value.
   iter15_FINAL's 10d override was driving turnover to 208% after the
   universe expansion. 21d cuts it to ~108% with IR improvement.
3. **Value-trap gate** (`apply_value_trap_gate` in `backtest.py`) —
   post-prediction multiplicative discount for cells matching
   `fin_pe_level_z < -0.5 & momentum_252d < -0.5 & oper_margin_accel > +0.5`.
   Empirically this profile produces −0.25%/20d fwd specific return
   (47.3% hit) and −1.99% in the P3 regime (2023–). Panel and model
   unchanged — only predictions are rescaled before MVO.

### Empirical basis for the gate (2924 days, 2015–2026)

| Profile | Obs | Mean 20d fwd α | Hit rate |
|---|---:|---:|---:|
| Cheap + Bad mom + accel>+0.5 (CRM-like) | 1,631 | **−0.25%** | **47.3%** |
| Cheap + Bad mom + accel≤+0.5 (pure value-trap) | 23,921 | +0.82% | 55.3% |
| Universe remainder | ~190k | +0.19% | 50.3% |

The accel leg flips what would otherwise be a modest mean-reversion
opportunity into a −0.25% alpha drag. The gate fires on 1.00% of cells
(1,261 / 125,837 in the 2015–2026 walk-forward).

### Two failed alternatives that preceded this

- **Variant A** (add `sent_momentum` + `sent_21d_accel` to whitelist,
  63 features total): IR 1.253 → 0.808, P1/P2/P3 all collapsed. The
  model destabilised during retrains (Degenerate model warnings
  2021-07 → 2022-07 with trees=1–4), consistent with CLAUDE.md warning
  that "the model is very sensitive to feature additions (iter 4, 5, 7,
  12, 13)."
- **Raw CLAUDE.md option of dropping `tg_upside`**: rejected by prior
  iter21 run (IR 1.119 vs iter19 1.597). Confirmed unchanged here.

Post-process approach was chosen precisely because it bypasses the
feature-space instability risk.

## Canonical metrics (source: `outputs/baseline_v4/metrics.json`)

| Metric | Value |
|---|---|
| Annual Return | 28.96% |
| Annual Vol | 22.46% |
| Sharpe Ratio | **1.289** |
| Active Return | 4.28% |
| Tracking Error | 3.26% |
| **Information Ratio** | **1.310** |
| Max Drawdown | −29.95% |
| Annual Turnover (two-way) | 109.5% |
| Annual Turnover (one-way) | 54.8% |
| Avg IC | 0.0355 |
| Annual TC | 11.0 bps |

## Sub-period IRs

| Period | IR | vs v3 | Status |
|---|---:|---:|---|
| P1 | **+1.537** | −0.124 | PASS (still strong) |
| P2 | **+0.171** | **+0.559** | **FLIPPED POSITIVE** (was −0.388) |
| P3 | **+1.911** | +0.233 | PASS |

**P2 crossing zero is the headline event**: the rate-shock regime
(2021–2023) that has been the bottleneck for every prior baseline is now
mildly positive. The value-trap gate was designed for P3 but captures
enough P2 value-trap alpha erosion that P2 flips as a side effect.

## Δ vs baseline_v3

| Metric | Δ |
|---|---:|
| IR | **+0.244** |
| Sharpe | +0.044 |
| Active return | +0.41pp |
| TE | −0.37pp |
| Turnover (two-way) | **−92.9pp** (202.4% → 109.5%) |
| P1 IR | −0.124 |
| P2 IR | **+0.559** (partial recovery, now positive) |
| P3 IR | +0.233 |

Turnover reduction is driven by the rebalance_freq change (10d → 21d),
not the gate. The IR lift combines gate effect + universe expansion +
turnover reduction.

## What changed between v3 and v4 — one paragraph

Universe grew to 65 tickers (+BAC, CSCO, INTC, ORCL, TSM). Rebalance
frequency restored to DEFAULT_CONFIG's 21 days (iter15_FINAL's 10d
override was dropped). A post-prediction value-trap gate was added in
`src/backtest.py::apply_value_trap_gate`, controlled by 5 new config
fields (`value_trap_gate_enabled`, `vtg_pe_z_threshold`,
`vtg_momentum_threshold`, `vtg_accel_threshold`, `vtg_scale`). Default
is OFF; the v4 baseline turns it ON through `variants/iter15_65tkr_reb21_vtg.yaml`.
Core feature whitelist, LightGBM hyperparameters, target engine,
portfolio optimiser, and overlay cascade (PEAD, growth_tilt) are all
unchanged.

## How to regenerate

```bash
python run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml --no-cache
# artifacts written to outputs/iter15_65tkr_reb21_vtg/
# baseline_v4/ is a byte-identical copy of that directory
```

## Definition of "beat baseline_v4"

A candidate must satisfy ALL of:

1. `information_ratio` ≥ **1.310**
2. `sub_periods.P1_ir` ≥ **+1.44** (do not lose P1 beyond −0.10 of 1.537)
3. **`sub_periods.P2_ir` ≥ +0.08** (hold or improve on flipping-positive)
4. `sub_periods.P3_ir` ≥ **+1.81** (do not lose P3 beyond −0.10)
5. `avg_annual_turnover` ≤ **1.20** (keep under +10% headroom from 1.095)

If ALL of 1–5 hold, the candidate is a promotion CANDIDATE. Final
promotion requires an independent reproduction run (re-run --no-cache
and confirm deltas hold within ±0.01 IR).

## Rejected paths (2026-04-24)

### Sentiment feature addition (Variant A)

- **Added**: `sent_momentum`, `sent_21d_accel` to `CORE_FEATURE_WHITELIST`.
- **Result**: IR 1.253 → 0.808 (−0.445). P1 −0.675, P2 −0.195, P3 −0.432.
- **Root cause**: LightGBM retrain destabilisation. 6 retrains in 2021-07
  → 2022-07 produced Degenerate models (1–4 trees). EWMA feature
  importance cold-start for the new columns was more destructive than
  the marginal value those features provided.
- **Lesson**: any sentiment intervention should be post-process, not
  a new whitelist column. See `apply_value_trap_gate` as a template.

## Historical anchors (do not confuse)

- `outputs/baseline_v3/` — replaced 2026-04-24. 60-ticker universe,
  reb 10d, no value-trap gate. Kept for historical record.
- `outputs/iter15_FINAL/` — currently holds the 65-ticker reb 10d run
  (IR 1.066) used as a pre-promotion reference. Not a true baseline;
  its label is reused only because `run_variant.py` writes there
  when given the `iter15_FINAL.yaml` manifest.
- `outputs/iter15_65tkr_reb21/` — intermediate step (universe expanded
  + reb 21d, no gate). IR 1.253. Kept for A/B forensics.
- `outputs/iter15_65tkr_reb21_sent/` — failed Variant A. IR 0.808.
  Kept as cautionary record of feature-space fragility.
- `outputs/iter15_65tkr_reb21_vtg/` — the promotion candidate.
  Byte-identical to `outputs/baseline_v4/` — this is the current
  canonical artifact directory (point the reader at `baseline_v4/`).
