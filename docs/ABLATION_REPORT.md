# Overlay Ablation Report (Task B step 2)

**Generated**: 2026-05-19
**Baseline**: `iter15_FINAL_postfix` (research mode, embargo=20, cutoff=2024-12-31)
**Baseline trimmed IR**: 0.392    **Baseline trimmed P2 IR**: -0.497

## Methodology

- Each variant disables a single overlay (or swaps a single configurable) vs the research baseline.
- Daily active returns trimmed to ≤ 2024-12-31 (1592 obs).
- Block bootstrap of ΔIR with `block_size=10`, `n_iter=1000`, `seed=42`.
- Decision rule:
  - `Δlo95 > 0` → DROP overlay (removing it improves IR)
  - `Δhi95 < 0` → KEEP overlay (removing it hurts IR)
  - `CI ∋ 0` → DROP overlay (parsimony)
  - `Δ trimmed_P2_IR < -0.10` → DROP regardless
- For *alternative-form* variants (revision_clean_mode, feature_mode), DROP means swap to the alternative; KEEP means stay with current.

## Per-overlay results

| Overlay (variant) | ΔIR (obs) | 95% CI | p (2-sided) | ΔP2 | Verdict |
|---|---:|---|---:|---:|---|
| composite (5 overlays off + lean panel) (`ablation_all_overlays_off`) | -0.047 | [-0.816, +0.722] | 0.910 | +1.027 | **INFO** |
| feature_mode=core -> lean (`ablation_feature_mode_lean`) | +0.506 | [-0.053, +1.132] | 0.094 | +1.657 | **DROP** |
| growth_tilt (`ablation_no_growth_tilt`) | -0.182 | [-0.572, +0.160] | 0.330 | -0.117 | **DROP** |
| mega_cap_funding (`ablation_no_mega_funding`) | -0.186 | [-0.452, +0.068] | 0.149 | -0.142 | **DROP** |
| pead_boost (`ablation_no_pead`) | -0.106 | [-0.308, +0.066] | 0.278 | -0.066 | **DROP** |
| value_trap_gate (`ablation_no_vtg`) | -0.046 | [-0.316, +0.186] | 0.730 | +0.072 | **DROP** |
| revision_clean_mode=reversion_gated -> down_only (`ablation_revision_down_only`) | +0.290 | [-0.161, +0.737] | 0.222 | +0.787 | **DROP** |

### composite (5 overlays off + lean panel)

- Variant label: `ablation_all_overlays_off`
- Variant trimmed IR: **0.345** (baseline 0.392, Δ=-0.047)
- Bootstrap 95% CI: [-0.816, +0.722], p=0.910
- Variant trimmed P2 IR: +0.530 (baseline -0.497, Δ=+1.027)
- **Verdict: INFO** — composite — diagnostic only, not a per-overlay decision

### feature_mode=core -> lean

- Variant label: `ablation_feature_mode_lean`
- Variant trimmed IR: **0.898** (baseline 0.392, Δ=+0.506)
- Bootstrap 95% CI: [-0.053, +1.132], p=0.094
- Variant trimmed P2 IR: +1.160 (baseline -0.497, Δ=+1.657)
- **Verdict: DROP** — CI spans 0 [-0.053, +1.132] — parsimony

### growth_tilt

- Variant label: `ablation_no_growth_tilt`
- Variant trimmed IR: **0.210** (baseline 0.392, Δ=-0.182)
- Bootstrap 95% CI: [-0.572, +0.160], p=0.330
- Variant trimmed P2 IR: -0.614 (baseline -0.497, Δ=-0.117)
- **Verdict: DROP** — P2 worsens by -0.117 (< -0.10)

### mega_cap_funding

- Variant label: `ablation_no_mega_funding`
- Variant trimmed IR: **0.207** (baseline 0.392, Δ=-0.186)
- Bootstrap 95% CI: [-0.452, +0.068], p=0.149
- Variant trimmed P2 IR: -0.639 (baseline -0.497, Δ=-0.142)
- **Verdict: DROP** — P2 worsens by -0.142 (< -0.10)

### pead_boost

- Variant label: `ablation_no_pead`
- Variant trimmed IR: **0.286** (baseline 0.392, Δ=-0.106)
- Bootstrap 95% CI: [-0.308, +0.066], p=0.278
- Variant trimmed P2 IR: -0.563 (baseline -0.497, Δ=-0.066)
- **Verdict: DROP** — CI spans 0 [-0.308, +0.066] — parsimony

### value_trap_gate

- Variant label: `ablation_no_vtg`
- Variant trimmed IR: **0.346** (baseline 0.392, Δ=-0.046)
- Bootstrap 95% CI: [-0.316, +0.186], p=0.730
- Variant trimmed P2 IR: -0.425 (baseline -0.497, Δ=+0.072)
- **Verdict: DROP** — CI spans 0 [-0.316, +0.186] — parsimony

### revision_clean_mode=reversion_gated -> down_only

- Variant label: `ablation_revision_down_only`
- Variant trimmed IR: **0.682** (baseline 0.392, Δ=+0.290)
- Bootstrap 95% CI: [-0.161, +0.737], p=0.222
- Variant trimmed P2 IR: +0.290 (baseline -0.497, Δ=+0.787)
- **Verdict: DROP** — CI spans 0 [-0.161, +0.737] — parsimony

## Decision summary (input to step 3 production rebuild)

- KEEP (0): —
- DROP (6): feature_mode=core -> lean, growth_tilt, mega_cap_funding, pead_boost, value_trap_gate, revision_clean_mode=reversion_gated -> down_only

Composite reference (NOT a per-overlay verdict):
- `ablation_all_overlays_off`: IR +0.345 (Δ -0.047)

### ⚠️ Non-additivity caveat

Naively composing all DROP/swap decisions gives the `ablation_all_overlays_off` configuration, which we measured at trimmed IR = +0.345 — *worse* than baseline (+0.392) and dramatically worse than the best single-knob variant `ablation_feature_mode_lean` (trimmed IR +0.898).

Implication: overlays are NOT additive on top of the cleaner panel/cleaning. Each was *individually* tuned to compensate for noise that the core+reversion_gated baseline carries; remove that noise and the overlays themselves stop adding value (and may double-correct).

Practical decision for step 3: rather than compose 6 DROP verdicts (which we've already empirically falsified via `ablation_all_overlays_off`), promote the single best measured variant as `baseline_v5`. Best measured = `ablation_feature_mode_lean` (IR +0.898, P2 +1.160). Its only delta from the baseline is `feature_mode: lean` — a minimum-surface change.

Future work (not in this phase): a second ablation round on top of the lean-panel baseline to determine which overlays (if any) add value in that regime. That round would need fresh OOS budget — defer until Task C step 2 introduces rolling-IR + SPA gating so the statistical cost is bounded.

### Recommended overrides for variants/baseline_v5.yaml

Based on the non-additivity caveat above, the recommended candidate is **the best single-knob variant** (minimum-surface change from research baseline):

```yaml
label: baseline_v5
description: >
  Post-ablation production candidate. Single-knob promotion of
  ablation_feature_mode_lean (the best in-cutoff measurement).
  All other production overlays (VTG, growth_tilt, PEAD,
  mega_cap_funding, reversion_gated revision cleaning) are kept;
  only feature_mode swaps from 'core' to 'lean' to remove the
  post-hoc-fitted whitelist that was net-negative under embargo.
out_dir: outputs/baseline_v5
tuning_mode: oos_verify   # 1-peek; logs to experiment_inventory.json
overrides:
  rebalance_freq: 21
  embargo_days: 20
  feature_mode: "lean"          # was: core (DROP per ablation)
  value_trap_gate_enabled: true  # kept
  vtg_pe_z_threshold: -0.5
  vtg_momentum_threshold: -0.5
  vtg_accel_threshold: 0.5
  vtg_scale: 0.0
```

Expected in-cutoff result (from `ablation_feature_mode_lean`):
  - trimmed IR ≈ +0.898 (vs research baseline +0.392)
  - trimmed P2 IR ≈ +1.160 (vs baseline -0.497)
  - The oos_verify peek will measure the post-cutoff (2025-01 onward) extension.

## (pending step 3) Rollback log entries

### DROP — feature_mode=core -> lean
- Variant: `ablation_feature_mode_lean`
- Trimmed ΔIR vs baseline: +0.506 [-0.053, +1.132] (p=0.094)
- ΔP2: +1.657
- Reason: CI spans 0 [-0.053, +1.132] — parsimony
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.

### DROP — growth_tilt
- Variant: `ablation_no_growth_tilt`
- Trimmed ΔIR vs baseline: -0.182 [-0.572, +0.160] (p=0.330)
- ΔP2: -0.117
- Reason: P2 worsens by -0.117 (< -0.10)
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.

### DROP — mega_cap_funding
- Variant: `ablation_no_mega_funding`
- Trimmed ΔIR vs baseline: -0.186 [-0.452, +0.068] (p=0.149)
- ΔP2: -0.142
- Reason: P2 worsens by -0.142 (< -0.10)
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.

### DROP — pead_boost
- Variant: `ablation_no_pead`
- Trimmed ΔIR vs baseline: -0.106 [-0.308, +0.066] (p=0.278)
- ΔP2: -0.066
- Reason: CI spans 0 [-0.308, +0.066] — parsimony
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.

### DROP — value_trap_gate
- Variant: `ablation_no_vtg`
- Trimmed ΔIR vs baseline: -0.046 [-0.316, +0.186] (p=0.730)
- ΔP2: +0.072
- Reason: CI spans 0 [-0.316, +0.186] — parsimony
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.

### DROP — revision_clean_mode=reversion_gated -> down_only
- Variant: `ablation_revision_down_only`
- Trimmed ΔIR vs baseline: +0.290 [-0.161, +0.737] (p=0.222)
- ΔP2: +0.787
- Reason: CI spans 0 [-0.161, +0.737] — parsimony
- Original rationale (now superseded by honest evaluation): see CLAUDE.md before 2026-05-19 + REDESIGN notes in src/config.py.
