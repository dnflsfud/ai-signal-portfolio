# Archive — legacy runners

Scripts here were removed from the live project tree on 2026-04-20 because
they exclusively drive **destructive experiment levers** that no longer
exist in `src/config.py` (see `../docs/rollback_log.md`).

These scripts will crash on import if executed against the current code
— the overrides they pass (e.g. `decile_funding_enabled`,
`regime_gate_enabled`, `regime_active_shrink_max`) are no longer valid
`PipelineConfig` fields.

They are preserved here purely for research archaeology — so someone
reviewing old experiment results can still read the exact overrides that
produced them.

## Contents

### Legacy runners (2026-04-20)

- `run_iter20.py` — iter20 grid over W (decile_funding) × X (regime_gate).
  Verdict: A (iter15-like) baseline was the best variant.

- `run_iter21_full.py` — iter21 deeper sweep of W + X + interaction gates.
  Verdict: all variants net-negative on IR vs iter15 baseline; closest
  was `H_regime_very_mild` at −0.025 IR.

### Archived variants — `variants/` subdirectory (2026-05-27)

Manifests retired during the post-`final-v1-promotion` cleanup. All five
used the deprecated `tuning_mode: production` alias and held no remaining
promotion or audit value. The active overlay/feature comparisons they
served are now captured by `variants/ablation_*.yaml` + `docs/ABLATION_REPORT.md`.

- `exp_signal_layer_v1.yaml`
- `exp_signal_layer_macro_only.yaml`
- `exp_signal_layer_regime_only.yaml`
  → Phase 2 signal-layer experiments. Marked SUPERSEDED by `docs/ROADMAP.md`
  after data-leakage-fix (sub-period IR targets re-derived on baseline_v5).

- `exp_revision_symmetric.yaml`
- `exp_revision_reversion_gated.yaml`
  → `revision_clean_mode` ablations. Redundant: `reversion_gated` is the
  current `DEFAULT_CONFIG` value, and the `down_only` alternative is
  preserved in `variants/ablation_revision_down_only.yaml`.

To resurrect: copy back to `variants/`, change `tuning_mode: production`
to `research`, and add a fresh OOS peek before any deploy claim.

## Do not resurrect without reading

Before re-enabling any lever these scripts depend on, review
`../docs/rollback_log.md` — the rollback reasons there reflect empirical
results on 2018-11 to 2026-04 walk-forward data, not opinion.
