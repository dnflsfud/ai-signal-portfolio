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

- `run_iter20.py` — iter20 grid over W (decile_funding) × X (regime_gate).
  Verdict: A (iter15-like) baseline was the best variant.

- `run_iter21_full.py` — iter21 deeper sweep of W + X + interaction gates.
  Verdict: all variants net-negative on IR vs iter15 baseline; closest
  was `H_regime_very_mild` at −0.025 IR.

## Do not resurrect without reading

Before re-enabling any lever these scripts depend on, review
`../docs/rollback_log.md` — the rollback reasons there reflect empirical
results on 2018-11 to 2026-04 walk-forward data, not opinion.
