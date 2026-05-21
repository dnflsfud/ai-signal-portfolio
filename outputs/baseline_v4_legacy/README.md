# outputs/baseline_v4/ — Deploy Baseline

**Status (2026-05-19)**: deploy baseline for the live `update_and_deploy.bat`
pipeline. NOT the research baseline for new variant promotion.

## What this directory is

This is the artifact set produced by running
`variants/iter15_65tkr_reb21_vtg.yaml` under `tuning_mode: deploy` — i.e.
without OOS hold-out (cutoff disabled, full sample through the latest data
date). The daily refresh flow (`update_and_deploy.bat`, `daily_update.py`,
`scripts/build_dashboard_data.py`, `streamlit_mobile.py`) reads from here.

## What this directory is NOT

It is **not** the comparison baseline for new strategy candidates. After the
`data-leakage-fix` phase (2026-05-19), all research/promotion decisions use:

- **Research baseline**: `outputs/iter15_FINAL_postfix/`
- **Manifest**: `variants/iter15_FINAL_postfix.yaml` (`tuning_mode: research`)
- Cutoff: `train_cutoff_date = "2024-12-31"`
- Embargo: `embargo_days = 20`

See `docs/BASELINE.md` § "Canonical Baseline (Research)" for the IR/sub-period
numbers under honest evaluation and the gate criteria.

## Why two baselines coexist

- **deploy** must use every available day of data to make the freshest possible
  daily predictions. It does not need cutoff discipline because nothing is
  being "tuned" against it — it just runs the locked production config.
- **research** must use cutoff so that selection bias accounting is meaningful.
  Without cutoff, every tuning iteration peeks at data that's supposed to be
  held out for OOS verification.

The two paths are kept structurally separate by the `tuning_mode` mechanism in
`run_variant.py` (data-leakage-fix Task A step 1, 2026-05-19).

## Eventual cutover

When a new research candidate (`baseline_v5` from Task B step 3, or later)
passes the OOS-verify peek and is promoted, this directory's contents will be
overwritten by the new candidate's deploy artifacts. Until then, the legacy
deploy strategy continues to ship.
