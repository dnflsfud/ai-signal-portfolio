# Variants — declarative experiment manifests

Each YAML file in this directory describes **one** PipelineConfig override set
plus metadata. Drive any experiment via:

```bash
python run_variant.py --variant variants/<name>.yaml
```

instead of adding another `run_iter{N}.py` script. The previous per-iteration
runner pattern fragmented entry points (8+ scripts at peak) and made config
drift easy — see `../archive/` for the legacy runners this replaces.

## Schema

```yaml
label: baseline_v5_deploy        # goes into outputs/<label>/
description: "..."               # free text; written into experiment_manifest
out_dir: outputs/baseline_v5_deploy   # default: outputs/{label}
tuning_mode: deploy              # {research, oos_verify, deploy}
                                 # - research:   cutoff=2024-12-31 enforced;
                                 #               every-day-of-the-week tuning
                                 #               candidate runs go here
                                 # - oos_verify: single peek after research
                                 #               win; logs n_oos_peeks+=1 to
                                 #               experiment_inventory.json
                                 # - deploy:     cutoff OFF; production daily
                                 #               flow (update_and_deploy.bat).
                                 #               Logs to outputs/deploy_log.txt
                                 #
                                 # Legacy aliases (DeprecationWarning):
                                 # - production / tuning → research semantics
overrides:                       # any PipelineConfig field (name: value)
  feature_mode: lean
  rebalance_freq: 21
  ...
```

Unknown fields under `overrides:` are rejected at load time so typos surface
early instead of silently doing nothing.

## Current canonical variants (2026-05-19 v2 cutover)

| Role | Manifest | tuning_mode | Output dir |
|---|---|---|---|
| **Production deploy** (`update_and_deploy.bat`) | `baseline_v5_deploy.yaml` | `deploy` | `outputs/baseline_v5_deploy/` (+ alias `outputs/baseline_v4/`) |
| **Research baseline** (gate denominator) | `iter15_FINAL_postfix.yaml` | `research` | `outputs/iter15_FINAL_postfix/` |
| **OOS verify peek** (promotion candidate of v5) | `baseline_v5.yaml` | `oos_verify` | `outputs/baseline_v5/` |
| Legacy deploy (audit / rollback) | `iter15_65tkr_reb21_vtg.yaml` | `deploy` | `outputs/baseline_v4_legacy/` |

See `../docs/BASELINE.md` for the full lineage and the 5 promotion gates.

## Other variants on disk

- `iter15_FINAL.yaml`, `iter15_FINAL_reproduce.yaml`, `iter15_65tkr_reb21.yaml`,
  `iter15_65tkr_reb21_sent.yaml` — historical anchors referenced by
  `docs/BASELINE.md` "Historical anchors" section. Kept for A/B forensics and
  cautionary failed-variant records.
- `ablation_*.yaml` (7 files) — Task B (overlay-ablation) artefacts.
  Reproducibility for `docs/ABLATION_REPORT.md`. Each disables one overlay
  vs the research baseline.
- `exp_pca_scale_daily_eq.yaml` — Task C step 1 (PCA scale A/B). Forensic
  documented in `docs/PCA_SCALE_FORENSICS.md`.
- `exp_baseline_v2_pp_cleaned.yaml` — `baseline_v3` ancestor; kept per
  `docs/BASELINE.md` "Historical anchors".
- `exp_revision_ma10d/ma21d/ma_dual.yaml` — Phase 2.4 revision-cleaning
  ablations. Kept for ablation traceability.

The following test variants were moved to `../archive/variants/` on 2026-05-27
because they (a) hold zero promotion value and (b) live under deprecated
`tuning_mode: production`:

- `exp_signal_layer_v1.yaml`, `exp_signal_layer_macro_only.yaml`,
  `exp_signal_layer_regime_only.yaml` — Phase 2 SUPERSEDED (`docs/ROADMAP.md`).
- `exp_revision_symmetric.yaml`, `exp_revision_reversion_gated.yaml` —
  redundant with current `revision_clean_mode` default ("reversion_gated")
  and the `down_only` ablation already in `variants/ablation_revision_down_only.yaml`.

See `../archive/README.md` for resurrection instructions.

## Tuning workflow (2026-05-19 v2)

1. Copy `iter15_FINAL_postfix.yaml` to a new file (e.g.
   `exp_p2_multi_horizon.yaml`). Keep `tuning_mode: research` so the
   2024-12-31 cutoff stays enforced.
2. Run: `python run_variant.py --variant variants/exp_p2_multi_horizon.yaml`.
3. Inspect `outputs/<label>/metrics.json`. Gate against the research baseline
   using the five primary gates in `docs/BASELINE.md` § "Gate criteria".
4. If the candidate passes all five gates, create a SECOND yaml with
   `tuning_mode: oos_verify` (everything else identical) and run ONCE
   for the holdout. `experiment_inventory.json.n_oos_peeks` increments.
5. If oos_verify also passes, create a THIRD yaml with `tuning_mode: deploy`
   and run that — it is the artefact `update_and_deploy.bat` will point to
   after the cutover. See `phases/final-v1-promotion/` for the template.

## Rules

- NEVER run a tuning variant with `tuning_mode: deploy` — that bypasses the
  cutoff and inflates selection bias.
- The legacy `tuning_mode: production` and `tuning_mode: tuning` aliases
  still load (with a `DeprecationWarning`) but new variants must pick from
  {`research`, `oos_verify`, `deploy`}.
- Do NOT commit a yaml that resurrects a rollback-listed lever without
  first reading `../docs/rollback_log.md`.
- Experiment directories follow `outputs/<label>/` convention. Never
  overwrite the top-level `outputs/`.
