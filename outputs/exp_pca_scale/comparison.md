# PCA target scale A/B (Task C step 1)

**Question**: Does dividing the PCA residual by sqrt(forward_horizon)
change downstream behavior? Cross-sectional Z-score normalization in
model_trainer should make this invariant — if IR changes, some module
downstream depends on absolute residual magnitude (signal stability
shrinkage, PEAD composition, EMA blending).

**Verdict**: |Δ trim_IR| = 0.4276 ≥ 0.02 → downstream IS sensitive — investigate.

## Headline comparison (cutoff-trimmed, 1592 obs)

| Metric | raw (baseline) | daily_eq | Δ |
|---|---:|---:|---:|
| trim IR | 0.3923 | 0.8200 | +0.4276 |
| trim active return | 1.26% | 2.87% | +1.62p |
| trim TE | 2.88% | 2.92% | +0.03p |
| trim P1 IR | +1.287 | +1.748 | +0.460 |
| trim P2 IR | -0.497 | +0.375 | +0.872 |
| trim P3 IR | +0.390 | +0.017 | -0.373 |
| Avg IC | 0.0463 | 0.0569 | +0.0106 |
| Turnover 2-way | 90.8% | 102.4% | +11.6p |
| rolling_ir_mean (full) | 0.588 | 0.846 | +0.258 |
| SPA p-value (full) | 0.0010 | 0.0150 | — |

## Decision

|Δ trim_IR| ≥ 0.02. Downstream modules ARE sensitive to absolute
residual magnitude — likely culprits: signal_stability_lambda,
PEAD boost composition, or prediction EMA blending. Promotion of
`daily_eq` is OUT OF SCOPE for this step (Task C step 1 is verification,
not promotion). Filed as future-work to identify and isolate the
sensitive module.

## Reproduce

```bash
python run_variant.py --variant variants/exp_pca_scale_daily_eq.yaml --no-cache
python scripts/step1_pca_compare.py
```
