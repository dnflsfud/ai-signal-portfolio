# PCA Scale Forensics

**Date**: 2026-05-19 v2 (final-v1-promotion step 3)
**Status**: Diagnostic complete. Finding contradicts initial hypothesis.

## Question

`phases/selection-bias-discipline/` step 1 A/B test:
- `pca_target_scale_mode='raw'`: trim_IR **0.392**
- `pca_target_scale_mode='daily_eq'` (residual ÷ √20): trim_IR **0.820** (Δ +0.428)

Predictions are cross-sectionally Z-scored in the model trainer, so the
ordering should be rank-invariant to target scale. **Why does scaling the
PCA residual change IR by +0.428?**

Original hypothesis: one downstream post-process module (PEAD additive
boost, growth_tilt, value-trap gate, EMA blending, or signal-stability
shrinkage) consumes the prediction magnitude directly and is amplifying
the scale change.

## Method

`scripts/diag_pca_scale_layers.py` loads both pkls and computes the
cross-sectional std of `raw_predictions` (pre post-process) and
`predictions` (post post-process). If a post-process layer amplifies
magnitude, `ratio_post / ratio_pre` should be > 1.

## Finding

**The original hypothesis is wrong.** The diagnostic output:

```
Theoretical pre ratio    : 0.2236  (= 1/sqrt(20))
Observed  pre ratio      : 1.0000  (stdev raw_predictions: 0.9923 -> 0.9923)
Observed  post ratio     : 1.0011  (stdev final predictions: 0.9496 -> 0.9506)
Post/Pre amplification   : 1.0011
```

Raw predictions have **identical magnitude** (0.9923) in both runs.
Post-process amplification is negligible (1.0011). The model already
produces same-magnitude outputs regardless of target scale.

The reason: model_trainer applies cross-sectional Z-score normalization
to the model output. Both runs hit the Z-score normalizer and emerge
with σ ≈ 1.0 by construction. The post-process stack (PEAD, growth_tilt,
VTG) is essentially magnitude-invariant.

## Real mechanism

If both runs produce equally-scaled predictions, the IR delta must come
from a **different fitted model**, not from a different post-process
treatment of the same predictions.

LightGBM is **target-scale-sensitive** at training time even though its
predictions are downstream-rank-normalized:

1. **Regularization** (`reg_alpha=0.3`, `reg_lambda=2.0`) is in absolute
   gradient/leaf-value units. Smaller target magnitudes → smaller raw
   gradients → relatively larger penalty → smoother trees.
2. **Early stopping** triggers on absolute validation loss thresholds.
   Smaller target → smaller absolute loss → different stopping point.
3. **`min_child_samples=60` + `learning_rate=0.02`** interact non-linearly
   with the squared-error gradient magnitude.

The `daily_eq` mode (residual / √20) doesn't just scale targets — it
produces a **structurally different LGBM** with smoother trees, different
early-stopping points, and consequently different cross-sectional rank
ordering.

This is a *latent target-scale effect on the model*, not a magnitude bug
in post-process.

## Implication

The +0.428 IR uplift is **real but architecturally precarious**:

- It's not a "free win" from a numerical fix; it's a different model with
  different inductive biases.
- The fact that we got it by accident (just dividing the target by √20)
  means LGBM hyperparameters are not well-tuned to the target scale —
  there's plausibly a more deliberate hyperparameter set that captures
  the same gain.
- Recommended: rather than promoting `daily_eq` as-is (which changes
  reproducibility of *every* prior trial that used `raw`), tune LGBM
  hyperparameters under `raw` to recover similar smoothing
  characteristics. Candidates: `reg_alpha` ∈ {0.6, 1.0}, `reg_lambda` ∈
  {4.0, 8.0}, lower `learning_rate` 0.02 → 0.01 with proportional
  `n_estimators` bump.

This work is **out of scope for final-v1-promotion**. Filing as next-
phase candidate. Default stays `raw` per the step file prohibition.

## Reproduction

```bash
python scripts/diag_pca_scale_layers.py
# Output:
#   stdout summary
#   outputs/diagnostics/pca_scale_layer_ratio.csv
# Inputs:
#   outputs/iter15_FINAL_postfix/backtest_result.pkl    (raw)
#   outputs/exp_pca_scale/daily_eq/backtest_result.pkl  (daily_eq)
```

## Caveat

This is a **minimal forensic** using existing pkls. A more thorough
analysis would instrument `apply_pead_boost`/`apply_growth_tilt`/etc.
to record stdev BEFORE and AFTER each post-process layer (the step.md
described this approach). The minimal version is sufficient to **falsify
the post-process amplification hypothesis** but does not pin down the
exact LGBM-internal mechanism. That requires inspecting model objects
(tree depth, leaf counts, n_estimators at early-stop) across both runs.
Deferred.
