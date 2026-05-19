# Step 1: pca-target-scale

PCA 잔차 타겟의 스케일 정합을 점검한다. 현재 `target_engine.py`는 *daily returns* 공분산으로 PCA를 적합한 뒤 *20일 cumulative forward return*을 그 eigenvector 공간에 투영한다. PCA가 선형이므로 *수학적으로는 look-ahead가 아님*이지만, 잔차 absolute magnitude가 horizon에 비례해 부풀어 cross-sectional rank 외의 후속 처리(예: signal stability shrinkage)가 영향을 받을 가능성이 있다. **A/B로 검증하고 통계적으로 의미 있을 때만 채택**한다.

## 읽어야 할 파일

- `src/target_engine.py` 전체 (특히 `compute_specific_returns` L88-143, `compute_specific_returns_regime_weighted` L146-276)
- `docs/AI_METHODOLOGY.md` §2 (타겟 정의)
- **이전 step 산출**: `src/backtest.py` `compute_config_fingerprint` (이 step의 변경이 fingerprint를 바꿔 자동 재계산을 trigger해야 함)
- `phases/selection-bias-discipline/index.json` step 0 summary
- `outputs/iter15_FINAL_postfix/metrics.json` (research baseline)

## 작업

### 1. 새 config 필드: `pca_target_scale_mode`

```python
# src/config.py PipelineConfig
# Target residual scale mode.
#   "raw"      — current default. specific_return is in 20d cumulative scale.
#   "daily_eq" — divide residual by sqrt(forward_horizon / 1) so the scale
#                matches a daily-equivalent return. Useful when downstream
#                modules assume daily magnitudes.
pca_target_scale_mode: str = "raw"
```

FINGERPRINT_KEYS (step0)에 `"pca_target_scale_mode"` 추가.

### 2. `compute_specific_returns` 수정

```python
# 기존 (L131-132)
spec = fwd_t - common
specific_ret.iloc[t] = spec.flatten()

# 변경
spec = fwd_t - common
if config.pca_target_scale_mode == "daily_eq":
    spec = spec / np.sqrt(max(horizon, 1))  # sqrt-scale to daily
specific_ret.iloc[t] = spec.flatten()
```

`compute_specific_returns_regime_weighted`에도 동일 분기 추가.

cross-sectional Z-score 정규화는 model_trainer.py에서 이미 일어나므로 절대 magnitude 변경이 *학습된 LightGBM*에는 영향이 없을 가능성 큼. 그러나 EWMA, EMA blending 등 raw 예측을 쓰는 downstream에는 영향 가능. 이게 A/B의 핵심.

### 3. A/B variant 작성

```yaml
# variants/exp_pca_scale_daily_eq.yaml
label: exp_pca_scale_daily_eq
description: >
  Test daily-equivalent residual scale for the 20d PCA target. Compares
  vs iter15_FINAL_postfix (raw scale). Downstream cross-sectional Z-score
  normalization should make this neutral; if IR changes meaningfully,
  some downstream module is sensitive to absolute residual magnitude.
out_dir: outputs/exp_pca_scale/daily_eq
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  value_trap_gate_enabled: true
  vtg_pe_z_threshold: -0.5
  vtg_momentum_threshold: -0.5
  vtg_accel_threshold: 0.5
  vtg_scale: 0.0
  pca_target_scale_mode: "daily_eq"
```

기존 baseline은 `pca_target_scale_mode: "raw"`로 동작 (default).

### 4. 실행 + 비교

```bash
python run_variant.py --variant variants/exp_pca_scale_daily_eq.yaml --no-cache
```

산출 후 `outputs/exp_pca_scale/comparison.md`에:

| Metric | iter15_FINAL_postfix (raw) | exp_pca_scale_daily_eq | Δ |
|---|---|---|---|
| IR | … | … | … |
| P1/P2/P3 IR | … | … | … |
| turnover | … | … | … |
| Avg IC | … | … | … |

해석:
- |ΔIR| < 0.02 → "downstream invariant — 채택 무의미, raw 유지"
- |ΔIR| ≥ 0.02 → 어떤 downstream이 raw magnitude에 의존하는지 진단 필요. 의심: `prediction_ema_alpha` blending (model_trainer.py L420-424), `signal_stability_lambda` (현재 0), PEAD boost composition. raw 유지 + 별도 조사 항목 (이 step에서 fix 시도 X).

### 5. 결정

- |ΔIR| < 0.02: `pca_target_scale_mode`를 `"raw"`(현 default)로 유지. config 필드는 **그대로 두되 default 변경 없음**. 향후 조사 위한 lever로 남김.
- |ΔIR| ≥ 0.02: raw 유지 + `outputs/exp_pca_scale/finding.md`에 sensitivity 진단 가설 기록. Task C 종료 후 별도 follow-up.

**어떤 경우에도 default를 `"daily_eq"`로 바꾸지 마라.** 이 step의 목적은 검증이지 promote가 아니다.

## Acceptance Criteria

```bash
# 1. 새 필드 노출 + 기본값 raw
python -c "from src.config import PipelineConfig; c=PipelineConfig(); assert c.pca_target_scale_mode == 'raw'"

# 2. fingerprint에 포함됐는지
python -c "
from src.backtest import FINGERPRINT_KEYS
assert 'pca_target_scale_mode' in FINGERPRINT_KEYS
"

# 3. A/B variant 산출
test -f variants/exp_pca_scale_daily_eq.yaml
test -f outputs/exp_pca_scale/daily_eq/metrics.json
test -f outputs/exp_pca_scale/comparison.md

# 4. 결정이 보고서에 명시
grep -E "decision|결정|채택|유지" outputs/exp_pca_scale/comparison.md

# 5. default가 raw로 유지
python -c "from src.config import DEFAULT_CONFIG; assert DEFAULT_CONFIG.pca_target_scale_mode == 'raw', 'do not promote'"
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - `compute_specific_returns_regime_weighted`에도 동일 scale 분기가 들어갔는가?
   - 새 variant도 `tuning_mode: research`로 cutoff 안에서 비교했는가?
   - `outputs/exp_pca_scale/`이 `outputs/baseline_v5/`와 격리되어 있는가?
3. `phases/selection-bias-discipline/index.json` step 1 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "pca_target_scale_mode field added (default raw, options raw|daily_eq). A/B run: ΔIR=<X.XXX>. Decision: <retain raw / further investigation needed>. Field kept as future lever; default unchanged."`
   - 실패/blocked → 사유

## 금지사항

- **default를 `daily_eq`로 바꾸지 마라.** 이유: 이 step은 검증 단계. promote는 별도 후속 task.
- **`compute_specific_returns_regime_weighted`를 빠뜨리지 마라.** 이유: regime-weighted 분기에서만 scale이 달라지면 silent inconsistency.
- **새 variant의 다른 키를 손대지 마라.** scale mode 외 변경은 단일 변수 비교를 깬다.
- **결과가 |ΔIR| ≥ 0.02여도 이 step에서 downstream을 고치지 마라.** 이유: scope 초과. `outputs/exp_pca_scale/finding.md`에 *진단 가설*만 적고 종료.
- **cross-sectional Z-score 정규화를 건드리지 마라.** 이유: 그건 라벨 자체가 아니라 model_trainer 안의 별개 단계.
