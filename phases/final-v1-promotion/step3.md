# Step 3: pca-scale-forensics

`phases/selection-bias-discipline/` step 1 의 A/B 결과:
- `pca_target_scale_mode='raw'` (default): trim_IR 0.392
- `pca_target_scale_mode='daily_eq'` (residual ÷ √horizon): trim_IR **0.820** (Δ +0.428)

전체 IR (`full_IR 1.017 → 0.734`) 은 오히려 줄지만 trim_IR + rolling_ir_mean (0.588 → 0.846)이 크게 개선된다. Residual의 단순 스케일 변화에 이 정도의 민감도는 **rank-only 신호이라면 일어날 수 없다** — 어떤 downstream module이 magnitude를 직접 사용하고 있다. 후보:

1. `prediction_ema_alpha=0.5` 의 EMA blending (이전 prediction과 magnitude로 blend)
2. PEAD boost (`pead_boost_weight=0.30`) 가 score에 더해짐 — 더하기 연산이므로 score 분산에 영향
3. Growth tilt (`growth_tilt_weight=0.25`)
4. Mega-cap funding mode (`mega_cap_funding_mode=True`) — score 기반 K=4 worst-scoring 선택, magnitude로 ordering
5. Value-trap gate (multiplicative; magnitude 직접 사용)
6. `signal_stability_lambda` (off by default — 만약 baseline_v5에서 켰다면 후보)

이 step은 instrumentation 으로 각 layer 의 score std/mean 을 record 하여 어느 layer가 magnitude 의존성의 핵심인지 격리한다.

## 읽어야 할 파일

- `src/backtest.py` — `apply_value_trap_gate`, `apply_pead_boost`, `apply_growth_tilt`, `apply_mega_cap_funding` (실제 함수명은 코드에서 확인) 의 호출 순서와 시그니처
- `src/target_engine.py` `compute_specific_returns`, `compute_specific_returns_regime_weighted` — `pca_target_scale_mode` 분기점
- `src/model_trainer.py` — EMA blending 위치 (prediction_ema_alpha 사용처)
- `src/config.py` — `pead_*`, `growth_tilt_*`, `mega_cap_*`, `prediction_ema_alpha`, `signal_stability_lambda` 필드 정의
- `variants/exp_pca_scale_daily_eq.yaml` (이전 A/B의 daily_eq variant)
- `variants/baseline_v5.yaml`
- `phases/selection-bias-discipline/index.json` step 1 summary

## 작업

### 1. Per-layer score snapshot 수집 instrumentation

`src/backtest.py` 의 post-prediction 파이프라인 (혹은 score가 layer를 거치는 위치) 에 다음 형태의 hook 을 추가:

```python
def _record_score_layer(score: pd.Series, layer_name: str, t: pd.Timestamp,
                        recorder: Optional[Dict[str, list]] = None) -> None:
    """Append per-date layer snapshot to recorder when present.

    recorder는 backtest 호출 시 옵션으로 주입. None이면 no-op.
    레이어를 거칠 때마다 (t, layer_name, std, mean, |q90-q10|, n_finite) 를 append.
    """
    if recorder is None:
        return
    s = pd.to_numeric(score, errors="coerce").dropna()
    if len(s) == 0:
        return
    recorder.setdefault("rows", []).append({
        "t": t,
        "layer": layer_name,
        "std": float(s.std(ddof=0)),
        "mean": float(s.mean()),
        "iqr": float(s.quantile(0.9) - s.quantile(0.1)),
        "n_finite": int(len(s)),
    })
```

호출 지점 (예시; 실제 함수명은 코드 확인 후):
- `model_raw_predictions` 직후
- `EMA_blended` 직후 (prediction_ema_alpha 적용 후)
- `pead_boosted` 직후
- `growth_tilted` 직후
- `value_trap_gated` 직후
- `mega_cap_funded` 직후 (가장 마지막)

`run_variant.py` 에 `--record-layer-stats` 플래그 추가 — True 면 backtest에 recorder dict 주입하고 종료 후 `outputs/<label>/layer_stats.csv` 로 저장.

### 2. 비교 A/B 재실행 (with instrumentation)

baseline_v5 와 같은 overlay set 위에서, `pca_target_scale_mode` 만 raw / daily_eq 로 다른 두 variant를 instrumentation 켜고 실행. 이미 `exp_pca_scale_daily_eq` artifact가 있어도, layer_stats.csv 가 없으면 재실행 필요:

```bash
python run_variant.py --variant variants/exp_pca_scale_raw.yaml --record-layer-stats
python run_variant.py --variant variants/exp_pca_scale_daily_eq.yaml --record-layer-stats
```

(`exp_pca_scale_raw.yaml` 가 없으면 `baseline_v5.yaml` 복제 후 `pca_target_scale_mode: "raw"` 명시.)

### 3. Diagnostic 비교 script

`scripts/diag_pca_scale_layers.py`:

```python
"""Compare per-layer score std between raw vs daily_eq across same dates."""
# Input: outputs/exp_pca_scale_raw/layer_stats.csv,
#        outputs/exp_pca_scale_daily_eq/layer_stats.csv
# Output: outputs/diagnostics/pca_scale_layer_ratio.csv with columns
#         (layer, mean_std_raw, mean_std_daily_eq, ratio, biggest_per_layer_ratio_change)
#
# Print sorted by |ratio change between consecutive layers| descending.
# The layer with the largest *delta in ratio* is the magnitude-sensitive
# point: e.g. if EMA blending input ratio is 0.45 (= 1/sqrt(20)) but output
# ratio is 0.78, then EMA blending is non-linearly amplifying magnitude.
```

산출물은 layer 별 stdev ratio (raw/daily_eq) 를 produce. ratio 가 layer 를 거치며 어떻게 변하는지가 진단의 본질.

### 4. 결과 문서화

`docs/PCA_SCALE_FORENSICS.md` 새 파일:

```markdown
# PCA Scale Forensics

## Question
Why does `pca_target_scale_mode='daily_eq'` (residual ÷ √20) cause
ΔIR_trim +0.428 if predictions are rank-based?

## Method
<step 1, 2, 3 결과 요약 — layer_stats.csv 어디서 ratio 가 가장 많이 변하는지>

## Finding
Layer X (`<함수명>`) 가 magnitude 에 sensitive. 그 layer 의 식은:
`<코드 인용 1줄>`. raw 입력은 ~σ=σ_raw 인데 그 layer 의 동작은
σ_input 이 N% 변하면 output IC 가 M% 변하는 비선형 응답이 있다.

## Recommendation
- Short term: 기존 default 유지 (`raw`). 본 step 산출물은 *진단*용.
- Long term: layer X 의 magnitude-invariant 버전 도입 (rank 변환, 또는
  rolling z-score 정규화) → 다음 phase 후보.

## Reproduction
<command 시퀀스>
```

## Acceptance Criteria

```bash
# 1) Instrumentation hook exists in src/backtest.py
grep -q "_record_score_layer" src/backtest.py

# 2) --record-layer-stats flag exists in run_variant.py
python run_variant.py --help | grep -q "record-layer-stats"

# 3) Both layer_stats.csv produced and non-empty
test -s outputs/exp_pca_scale_raw/layer_stats.csv
test -s outputs/exp_pca_scale_daily_eq/layer_stats.csv

# 4) Diagnostic output produced
python scripts/diag_pca_scale_layers.py
test -s outputs/diagnostics/pca_scale_layer_ratio.csv

# 5) Forensics doc exists with Finding section
test -f docs/PCA_SCALE_FORENSICS.md
grep -q "## Finding" docs/PCA_SCALE_FORENSICS.md
```

## 검증 절차

1. AC 1–5 모두 PASS.
2. `docs/PCA_SCALE_FORENSICS.md` 의 Finding 섹션이 1개 구체적 모듈을 지목하는지 확인. "여러 가능성이 있다" 식 hedging 으로 끝나면 step 미완.
3. 지목된 모듈을 `src/<module>.py` 에서 직접 1줄 인용하여 magnitude 의존성을 코드 수준에서 입증.
4. `phases/final-v1-promotion/index.json` step 3 status `completed`. summary 에 지목된 모듈명 + ratio 변화 수치를 1줄로 명시.

## 금지사항

- **`pca_target_scale_mode` default 를 바꾸지 마라.** 이유: `phases/selection-bias-discipline/` step 1 의 prohibition 을 유지한다. 이 step 은 진단만 한다.
- **다른 overlay (PEAD, growth_tilt, mega_cap_funding) 의 default 를 끄지 마라.** 이유: 격리는 *instrumentation 으로* 한다. overlay 를 끄고 비교하는 것은 ablation 으로 별도 phase에서 처리되었어야 하는 것이고, 이미 `phases/overlay-ablation/` 에서 했다.
- **OOS peek 을 추가로 소비하지 마라.** 이유: 이 step 은 모두 `tuning_mode: research` 환경에서 진단만 한다. `oos_verify` 호출 금지.
- 기존 테스트를 깨뜨리지 마라.
