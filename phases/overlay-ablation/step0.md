# Step 0: ablation-variants

Task A 환경(walk-forward embargo + cutoff=2024-12-31) 위에서, 현 production에 켜져 있는 in-sample fit 의심 overlay 6개를 개별/전체 ablation할 manifest 7개를 작성한다.

## 읽어야 할 파일

- `variants/iter15_FINAL_postfix.yaml` (Task A에서 생성한 새 canonical baseline manifest)
- `outputs/iter15_FINAL_postfix/metrics.json` (비교 기준)
- `src/config.py` ablation 대상 overlay 필드 위치:
  - `value_trap_gate_enabled` 등 vtg_* (~L391-405)
  - `growth_tilt_enabled` 등 growth_tilt_* (~L297-315)
  - `pead_boost_enabled` 등 pead_* (~L283-293)
  - `mega_cap_protection_enabled`, `mega_cap_funding_mode`, `mega_cap_funding_k`, `mega_cap_funding_score_max` (~L319-347)
  - `revision_clean_mode` (~L99-124)
  - `feature_mode` (~L168-179)
- `docs/BASELINE.md` 갱신본 (Task A step3)
- **이전 task 산출**: `phases/data-leakage-fix/index.json` 전 step completed 확인
- `phases/overlay-ablation/index.json`

Task A가 모두 completed가 아니면 즉시 blocked로 보고하라 (`tuning_mode: research`가 동작 안 함).

## 작업

### variant 7개 작성

전부 다음 공통 헤더를 따른다:
```yaml
tuning_mode: research              # cutoff 강제 ON
out_dir: outputs/ablation/<label>  # 격리
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  # + variant 별 ablation 키
```

#### 1. `variants/ablation_no_vtg.yaml`
```yaml
label: ablation_no_vtg
description: >
  iter15_FINAL_postfix - value_trap_gate. Tests whether the empirically-fit
  cheap/bad_mom/accel gate survives OOS discipline. Prior to leakage fix it
  was credited with +0.2 P3 IR.
out_dir: outputs/ablation/ablation_no_vtg
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  value_trap_gate_enabled: false
```

#### 2. `variants/ablation_no_growth_tilt.yaml`
```yaml
label: ablation_no_growth_tilt
description: >
  Disable growth/revision post-prediction tilt (boost_weight=0.25, EPS:Sales
  50:50). The (50:50, weight=0.25) point was selected after grid testing
  multiple combinations on the full sample — test whether the lift survives
  OOS discipline.
out_dir: outputs/ablation/ablation_no_growth_tilt
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  growth_tilt_enabled: false
```

#### 3. `variants/ablation_no_pead.yaml`
```yaml
label: ablation_no_pead
description: >
  Disable Post-Earnings Announcement Drift boost (weight=0.30, decay=7d,
  cutoff=21d). Magic constants without OOS validation. Hypothesis: boost
  is fitted to the same sample its parameters were tuned on.
out_dir: outputs/ablation/ablation_no_pead
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  pead_boost_enabled: false
```

#### 4. `variants/ablation_no_mega_funding.yaml`
```yaml
label: ablation_no_mega_funding
description: >
  Keep mega_cap_protection (asymmetric bounds) but disable the K=4 concentrated
  funding mode. The k=4 + score_max=0.0 numbers were chosen from the
  'MSFT/AVGO problem' diagnosis on full sample.
out_dir: outputs/ablation/ablation_no_mega_funding
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  mega_cap_funding_mode: false
```

#### 5. `variants/ablation_revision_down_only.yaml`
```yaml
label: ablation_revision_down_only
description: >
  Roll back revision_clean_mode from reversion_gated (current production)
  to down_only (iter15 original). reversion_gated was selected because it
  beat down_only on the same OOS window — picking-the-winner bias.
out_dir: outputs/ablation/ablation_revision_down_only
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  revision_clean_mode: "down_only"
```

#### 6. `variants/ablation_feature_mode_lean.yaml`
```yaml
label: ablation_feature_mode_lean
description: >
  Replace hand-picked 'core' (61-feature whitelist, post-hoc selected from
  full-sample importance) with 'lean' (~80 features, no post-hoc cherry pick).
  Tests how much of IR depends on the whitelist being fitted to the data.
out_dir: outputs/ablation/ablation_feature_mode_lean
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  feature_mode: "lean"
```

#### 7. `variants/ablation_all_overlays_off.yaml`
```yaml
label: ablation_all_overlays_off
description: >
  Pure LightGBM signal + MVO. Disables VTG, growth_tilt, PEAD, mega_cap_funding,
  bm_proportional cap. Reverts revision cleaning to down_only and uses lean
  feature panel. The "honest baseline" — what does the alpha stack look like
  without any of the post-hoc fitted overlays?
out_dir: outputs/ablation/ablation_all_overlays_off
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_cutoff_date: "2024-12-31"
  value_trap_gate_enabled: false
  growth_tilt_enabled: false
  pead_boost_enabled: false
  mega_cap_funding_mode: false
  mega_cap_protection_enabled: false
  revision_clean_mode: "down_only"
  feature_mode: "lean"
```

### 실행 가능성 검증 (smoke만)

이 step은 manifest 작성만이다. 실제 backtest는 step1에서 일괄 실행한다. 단 각 YAML이 `run_variant.py`의 manifest 검증을 통과하는지만 확인:

```bash
for f in variants/ablation_*.yaml; do
    python -c "
from pathlib import Path
from run_variant import load_manifest, compose_config
m = load_manifest(Path('$f'))
cfg = compose_config(m)
assert cfg.enforce_oos_holdout is True, '$f: cutoff not enforced'
print('OK', '$f')
"
done
```

(`load_manifest`/`compose_config`는 unknown override 키를 거부하므로 오타 자동 검출.)

## Acceptance Criteria

```bash
# 1. 7개 manifest 존재
for v in no_vtg no_growth_tilt no_pead no_mega_funding revision_down_only feature_mode_lean all_overlays_off; do
    test -f "variants/ablation_${v}.yaml" || { echo "MISSING: $v"; exit 1; }
done

# 2. 모두 manifest 검증 통과 + research 모드
python -c "
from pathlib import Path
from run_variant import load_manifest, compose_config
labels = ['no_vtg','no_growth_tilt','no_pead','no_mega_funding','revision_down_only','feature_mode_lean','all_overlays_off']
for lab in labels:
    p = Path(f'variants/ablation_{lab}.yaml')
    m = load_manifest(p)
    cfg = compose_config(m)
    assert m['tuning_mode'] == 'research', f'{lab} not research'
    assert cfg.enforce_oos_holdout is True, f'{lab} holdout off'
    assert cfg.embargo_days == 20, f'{lab} embargo missing'
print('all 7 manifests valid')
"

# 3. 모든 manifest가 out_dir을 outputs/ablation/ 하위로 격리
grep -l "out_dir: outputs/" variants/ablation_*.yaml | xargs grep -L "outputs/ablation/" && exit 1 || true

# 4. 어떤 backtest도 아직 안 돌렸어야 함
test ! -d outputs/ablation/ablation_no_vtg/ || { echo "Ran prematurely"; exit 1; }
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - 모든 manifest가 `iter15_FINAL_postfix`와 동일한 baseline 위에서 *한 개만* 변경하는가? (all_overlays_off 제외)
   - 각 overrides 키가 `src/config.py`에 실제 존재하는가? (manifest 검증으로 자동 확인됨)
3. `phases/overlay-ablation/index.json` step 0 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "7 ablation manifests written under variants/ablation_*.yaml. All pass research-mode validation (cutoff=2024-12-31 enforced, embargo_days=20). Single-key variants: no_vtg/no_growth_tilt/no_pead/no_mega_funding/revision_down_only/feature_mode_lean. Composite: all_overlays_off."`
   - 실패/blocked → 사유

## 금지사항

- **backtest를 이 step에서 실행하지 마라.** 이유: step1 runner가 일괄 처리하며 진행 로그도 거기서 관리. 미리 돌리면 cache 상태가 꼬인다.
- **`tuning_mode: deploy` 또는 `oos_verify`로 만들지 마라.** 이유: ablation은 *cutoff 이전 데이터에서만* 비교해야 공정. peek은 step3 단 1회만 (그것도 baseline_v5 결정 후).
- **`outputs/iter15_FINAL_postfix/`와 같은 디렉토리를 쓰지 마라.** 이유: artifact가 덮어써져서 비교 불가.
- **override에 ablation 무관 키를 끼워넣지 마라** (예: max_te_annual 조정). 이유: ablation은 *단일 변수* 효과 측정. 다변수 조정은 step3에서.
- **`variants/iter15_FINAL_postfix.yaml`을 수정하지 마라.** 이유: 비교 기준선.
