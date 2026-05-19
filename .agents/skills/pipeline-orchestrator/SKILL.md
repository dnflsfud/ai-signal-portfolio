---
name: pipeline-orchestrator
description: "AI Signal 포트폴리오 전체 파이프라인을 오케스트레이션한다. '파이프라인 실행', '전체 실행', '백테스트 파이프라인', 'run pipeline', '하네스 실행' 요청 시 반드시 이 스킬을 사용. 데이터→피처→타겟→모델→백테스트→Selection Bias 검증→어트리뷰션→대시보드 전 과정을 6개 전문 에이전트로 조율한다."
---

# AI Signal Pipeline Orchestrator

6개 전문 에이전트를 순차/병렬로 조율하여 전체 포트폴리오 파이프라인을 실행한다.

## 실행 모드: 서브 에이전트

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 스킬 | 출력 |
|---------|--------------|------|------|------|
| data-pipeline | data-pipeline | 데이터/피처/타겟 | data-pipeline | prepared_data.pkl |
| model-trainer | model-trainer | LightGBM 학습 | model-training | checkpoint_phase4.pkl |
| backtester | backtester | MVO+시뮬레이션 | backtesting | backtest_result.pkl |
| **selection-bias-checker** | **selection-bias-checker** | **다중 비교 편향 검증** | **selection-bias-check** | **selection_bias_report.md** |
| attribution-analyst | attribution-analyst | SHAP+분해 | attribution-analysis | csv/shap_*.csv |
| dashboard-publisher | dashboard-publisher | CSV+대시보드 | dashboard-publish | outputs/csv/*.csv |

## 워크플로우

### Phase 1: 준비
1. 사용자 입력 분석 — data_path, output_dir, PipelineConfig 파라미터 확인
2. `_workspace/` 생성
3. PipelineConfig 구성 (사용자 오버라이드 반영)

### Phase 2: 데이터 파이프라인 (순차)

```
Agent(
  prompt: "Phase 1~3 실행: 데이터 로드, 피처 엔지니어링, PCA 타겟 생성.
           data_path: {data_path}
           PipelineConfig: {config_params}
           결과를 outputs/prepared_data.pkl로 저장하라.",
  subagent_type: "data-pipeline",
  model: "opus"
)
```

출력: `outputs/prepared_data.pkl`, `outputs/checkpoints/checkpoint_phase1~3.pkl`

### Phase 3: 모델 학습 (순차, Phase 2 완료 후)

```
Agent(
  prompt: "Phase 4 실행: prepared_data.pkl을 로드하여 walk-forward LightGBM 학습.
           결과를 checkpoint_phase4.pkl로 저장하라.
           Raw IC > 0.015 검증 포함.",
  subagent_type: "model-trainer",
  model: "opus"
)
```

출력: `outputs/checkpoints/checkpoint_phase4.pkl`

### Phase 4: 백테스트 (순차, Phase 3 완료 후)

```
Agent(
  prompt: "Phase 5-6 실행: prepared_data와 Phase 4 모델을 로드하여 MVO 백테스트 실행.
           run_backtest에 precomputed 데이터를 전달하라.
           validate_backtest로 검증 후 backtest_result.pkl 저장.",
  subagent_type: "backtester",
  model: "opus"
)
```

출력: `outputs/backtest_result.pkl`

### Phase 5: Selection Bias 검증 (Phase 4 완료 후 — GATE)

백테스트 결과의 통계적 유의성과 다중 비교 편향을 검증하는 **필수 게이트**다.
이 검증이 FAIL이면 파이프라인은 계속 진행하되, 최종 보고서에 경고가 표시된다.

```
Agent(
  prompt: "backtest_result.pkl을 로드하여 Selection Bias 검증을 실행하라.
           검증 항목: Deflated Sharpe Ratio, Minimum Track Record Length,
           Grid Search Haircut, Universe Survivorship, Sub-period Stability.
           시행 횟수 N: {grid_search 조합 수, 기본값 1}.
           결과를 outputs/reports/selection_bias_report.md에 저장하라.",
  subagent_type: "selection-bias-checker",
  model: "opus"
)
```

출력: `outputs/reports/selection_bias_report.md`, `outputs/csv/selection_bias_metrics.csv`

**게이트 판정:**
- **PASS**: 모든 항목 통과 → Phase 6 진행
- **WARN**: 1~2개 경고 → Phase 6 진행 + 최종 보고서에 경고 포함
- **FAIL**: 1개 이상 실패 → Phase 6 진행하되, 사용자에게 경고 출력:
  "Selection Bias 검증 FAIL — 관측된 성과가 다중 비교 편향에 의한 것일 수 있습니다.
   상세 내용: outputs/reports/selection_bias_report.md"

### Phase 6: 어트리뷰션 + 대시보드 (병렬, Phase 5 완료 후)

Attribution과 Dashboard의 CSV 기본 export는 병렬 실행 가능:

```
# 병렬 실행 1: Attribution
Agent(
  prompt: "backtest_result.pkl과 prepared_data를 로드하여 SHAP + Li 3-component 어트리뷰션 실행.
           결과를 outputs/csv/에 저장하라.",
  subagent_type: "attribution-analyst",
  model: "opus",
  run_in_background: true
)

# 병렬 실행 2: Dashboard CSV Export
Agent(
  prompt: "backtest_result.pkl을 로드하여 14개 CSV 파일을 outputs/csv/에 내보내라.
           어트리뷰션 결과가 필요한 CSV는 attribution 완료 후 별도 생성.",
  subagent_type: "dashboard-publisher",
  model: "opus",
  run_in_background: true
)
```

### Phase 7: 통합 및 정리
1. 두 에이전트 완료 대기
2. 모든 CSV 파일 존재 확인 (`outputs/csv/` 내 14개 파일)
3. Selection Bias 보고서 포함 여부 확인
4. backtest_result.pkl의 summary() 출력
5. **Selection Bias verdict 출력** — PASS/WARN/FAIL + 핵심 수치
6. 검증 결과 요약 보고
7. `_workspace/` 보존

## 데이터 흐름

```
[data-pipeline]          → prepared_data.pkl + checkpoints
       ↓
[model-trainer]          → checkpoint_phase4.pkl (models, predictions)
       ↓
[backtester]             → backtest_result.pkl
       ↓
[selection-bias-checker] → selection_bias_report.md  ← GATE
       ↓ (PASS/WARN: 계속, FAIL: 경고 후 계속)
       ├── [attribution-analyst]  → shap/linear CSV     ──┐
       │                                                   ├→ 통합 확인
       └── [dashboard-publisher]  → 14 CSV + bias report ──┘
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| data-pipeline 실패 | 전체 중단, 데이터 경로/형식 확인 안내 |
| model-trainer 실패 | 전체 중단, 하이퍼파라미터/데이터 품질 확인 안내 |
| backtester 실패 (optimizer) | 1회 재시도 (제약 완화), 재실패 시 중단 |
| attribution 실패 | 경고 + dashboard는 attribution 없이 진행 |
| dashboard 실패 | 경고 + backtest_result.pkl은 보존 |
| 체크포인트 서명 불일치 | 즉시 중단, 파일 무결성 확인 요청 |

## 테스트 시나리오

### 정상 흐름
1. 사용자가 "파이프라인 실행" 요청
2. Phase 2: data-pipeline이 ~350 피처 + PCA 타겟 생성 → prepared_data.pkl
3. Phase 3: model-trainer가 walk-forward 학습 → IC > 0.03 확인
4. Phase 4: backtester가 MVO 시뮬레이션 → IR >= 1.0 확인
5. Phase 5: selection-bias-checker가 Deflated SR, MinTRL, Sub-period 검증 → **PASS**
6. Phase 6: attribution + dashboard 병렬 실행 → 14 CSV + bias report 생성
7. Phase 7: 전체 요약 보고 (Selection Bias: PASS 포함)

### 에러 흐름 1: Optimizer 실패
1. Phase 4에서 optimizer 실패율 15% (임계값 10% 초과)
2. 1회 재시도: MAX_TE_ANNUAL 1.5배 완화하여 재실행
3. 재시도 후에도 실패: 현재까지 결과 저장, 사용자에게 제약 조정 권고
4. 최종 보고서에 "Optimizer 실패율 높음 — 제약 완화 필요" 명시

### 에러 흐름 2: Selection Bias FAIL
1. Phase 5에서 Deflated SR p-value = 0.23 (> 0.05 → FAIL)
2. Sub-period Stability: Period 2 IR = -0.3 (UNSTABLE)
3. 파이프라인은 계속 진행 (Phase 6, 7 실행)
4. 사용자에게 경고: "Selection Bias 검증 FAIL — DSR p=0.23, Sub-period 불안정"
5. 최종 보고서에 selection_bias_report.md 내용 요약 포함
6. 사용자에게 권고: "Grid search 조합 수를 줄이거나, out-of-sample 기간 추가 필요"

### 하네스 모드 (단계별 실행)
사용자가 특정 Phase만 실행 요청 시:
- "Phase 1~3만 실행" → data-pipeline 에이전트만 호출
- "Phase 4부터 재개" → checkpoint 로드 + model-trainer부터 실행
- "어트리뷰션만 재실행" → backtest_result.pkl 로드 + attribution 에이전트만 호출
