---
name: model-training
description: "LightGBM walk-forward 모델 학습과 예측 생성을 실행한다. '모델 훈련', '모델 학습', 'Phase 4 실행', '예측 생성', 'walk-forward 실행' 요청 시 이 스킬을 사용."
---

# Model Training Skill

LightGBM walk-forward 방식으로 모델을 훈련하고 cross-sectional 예측을 생성한다.

## 워크플로우

### Step 1: 데이터 로드
```python
from src.backtest import load_checkpoint
prepared = load_checkpoint("phase2")  # panel, feature_names
targets_cp = load_checkpoint("phase3")  # targets
```

### Step 2: Walk-Forward 학습
```python
from src.model_trainer import walk_forward_train
models, predictions, raw_predictions = walk_forward_train(
    panel, targets, feature_names, all_dates
)
```
- 756일(3년) rolling window
- 63일(3개월)마다 재훈련
- Early stopping: validation 126일, patience 50
- EMA 스무딩 alpha=0.5

### Step 3: 품질 검증
- Raw IC(sample 50) > 0.015 확인
- Degenerate 모델(trees < 10) 비율 < 10% 확인
- 모델 수 > 0 확인

### Step 4: 저장
```python
from src.backtest import save_checkpoint
save_checkpoint("phase4", {
    "models": models,
    "predictions": predictions,
    "raw_predictions": raw_predictions,
})
```

## LightGBM 하이퍼파라미터
| 파라미터 | 값 | 이유 |
|---------|-----|------|
| learning_rate | 0.008 | 낮은 LR + 많은 트리 → 안정적 |
| num_leaves | 63 | 충분한 복잡도 |
| max_depth | 7 | 과적합 방지 |
| min_child_samples | 40 | 소규모 유니버스 고려 |
| subsample | 0.7 | 행 샘플링 |
| colsample_bytree | 0.4 | 피처 다양성 |
| reg_alpha/lambda | 0.5/2.0 | L1/L2 정규화 |
| n_estimators | 1500 | early stopping으로 제어 |

## 모니터링 지표
- 재훈련 전후 예측 상관: ~0.95 목표
- Raw IC: > 0.03 목표
- 유효 예측 관측치 수
