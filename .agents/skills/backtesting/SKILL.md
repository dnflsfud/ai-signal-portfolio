---
name: backtesting
description: "cvxpy MVO 포트폴리오 최적화와 walk-forward 백테스트를 실행한다. '백테스트 실행', 'Phase 5-6 실행', '포트폴리오 구축', 'grid search', '파라미터 탐색' 요청 시 이 스킬을 사용."
---

# Backtesting Skill

MVO 포트폴리오 최적화 + walk-forward 시뮬레이션을 실행하고 성과를 측정한다.

## 워크플로우

### Step 1: 백테스트 실행
```python
from src.backtest import run_backtest
from src.data_loader import UniverseData
from src.config import PipelineConfig

config = PipelineConfig()  # 또는 커스텀 파라미터
data = UniverseData("./data/ai_signal_data.xlsx")
result = run_backtest(data, config=config)
```

run_backtest는 내부적으로:
1. 피처 생성 (precomputed 전달 가능)
2. 타겟 생성 (precomputed 전달 가능)
3. 모델 학습 (precomputed 전달 가능)
4. simulate_portfolio()로 포트폴리오 시뮬레이션

### Step 2: 결과 검증
```python
from src.backtest import validate_backtest
validation = validate_backtest(result)
```
- IC > 0.015
- Annual Turnover < 400%
- Optimizer 실패율 < 10%

### Step 3: 성과 확인
```python
metrics = result.compute_metrics()  # geometric annualization
print(result.summary())
```

## Variant 탐색 모드 (legacy grid_search.py 대체)
파라미터 sweep은 variants/*.yaml 매니페스트로 관리:
```bash
# 각 후보 yaml에 대해
python run_variant.py --variant variants/exp_ra_sweep_03.yaml
python run_variant.py --variant variants/exp_ra_sweep_05.yaml
python run_variant.py --variant variants/exp_ra_sweep_07.yaml
# 결과: outputs/<label>/metrics.json 비교
```
yaml 안에 `overrides:` 블록으로 risk_aversion, turnover_penalty, max_te_annual 등 PipelineConfig 필드를 override.

## 하네스 모드 (사전 계산 데이터 사용)
```python
result = run_backtest(
    data,
    precomputed_panel=panel,
    precomputed_targets=targets,
    precomputed_models=models,
    precomputed_predictions=predictions,
    precomputed_raw_predictions=raw_predictions,
    config=config,
)
```

## 성과 목표
| 지표 | 목표 | 비고 |
|------|------|------|
| Information Ratio | >= 1.0 | vs EW 벤치마크 |
| Average IC | > 0.03 | Spearman rank |
| Annual Turnover | 150~200% | 편도 기준 |
| Max Drawdown | > -30% | 절대값 |
| Sharpe Ratio | > 1.0 | |
