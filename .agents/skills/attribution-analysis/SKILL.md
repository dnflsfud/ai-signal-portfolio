---
name: attribution-analysis
description: "SHAP 기반 모델 해석과 Li et al. 3-component 비선형 분해를 실행한다. '어트리뷰션 실행', 'SHAP 분석', '선형/비선형 분해', '피처 중요도 분석', '모델 해석' 요청 시 이 스킬을 사용."
---

# Attribution Analysis Skill

포트폴리오 성과의 원인을 SHAP과 Li et al. 분해로 분석한다.

## 워크플로우

### Step 1: 어트리뷰션 실행
```python
from src.attribution import run_attribution
attr_result = run_attribution(result, data)
```

내부적으로:
1. 리밸런싱 시점별 SHAP TreeExplainer 실행
2. Li et al. 3-component 분해 (Linear / Marginal NL / Interaction)
3. 피처 그룹별 기여도 집계

### Step 2: 결과 해석

**SHAP Feature Importance** — 피처별 평균 |SHAP value|
```python
shap_importance = attr_result["shap_importance"]  # DataFrame
```

**Linear/Nonlinear Ratio** — 목표 ~50/50
```python
linear_ratios = attr_result["linear_ratios"]  # Series over time
```

**Feature Group Contributions**
| 그룹 | 목표 비중 |
|------|----------|
| Price | ~40% |
| Accounting | ~20% |
| Sellside | ~25% |
| Conditioning | ~7-8% |
| Factor | ~5-8% |

### Step 3: 시장 레짐 분석
```python
from src.analytics import classify_market_direction, classify_volatility_regime
direction = classify_market_direction(returns_21d)
vol_regime = classify_volatility_regime(annualized_vol)
```

## Li et al. 3-Component 분해 해석

| 성분 | 의미 | 비율이 높으면 |
|------|------|-------------|
| Linear | OLS 기울기로 설명 가능한 부분 | 전통적 팩터 노출 |
| Marginal NL | 개별 피처의 비선형 효과 | 단일 피처 threshold 효과 |
| Interaction | 피처 간 상호작용 | 복합 조건 의존 신호 |

## 주의사항
- total_var < 1e-10 → NaN 반환 (0.33/0.34 sentinel 금지)
- float 비교 시 == 0 대신 < 1e-10
- RandomState(42)는 함수 외부에서 1회 생성
