---
name: selection-bias-check
description: "백테스트 결과의 Selection Bias(다중 비교 편향)를 검증한다. 'selection bias 검사', '편향 검증', 'deflated sharpe', '다중 비교 보정', 'data snooping 체크', 'grid search 편향', '생존 편향', 'backtest 신뢰성', 'haircut sharpe' 요청 시 반드시 이 스킬을 사용."
---

# Selection Bias Check Skill

Multiple Comparisons Problem에서 발생하는 Selection Bias를 통계적으로 검증한다.

## 배경: 왜 Selection Bias를 검사하는가?

Grid search에서 100개 파라미터 조합을 시도하면, 최적 조합의 Sharpe Ratio는 **실제보다 과대추정**된다. 이는 "100명이 동전을 던져서 가장 많이 앞면 나온 사람을 실력자로 부르는 것"과 같다. Deflated Sharpe Ratio는 이 과대추정을 통계적으로 보정한다.

## 워크플로우

### Step 1: 데이터 수집
```python
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# 백테스트 결과 로드
with open("outputs/backtest_result.pkl", "rb") as f:
    result = pickle.load(f)

port = result.portfolio_returns.dropna()
bm = result.benchmark_returns.dropna()
active = port - bm
```

### Step 2: 시행 횟수(N) 결정

N = 파이프라인에서 시도한 **모든 선택의 곱**:
```python
n_grid_combos = 1      # grid_search를 실행했다면 해당 조합 수
n_pca_variants = 2     # PCA n_remove=2 vs n_remove=5
n_rebal_variants = 1   # rebalance_freq 변형 수
n_feature_variants = 1 # 피처 셋 변형 수

N = max(n_grid_combos * n_pca_variants * n_rebal_variants * n_feature_variants, 1)
```

### Step 3: Deflated Sharpe Ratio (DSR) 계산
```python
T = len(port)  # 관측 일수
SR = active.mean() / active.std() * np.sqrt(252)  # annualized
skew = active.skew()
kurt = active.kurtosis() + 3  # excess → raw kurtosis

# SR의 표준오차
sigma_SR = np.sqrt((1 - skew * SR / np.sqrt(252) + (kurt - 1) / 4 * (SR / np.sqrt(252))**2) / T)

# 다중 비교 보정: N번 시행 시 기대 최대 SR
from scipy import stats
E_max_SR = sigma_SR * stats.norm.ppf(1 - 1 / N) if N > 1 else 0

# Deflated SR
DSR = (SR - E_max_SR) / sigma_SR if sigma_SR > 0 else 0
p_value = 1 - stats.norm.cdf(DSR)
dsr_pass = p_value < 0.05
```

### Step 4: Minimum Track Record Length (MinTRL)
```python
z_alpha = stats.norm.ppf(0.95)  # 5% significance
SR_annual = SR
SR_daily = SR / np.sqrt(252)

MinTRL_days = 1 + (1 - skew * SR_daily + (kurt - 1) / 4 * SR_daily**2) * (z_alpha / SR_daily)**2
MinTRL_years = MinTRL_days / 252

sufficient = T > MinTRL_days
```

### Step 5: Grid Search Haircut
```python
haircut = sigma_SR * np.sqrt(2 * np.log(N)) if N > 1 else 0
adjusted_SR = SR - haircut
grid_pass = adjusted_SR > 0
```

### Step 6: Universe Survivorship Bias
```python
# 종목별 데이터 시작일 확인
from src.data_loader import UniverseData, TICKERS
data = UniverseData("./data/ai_signal_data.xlsx")

backtest_start = port.index[0]
late_entrants = []
for ticker in TICKERS:
    first_valid = data.returns[ticker].first_valid_index()
    if first_valid is not None and first_valid > backtest_start + pd.Timedelta(days=30):
        late_entrants.append((ticker, first_valid.strftime('%Y-%m-%d')))

survivorship_clean = len(late_entrants) == 0
```

### Step 7: Sub-period Stability
```python
n = len(active)
third = n // 3
periods = [
    ("Period 1", active.iloc[:third]),
    ("Period 2", active.iloc[third:2*third]),
    ("Period 3", active.iloc[2*third:]),
]

period_results = []
for name, sub in periods:
    sub_ir = sub.mean() / sub.std() * np.sqrt(252) if sub.std() > 0 else 0
    period_results.append((name, sub.index[0], sub.index[-1], sub_ir))

all_positive = all(r[3] > 0 for r in period_results)
```

### Step 8: Feature Snooping Check (선택사항)
- 전체 피처 IC vs 랜덤 피처 셔플 IC 비교
- references/feature_snooping_methodology.md 참조

### Step 9: 보고서 생성
```python
report = f"""# Selection Bias Analysis Report

## 1. Summary Verdict
- **{verdict}** — DSR p={p_value:.4f}, Adjusted SR={adjusted_SR:.2f}, MinTRL={MinTRL_years:.1f}yr

## 2. Deflated Sharpe Ratio
- Observed SR: {SR:.2f}
- Number of trials (N): {N}
- Expected max SR under null: {E_max_SR:.2f}
- Deflated SR: {DSR:.2f} (p-value: {p_value:.4f})
- Verdict: **{"PASS" if dsr_pass else "FAIL — 다중 비교 보정 후 유의하지 않음"}**

## 3. Minimum Track Record Length
- Required: {MinTRL_years:.1f} years ({MinTRL_days:.0f} trading days)
- Available: {T/252:.1f} years ({T} trading days)
- Verdict: **{"SUFFICIENT" if sufficient else "INSUFFICIENT — 데이터 부족"}**

## 4. Grid Search Bias
- Combinations tested: {N}
- Haircut: {haircut:.2f}
- Adjusted SR: {adjusted_SR:.2f}
- Verdict: **{"PASS" if grid_pass else "WARN — 보정 후 SR <= 0"}**

## 5. Universe Survivorship
- Late entrants: {late_entrants if late_entrants else "None"}
- Verdict: **{"CLEAN" if survivorship_clean else "WARN — 생존 편향 의심 종목 존재"}**

## 6. Sub-period Stability
{period_table}
- Verdict: **{"STABLE" if all_positive else "UNSTABLE — 시기 의존적 성과"}**
"""

Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/reports/selection_bias_report.md").write_text(report, encoding="utf-8")
```

## 판정 기준 요약

| 검증 항목 | PASS 조건 | WARN 조건 | FAIL 조건 |
|----------|-----------|-----------|-----------|
| Deflated SR | p < 0.05 | 0.05 <= p < 0.10 | p >= 0.10 |
| MinTRL | T > MinTRL | T > MinTRL * 0.8 | T < MinTRL * 0.8 |
| Grid Haircut | Adj SR > 0.5 | 0 < Adj SR <= 0.5 | Adj SR <= 0 |
| Survivorship | late entrants = 0 | 1-2 late entrants | 3+ late entrants |
| Sub-period | 3/3 IR > 0 | 2/3 IR > 0 | 1/3 이하 |

## 전체 Verdict 결정
- **PASS**: 모든 항목 PASS
- **WARN**: 1~2개 WARN, FAIL 없음
- **FAIL**: 1개 이상 FAIL

## 참고 문헌
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
- 상세 수학 유도: `references/deflated_sharpe_derivation.md`
