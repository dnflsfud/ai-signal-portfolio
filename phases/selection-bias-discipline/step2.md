# Step 2: rolling-ir-eval

3-구간(P1/P2/P3) 다중 IR 평가를 **단일 통계**로 보강한다. 현재 운영은 "P1 양수 + P2 양수 + P3 양수"라는 3개 상관 목표를 동시 최적화해 N_trials를 실질적으로 3배 증폭한다. 이 step은 (a) rolling 252d IR 분포 통계, (b) Hansen (2005) SPA p-value의 단순화 구현을 추가하고 `run_variant.py` summary에 노출한다.

## 읽어야 할 파일

- `src/utils.py` `compute_performance_metrics` (현재 metrics 계산 위치)
- `src/harness.py` `sub_ir`, `sub_period_irs`
- `run_variant.py` `_summarize`
- `src/analytics.py` (이미 존재하면 — 없으면 신규)
- **이전 step 산출 확인**: `phases/selection-bias-discipline/index.json` step 0, 1 summary
- 참고 (외부 — 읽을 필요 없음, 구현 가이드만):
  - Hansen P.R. (2005), "A Test for Superior Predictive Ability", JBES
  - López de Prado (2018), Ch. 14: bootstrap SPA simplification

## 작업

### 1. `src/analytics.py`에 두 함수 추가 (파일 없으면 신규)

```python
import numpy as np
import pandas as pd

def rolling_ir(active_returns: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """Rolling annualised IR.

    IR_t = mean(active_t-window+1 ... t) / std(...) * sqrt(252)

    Returns a Series indexed identically to active_returns; NaN for the
    first window-1 dates (or wherever std==0 or insufficient observations).
    """

def rolling_ir_stats(active_returns: pd.Series, window: int = 252) -> dict:
    """Summary of rolling_ir distribution.

    Returns:
      {
        'rolling_ir_mean': float,
        'rolling_ir_median': float,
        'rolling_ir_min': float,
        'rolling_ir_max': float,
        'rolling_ir_pos_frac': float,    # fraction of dates with rolling_ir > 0
        'rolling_ir_window': int,
      }
    """

def spa_pvalue(
    active_returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 10,
    seed: int = 42,
) -> float:
    """Simplified single-strategy Hansen SPA test.

    Null: E[active] <= 0 (strategy does not beat benchmark).
    Block-bootstrap t-stat of mean active return; p-value = fraction of
    bootstrap t-stats >= observed t-stat under the null-centred resampling.

    Returns float in [0, 1].

    Note: this is not the full multi-strategy SPA (which would require all
    candidate strategy returns to bound the data-snooping inflation). For
    our use case the candidate set is a single strategy vs benchmark; the
    multi-strategy version comes in step3 if needed.
    """
```

이 단순화 SPA는 *N_trials를 명시적으로 입력받지 않는다*. N_trials 가중은 별도 `haircut_sharpe`로 처리한다 (이 step에서는 추가하지 마라 — `run_selection_bias.py`에 이미 존재).

### 2. `src/utils.py` `compute_performance_metrics` 확장

기존 반환 dict에 다음 키를 추가:

```python
from src.analytics import rolling_ir_stats, spa_pvalue

def compute_performance_metrics(port_returns, bm_returns, ...):
    metrics = {...}  # 기존
    active = port_returns - bm_returns
    metrics.update(rolling_ir_stats(active.dropna(), window=252))
    metrics["spa_pvalue"] = spa_pvalue(active.dropna(), n_bootstrap=1000, block_size=10, seed=42)
    return metrics
```

`run_backtest`의 `result.compute_metrics()`가 자동으로 새 키를 가지게 된다.

### 3. `run_variant.py` `_summarize` 출력 보강

```python
def _summarize(metrics, baseline_path):
    ...
    print(f"  IR          : {ir:.3f}")
    print(f"  Rolling IR  : mean={metrics.get('rolling_ir_mean', float('nan')):.3f}, "
          f"min={metrics.get('rolling_ir_min', float('nan')):.3f}, "
          f"pos_frac={metrics.get('rolling_ir_pos_frac', float('nan')):.2f}")
    print(f"  SPA p-value : {metrics.get('spa_pvalue', float('nan')):.4f}")
    ...
```

### 4. 단위 테스트 `tests/test_analytics.py`

```python
def test_rolling_ir_constant_alpha():
    """일정한 양수 active return → rolling_ir가 점근적으로 안정한 양수."""
def test_rolling_ir_zero_alpha():
    """평균 0인 noise → rolling_ir 분포가 0 중심, pos_frac ≈ 0.5."""
def test_spa_pvalue_strong_alpha():
    """일정한 양수 active → spa_pvalue ≈ 0 (널 기각)."""
def test_spa_pvalue_zero_alpha():
    """평균 0인 noise → spa_pvalue 균등분포 (대략 0.4-0.6)."""
def test_spa_pvalue_reproducible():
    """동일 seed → 동일 p-value."""
```

### 5. 기존 baseline metrics 재산출

```bash
# iter15_FINAL_postfix와 baseline_v5 (있다면)의 metrics.json에 새 키 채우기
python -c "
import pickle, json
for label in ['iter15_FINAL_postfix', 'baseline_v5']:
    pkl = f'outputs/{label}/backtest_result.pkl'
    try:
        r = pickle.load(open(pkl, 'rb'))
    except FileNotFoundError:
        print(f'skip {label} (no result)')
        continue
    m = r.compute_metrics()
    j = json.load(open(f'outputs/{label}/metrics.json'))
    j['metrics'].update({k: m[k] for k in m if k.startswith('rolling_ir_') or k == 'spa_pvalue'})
    json.dump(j, open(f'outputs/{label}/metrics.json','w'), indent=2, default=str)
    print(label, 'rolling_ir_mean=', m['rolling_ir_mean'], 'spa_p=', m['spa_pvalue'])
"
```

## Acceptance Criteria

```bash
# 1. analytics 모듈 + 함수
test -f src/analytics.py
python -c "
from src.analytics import rolling_ir, rolling_ir_stats, spa_pvalue
import pandas as pd, numpy as np
np.random.seed(0)
a = pd.Series(np.random.randn(500)*0.001 + 0.0002, index=pd.date_range('2020-01-01', periods=500, freq='B'))
ri = rolling_ir(a)
assert len(ri) == 500
stats = rolling_ir_stats(a)
assert {'rolling_ir_mean','rolling_ir_min','rolling_ir_pos_frac','rolling_ir_window'} <= set(stats)
p = spa_pvalue(a, n_bootstrap=200)
assert 0.0 <= p <= 1.0
"

# 2. compute_performance_metrics 신규 키 노출
python -c "
import pickle
r = pickle.load(open('outputs/iter15_FINAL_postfix/backtest_result.pkl','rb'))
m = r.compute_metrics()
for k in ['rolling_ir_mean','rolling_ir_min','rolling_ir_pos_frac','spa_pvalue']:
    assert k in m, f'missing {k}'
print({k: m[k] for k in ['rolling_ir_mean','rolling_ir_min','rolling_ir_pos_frac','spa_pvalue']})
"

# 3. 단위 테스트
python -m pytest tests/test_analytics.py -v

# 4. run_variant summary 출력 (1회 fresh run — research mode, 캐시 hit 가능)
python run_variant.py --variant variants/iter15_FINAL_postfix.yaml 2>&1 | tee /tmp/summary.log
grep -q "Rolling IR" /tmp/summary.log
grep -q "SPA p-value" /tmp/summary.log

# 5. metrics.json에 새 키 영구화
python -c "
import json
m = json.load(open('outputs/iter15_FINAL_postfix/metrics.json'))['metrics']
assert 'rolling_ir_mean' in m and 'spa_pvalue' in m
"
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - `src/analytics.py`가 dependency 사이클을 만들지 않는가? (utils.py가 analytics.py를 import — analytics.py는 stdlib + pandas/numpy만)
   - `spa_pvalue`가 deterministic (seed=42 고정)인가?
   - `rolling_ir_stats`가 `window=252` 디폴트로 daily series 가정인가? (영업일 252)
3. `phases/selection-bias-discipline/index.json` step 2 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "src/analytics.py: rolling_ir, rolling_ir_stats, spa_pvalue. compute_performance_metrics extended. run_variant _summarize prints new stats. iter15_FINAL_postfix rolling_ir_mean=<X.XXX>, min=<>, pos_frac=<>, spa_pvalue=<>. baseline_v5 (if exists) similarly back-filled. tests/test_analytics.py 5 cases."`
   - 실패/blocked → 사유

## 금지사항

- **N_trials 가중 (haircut Sharpe)을 이 step에 끼워넣지 마라.** 이유: 이미 `run_selection_bias.py`에 존재. 중복 구현은 정확도 분기를 만든다.
- **`spa_pvalue`를 multi-strategy SPA로 확장하지 마라.** 이유: 현재 ablation/baseline 비교는 단일 strategy. 다중 strategy SPA는 ablation summary 전체를 입력으로 받아야 해 다른 인터페이스.
- **block_size를 fingerprint나 metrics에 포함하지 마라.** 이유: 함수 인자로만 통제. 값을 metrics에 노출하면 비교 시 혼동.
- **`compute_performance_metrics`의 기존 키를 *변경*하지 마라.** 이유: streamlit_mobile.py, dashboard_data.pkl, 외부 노트북이 참조한다. 새 키 *추가*만 허용.
- **`subprocess.run` 으로 `run_selection_bias.py`를 호출하지 마라.** 이유: 결합도 ↑. 두 도구는 독립적으로 유지.
