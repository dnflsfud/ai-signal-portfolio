# Step 1: wire-tc-vectorized

Step 0에서 추가한 `fx_surcharge_per_ticker`를 실제 백테스트의 TC 계산에 연결한다.
스칼라 곱(`turnover * one_way_tc`)을 per-ticker 벡터 곱으로 갈아치우고, `annual_tc`
metric도 누적된 실제 TC를 사용하도록 개선한다.

## 읽어야 할 파일

- `src/backtest.py`:
  - line 1264 부근 — walk_forward_simulate 내부 TC 적용 지점
  - line 667 부근 — `BacktestResult.compute_metrics` 의 `annual_tc` 계산
  - `BacktestResult` dataclass 정의 (tc_costs 필드 추가 위치 파악)
  - `walk_forward_simulate` 함수 signature (tickers 변수 접근 경로 확인)

## 작업

### 1. `walk_forward_simulate` TC 계산 벡터화 (line ~1259-1265)

**Before**:
```python
turnover = float(np.sum(np.abs(new_weights - old_weights)))
tc_cost = turnover * one_way_tc
port_ret -= tc_cost
```

**After**:
```python
delta_w = np.abs(new_weights - old_weights)
turnover = float(np.sum(delta_w))  # two-way L1 - unchanged semantic
fx_surcharge = getattr(config, "fx_surcharge_per_ticker", {})
if fx_surcharge:
    # tickers: column order of weights array. Verify from outer scope.
    fx_vec = np.array([fx_surcharge.get(t, 0.0) for t in tickers], dtype=float)
    tc_per_ticker = one_way_tc + fx_vec
    tc_cost = float(np.sum(delta_w * tc_per_ticker))
else:
    tc_cost = turnover * one_way_tc  # legacy path (backward-compat)
port_ret -= tc_cost
```

`tickers` 변수가 해당 컨텍스트에서 무엇으로 노출되는지 정확히 확인:
- 함수 signature에 있으면 그대로 사용
- 없으면 `new_weights` 가 `pd.Series` 라면 `new_weights.index.tolist()` 사용
- numpy array라면 outer scope `panel.columns` 또는 동등한 ticker order 사용

`config` 가 walk_forward_simulate scope에서 접근 가능한지 확인 — 보통 함수 인자로
전달됨. 아니면 가까운 caller에서 dict만 끌어와서 전달하는 패턴 사용.

### 2. 일별 TC 누적 + `BacktestResult.tc_costs` 필드

`walk_forward_simulate` 본문 상단 (loop 시작 전):
```python
daily_tc_costs = []
```

매 step 끝부분 (port_ret 계산 직후):
```python
daily_tc_costs.append(tc_cost)
```

함수 return / BacktestResult 구성 부분:
```python
# attach
result.tc_costs = pd.Series(daily_tc_costs, index=date_index_used)
```

`BacktestResult` dataclass에 새 optional 필드 추가:
```python
tc_costs: Optional[pd.Series] = None  # daily actual TC, populated by walk_forward_simulate
```

`date_index_used` 는 portfolio returns와 동일 index. 정확한 변수명은 실제 코드
확인 시 결정.

### 3. `compute_metrics` `annual_tc` 갱신 (line ~667)

**Before**:
```python
annual_tc = avg_turnover_two_way * ONE_WAY_TC
```

**After**:
```python
# Prefer accumulated actual TC (handles per-ticker rates correctly).
# Fallback to legacy scalar approximation for old pkls without tc_costs.
if hasattr(self, "tc_costs") and self.tc_costs is not None and len(self.tc_costs) > 0:
    total_tc = float(self.tc_costs.sum())
    annual_tc = total_tc * (252.0 / len(self.tc_costs))
else:
    annual_tc = avg_turnover_two_way * ONE_WAY_TC
```

### 4. 변경 없음

- `turnover` 정의 (two-way L1) 그대로
- portfolio_optimizer 의 turnover_penalty objective 그대로
- 다른 metric (`avg_annual_turnover`, `avg_annual_turnover_one_way`, `sharpe_ratio`,
  `information_ratio`, `active_return`) 의 계산 코드는 손대지 않음. 단, `active_return`
  은 `port_ret` 의 누적이므로 TC 차감 결과 자동 반영.

## Acceptance Criteria

```bash
# 1. Import 무결성
python -c "
from src.backtest import BacktestResult
import inspect
src = inspect.getsource(BacktestResult)
assert 'tc_costs' in src, 'tc_costs field not added to BacktestResult'
print('OK: BacktestResult has tc_costs field')
"

# 2. Backward-compat: empty dict matches scalar (no per-ticker)
python -c "
from dataclasses import replace
from src.config import DEFAULT_CONFIG
cfg_empty = replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={})
# This must execute the legacy path; runtime check via tests (Step 2a)
print('OK: empty dict cfg replaces successfully')
"

# 3. Default cfg vector math sanity
python -c "
import numpy as np
from src.config import DEFAULT_CONFIG
fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
tickers = ['AAPL', '000660', 'MSFT']
delta_w = np.array([0.05, 0.05, 0.05])
fx_vec = np.array([fx.get(t, 0.0) for t in tickers])
tc_per_ticker = DEFAULT_CONFIG.one_way_tc + fx_vec
tc_cost = float(np.sum(delta_w * tc_per_ticker))
expected = 0.05 * 0.001 + 0.05 * 0.0013 + 0.05 * 0.001  # = 0.000165
assert abs(tc_cost - expected) < 1e-9, f'{tc_cost} vs {expected}'
print(f'OK: tc_cost={tc_cost:.6f} matches expected {expected:.6f}')
"

# 4. 기존 테스트 전체 통과
python -m pytest tests/ -x --tb=short
```

## 검증 절차

1. AC 4개 모두 통과 확인.
2. `phases/fx-cost-modeling/index.json` step 1 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "TC vectorized at walk_forward_simulate
     (per-ticker vec) + accumulated daily_tc_costs on BacktestResult. annual_tc metric
     now uses actual TC sum with legacy fallback. tests/ all pass."`
   - 실패 → 구체적 에러 + ticker 변수 매핑 문제 명시.

## 금지사항

- **`turnover` 정의를 바꾸지 마라.** two-way L1 의미 보존 (다른 metric/리포트에서 사용).
- **`port_ret` 차감 외 다른 위치에서 TC 적용하지 마라.** 이중 차감 위험.
- **legacy pkl 호환성을 깨뜨리지 마라.** `tc_costs is None` fallback 필수.
- **`fx_surcharge_per_ticker` 가 dict가 아닌 경우 silent default 처리하지 마라.**
  `getattr(config, ..., {})` 만 사용 (None은 fail-fast가 낫지만 dict 기본값으로 안전화).
- **`walk_forward_train` 은 건드리지 마라.** 학습 흐름은 step 0의 embargo 작업과 완전 분리.
