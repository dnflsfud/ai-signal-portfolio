# Step 2: validate-and-docs

단위 테스트 작성 + baseline_v5_deploy 재실행으로 metrics diff 측정 + docs 갱신.

## 읽어야 할 파일

- `tests/test_model_trainer_embargo.py` — 단위 테스트 스타일 참고
- `CLAUDE.md` "핵심 파라미터" 표 (한국어 표) 및 "Post-prediction 조정 모듈"
- `AGENTS.md` (CLAUDE.md의 사본 — 동기 유지 필수)
- `docs/AI_METHODOLOGY.md` §6 (TC 단락 위치)
- `docs/BASELINE.md` headline metric 표

## 작업

### 1. `tests/test_fx_surcharge.py` 신규 작성

```python
"""Unit tests for fx_surcharge_per_ticker FX cost layer."""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from src.config import DEFAULT_CONFIG, PipelineConfig


def test_default_config_has_krw_tickers():
    """DEFAULT_CONFIG should register both KRX names at 3 bp each."""
    fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
    assert fx == {"000660": 0.0003, "005930": 0.0003}, fx


def test_negative_surcharge_raises():
    """__post_init__ rejects negative values."""
    with pytest.raises(ValueError, match="must be >= 0"):
        replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={"X": -0.001})


def test_excessive_surcharge_warns():
    """> 100 bp should warn but not raise (operator override allowed)."""
    with pytest.warns(UserWarning, match="unusually high"):
        replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={"X": 0.02})


def test_tc_cost_vector_math_krw():
    """Synthetic walk-1: 5%p delta on a KRW ticker incurs 10+3 = 13 bp."""
    fx = DEFAULT_CONFIG.fx_surcharge_per_ticker
    one_way_tc = DEFAULT_CONFIG.one_way_tc
    tickers = ["AAPL", "000660", "MSFT"]
    delta_w = np.array([0.05, 0.05, 0.05])
    fx_vec = np.array([fx.get(t, 0.0) for t in tickers])
    tc_per_ticker = one_way_tc + fx_vec
    tc_cost = float(np.sum(delta_w * tc_per_ticker))
    # AAPL: 0.05 * 0.0010 = 5e-5
    # 000660: 0.05 * (0.0010 + 0.0003) = 6.5e-5
    # MSFT: 0.05 * 0.0010 = 5e-5
    # total = 1.65e-4
    assert abs(tc_cost - 1.65e-4) < 1e-9, tc_cost


def test_empty_dict_matches_scalar_path():
    """fx_surcharge_per_ticker={} should reduce to legacy scalar TC."""
    cfg = replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={})
    assert cfg.fx_surcharge_per_ticker == {}
    # walk_forward_simulate uses legacy `turnover * one_way_tc` when dict empty.
    # Full backtest equivalence verified in Step 2b regression run.
```

### 2. baseline_v5_deploy 재실행 + diff

```bash
# (a) backup current metrics
cp outputs/baseline_v5_deploy/metrics.json outputs/baseline_v5_deploy/metrics.json.bak

# (b) regenerate (no cache, no dashboard push)
python run_variant.py --variant variants/baseline_v5_deploy.yaml --no-cache

# (c) diff
python -c "
import json
before = json.load(open('outputs/baseline_v5_deploy/metrics.json.bak'))['metrics']
after  = json.load(open('outputs/baseline_v5_deploy/metrics.json'))['metrics']
keys = ['annual_return', 'active_return', 'tracking_error',
        'information_ratio', 'sharpe_ratio', 'annual_tc',
        'avg_annual_turnover']
print(f'{\"metric\":22} {\"before\":>12} {\"after\":>12} {\"delta\":>12}')
print('-' * 62)
for k in keys:
    b, a = before[k], after[k]
    print(f'{k:22} {b:12.6f} {a:12.6f} {a-b:+12.6f}')
"
```

**Acceptance**:
- `annual_tc` 가 **증가** (positive delta)
- `active_return` 이 **감소** (negative delta, magnitude ≈ annual_tc delta)
- `information_ratio` 변화 절댓값 < 0.01
- `sharpe_ratio` 변화 절댓값 < 0.005
- `avg_annual_turnover` 변화 0 (turnover 정의 자체는 안 바뀜)

### 3. 문서 갱신

#### `CLAUDE.md` (그리고 동일 적용 `AGENTS.md`)

"핵심 파라미터" 표에서 `one_way_tc` 행 아래 추가:
```
| **fx_surcharge_per_ticker** | **{000660: 0.0003, 005930: 0.0003}** | KRW 상장 종목의 KRW↔USD spot bid-ask + slippage 보정 (편도 3bp 추가) |
```

성과 헤드라인 박스의 `Annual TC`, `Active Return`, `Information Ratio` 등 변경된
숫자를 step 2 의 diff 출력에 맞춰 갱신.

#### `docs/AI_METHODOLOGY.md` §6 (또는 TC 단락 위치)

TC 설명 단락에 1-2 문장 추가:
```
또한 KRW 상장 종목 (000660 SK Hynix, 005930 Samsung Electronics)에 대해서는
KRW↔USD spot 변환의 bid-ask + slippage 명목으로 편도 3bp 의 FX surcharge를
one_way_tc 위에 추가 적용한다. round-trip 6bp 가 KRW 기관 흐름의
보수-현실적 mid-point 추정치이며, 향후 KRX 종목 추가시 동일 dict에 등록한다.
```

#### `docs/BASELINE.md` headline 표

`Annual Turnover` / `Annual TC` / `Active Return` / `Information Ratio` 값을
새 metrics.json 값으로 미세 갱신. 메모 한 줄: "fx-cost-modeling phase 적용
이후 수치 (KRW FX surcharge 6bp round-trip 반영)."

#### `phases/index.json` 최상위에 phase 추가

```json
{
  "dir": "fx-cost-modeling",
  "status": "completed",
  "completed_at": "<ISO timestamp>"
}
```

## Acceptance Criteria

```bash
# 1. 단위 테스트 pass
python -m pytest tests/test_fx_surcharge.py -v
# 5 passed expected

# 2. 기존 테스트 회귀 없음
python -m pytest tests/ -x --tb=short

# 3. baseline 재실행 결과 acceptance
# (Step 2의 diff 출력 직접 검증 — annual_tc 증가, IR 변화 < 0.01)

# 4. 문서 grep 확인
grep -l "fx_surcharge_per_ticker" CLAUDE.md AGENTS.md docs/AI_METHODOLOGY.md docs/BASELINE.md
```

## 검증 절차

1. AC 4개 통과.
2. `phases/fx-cost-modeling/index.json` step 2 + top-level `phases/index.json`
   동시 업데이트:
   - 성공 → `"status": "completed"`, summary에 실제 측정된 `delta_annual_tc`
     + `delta_active_return` + `delta_IR` 수치 포함.
   - 실패 → 구체적 에러 + 어떤 acceptance가 깨졌는지 명시.

## 금지사항

- **새 baseline 산출 전에 docs 숫자를 갱신하지 마라.** 실제 측정값 ≠ 추정값.
- **outputs/baseline_v4/ 를 건드리지 마라.** v4 는 alias dir, 이 phase 범위 밖.
- **CLAUDE.md ↔ AGENTS.md 동기 깨뜨리지 마라.** 같은 변경을 양쪽에 적용.
- **테스트가 단순히 "code runs without error" 만 검사하지 않게 하라.**
  반드시 수치 정확성 (vector math, validation 에러 메시지) 검증.
