# Step 0: config-fx-surcharge

KRW 상장 종목 (000660 SK Hynix, 005930 Samsung Electronics)의 FX 변환 비용
(KRW↔USD spot bid-ask + slippage)을 모델에 반영하기 위한 config 필드를 추가한다.

현 `one_way_tc` (스칼라 10bp)는 모든 종목에 동일 적용. KRW 종목에는 USD-base
포트폴리오에서 매매시 추가 ~3bp/side FX 비용이 발생함. 본 step은 config 표면만
열고, 실제 wire-in은 step 1에서.

## 읽어야 할 파일

- `src/config.py` `PipelineConfig` 전체 (특히 line 391~ Backtest 섹션, line 397
  `one_way_tc`, line 366~ `mega_cap_funding_*` 패턴 — 유사 디자인 컨벤션)
- `CLAUDE.md` "핵심 파라미터" 표 (현 one_way_tc 정의 확인)

## 작업

### 1. `src/config.py`에 새 필드 + validation 추가

`one_way_tc: float = 0.0010` 직후 (Backtest 섹션 내):

```python
# FX surcharge applied on top of one_way_tc for non-USD listed tickers
# (KRW<->USD spot bid-ask + slippage). Charged per one-way turnover unit
# of the affected ticker. Round-trip cost = 2 * surcharge. Empirical KRW
# institutional flow: ~3 bp per side is conservative-realistic mid-point.
# Default ON with 000660 + 005930 since both are KRX-listed and the only
# non-USD tickers in the current universe. Extend dict to add more KRX
# names. Set to {} to disable.
fx_surcharge_per_ticker: Dict[str, float] = field(
    default_factory=lambda: {
        "000660": 0.0003,  # SK Hynix (KRX)
        "005930": 0.0003,  # Samsung Electronics (KRX)
    }
)
```

`__post_init__` 끝부분에 validation 추가:

```python
# Validate fx_surcharge_per_ticker
import warnings as _w
for _tkr, _sur in self.fx_surcharge_per_ticker.items():
    if _sur < 0:
        raise ValueError(
            f"fx_surcharge_per_ticker[{_tkr!r}]={_sur} must be >= 0"
        )
    if _sur > 0.01:
        _w.warn(
            f"fx_surcharge_per_ticker[{_tkr!r}]={_sur} > 100bp - "
            "unusually high; verify empirical justification.",
            stacklevel=2,
        )
```

`Dict`가 import되어 있는지 확인 — 이미 `typing` import 있으면 그대로, 없으면
`from typing import Dict` 추가.

### 2. 변경 없음

다음은 절대 건드리지 마라:
- `one_way_tc` 기본값 (= 0.0010 유지)
- 기존 dataclass field 순서/이름
- `src/backtest.py` (step 1에서 처리)
- 다른 config 필드의 default

## Acceptance Criteria

```bash
# 1. Import 무결성 + DEFAULT 등록 확인
python -c "
from src.config import DEFAULT_CONFIG
assert isinstance(DEFAULT_CONFIG.fx_surcharge_per_ticker, dict)
assert DEFAULT_CONFIG.fx_surcharge_per_ticker == {'000660': 0.0003, '005930': 0.0003}, \\
    DEFAULT_CONFIG.fx_surcharge_per_ticker
print('OK: fx_surcharge_per_ticker registered with', len(DEFAULT_CONFIG.fx_surcharge_per_ticker), 'tickers')
"

# 2. Validation: negative raises
python -c "
from dataclasses import replace
from src.config import DEFAULT_CONFIG
try:
    replace(DEFAULT_CONFIG, fx_surcharge_per_ticker={'X': -0.001})
    raise AssertionError('expected ValueError on negative surcharge')
except ValueError as e:
    print('OK: negative raises -', e)
"

# 3. 기존 테스트 전체 통과
python -m pytest tests/ -x --tb=short
```

## 검증 절차

1. 위 AC 3개 모두 통과 확인.
2. 결과에 따라 `phases/fx-cost-modeling/index.json` step 0 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "fx_surcharge_per_ticker dict
     added to PipelineConfig with __post_init__ validation. DEFAULT registers
     000660 + 005930 @ 3bp each. tests/ all pass."`
   - 실패 → `"status": "error"`, `"error_message": "<구체적 실패>"`

## 금지사항

- **`one_way_tc` 기본값을 바꾸지 마라.** 이유: FX surcharge는 ADD-ON 이지
  replacement가 아님. 기존 모든 USD 종목의 TC semantics 보존.
- **MVO objective에 FX 비용을 주입하지 마라.** 이유: 본 phase는 measurement
  layer만 변경. strategy behavior 변경은 별도 phase.
- **default dict를 빈 dict로 두지 마라.** 이유: 사용자 결정에 따라 default ON.
  empty default는 silent silence (기존 누락 그대로 유지) 가 되어 회귀 위험.
