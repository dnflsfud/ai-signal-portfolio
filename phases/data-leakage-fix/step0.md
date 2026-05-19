# Step 0: walk-forward-embargo

20일 forward target과 walk-forward 학습 윈도우 사이의 **라벨 누수**를 제거한다. 현재 `train_end`/`val_start`/`val_end`/`predict_idx` 사이의 갭이 모두 0이라, 마지막 학습 샘플의 라벨이 검증 구간으로, 마지막 검증 샘플의 라벨이 실제 예측 구간으로 침범한다.

## 읽어야 할 파일

먼저 아래 파일들을 읽고 walk-forward 학습 흐름과 타겟 정의를 파악하라:

- `CLAUDE.md` (Phase 3 타겟 + Phase 4 모델 섹션)
- `docs/AI_METHODOLOGY.md` §2-§4
- `src/config.py` `PipelineConfig` 전체 (특히 `forward_horizon`, `train_window`, `retrain_freq`, `val_window`)
- `src/target_engine.py` `compute_specific_returns` — 타겟이 `[t, t+forward_horizon]` 누적 수익률임을 직접 확인하라
- `src/model_trainer.py` 전체 (특히 `walk_forward_train` 라인 ~320-400 및 `_prepare_train_data`)

타겟이 forward 20일 수익률이므로 train/val/predict 사이에 최소 `forward_horizon` 일의 갭이 필요한 이유를 코드 안에서 확인한 뒤 작업하라.

## 작업

### 1. `src/config.py`에 embargo 파라미터 추가

`PipelineConfig`에 아래 필드를 `forward_horizon` 직후에 추가:

```python
# Walk-forward train/val/predict 사이의 embargo 갭 (영업일).
# forward_horizon과 같게 두는 것이 표준 (López de Prado 2018).
# 0으로 두면 기존 동작 (라벨 누수). 변경 권장하지 않음.
embargo_days: int = 20
```

`__post_init__`에 sanity assert 추가:

```python
if self.embargo_days < 0:
    raise ValueError(f"embargo_days must be >= 0, got {self.embargo_days}")
if self.embargo_days < self.forward_horizon:
    import warnings
    warnings.warn(
        f"embargo_days={self.embargo_days} < forward_horizon={self.forward_horizon}: "
        "label leak possible.", stacklevel=2)
```

### 2. `src/model_trainer.py` `walk_forward_train` embargo 적용

현재 (라인 ~330):
```python
train_end = t_idx - val_window
val_start = t_idx - val_window
val_end = t_idx
```

변경:
```python
embargo = config.embargo_days
# train labels: [d, d+forward_horizon] must end strictly BEFORE val_start.
train_end = t_idx - val_window - embargo
# val labels: must end strictly BEFORE predict bar (t_idx).
val_start = t_idx - val_window
val_end = t_idx - embargo
```

엣지 케이스: `train_end <= train_start` 또는 `val_end <= val_start`이면 해당 retrain을 스킵하고 직전 모델 재사용(`current_model`, `current_features`, `current_fw`). 이 fallback은 기존 degenerate-model fallback과 동일한 코드 경로 사용.

예측 시점 `t_idx`는 절대 변경하지 마라. 예측은 항상 모든 가용 데이터 끝에서 일어나야 한다.

### 3. 단위 테스트 추가 `tests/test_model_trainer_embargo.py`

`pytest` 기준. 다음 케이스를 커버:

```python
def test_embargo_drops_last_train_samples_within_horizon():
    """train_end가 (t_idx - val_window - embargo)인지 확인."""
    # 합성 데이터: all_dates 길이 1500, train_window=1260, val_window=126,
    # forward_horizon=20, embargo=20.
    # t_idx=1400일 때:
    #   train_end == 1400 - 126 - 20 == 1254
    #   val_start == 1274, val_end == 1380
    # 핵심: train 마지막 라벨의 forward window 끝(1254+20=1274) == val_start.

def test_embargo_zero_matches_legacy():
    """embargo_days=0이면 기존(누수 있는) 동작과 동일."""

def test_embargo_skip_when_window_too_narrow():
    """embargo가 너무 커서 train_end <= train_start인 경우 retrain skip."""
```

테스트는 실제 LightGBM 학습 없이 윈도우 계산만 검증하면 충분 (`walk_forward_train`을 부르지 말고 윈도우 산출 로직을 helper로 추출하거나, monkey-patch로 `train_model`을 mock).

깔끔한 방법: `walk_forward_train`의 윈도우 산출 부분을 작은 helper `_compute_window_bounds(t_idx, train_window, val_window, embargo)`로 추출해 helper만 테스트.

### 4. 변경 없음

다음은 절대 건드리지 마라:
- `src/target_engine.py` PCA 잔차 생성 — 타겟 정의는 그대로
- `retrain_freq`, `train_window`, `val_window` 값 자체
- 예측 단계 (`predict_cross_sectional`)
- EWMA tracker

## Acceptance Criteria

```bash
# 1. 컴파일 / 임포트 무결성
python -c "from src.config import PipelineConfig; c = PipelineConfig(); assert c.embargo_days == 20"
python -c "from src.model_trainer import walk_forward_train; import inspect; src = inspect.getsource(walk_forward_train); assert 'embargo' in src, 'embargo not wired'"

# 2. 단위 테스트
python -m pytest tests/test_model_trainer_embargo.py -v

# 3. 스모크 — production variant 1회 실행 (no-cache로 새 학습)
python run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml --no-cache
# 종료 코드 0 + outputs/iter15_65tkr_reb21_vtg/metrics.json 갱신 확인
test -f outputs/iter15_65tkr_reb21_vtg/metrics.json
```

## 검증 절차

1. 위 AC 커맨드를 실행한다.
2. 아키텍처 체크리스트:
   - `CLAUDE.md` "기술 스택" / "디렉토리 구조" 위반 없는가?
   - PCA 타겟 정의는 그대로인가? (`src/target_engine.py` diff 없어야 함)
   - 새 필드 `embargo_days`가 `DEFAULT_CONFIG`에 노출됐는가?
3. 결과에 따라 `phases/data-leakage-fix/index.json` step 0 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "embargo_days(=20) added to PipelineConfig. walk_forward_train now opens forward_horizon-day gaps between train/val and val/predict. tests/test_model_trainer_embargo.py covers 3 cases. iter15_65tkr_reb21_vtg smoke OK — new metrics.json: IR=<X.XXX>."`
   - 3회 시도 실패 → `"status": "error"`, `"error_message": "<구체적 실패 (예: pytest 실패한 케이스명 + 에러 로그)>"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "<구체 사유>"`

## 금지사항

- **타겟 정의를 바꾸지 마라.** 이유: PCA 잔차 자체는 정확. 라벨 누수는 *시간 경계*의 문제이지 타겟 형식의 문제가 아니다.
- **예측 인덱스 `t_idx`를 미루지 마라.** 이유: production은 항상 최신 데이터를 예측해야 한다. embargo는 학습/검증 데이터의 *끝*만 자른다.
- **`forward_horizon`을 줄이지 마라.** 이유: 20일은 PCA 잔차 + MVO turnover와 묶인 시스템 상수.
- **기존 테스트를 깨뜨리지 마라.** `tests/` 하위 다른 파일이 있으면 모두 통과해야 한다.
- **degenerate fallback과 embargo skip을 하나의 분기로 합치지 마라.** 이유: 두 사유의 진단 로그가 섞이면 운영 시 원인 분리 불가.
