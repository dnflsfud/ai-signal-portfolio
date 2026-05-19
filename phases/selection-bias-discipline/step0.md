# Step 0: checkpoint-fingerprint

Phase 1/2/4 checkpoint (`outputs/.cache/phase{1,2,4}.pkl` 등)에 *생성 시점의 config fingerprint*를 동봉하고 로드 시 mismatch면 자동 폐기/재계산되게 한다. 현재는 `SAFE_FOR_CACHE_REUSE` 화이트리스트(`run_variant.py`)만으로 가드되어, default config 변경이 silent하게 stale cache를 통과시킬 수 있다.

## 읽어야 할 파일

- `src/backtest.py` 안의 `save_checkpoint`, `load_checkpoint` 함수 (위치 확인: `grep -n "def save_checkpoint\|def load_checkpoint" src/backtest.py`)
- `run_variant.py` `SAFE_FOR_CACHE_REUSE` 화이트리스트와 캐시 로딩 분기 (라인 ~206-280)
- `src/config.py` `PipelineConfig` 필드 목록 (fingerprint에 포함할 키 결정용)
- `outputs/.cache/` 또는 `outputs/iter15_FINAL_postfix/` 등 실제 checkpoint pkl이 어디 저장되는지 확인
- **이전 task 산출**: `phases/data-leakage-fix/`, `phases/overlay-ablation/` 모두 completed 확인
- `phases/selection-bias-discipline/index.json`

## 작업

### 1. Fingerprint 정의

caching이 *반드시* 깨져야 하는 config 키 집합을 하나의 함수로 정의:

```python
# src/backtest.py (또는 신규 src/checkpoint_fingerprint.py)

# Phase 1/2/4 결과에 영향을 주는 config 키. 이 키들이 바뀌면 캐시는 무효.
# SAFE_FOR_CACHE_REUSE의 *여집합*에 해당.
FINGERPRINT_KEYS = (
    "data_path",
    "feature_mode",
    "pca_components", "pca_n_remove", "pca_lookback", "forward_horizon",
    "regime_aware_pca_lookback", "pca_lookback_short", "pca_lookback_long",
    "regime_pca_weighted_enabled", "regime_pca_vix_threshold",
    "macro_cross_enabled",
    "multi_horizon_targets_enabled", "multi_horizon_weights",
    "revision_clean_mode", "revision_clean_threshold",
    "revision_clean_extreme_threshold", "revision_clean_reversion_ratio",
    "train_window", "retrain_freq", "val_window",
    "embargo_days",                    # Task A에서 도입
    "enforce_oos_holdout", "train_cutoff_date",
    "prediction_ema_alpha",
    "lgbm_params",                     # dict — 그대로 직렬화
    "ewma_enabled", "ewma_alpha", "ewma_drop_pct", "ewma_min_features",
    "ewma_min_retrains",
)

def compute_config_fingerprint(config: PipelineConfig) -> str:
    """Stable hex digest of FINGERPRINT_KEYS subset of config.

    Implementation:
      from dataclasses import asdict
      sub = {k: asdict(config)[k] for k in FINGERPRINT_KEYS}
      blob = json.dumps(sub, sort_keys=True, default=str)
      return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]   # 16 hex chars enough
    """
```

### 2. `save_checkpoint` / `load_checkpoint` 패치

기존 checkpoint payload (예: `{"data": ..., "panel": ...}`)에 `_fingerprint` 필드를 추가:

```python
def save_checkpoint(phase: str, payload: dict, config: PipelineConfig) -> None:
    payload = dict(payload)  # shallow copy
    payload["_fingerprint"] = compute_config_fingerprint(config)
    payload["_fingerprint_keys_version"] = 1
    payload["_saved_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    # 기존 pickle.dump 로직 유지
    ...

def load_checkpoint(phase: str, config: PipelineConfig | None = None) -> dict | None:
    """Load checkpoint. If config is provided and fingerprint mismatches, return None."""
    # 기존 로딩 로직
    payload = ...
    if payload is None:
        return None
    if config is not None:
        expected = compute_config_fingerprint(config)
        actual = payload.get("_fingerprint")
        if actual != expected:
            logger.warning(
                "[checkpoint] %s fingerprint mismatch (saved=%s, current=%s) — discarding cache.",
                phase, actual, expected,
            )
            return None
    return payload
```

**핵심 규칙**:
- `config=None`으로 호출되면 backwards-compat: fingerprint 무시 (호환성 위해 유지).
- `config` 인자가 들어오면 무조건 검증. mismatch면 `None` 반환 → 호출자가 재계산.

### 3. `run_variant.py` 호출 측 수정

```python
# 기존
cp1 = load_checkpoint("phase1")
cp2 = load_checkpoint("phase2")
cp4 = load_checkpoint("phase4")

# 변경
cp1 = load_checkpoint("phase1", config=cfg)
cp2 = load_checkpoint("phase2", config=cfg)
cp4 = load_checkpoint("phase4", config=cfg)
```

`SAFE_FOR_CACHE_REUSE` 화이트리스트는 *유지*. fingerprint는 그것의 백업/세컨드 라인이다. 두 가드가 다 통과해야 캐시 재사용.

### 4. 마이그레이션: 기존 캐시 일괄 무효화

기존 캐시는 `_fingerprint` 필드가 없다 → 자동으로 mismatch → 다음 실행 시 자동 재계산. 따라서 명시적 삭제 불필요. 단 step0 종료 시 한 번 강제 무효화하기 위해 다음을 권장:

```bash
# 기존 캐시 archived (지우지 말고 보관)
mkdir -p outputs/.cache/legacy_prefingerprint
mv outputs/.cache/phase*.pkl outputs/.cache/legacy_prefingerprint/ 2>/dev/null || true
```

위 캐시 디렉토리 경로는 실제 `save_checkpoint` 로직에서 확인 후 정확히 맞춰라.

### 5. 단위 테스트 `tests/test_checkpoint_fingerprint.py`

```python
def test_fingerprint_changes_on_key_change():
    """lgbm_params 변경 시 hex digest가 달라지는가."""
def test_fingerprint_stable_on_safe_key_change():
    """rebalance_freq (SAFE_FOR_CACHE_REUSE 멤버) 변경 시 fingerprint 불변."""
def test_load_rejects_stale_cache(tmp_path):
    """save 후 config을 바꿔 load → None 반환."""
def test_load_accepts_matching_cache(tmp_path):
    """save 후 동일 config로 load → payload 반환."""
def test_load_without_config_is_backwards_compat(tmp_path):
    """config=None이면 fingerprint 검증 스킵."""
```

`tmp_path`에 임시 cache dir를 만들고 monkey-patch로 cache 경로를 거기로 돌려라.

## Acceptance Criteria

```bash
# 1. 새 helper 노출
python -c "
from src.backtest import compute_config_fingerprint, FINGERPRINT_KEYS
from src.config import PipelineConfig
fp = compute_config_fingerprint(PipelineConfig())
assert isinstance(fp, str) and len(fp) >= 16
# 동일 config는 동일 fingerprint
assert compute_config_fingerprint(PipelineConfig()) == fp
# SAFE 키 변경은 fingerprint 불변
import dataclasses
c2 = dataclasses.replace(PipelineConfig(), rebalance_freq=42)
assert compute_config_fingerprint(c2) == fp, 'rebalance_freq leaked into fingerprint'
# FINGERPRINT 키 변경은 fingerprint 변동
c3 = dataclasses.replace(PipelineConfig(), train_window=999)
assert compute_config_fingerprint(c3) != fp, 'train_window did not change fingerprint'
print('fingerprint behavior OK')
"

# 2. 단위 테스트
python -m pytest tests/test_checkpoint_fingerprint.py -v

# 3. run_variant 통합 — research mode로 1회 실행, 캐시 생성 확인
python run_variant.py --variant variants/iter15_FINAL_postfix.yaml --no-cache
# 2번째 실행은 캐시 hit → 빠르게 종료
time python run_variant.py --variant variants/iter15_FINAL_postfix.yaml | tee /tmp/run2.log
grep -q "reusing Phase 1/2/4 checkpoints" /tmp/run2.log

# 4. config 변경 시 캐시 거부 (fingerprint mismatch)
# train_window override한 임시 variant
cat > /tmp/test_fp_reject.yaml <<EOF
label: test_fp_reject
out_dir: outputs/test_fp_reject
tuning_mode: research
overrides:
  rebalance_freq: 21
  embargo_days: 20
  train_window: 1300
EOF
python run_variant.py --variant /tmp/test_fp_reject.yaml 2>&1 | tee /tmp/run3.log
grep -q "fingerprint mismatch" /tmp/run3.log || grep -q "cache DISABLED" /tmp/run3.log
# train_window가 SAFE_FOR_CACHE_REUSE에 없으므로 어차피 unsafe로 잡힘 — 두 가드 중 하나 동작 OK

# 5. legacy 캐시 자동 무효화
test -d outputs/.cache/legacy_prefingerprint || echo "WARN: legacy cache may need manual mv"
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - `FINGERPRINT_KEYS`와 `SAFE_FOR_CACHE_REUSE`가 서로의 **여집합**인가? 교집합이 있으면 둘 다 문서화된 의도와 충돌.
   - `lgbm_params`처럼 dict 값은 `json.dumps(..., sort_keys=True, default=str)`로 결정적 직렬화 되는가? (`random_state`는 fingerprint 영향)
   - `multi_horizon_weights` 같은 dict도 같은 방식으로 안정적인가?
3. `phases/selection-bias-discipline/index.json` step 0 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "compute_config_fingerprint + FINGERPRINT_KEYS added. save/load_checkpoint store and verify fingerprint. run_variant.py passes cfg to load_checkpoint. tests/test_checkpoint_fingerprint.py (5 cases). Legacy caches archived to outputs/.cache/legacy_prefingerprint/."`
   - 실패/blocked → 사유

## 금지사항

- **`SAFE_FOR_CACHE_REUSE`를 제거하지 마라.** 이유: fingerprint는 *값 변경* 가드일 뿐이다. SAFE 화이트리스트는 *의미적* 가드 (예: feature_mode가 cache에 영향을 줘야 함을 명시). 두 가드 병존.
- **legacy 캐시를 삭제하지 마라.** rename/archive만. 이유: 디버깅 시 비교 필요.
- **fingerprint를 plaintext로 저장하지 마라.** SHA256 hex만. 이유: 누출 위험은 없지만 일관성.
- **fingerprint mismatch를 silent fallback으로 처리하지 마라.** 반드시 `logger.warning`. 이유: 운영자가 인지해야 한다.
- **`daily_update.py`를 이 step에서 건드리지 마라.** 이유: scope 초과. daily_update는 별도 cache 메커니즘(daily_state.pkl) 사용.
