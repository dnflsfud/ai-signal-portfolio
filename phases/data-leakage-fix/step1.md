# Step 1: oos-holdout-default

OOS holdout 메커니즘은 이미 코드에 존재하지만 **default OFF**이고 production variant가 명시적으로 OFF로 강제한다 (`run_variant.py:124`). 이 step은 모드 의미를 뒤집어 *연구/튜닝은 cutoff 강제*가 되도록 만든다. 단 한 번의 "peek"만 허용하고 그 사실을 selection-bias 회계에 기록한다.

## 읽어야 할 파일

먼저 아래 파일들을 읽고 현재 OOS holdout 흐름을 파악하라:

- `src/config.py` `PipelineConfig.enforce_oos_holdout`, `train_cutoff_date` 정의 (라인 ~407-429)
- `src/model_trainer.py` `walk_forward_train`의 OOS cutoff 분기 (라인 ~280-296)
- `run_variant.py` `VALID_TUNING_MODES`, `load_manifest`, `compose_config` 전체
- **이전 step에서 수정된 파일**: `src/config.py` (`embargo_days` 추가됨), `src/model_trainer.py` (embargo 로직)
- `phases/data-leakage-fix/index.json` step 0 summary 확인

이전 step의 embargo 변경이 이 step의 cutoff 로직과 충돌하지 않는지 (둘 다 `walk_forward_train` 안에서 일어남) 확인하라.

## 작업

### 1. `src/config.py` 디폴트 변경

```python
# (기존)
enforce_oos_holdout: bool = False
train_cutoff_date: Optional[str] = None

# (변경 후)
enforce_oos_holdout: bool = True
train_cutoff_date: Optional[str] = "2024-12-31"
```

코멘트도 갱신: 이 디폴트는 "어떤 새 variant도 명시적 opt-out 없이는 2024-12-31 이후 데이터를 학습/예측에 쓸 수 없다"는 의미임을 명시.

### 2. `run_variant.py` `VALID_TUNING_MODES` 재정의

기존 `production`/`tuning`/`oos_verify` → 다음 4개로 확장:

| 모드 | enforce_oos_holdout | 용도 | 부수효과 |
|---|---|---|---|
| `research` | **True** (강제) | 신규 variant 탐색의 표준 모드 (구 `tuning`과 동일) | `train_cutoff_date` 누락 시 abort |
| `oos_verify` | **False** (강제) | candidate 1개에 대해 **1회만** 허용되는 peek | `experiment_inventory.json`의 `n_oos_peeks` +=1 기록 |
| `deploy` | False (강제) | 실제 배포되는 daily_update / production 운영 경로. cutoff 무시. | `outputs/deploy_log.txt`에 실행 시각 + label 기록 |
| `production` | **DEPRECATED** | 구 manifest 호환. 로드 시 warning + `research` 의미로 fallback. | — |

`compose_config` 변경 핵심:

```python
def compose_config(manifest: Dict[str, Any]) -> PipelineConfig:
    overrides = dict(manifest.get("overrides") or {})
    tuning_mode = manifest.get("tuning_mode", "research")  # was "production"

    if tuning_mode == "production":
        logger.warning(
            "tuning_mode='production' is DEPRECATED. Mapping to 'research'. "
            "Use 'deploy' for the actually-deployed daily run."
        )
        tuning_mode = "research"

    if tuning_mode == "research":
        overrides["enforce_oos_holdout"] = True
        if not overrides.get("train_cutoff_date"):
            # config default of "2024-12-31" applies; just verify it's set
            pass
        # explicit cutoff in manifest overrides default — ok
    elif tuning_mode == "oos_verify":
        overrides["enforce_oos_holdout"] = False
        _log_oos_peek(manifest["label"], manifest_path=...)  # writes inventory
    elif tuning_mode == "deploy":
        overrides["enforce_oos_holdout"] = False
        _log_deploy(manifest["label"])
    else:
        raise ValueError(f"unknown tuning_mode: {tuning_mode}")
    ...
```

### 3. `experiment_inventory.json` peek 카운터 추가

해당 파일이 이미 존재한다 (repo root). `_log_oos_peek` helper는:

```python
def _log_oos_peek(label: str, manifest_path: Path) -> None:
    """Append peek record to experiment_inventory.json."""
    inv_path = Path("experiment_inventory.json")
    inv = json.loads(inv_path.read_text(encoding="utf-8")) if inv_path.exists() else {}
    inv.setdefault("oos_peeks", []).append({
        "label": label,
        "manifest": str(manifest_path),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_hash": _git_hash(),  # reuse src.config._git_hash
    })
    inv["n_oos_peeks"] = len(inv["oos_peeks"])
    inv_path.write_text(json.dumps(inv, indent=2, sort_keys=True), encoding="utf-8")
```

호출 위치는 `compose_config` 안. 즉 **실행 전에 기록**된다 (peek가 실패해도 시도는 카운트).

### 4. 기존 variant manifest의 `tuning_mode` 일괄 마이그레이션

다음 변환만 적용:
- `variants/*.yaml` 중 `tuning_mode: production` → `tuning_mode: deploy` (실배포 manifest인 경우) 또는 `tuning_mode: research` (탐색용 manifest인 경우)

분류 기준:
- **deploy**: `variants/iter15_65tkr_reb21_vtg.yaml`, `variants/iter15_FINAL.yaml`, `variants/baseline_v4*.yaml` (실제 production manifest)
- **research**: 그 외 모든 `iter*_*`, `exp_*`, `ablation_*`

확신 없는 manifest는 그대로 두라 (deprecated warning이 뜨지만 동작은 한다).

`variants/`를 모두 훑어 분류 후 git diff로 단순 키 치환만 수행하라. 다른 필드는 절대 건드리지 마라.

### 5. `run_variant.py` `_summarize`에 peek 카운터 출력

```python
inv_path = Path("experiment_inventory.json")
if inv_path.exists():
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    n_peeks = inv.get("n_oos_peeks", 0)
    print(f"  OOS peeks so far: {n_peeks}")
```

## Acceptance Criteria

```bash
# 1. 기본값 확인
python -c "from src.config import PipelineConfig; c = PipelineConfig(); assert c.enforce_oos_holdout is True and c.train_cutoff_date == '2024-12-31'"

# 2. 모드 의미 검증
python -c "
from run_variant import compose_config
m = {'label': 't', 'overrides': {}, 'tuning_mode': 'research'}
cfg = compose_config(m); assert cfg.enforce_oos_holdout is True
m['tuning_mode'] = 'deploy'
cfg = compose_config(m); assert cfg.enforce_oos_holdout is False
"

# 3. deprecated mode warning
python -c "
import warnings, logging
from run_variant import compose_config
with warnings.catch_warnings(record=True):
    cfg = compose_config({'label': 't', 'overrides': {}, 'tuning_mode': 'production'})
    assert cfg.enforce_oos_holdout is True, 'production should fall back to research'
" 2>&1 | grep -i "deprecat" || echo "WARN: deprecation message not surfaced"

# 4. peek 기록 (실제 backtest 안 돌리고 inventory만 검증)
python -c "
from run_variant import compose_config
import json, pathlib
before = json.loads(pathlib.Path('experiment_inventory.json').read_text())
compose_config({'label': 'test_peek_dryrun', 'overrides': {}, 'tuning_mode': 'oos_verify'})
after = json.loads(pathlib.Path('experiment_inventory.json').read_text())
assert after.get('n_oos_peeks', 0) == before.get('n_oos_peeks', 0) + 1, 'peek not counted'
"
# 위 dryrun 기록은 테스트 후 수동으로 inventory에서 제거하지 말 것 — 회계 정직성

# 5. 기존 production manifest 동작 확인
grep -l "tuning_mode: production" variants/ && echo "ERROR: production mode remains in variants" && exit 1 || true
```

## 검증 절차

1. AC 커맨드 전부 통과.
2. 아키텍처 체크리스트:
   - `CLAUDE.md`의 production 운영 흐름(daily_update/update_and_deploy.bat)이 `deploy` 모드로 정상 동작하는가? 실제 호출 1회 (`python run_variant.py --variant variants/iter15_65tkr_reb21_vtg.yaml`) 시 cutoff 무시 + deploy_log.txt에 기록되는가?
   - `experiment_inventory.json` peek 카운터가 atomic하게 증가하는가?
3. `phases/data-leakage-fix/index.json` step 1 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "OOS holdout default flipped to ON (cutoff=2024-12-31). tuning_mode redefined as research/oos_verify/deploy/production-deprecated. N variants migrated. peek counter wired to experiment_inventory.json."`
   - 실패/blocked → 사유 명기

## 금지사항

- **`enforce_oos_holdout`의 강제 OFF 경로(`deploy`, `oos_verify`)에서 cutoff를 *조용히* 우회하지 마라.** 이유: discipline의 본질은 "cutoff를 어겼다는 사실이 남는 것". `oos_verify`는 N_trials에 가산, `deploy`는 별도 로그.
- **기존 `tuning_mode: production` manifest를 자동으로 `research`로 만들지 마라.** 이유: 실배포 manifest가 cutoff에 잡혀 daily_update가 망가질 수 있다. 분류는 수동으로 신중하게.
- **`daily_update.py`를 이 step에서 건드리지 마라.** 이유: scope-out. cutoff 정책은 manifest 레벨에서만 통제.
- **`experiment_inventory.json`의 과거 기록을 수정하지 마라.** 이유: selection-bias 회계의 무결성. 새 record만 append.
