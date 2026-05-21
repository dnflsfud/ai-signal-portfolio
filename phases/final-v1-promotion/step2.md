# Step 2: production-cutover

production daily 경로 (`update_and_deploy.bat` → `update_and_deploy.py`)는 현재 leaky 환경의 `variants/iter15_65tkr_reb21_vtg.yaml` (`tuning_mode: deploy`)를 사용한다. 이 step은 그 entry point를 **새 baseline_v5 deploy variant**로 옮긴다. step 0이 PROMOTION-ELIGIBLE을 확인하고 step 1이 selection-bias 측정을 마친 상태에서만 진행한다.

핵심 설계 결정 (변경 불가):
- **새 deploy variant는 `tuning_mode: deploy`** — research 모드의 cutoff는 production에서 적용되지 않는다 (`docs/BASELINE.md` "deploy vs research separation" 참조).
- **`outputs/baseline_v4/` alias는 유지** — dashboard `streamlit_mobile.py` 가 그 경로를 hard-code. 새 deploy variant 결과를 그 디렉토리로 promote 한다.
- **`outputs/baseline_v4_legacy/`** 로 기존 leaky baseline_v4 artifact를 1회 백업해 audit trail 보존.

## 읽어야 할 파일

- `update_and_deploy.bat` (entry point)
- `update_and_deploy.py` (full / incremental 분기, variant 호출)
- `daily_update.py` (incremental 모드 동작)
- `variants/iter15_65tkr_reb21_vtg.yaml` (현재 production variant)
- `variants/baseline_v5.yaml` (research candidate)
- `docs/UPDATE_AND_DEPLOY_FLOW.md`
- `streamlit_mobile.py` (dashboard 가 어떤 pkl/csv를 읽는지)
- `scripts/build_dashboard_data.py`
- `phases/final-v1-promotion/index.json` step 0, 1 summary
- `outputs/baseline_v4/README.md`

## 작업

### 1. 새 deploy variant 생성

`variants/baseline_v5_deploy.yaml`:

```yaml
label: baseline_v5_deploy
description: >
  Production deploy version of baseline_v5. Same overrides as baseline_v5.yaml
  but tuning_mode: deploy (cutoff OFF, no peek consumed). Used by
  update_and_deploy.bat after final-v1-promotion phase. Compare to legacy
  iter15_65tkr_reb21_vtg.yaml — this replaces it.
out_dir: outputs/baseline_v5_deploy
tuning_mode: deploy
overrides:
  # MIRROR baseline_v5.yaml overrides exactly — copy them in directly,
  # do not import. The 'oos_verify' field on baseline_v5.yaml is replaced
  # by 'deploy' here; everything else identical.
  <COPY ALL overrides FROM baseline_v5.yaml HERE>
```

`<COPY ALL ...>`는 baseline_v5.yaml 의 overrides 블록을 그대로 복사. tuning_mode만 다르고 동일해야 한다.

### 2. 기존 baseline_v4 artifact 백업

```bash
cp -r outputs/baseline_v4 outputs/baseline_v4_legacy
echo "Archived 2026-05-XX from pre-cutover production path. See phases/final-v1-promotion/." > outputs/baseline_v4_legacy/ARCHIVE_NOTE.md
```

(Windows 상에서는 PowerShell `Copy-Item -Recurse outputs/baseline_v4 outputs/baseline_v4_legacy`)

### 3. `update_and_deploy.py` 패치

현재 코드에서 `iter15_65tkr_reb21_vtg.yaml` 을 호출하는 부분을 찾아 `baseline_v5_deploy.yaml`로 변경. 변수가 있다면 1줄 수정, hard-code면 그 위에 명시적 상수로 분리:

```python
DEPLOY_VARIANT_MANIFEST = "variants/baseline_v5_deploy.yaml"  # was iter15_65tkr_reb21_vtg.yaml until 2026-05-XX
DEPLOY_OUT_ALIAS = "outputs/baseline_v4"  # dashboard hard-coded path; do not rename
```

그리고 variant 실행 후 `outputs/baseline_v5_deploy/` → `outputs/baseline_v4/` rsync/copy 단계가 이미 있어야 dashboard 호환. 없으면 추가.

### 4. dashboard wiring 점검

`scripts/build_dashboard_data.py` 가 `--run outputs/baseline_v4` 로 호출되는지 확인. 그렇다면 변경 불필요. 만약 hard-coded variant name (`iter15_65tkr_reb21_vtg`) 가 있으면 alias 사용으로 바꿔라.

### 5. 1회 end-to-end smoke test

```bash
# full mode 한 번 실제로 돌려 production path 동작 확인
update_and_deploy.bat
# 또는 PowerShell에서:
#   .\update_and_deploy.bat
```

성공 조건:
- 에러 코드 0 종료
- `outputs/baseline_v4/backtest_result.pkl` 의 mtime 이 갱신됨
- `outputs/baseline_v4/metrics.json` 의 `information_ratio` 가 baseline_v5 의 deploy 환경 (cutoff OFF) 값과 일치 (≈ 1.3 근처 예상; 정확한 숫자는 v5 deploy variant 첫 실행 후 확정)
- `dashboard_data.pkl` 가 갱신됨

### 6. 문서 업데이트

- `docs/UPDATE_AND_DEPLOY_FLOW.md` 의 variant 이름 갱신, "Cutover history" 섹션을 추가 (legacy → baseline_v5_deploy 전환 날짜 + 이유)
- `outputs/baseline_v4/README.md` 에 새 entry: "이제 baseline_v4 directory는 baseline_v5_deploy의 결과물을 alias로 가진다. legacy artifact는 baseline_v4_legacy/ 에 보존."
- `CLAUDE.md` "프로덕션 구성" 표의 deploy variant 참조를 변경

## Acceptance Criteria

```bash
# 1) New deploy variant exists with tuning_mode: deploy
test -f variants/baseline_v5_deploy.yaml
grep -q "tuning_mode: deploy" variants/baseline_v5_deploy.yaml

# 2) Legacy backup created
test -d outputs/baseline_v4_legacy
test -f outputs/baseline_v4_legacy/ARCHIVE_NOTE.md

# 3) update_and_deploy.py references new variant
grep -q "baseline_v5_deploy" update_and_deploy.py

# 4) End-to-end smoke test succeeds
update_and_deploy.bat
# (Manual check: exit code 0, baseline_v4 pkl mtime updated)

# 5) Docs updated
grep -q "baseline_v5_deploy" docs/UPDATE_AND_DEPLOY_FLOW.md
grep -q "baseline_v4_legacy" outputs/baseline_v4/README.md
```

## 검증 절차

1. AC 1–3, 5 PASS 후 AC 4 실행.
2. AC 4 smoke test 성공 후 산출물 점검:
   - `outputs/baseline_v4/metrics.json` 의 IR 이 baseline_v5_deploy 첫 실행 IR과 매치 (오차 < 0.005).
   - `dashboard_data.pkl` mtime 갱신.
3. dashboard repo (`cc2-dashboard`)로 push 가 자동 발생했는지 확인. 발생했다면 streamlit cloud 재배포가 트리거됨.
4. `phases/final-v1-promotion/index.json` step 2 status `completed`, summary에 cutover 완료 + 새 deploy IR + push 여부 명시.

## 금지사항

- **`outputs/baseline_v4/`를 삭제하지 마라.** 이유: dashboard hard-code 경로다. **alias로 재사용**하되 디렉토리 자체는 유지.
- **`outputs/baseline_v4_legacy/`를 git ignore 하지 마라.** 이유: legacy artifact의 audit trail은 commit 되어야 한다 (적어도 README + metrics.json은).
- **embargo_days / train_cutoff_date 를 deploy variant에서 ON 으로 두지 마라.** 이유: production daily는 전체 데이터 사용이 정상 동작이다. `tuning_mode: deploy` 가 `compose_config` 에서 자동으로 cutoff를 OFF 시키는 것을 신뢰하라. 명시적 override 추가 금지.
- **`iter15_65tkr_reb21_vtg.yaml` 을 삭제하지 마라.** 이유: deploy 경로에서 빠지더라도 legacy 재현 (`outputs/baseline_v4_legacy/` 재생성용) 으로 보존한다.
- 기존 테스트를 깨뜨리지 마라.
