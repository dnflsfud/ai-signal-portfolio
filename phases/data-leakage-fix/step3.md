# Step 3: propagate-baseline-docs

새 baseline (`iter15_FINAL_postfix`)을 canonical로 등록한다. CLAUDE.md, BASELINE.md, ROADMAP.md, AI_METHODOLOGY.md의 수치/표기를 일관되게 갱신한다.

## 읽어야 할 파일

- `outputs/iter15_FINAL_postfix/metrics.json` (Task A step 2 산출)
- `outputs/iter15_FINAL_postfix/comparison.md` (legacy vs postfix Δ)
- `outputs/iter15_FINAL_postfix/experiment_manifest.json` (config snapshot + git hash)
- `docs/BASELINE.md` 현재 내용
- `docs/ROADMAP.md` 현재 내용
- `docs/AI_METHODOLOGY.md` 현재 내용
- `CLAUDE.md` "최종 성과" / "Selection Bias 검증" / "검증 체크리스트" 섹션
- **이전 step 산출물**: `variants/iter15_FINAL_postfix.yaml`
- `phases/data-leakage-fix/index.json` step 2 summary

## 작업

### 1. `docs/BASELINE.md` 갱신

- canonical baseline 정의를 `iter15_65tkr_reb21_vtg` → `iter15_FINAL_postfix`로 교체
- 이전 baseline의 수치는 "legacy (pre-leak-fix)" 박스에 보존 (지우지 마라)
- 새 baseline의 metrics + sub-period IR 표 추가
- gate criteria 갱신:
  - **rule**: 새 variant는 `tuning_mode: research`로 실행되어야 한다 (cutoff 강제)
  - **rule**: production promote는 단일 `tuning_mode: oos_verify` 실행 후 IR ≥ baseline 시에만 (이 정책은 Task B step3에서 시행)

### 2. `CLAUDE.md` 갱신

다음 3개 섹션:

a) **"## 핵심 파라미터"** 표에 추가:
- `embargo_days = 20` (Walk-forward, NEW)
- `enforce_oos_holdout = True` (NEW default)
- `train_cutoff_date = "2024-12-31"` (NEW default)

b) **"### 최종 성과"** 박스 교체:
- 출처: `outputs/iter15_FINAL_postfix/metrics.json`
- 모든 수치를 새 baseline으로
- 표 위에 한 줄 주의: *"Window restricted to ≤ 2024-12-31 due to OOS hold-out. 2025-01-01 이후는 reserved OOS이며 향후 oos_verify peek로만 측정."*

c) **"### Selection Bias 검증"** 박스:
- 이 결과는 *embargo + cutoff 적용 이전* 환경의 측정이므로 stale임을 명시
- 재측정 일정은 Task C step2의 rolling-IR 도입 후로 미룬다고 적어라
- 기존 DSR=0.17, p=0.43 수치는 그대로 두되 "legacy environment" 박스로 감싸라

d) **"### 검증 체크리스트"** 1번 갱신:
- ✅ Look-ahead bias 없음 (backtest 실행 타이밍 + walk-forward embargo)
- 9번 갱신: IR 수치를 새 baseline으로

### 3. `docs/ROADMAP.md` 갱신

- "## Phase 1 — Completed (2026-04-20)" 항목 아래에 새 항목 추가:
  ```markdown
  ## Phase 2.6 — Data leakage fix (2026-MM-DD, COMPLETED)
  - ✅ walk-forward embargo (forward_horizon=20)
  - ✅ OOS hold-out default ON (cutoff 2024-12-31)
  - ✅ tuning_mode redefined: research / oos_verify / deploy
  - ✅ Canonical baseline recomputed → iter15_FINAL_postfix (IR=<X.XXX>, legacy 1.310, Δ=<+/-Y.YYY>)
  - Knock-on: prior IR comparisons in this doc are stale until Task B
    (overlay-ablation) and Task C (selection-bias-discipline) complete.
  ```

- 기존 Phase 2 P2 IR 목표 (+0.107 → +0.40 등)에 "**SUPERSEDED — Task B ablation으로 대체**" 주석. 수치는 지우지 말고 strikethrough 또는 박스.

### 4. `docs/AI_METHODOLOGY.md` 갱신

- §4 "모델: LightGBM walk-forward" 끝에 다음 문단 추가:
  ```markdown
  ### 라벨 누수 방지 (2026-MM-DD 추가)
  walk-forward의 train_end ~ val_start, val_end ~ predict 사이에
  `embargo_days = forward_horizon = 20` 영업일 갭을 강제한다. 갭 없이는
  20일 forward 타겟의 라벨 윈도우가 검증/예측 구간으로 침범해 early
  stopping이 미래 정보로 결정된다 (López de Prado 2018).
  ```
- 핵심 파라미터 박스에 `embargo_days: 20` 추가

### 5. `outputs/baseline_v4/` 처리 — 신중

**현재 production이 가리키는 디렉토리**다. 다음 옵션 중 선택:

- **옵션 A (권장)**: `baseline_v4/`는 *deploy* 모드의 산출물이므로 그대로 둔다. 단 `baseline_v4/README.md` (없으면 생성)에 "이 디렉토리는 cutoff 무시 deploy run의 산출. 연구/평가 기준선은 `iter15_FINAL_postfix/`" 한 줄 명기.
- **옵션 B**: `baseline_v4/`를 `legacy_baseline_v4/`로 rename. 단 `update_and_deploy.bat`, `daily_update.py`, `streamlit_mobile.py`, `scripts/build_dashboard_data.py`에서 경로 참조를 모두 수정해야 함. **이 step에서는 시도하지 마라** (scope 초과).

옵션 A로 가라.

## Acceptance Criteria

```bash
# 1. 문서가 새 baseline 수치를 반영
NEW_IR=$(python -c "import json; print(json.load(open('outputs/iter15_FINAL_postfix/metrics.json'))['metrics']['information_ratio'])")
grep -q "$NEW_IR" docs/BASELINE.md   # exact float may not match; allow short form
grep -q "iter15_FINAL_postfix" docs/BASELINE.md
grep -q "iter15_FINAL_postfix" CLAUDE.md
grep -q "embargo_days" CLAUDE.md
grep -q "embargo" docs/AI_METHODOLOGY.md
grep -q "Phase 2.6" docs/ROADMAP.md || grep -q "data leakage fix" docs/ROADMAP.md

# 2. legacy 수치 박스가 보존됐는지 (지우지 않았는지)
grep -q "1.310" CLAUDE.md   # legacy IR 값이 어딘가에 legacy 박스로 살아 있어야

# 3. baseline_v4/README.md 옵션 A 확인
test -f outputs/baseline_v4/README.md
grep -q "iter15_FINAL_postfix" outputs/baseline_v4/README.md

# 4. production manifest는 건드리지 않았는지
git diff variants/iter15_65tkr_reb21_vtg.yaml | head -1   # 비어야 함

# 5. CLAUDE.md/AGENTS.md sync 확인 (CLAUDE.md = AGENTS.md 사본 정책이 있다면)
diff CLAUDE.md AGENTS.md || echo "WARN: drift between CLAUDE.md and AGENTS.md — sync if policy demands"
```

## 검증 절차

1. AC 커맨드 통과.
2. 아키텍처 체크리스트:
   - `CLAUDE.md` "Selection Bias 검증" 박스의 수치가 stale로 마킹됐는가?
   - `docs/BASELINE.md`가 새/legacy 둘 다 노출하고 있는가?
   - ROADMAP의 Phase 2 P2 IR 목표가 SUPERSEDED 처리됐는가?
3. `phases/data-leakage-fix/index.json` step 3 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "Docs propagated: BASELINE.md/CLAUDE.md/ROADMAP.md/AI_METHODOLOGY.md updated to iter15_FINAL_postfix as canonical baseline (IR=<X.XXX>). Legacy 1.310 preserved as 'pre-leak-fix' reference. baseline_v4/README.md notes deploy vs research distinction. Task A complete — Task B unblocked."`
   - 실패/blocked → 사유

## 금지사항

- **legacy 수치를 모두 지우지 마라.** 이유: 누수 환경 vs 정직한 환경의 Δ가 selection-bias 회계의 핵심 입력. 보존해야 한다.
- **`outputs/baseline_v4/`를 삭제하거나 rename하지 마라.** 이유: production 운영 경로(`daily_update.py`, `streamlit_mobile.py`, `scripts/build_dashboard_data.py`)가 이 경로를 참조한다.
- **`variants/iter15_65tkr_reb21_vtg.yaml`을 수정하지 마라.** 이유: deploy 모드 manifest는 그대로 유지. 새 baseline은 별도 manifest(`iter15_FINAL_postfix.yaml`)로만 존재.
- **새 manifest를 promote 스크립트로 자동 복사하지 마라.** 이유: production promote는 Task B step3에서 ablation 통과 후 결정.
- **CLAUDE.md만 수정하고 AGENTS.md를 빠뜨리지 마라** (또는 그 반대). 두 파일은 mirror 관계 (CLAUDE.md에 명시).
