# Step 1: classify-and-propose

이 step은 **READ-ONLY**다. 어떤 source 파일도 수정하지 마라. 산출물은 `phases/update-deploy-cleanup/_research/cleanup_proposal.md` 단 하나다.

이 step은 **user gate** 역할도 한다. 완료 후 사용자가 산출물을 직접 검토하기 전까지 step2/step3는 실행되지 않는다.

## 목표

repo root + scripts/의 모든 .py 파일을 다음 세 가지 verdict로 분류한 표를 evidence와 함께 작성한다:

- **KEEP** — load-bearing (production 흐름이나 운영 워크플로의 일부)
- **DELETE** — 옛 버전, 실험, 테스트 — 외부 reference 0개
- **NEEDS_USER_DECISION** — 자동 분류로는 판단 불가 (사용자가 직접 결정)

## 읽어야 할 파일

먼저 다음 파일들을 반드시 읽어라:

### 이전 step 산출물 (필수)
- `phases/update-deploy-cleanup/_research/bat_flow_trace.md` — 어떤 스크립트가 production 흐름에 묶여 있는지
- `phases/update-deploy-cleanup/_research/script_inventory.md` — 분류 대상 전체 목록

### 사전 분류 가이드 (외부 입력)
이전 분석에서 다음이 확인되었다 — 반드시 이대로 분류하라:

**확정 KEEP (절대 DELETE로 분류하지 마라):**
| 파일 | 이유 |
|---|---|
| `update_and_deploy.bat`, `update_and_deploy.py` | production entry-point |
| `run_variant.py` | Stage 2 full mode 백엔드 |
| `daily_update.py` | Stage 2 incremental mode 백엔드 |
| `scripts/build_dashboard_data.py` | Stage 3에서 호출 |
| `streamlit_mobile.py` | Stage 4/5 대시보드 |
| `run_selection_bias.py` | Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) 검증 — `CLAUDE.md`, `AGENTS.md`, `experiment_inventory.json`, `main.py`에서 참조 |
| `streamlit_app.py` | legacy desktop 대시보드. `CLAUDE.md`, `AGENTS.md`, `run_all.py`, `main.py`에서 참조 — 함부로 삭제하면 docs reference 깨짐 |

**확정 DELETE 후보 (외부 reference grep으로 0개 확인 시 DELETE로 확정):**
- `compare_improvements.py`, `compare_max_weight.py`, `compare_pca.py`
- `experiment_monthly_rebal.py`, `experiment_te_sensitivity.py`
- `grid_search.py`
- `feature_pnl_attribution.py`
- `run_finalize_iter15.py`
- `scripts/compare_revision_variants.py`
- `scripts/diagnose_revision_spikes.py`
- `scripts/regen_csv_from_pkl.py`
- `scripts/render_baseline_metrics.py`

**검증 필요 (grep 결과로 verdict 결정):**
- `main.py` — run_variant.py로 대체되었는지? (단, `if 'main' in CLAUDE.md or AGENTS.md`는 무시 — "main" 단어는 너무 광범위)
- `run_all.py` — update_and_deploy.py로 대체되었는지?
- `export_csv.py` — build_dashboard_data.py로 대체되었는지?
- `prepare_data.py` ↔ `run_backtest_harness.py` — 한 쌍. 둘 다 살리거나 둘 다 죽이거나
- `app.py` (worktree에만 있을 수도 — git ls-files로 존재 여부 확인)

## 작업

### 작업 1: cross-reference grep 매트릭스 작성

각 후보 파일의 stem(예: `compare_pca`)에 대해 다음 grep을 모두 실행하고 결과를 기록한다:

```bash
# 1. Python import 그래프
grep -rn "import compare_pca\|from compare_pca" --include="*.py" .

# 2. .claude/skills/ — pipeline-orchestrator 등이 부르는지
grep -rn "compare_pca" .claude/skills/ 2>/dev/null

# 3. .claude/agents/
grep -rn "compare_pca" .claude/agents/ 2>/dev/null

# 4. docs 본문
grep -rn "compare_pca" docs/

# 5. 루트 markdown
grep -n "compare_pca" CLAUDE.md AGENTS.md DEPLOY.md README.md 2>/dev/null

# 6. JSON 설정
grep -n "compare_pca" experiment_inventory.json 2>/dev/null

# 7. requirements (entry-point가 박혀 있을 가능성 낮으나 확인)
grep -n "compare_pca" requirements*.txt 2>/dev/null

# 8. bat / yaml
grep -rn "compare_pca" --include="*.bat" --include="*.yaml" --include="*.yml" .
```

위 8개 grep을 후보 파일 ~20개에 대해 모두 돌리는 것이 burden이라면, 한 번에 묶어서 처리해도 된다:

```bash
# 모든 후보를 한 번에 (alternation 패턴)
PATTERN='compare_improvements|compare_max_weight|compare_pca|experiment_monthly_rebal|experiment_te_sensitivity|grid_search|feature_pnl_attribution|run_finalize_iter15|main|run_all|export_csv|prepare_data|run_backtest_harness|app|compare_revision_variants|diagnose_revision_spikes|regen_csv_from_pkl|render_baseline_metrics'
grep -rEn "($PATTERN)\.py|import ($PATTERN)|from ($PATTERN)" \
    --include="*.py" --include="*.md" --include="*.json" --include="*.bat" --include="*.yaml" \
    --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=outputs --exclude-dir=data .
```

**중요:** `experiment_inventory.json` 안의 reference는 historical audit trail이므로 "external ref"로 카운트하지 않는다 — 그 파일의 reference는 *그 스크립트의 출력이 이미 N_trials에 포함되었다*는 뜻이지, 스크립트 자체가 살아 있어야 한다는 뜻이 아니다. 단, **proposal 표의 evidence 컬럼에는 명시**: "experiment_inventory.json:행번호 (audit trail only — does not block delete)".

### 작업 2: `_research/cleanup_proposal.md` 작성

다음 형식으로 작성:

```markdown
# Cleanup Proposal

## 요약
- KEEP:    N개
- DELETE:  M개
- NEEDS_USER_DECISION: K개

## 분류 표

| File | Last Mod | Docstring | has __main__ | External Refs (file:line) | Verdict | Reason |
|---|---|---|---|---|---|---|
| `compare_pca.py` | 2026-?? | "..." | yes | (none — verified by grep above) | DELETE | experimental sweep, 외부 import 0개. experiment_inventory.json:24-28에 historical audit만 |
| `run_selection_bias.py` | 2026-?? | "..." | yes | CLAUDE.md:347, AGENTS.md:347, main.py:335, experiment_inventory.json:1 | KEEP | DSR 검증 운영 워크플로의 일부 |
| ... | ... | ... | ... | ... | ... | ... |

## DELETE 행 재현 grep 명령

각 DELETE 행에 대해 reference 0개를 재확인할 수 있는 명령:

```bash
# compare_pca.py
grep -rEn "import compare_pca|from compare_pca|compare_pca\.py" \
    --include="*.py" --include="*.md" --include="*.bat" --include="*.yaml" \
    --exclude-dir=.git --exclude-dir=outputs .  
# 기대: experiment_inventory.json 외 0건

# experiment_te_sensitivity.py
...
```

## NEEDS_USER_DECISION (있다면)

| File | 자동 분류 불가 사유 | 사용자 확인 질문 |
|---|---|---|
| ... | ... | ... |

## KEEP 그룹 (참고용 목록)
- 운영 entry: `update_and_deploy.{bat,py}`
- backtest: `run_variant.py`, `daily_update.py`
- dashboard: `streamlit_mobile.py`, `scripts/build_dashboard_data.py`
- legacy but referenced: `streamlit_app.py`, `run_selection_bias.py`
- (additional KEEPs from grep analysis...)

## DELETE 그룹 (정리 대상)
- `compare_*.py` (3개)
- `experiment_*.py` (2개)
- `grid_search.py`, `feature_pnl_attribution.py`, `run_finalize_iter15.py`
- `scripts/{compare_revision_variants, diagnose_revision_spikes, regen_csv_from_pkl, render_baseline_metrics}.py` (4개)
- (additional DELETEs from grep verification...)
```

## Acceptance Criteria

```bash
test -f phases/update-deploy-cleanup/_research/cleanup_proposal.md

# 모든 후보 파일이 표에 등재됐는지
python -c "
import pathlib, subprocess
tracked = subprocess.check_output(['git', 'ls-files', '*.py', 'scripts/*.py'], text=True).splitlines()
tracked = [t for t in tracked if not t.startswith('src/') and not t.startswith('archive/') and not t.startswith('.claude/')]
body = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
missing = [t for t in tracked if t not in body and pathlib.Path(t).name not in body and pathlib.Path(t).stem not in body]
assert not missing, f'missing from proposal: {missing}'
print(f'OK — all {len(tracked)} scripts classified')
"

# KEEP / DELETE / NEEDS_USER_DECISION 단어가 표 안에 모두 등장
for v in KEEP DELETE NEEDS_USER_DECISION; do
  grep -q "$v" phases/update-deploy-cleanup/_research/cleanup_proposal.md || (echo "MISSING verdict: $v"; exit 1)
done

# 확정 KEEP들이 정말 KEEP으로 분류됐는지
for f in update_and_deploy.py run_variant.py daily_update.py streamlit_mobile.py run_selection_bias.py streamlit_app.py; do
  python -c "
import re, pathlib
body = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
# '$f' 줄을 찾아 KEEP 단어가 같은 줄에 있는지
for line in body.splitlines():
    if '$f' in line and '|' in line:
        assert 'KEEP' in line, f'$f is NOT classified as KEEP: {line!r}'
print('OK $f → KEEP')
"
done
```

## status 처리 (중요 — user gate 역할)

이 step 완료 시 `phases/update-deploy-cleanup/index.json`의 step 1을 다음과 같이 업데이트한다:

```json
{
  "step": 1,
  "name": "classify-and-propose",
  "status": "completed",
  "summary": "Cleanup proposal ready at _research/cleanup_proposal.md (KEEP=N, DELETE=M, NEEDS_USER_DECISION=K). USER MUST REVIEW the proposal before running step2/step3."
}
```

`summary` 문장이 **반드시 "USER MUST REVIEW" 를 포함**해야 한다 — 이게 다음 step을 자동 실행하지 않게 하는 신호.

execute.py가 자동으로 step2를 시작하더라도 사용자가 산출물을 보고 OK하지 않으면 step2의 doc은 잘못된 KEEP/DELETE 가정 위에서 작성될 위험이 있다. 그래서 step1 완료 후에는 execute.py를 일단 멈추고 사용자가 `_research/cleanup_proposal.md`를 검토 → 분류 OK → 다시 execute.py로 step2/step3 진행하는 흐름을 권장한다.

## 검증 절차

1. 위 AC 커맨드를 모두 실행한다.
2. 분류 sanity check:
   - DELETE 행 중 random 1~2개를 선택해 grep을 직접 실행 → 정말 reference 0개인지 재확인
   - KEEP 행 중 `streamlit_app.py`를 grep → CLAUDE.md / AGENTS.md / run_all.py / main.py에서 정말 발견되는지 확인
3. 결과에 따라 step 1 status를 업데이트:
   - 성공 → `"completed"` + 위 형식의 summary
   - AC 실패 3회 → `"error"` + `error_message`
   - 자동 분류로 결정 못 하는 케이스가 너무 많음 (5개 이상) → `"blocked"` + `blocked_reason`

## 금지사항

- **어떤 source 파일도 수정하지 마라.** 이 step은 read + cleanup_proposal.md 작성만.
- **명시적 grep evidence 없이 "DELETE" 판정을 내리지 마라.** 직감으로 분류 X.
- **`experiment_inventory.json`의 reference를 "external ref"로 카운트해서 DELETE를 KEEP으로 바꾸지 마라.** 그 JSON은 historical audit trail이며, 그 안에 거론된 스크립트가 *지금 살아 있어야* 한다는 뜻이 아니다.
- **`src/*` 파일은 분류 대상에서 제외.** src/는 라이브러리이며 자동 분류로는 의존성을 다 못 잡는다. 이번 정리에서 건드리지 않는다.
- **확정 KEEP 목록을 무시하지 마라.** 특히 `run_selection_bias.py`, `streamlit_app.py` — grep 결과만 보면 "참조 적음"으로 보일 수 있으나 docs에서 운영 가이드의 일부.
