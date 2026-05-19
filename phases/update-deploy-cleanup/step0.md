# Step 0: audit-and-trace

이 step은 **READ-ONLY**다. 어떤 source 파일도 수정하지 마라. 산출물은 `phases/update-deploy-cleanup/_research/` 하위 두 개의 markdown 파일이다.

## 목표

`update_and_deploy.bat`이 실행되었을 때 일어나는 모든 일의 매핑을 두 산출물로 남긴다:

1. **bat → py → run_variant.py → src.backtest.run_backtest의 전체 호출 트리** (mermaid + line-by-line)
2. **repo root + scripts/의 모든 .py 파일에 대한 1줄 인벤토리** (last-modified, docstring, has-main)

이 산출물은 step1(분류), step2(문서), step3(정리)의 입력이 된다.

## 읽어야 할 파일

먼저 아래 파일들을 읽고 production 운영 흐름을 파악하라:

### Bat 흐름 추적용
- `update_and_deploy.bat`  ← 34줄, 단순 venv 활성 + python 호출 래퍼
- `update_and_deploy.py`  ← 5-stage orchestrator (462줄)
- `run_variant.py`  ← Stage 2 full mode 백엔드
- `daily_update.py`  ← Stage 2 incremental mode 백엔드
- `scripts/build_dashboard_data.py`  ← Stage 3 dashboard 페이로드 빌더
- `src/backtest.py`  ← `run_backtest` 함수 시그니처 + 8-phase 메인 루프 (전체 읽지 말고 `def run_backtest` 부근 + Phase anchor만)
- `src/data_loader.py`  ← `UniverseData` 시그니처
- `variants/iter15_65tkr_reb21_vtg.yaml`  ← production manifest
- `CLAUDE.md`  ← Phase 1~8 아키텍처 섹션
- `DEPLOY.md`  ← Streamlit Cloud 배포 절차

### 인벤토리용 (전체 읽을 필요 없음 — 첫 30줄 docstring만 + git log -1 으로 last_mod 확인)

repo root .py 전체:
```
app.py (worktree에만 있음 — 부모에는 없을 수 있음)
compare_improvements.py
compare_max_weight.py
compare_pca.py
daily_update.py
experiment_monthly_rebal.py
experiment_te_sensitivity.py
export_csv.py
feature_pnl_attribution.py
grid_search.py
main.py
prepare_data.py
run_all.py
run_backtest_harness.py
run_finalize_iter15.py
run_selection_bias.py
run_variant.py
streamlit_app.py
streamlit_mobile.py
update_and_deploy.py
```

scripts/*.py 전체:
```
scripts/build_dashboard_data.py
scripts/compare_revision_variants.py
scripts/diagnose_revision_spikes.py
scripts/regen_csv_from_pkl.py
scripts/render_baseline_metrics.py
```

위 목록은 `git ls-files '*.py' 'scripts/*.py'`로 한 번 재확인한 뒤 차이가 있으면 산출물에 반영하라.

## 작업

### 작업 1: `_research/bat_flow_trace.md` 작성

다음 섹션을 모두 포함한다:

```markdown
# update_and_deploy.bat 흐름 추적

## 한 줄 요약
새 가격 데이터 → 전체 backtest 재실행 → dashboard 페이로드 빌드 → cc2-dashboard repo로 git push → Streamlit Cloud 자동 재배포

## Sequence Diagram

\`\`\`mermaid
sequenceDiagram
    participant U as User
    participant BAT as update_and_deploy.bat
    participant PY as update_and_deploy.py
    participant RV as run_variant.py
    participant DU as daily_update.py
    participant BT as src.backtest.run_backtest
    participant BD as scripts/build_dashboard_data.py
    participant DASH as cc2-dashboard repo
    ...
\`\`\`

## Stage 1: stage_data_check
... (line-by-line, what it reads, what it warns on, fail vs warn)

## Stage 2: backtest (full vs incremental 분기)
### Stage 2a: --mode full → run_variant.py
... (manifest load → SAFE_FOR_CACHE_REUSE 검사 → run_backtest → promote)

### Stage 2b: --mode incremental → daily_update.py
... (DailyState 로드 → drift-only vs rebalance day → 리밸런싱 시 (a~d) production parity 단계)

## Stage 3: stage_build_dashboard
...

## Stage 4: --smoke (optional)
...

## Stage 5: stage_deploy
...

## subprocess.run 호출 카탈로그
| 호출 위치 (file:line) | 명령 | 목적 |
|---|---|---|
| update_and_deploy.py:122 | `subprocess.run(cmd, ...)` (`run()` helper) | Stage 2/3에서 외부 Python 실행 |
| update_and_deploy.py:301 | `subprocess.run([... "streamlit", "run", "streamlit_mobile.py"])` | Stage 4 smoke |
| update_and_deploy.py:316 | `subprocess.run(["git", ...])` (`_git()` helper) | Stage 5 git 명령 |
| ... | ... | ... |

## Input/Output 매핑 표
| Stage | Input 파일 | 호출 스크립트 | Output 파일 |
|---|---|---|---|
| 1 | `data/ai_signal_data.xlsx` (Daily_Returns 시트) | (없음, pd.read_excel만) | (없음, console 출력) |
| 2a | 위 + `variants/iter15_65tkr_reb21_vtg.yaml` | `run_variant.py` | `outputs/iter15_65tkr_reb21_vtg/{metrics.json, backtest_result.pkl, experiment_manifest.json}` → promote → `outputs/baseline_v4/` |
| 2b | 위 + `outputs/daily_state.pkl` | `daily_update.py` | `outputs/daily_state.pkl` (갱신) + `outputs/csv/*.csv` |
| 3 | `outputs/baseline_v4/backtest_result.pkl` (~65MB) + data xlsx | `scripts/build_dashboard_data.py` | `outputs/baseline_v4/dashboard_data.pkl` (~3MB) |
| 4 | `outputs/baseline_v4/dashboard_data.pkl` | `streamlit run streamlit_mobile.py` | (none, blocking server) |
| 5 | 위 + `streamlit_mobile.py` + `requirements_dashboard.txt` | `git add/commit/push` | cc2-dashboard repo의 같은 경로들 |
```

### 작업 2: `_research/script_inventory.md` 작성

표 형태:

```markdown
# Script Inventory (root *.py + scripts/*.py)

## Repo root *.py

| File | Last Mod (YYYY-MM-DD) | First Docstring Line | has __main__ block |
|---|---|---|---|
| compare_improvements.py | 2026-?? | "..." | yes/no |
| ... | ... | ... | ... |

## scripts/*.py

| File | Last Mod | First Docstring Line | has __main__ block |
|---|---|---|---|
| ... | ... | ... | ... |
```

각 행을 채우는 방법:
- **Last Mod**: `git log -1 --format=%ad --date=short -- <file>`
- **First Docstring Line**: 파일 첫 30줄 중 `"""` 또는 `'''` 첫 라인 다음의 한 줄 (없으면 "(no docstring)")
- **has __main__ block**: `grep -l 'if __name__ == .__main__.' <file>` (있으면 yes)

## Acceptance Criteria

```bash
test -f phases/update-deploy-cleanup/_research/bat_flow_trace.md
test -f phases/update-deploy-cleanup/_research/script_inventory.md

# bat_flow_trace.md 구조 검증
grep -c "^## Stage" phases/update-deploy-cleanup/_research/bat_flow_trace.md   # >= 5
grep -c "subprocess.run" phases/update-deploy-cleanup/_research/bat_flow_trace.md   # >= 3
grep -q '```mermaid' phases/update-deploy-cleanup/_research/bat_flow_trace.md

# script_inventory.md 구조 검증 (모든 .py가 등재됐는지)
python -c "
import pathlib, subprocess
tracked = subprocess.check_output(['git', 'ls-files', '*.py', 'scripts/*.py'], text=True).splitlines()
tracked = [t for t in tracked if not t.startswith('src/') and not t.startswith('archive/') and not t.startswith('.claude/')]
body = pathlib.Path('phases/update-deploy-cleanup/_research/script_inventory.md').read_text(encoding='utf-8')
missing = [t for t in tracked if t not in body and pathlib.Path(t).name not in body]
assert not missing, f'missing from inventory: {missing}'
print(f'OK — {len(tracked)} scripts inventoried')
"
```

## 검증 절차

1. 위 AC 커맨드를 실행한다.
2. 아키텍처 체크리스트:
   - `bat_flow_trace.md`의 Stage 1~5 설명이 `update_and_deploy.py`의 실제 코드 (Stage 함수의 line range)와 일치하는가?
   - mermaid sequence가 실제 호출 순서를 반영하는가?
   - IO 매핑 표의 파일 경로가 모두 실제 파일과 일치하는가?
3. 결과에 따라 `phases/update-deploy-cleanup/index.json`의 step 0을 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "Bat flow trace + script inventory written to _research/. N scripts catalogued (root M, scripts/ K)."`
   - AC 실패 3회 → `"status": "error"`, `"error_message": "구체적 실패 내용 (예: bat_flow_trace.md missing Stage 3 anchor)"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"`

## 금지사항

- **어떤 source 파일도 수정하지 마라.** 이유: 이 step은 순수 read + 산출물 작성 단계. 수정은 step3에서만.
- **`outputs/`, `data/`, `variants/` 디렉토리를 건드리지 마라.** 이유: 운영 데이터/결과물.
- **추측으로 작성하지 마라.** 모든 라인-넘버, 함수명, 파일 경로는 실제 read한 파일에서 가져와라. 확신 없으면 "(unverified)"로 표시.
- **`src/*` 모듈을 전부 읽지 마라.** `src/backtest.py`의 `run_backtest` 함수 부근만 읽으면 충분. 전체 읽으면 context가 터진다.
