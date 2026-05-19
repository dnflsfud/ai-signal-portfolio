# Step 3: apply-cleanup

이 step은 **destructive하지만 git으로 reversible**하다. step1의 DELETE 분류를 실제로 `git rm`으로 적용하고, 죽는 파일을 언급하던 doc/markdown에서 dead reference를 정리한 뒤 import smoke로 안전성을 검증한다.

## 목표

`_research/cleanup_proposal.md`의 DELETE 행에 적힌 모든 .py 파일을 repo에서 완전히 제거한다. CLAUDE.md / AGENTS.md / DEPLOY.md / docs/* / .claude/skills/ 에서 죽는 파일을 언급하는 줄을 정리한다. 마지막으로 `update_and_deploy.py` import가 깨지지 않았음을 smoke로 확인한다.

## 읽어야 할 파일

먼저 다음을 모두 읽어라:

### 이전 step 산출물 (필수)
- `phases/update-deploy-cleanup/_research/cleanup_proposal.md` — **DELETE 행만 사용**. KEEP / NEEDS_USER_DECISION은 무시
- `docs/UPDATE_AND_DEPLOY_FLOW.md` — step2에서 작성된 doc. 여기에 DELETE 후보가 등장하면 안 되지만 만에 하나 leaked가 있으면 같이 정리해야 함

### dead-reference 정리 대상 (read 후 필요 시 edit)
- `CLAUDE.md`
- `AGENTS.md`
- `DEPLOY.md`
- `docs/BASELINE.md`, `docs/FEATURE_CATALOG.md`, `docs/ROADMAP.md`, `docs/rollback_log.md`
- `.claude/skills/**/SKILL.md`
- `.claude/agents/**/*.md`

`experiment_inventory.json`은 **절대 수정하지 마라** (historical audit trail).

## 작업

### 작업 순서 (반드시 순서대로 실행)

#### 1. DELETE 목록 추출

`_research/cleanup_proposal.md`에서 verdict가 "DELETE"인 행만 추출한다:

```python
import re, pathlib
proposal = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
# 표 형식: | file | last_mod | docstring | has_main | refs | DELETE | reason |
# file 컬럼은 백틱으로 감싸여 있을 수 있음
delete_files = re.findall(r'\| \`?([\w_/]+\.py)\`? \|.*\| DELETE \|', proposal)
# 중복 제거 + 정렬
delete_files = sorted(set(delete_files))
print(f'DELETE 대상: {len(delete_files)}개')
for f in delete_files:
    print(f'  - {f}')
```

추출이 비었거나 형식이 안 맞으면 즉시 `error` 처리하고 step1로 돌아가라.

#### 2. 안전 가드

다음 파일은 절대 DELETE 목록에 들어가면 안 됨 — 들어 있으면 즉시 abort:
- `update_and_deploy.py`, `update_and_deploy.bat`
- `run_variant.py`, `daily_update.py`
- `streamlit_mobile.py`, `scripts/build_dashboard_data.py`
- `run_selection_bias.py`, `streamlit_app.py`
- `requirements*.txt`
- `src/**/*.py`

```python
forbidden = {
    'update_and_deploy.py', 'run_variant.py', 'daily_update.py',
    'streamlit_mobile.py', 'scripts/build_dashboard_data.py',
    'run_selection_bias.py', 'streamlit_app.py',
}
for f in delete_files:
    assert f not in forbidden, f'CRITICAL: {f} is load-bearing — abort cleanup'
    assert not f.startswith('src/'), f'CRITICAL: src/ files cannot be deleted — abort'
```

#### 3. `git rm` 실행

각 파일을 git tracking에서 제거 + working tree에서 삭제:

```bash
for f in <DELETE 목록>; do
  git rm "$f"
done
```

만약 파일이 untracked인 경우 (drop된 상태) `git rm`이 실패할 수 있음 — 그 경우 `rm "$f"`로 fallback.

#### 4. dead reference 정리

각 DELETE 파일의 stem(예: `compare_pca`)에 대해:

```bash
# 1. 모든 reference 찾기 (.git, outputs, data, archive 제외)
PATTERN='compare_pca|experiment_te_sensitivity|...'  # 모든 DELETE stem 합친 alternation
grep -rEln "($PATTERN)" \
    --include="*.md" --include="*.json" \
    --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data \
    --exclude-dir=archive --exclude-dir=__pycache__ \
    --exclude="experiment_inventory.json"  \
    .
```

찾아진 각 파일에 대해 Edit/MultiEdit로 다음 처리:
- 단순 언급(예: "see `compare_pca.py` for details") → 줄 삭제
- 표 안의 row → 행 삭제
- 운영 가이드의 일부 → 만약 운영 가이드에서 여전히 필요한 정보면 다른 KEEP 스크립트로 우회. 그게 불가능하면 step1으로 돌아가서 KEEP으로 재분류.

**`experiment_inventory.json`은 건드리지 마라.** historical audit trail이며, 그 안의 reference는 *과거에 그 스크립트가 돌았었다*는 사실 기록.

#### 5. import smoke

```bash
# 핵심 entry-point들이 import 가능한지
cd <repo_root>
python -c "import update_and_deploy"  || (echo "FAIL: update_and_deploy import broken"; exit 1)
python -c "import run_variant"        || (echo "FAIL: run_variant import broken"; exit 1)
python -c "import daily_update"       || (echo "FAIL: daily_update import broken"; exit 1)
python -c "import streamlit_mobile"   || (echo "FAIL: streamlit_mobile import broken"; exit 1)
echo "SMOKE OK"
```

각 import가 실패하면 즉시 `git restore`로 되돌리고 error 처리.

#### 6. (best-effort) full bat smoke

시간 여유 있으면 (~3-4분):
```bash
python update_and_deploy.py --no-build --no-deploy --no-smoke
# 또는 dry-run에 가까운:
python update_and_deploy.py --no-build --no-deploy --no-smoke --mode incremental
```

이건 **AC가 아니라 best-effort**. CI 환경에서 venv 의존성이 없거나 데이터 파일이 없으면 skip해도 OK.

## Acceptance Criteria

```bash
# AC 1: DELETE 목록의 모든 파일이 git tracked에서 사라졌는지
python -c "
import re, pathlib, subprocess
proposal = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
delete_files = sorted(set(re.findall(r'\| \`?([\w_/]+\.py)\`? \|.*\| DELETE \|', proposal)))
tracked = set(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
still_there = [f for f in delete_files if f in tracked]
assert not still_there, f'still tracked: {still_there}'
# 중복 안전 가드
forbidden = {'update_and_deploy.py', 'run_variant.py', 'daily_update.py', 'streamlit_mobile.py',
             'scripts/build_dashboard_data.py', 'run_selection_bias.py', 'streamlit_app.py'}
for f in delete_files:
    assert f not in forbidden, f'safety violation: {f} should not have been deleted'
    assert not f.startswith('src/'), f'safety violation: src/ touched'
print(f'OK — {len(delete_files)} files removed, no forbidden file deleted')
"

# AC 2: import smoke
python -c "
import importlib
for m in ['update_and_deploy', 'run_variant', 'daily_update', 'streamlit_mobile']:
    importlib.import_module(m)
    print(f'OK — import {m}')
"

# AC 3: dead-reference scan (정보성 — WARN만 출력)
python -c "
import re, pathlib
proposal = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
delete_stems = sorted(set(pathlib.Path(f).stem for f in re.findall(r'\| \`?([\w_/]+\.py)\`? \|.*\| DELETE \|', proposal)))
docs_to_check = ['CLAUDE.md', 'AGENTS.md', 'DEPLOY.md'] + [str(p) for p in pathlib.Path('docs').glob('*.md')]
for md in docs_to_check:
    if not pathlib.Path(md).exists():
        continue
    body = pathlib.Path(md).read_text(encoding='utf-8')
    leaked = [s for s in delete_stems if f'{s}.py' in body or f'\`{s}\`' in body]
    if leaked:
        print(f'WARN: {md} still mentions {leaked}')
print('Dead-ref scan complete (WARN are informational; not all need fixing if context clearly historical)')
"

# AC 4: experiment_inventory.json 보존
test "$(git diff --name-only HEAD experiment_inventory.json)" = "" || \
    (echo "ERROR: experiment_inventory.json was modified — must be preserved"; exit 1)
```

## 검증 절차

1. AC 1~4 모두 pass
2. `git status` 검사:
   - DELETE 목록 외의 파일이 staged 상태로 들어가 있지 않은지
   - markdown 정리는 동일 commit에 묶여도 OK (혹은 후속 commit으로 분리해도 OK)
3. step 3 status 업데이트:
   - 성공 → `"completed"` + `"summary": "Removed N legacy/test scripts via git rm. Cleaned dead refs in K markdown files. Import smoke OK."`
   - import smoke 실패 → 즉시 `git restore` 후 `"error"` + 어떤 import가 깨졌는지
   - dead-ref WARN이 너무 많아 (>5) 사용자 판단 필요 → `"blocked"` + `"blocked_reason": "DELETE 후보가 docs에 광범위하게 인용됨 — 수동 검토 필요"`

## 금지사항

- **`src/*` 파일을 절대 삭제하지 마라.** 이유: src/는 모든 production 코드의 라이브러리. 자동 import 분석으로는 dynamic import / config-driven import / pickle 역참조를 다 못 잡는다.
- **`data/`, `outputs/`, `variants/`, `archive/` 디렉토리를 건드리지 마라.** 이유: 운영 데이터 / 결과물 / 변경 이력.
- **`experiment_inventory.json`을 수정하지 마라.** 이유: historical audit trail. Deflated Sharpe Ratio 계산용 N_trials 기록이며, 거기 거론된 스크립트가 *지금* 살아 있어야 한다는 뜻이 아니다.
- **`--no-verify`, `--force`, `git reset --hard`, `git push --force` 등 destructive git 옵션을 사용하지 마라.** 모든 변경은 normal commit으로.
- **DELETE 목록에 없는 파일을 추가로 지우지 마라.** step1에서 분류 안 된 파일이 발견되면 step1로 돌려보내라 (status="blocked").
- **AC가 한 번 통과한 뒤에는 markdown을 추가로 수정하지 마라.** dead-ref scan 결과의 WARN은 정보성이며, doc 작성자 의도에 따라 historical reference로 남겨도 무방한 케이스가 있다.
