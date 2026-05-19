# Step 2: write-pipeline-doc

이 step의 산출물은 **`docs/UPDATE_AND_DEPLOY_FLOW.md` 단 하나**다. cleanup 후 살아남는 production 운영 로직을 단계별로 상세하게 설명하는 문서.

source 코드는 일절 수정하지 않는다 (수정은 step3).

## 목표

cc2_harness 운영자가 **이 문서 하나만 읽으면** `update_and_deploy.bat`이 처음부터 끝까지 무엇을 하는지, 매일/주간 운영을 어떻게 하는지, 실패 시 어디서 복구하는지를 모두 파악할 수 있어야 한다.

## 읽어야 할 파일

먼저 다음 파일들을 모두 읽어라:

### 이전 step 산출물 (필수)
- `phases/update-deploy-cleanup/_research/bat_flow_trace.md` — 본 문서의 sequence diagram + IO 표의 원본
- `phases/update-deploy-cleanup/_research/script_inventory.md`
- `phases/update-deploy-cleanup/_research/cleanup_proposal.md` — **DELETE 분류된 스크립트는 본 문서에 절대 등장하면 안 됨**

### Source 파일 (재인용 / 코드 anchor 참조용)
- `update_and_deploy.bat`
- `update_and_deploy.py`
- `run_variant.py`
- `daily_update.py`
- `scripts/build_dashboard_data.py`
- `src/backtest.py` (run_backtest 시그니처 + Phase anchor)
- `src/data_loader.py` (UniverseData 시그니처)
- `variants/iter15_65tkr_reb21_vtg.yaml`
- `CLAUDE.md` (Phase 1~8 섹션 — 이 문서의 Phase 절 작성에 사용)
- `DEPLOY.md` (Streamlit Cloud 배포 절차 — 이 문서의 Stage 5 절에 통합)

## 작업

`docs/UPDATE_AND_DEPLOY_FLOW.md` 작성. 한국어로 작성하되, 코드 anchor는 영문 식별자 그대로 둔다.

### 문서 구조 (목차 그대로 따를 것)

```markdown
# `update_and_deploy.bat` — 최종 파이프라인 흐름

## 1. 개요

### 한 줄 요약
새 가격 데이터(xlsx) → 전체 백테스트 재실행 → dashboard 페이로드 빌드 → cc2-dashboard repo로 git push → Streamlit Cloud 자동 재배포까지 한 번의 더블클릭으로.

### 사용 시나리오
| 모드 | 시나리오 | 소요 시간 | 갱신 대상 |
|---|---|---|---|
| `--mode full` (default) | 매주~격주 정식 갱신 | ~3-4분 | `outputs/baseline_v4/backtest_result.pkl` 전체 재계산 + dashboard 갱신 |
| `--mode incremental` | 매일 가격만 추가 (가벼움) | ~10초~1분 | `outputs/daily_state.pkl` + `outputs/csv/*.csv` 만. dashboard pkl은 갱신 X |

### 요구 사항
- Python 3.10+, requirements.txt 설치된 venv
- 데이터 파일: `data/ai_signal_data.xlsx` (실 경로는 `update_and_deploy.py:57` `DEFAULT_DATA_FILE`)
- cc2-dashboard repo 로컬 clone (기본 `C:/Users/westl/Documents/cc2-dashboard`, env `CC2_DASHBOARD_REPO`로 override)

---

## 2. Entry: `update_and_deploy.bat`

전체 코드 (34줄 단순 래퍼):

\`\`\`batch
@echo off
REM ... (실제 bat 내용 그대로 인용)
\`\`\`

| 동작 | 설명 |
|---|---|
| `cd /d "%~dp0"` | bat 파일이 위치한 디렉토리로 이동 (어디서 더블클릭하든 동일 작업 디렉토리 보장) |
| `if exist "..\..\..\..\Scripts\activate.bat"` | 4단계 위 venv 자동 활성. 경로는 `cc2_harness\.\..\..\..\Scripts\activate.bat`을 가리킴 |
| `python update_and_deploy.py %*` | bat에 전달된 모든 인자(예: `--mode incremental`, `--no-push`)를 그대로 forward |
| ERRORLEVEL 검사 | 비-0 종료 시 메시지 + `pause` (사용자가 콘솔 닫기 전 에러 확인 가능) |

---

## 3. Orchestrator: `update_and_deploy.py`

5-stage sequence diagram:

\`\`\`mermaid
sequenceDiagram
    participant U as User
    participant BAT as update_and_deploy.bat
    participant PY as update_and_deploy.py
    participant Stage1 as Stage 1<br/>data_check
    participant Stage2 as Stage 2<br/>backtest
    participant Stage3 as Stage 3<br/>build_dashboard
    participant Stage4 as Stage 4<br/>smoke (optional)
    participant Stage5 as Stage 5<br/>deploy
    participant DASH as cc2-dashboard repo
    participant SC as Streamlit Cloud

    U->>BAT: 더블클릭
    BAT->>PY: python update_and_deploy.py
    PY->>Stage1: stage_data_check()
    PY->>Stage2: stage_backtest_full() (default)
    PY->>Stage3: stage_build_dashboard()
    PY->>Stage4: (skip — --smoke OFF)
    PY->>Stage5: stage_deploy()
    Stage5->>DASH: git push
    DASH->>SC: webhook
    SC-->>U: ~30초 후 재배포 완료
\`\`\`

### CLI 옵션 표 (`python update_and_deploy.py --help`)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--mode {incremental,full}` | `full` | Stage 2 분기. full = run_variant.py, incremental = daily_update.py |
| `--data <path>` | `DEFAULT_DATA_FILE` | xlsx 데이터 파일 |
| `--variant <yaml>` | `variants/iter15_65tkr_reb21_vtg.yaml` | full 모드에서 사용할 manifest |
| `--run-dir <path>` | `outputs/baseline_v4` | canonical backtest 결과 디렉토리 (promote 대상) |
| `--dashboard-repo <path>` | `C:/Users/westl/Documents/cc2-dashboard` | 배포 대상 repo |
| `--no-push` | (off) | 로컬 commit만, push 생략 |
| `--no-build` | (off) | Stage 3 건너뜀 |
| `--no-deploy` | (off) | Stage 5 건너뜀 |
| `--smoke` | (off) | Stage 4 활성 (blocking streamlit 서버) |

---

## 4. Stage 1 — Data sanity check

함수: `stage_data_check(data_file)` (`update_and_deploy.py:153-185`)

| 단계 | 내용 |
|---|---|
| 1 | 파일 존재 확인 → 없으면 `fatal()` 종료 (exit 1) |
| 2 | 파일 크기 + mtime 출력 |
| 3 | `_peek_last_data_date()`로 Daily_Returns 시트 마지막 날짜 read |
| 4 | age 분기: <=1d "fresh" / <=7d "OK" / >7d **WARN** (fail은 아님) |

`_peek_last_data_date`는 `pd.read_excel(usecols=[0])`로 가벼운 read만 수행 — 전체 UniverseData 로드 X.

---

## 5. Stage 2a — Full mode (`run_variant.py`)

함수: `stage_backtest_full(variant, run_dir)` (`update_and_deploy.py:238-263`)

\`\`\`mermaid
flowchart TB
    A[run_variant.py --variant ...vtg.yaml] --> B{checkpoint 안전성<br/>SAFE_FOR_CACHE_REUSE}
    B -- 안전 --> C[Phase 1/2/4 캐시 reuse<br/>+ run_backtest 호출]
    B -- 위험 OR --no-cache --> D[전체 파이프라인 재실행]
    C --> E[outputs/iter15_65tkr_reb21_vtg/<br/>{metrics.json, backtest_result.pkl, manifest}]
    D --> E
    E --> F[_promote_variant_artifacts<br/>→ outputs/baseline_v4/]
\`\`\`

### Manifest 로드
`run_variant.py:81-108` `load_manifest()` — yaml 파싱 + `tuning_mode` 검증 + unknown override key 거부

### Checkpoint 안전성 검사
`run_variant.py:206-247` — `SAFE_FOR_CACHE_REUSE` set에 포함된 override key만 캐시와 호환. Phase 1/2/4 (data/feature/training) 자체를 바꾸는 override가 있으면 자동으로 cache disable.

### `src.backtest.run_backtest` 호출
`run_variant.py:269-284`에서 호출. 내부 흐름은 §7에서 자세히.

### Promote
`update_and_deploy.py:217-235` `_promote_variant_artifacts()` — `outputs/<variant>/`의 `backtest_result.pkl`, `metrics.json`, `experiment_manifest.json`을 `outputs/baseline_v4/`로 `shutil.copy2`. variant 디렉토리는 audit trail로 보존.

---

## 6. Stage 2b — Incremental mode (`daily_update.py`)

함수: `stage_backtest_incremental(data_file)` (`update_and_deploy.py:191-214`)

### DailyState (pickle 스키마)
`daily_update.py:64-92` `@dataclass DailyState` (schema_version=2):
- `weights`, `tickers` — 현재 비중
- `days_since_rebal`, `rebalance_freq` — 리밸런싱 카운터
- `port_rets`, `bm_rets`, `spx_rets`, `turnovers` — 누적 시계열
- `rebal_weights`, `daily_weights`, `bm_rebal_weights` — 비중 이력
- `models`, `predictions` — 학습 캐시
- `ic_values`, `sector_map`

### 분기
\`\`\`mermaid
flowchart TB
    A[daily_update.py 실행] --> B{daily_state.pkl 존재?}
    B -- No --> C[--full-init<br/>전체 백테스트 후 상태 저장]
    B -- Yes --> D[load_state]
    D --> E[새 영업일 loop]
    E --> F{rebalance day?<br/>days_since_rebal >= rebalance_freq}
    F -- No --> G[Drift only<br/>Step 1: 진입 비중으로 PnL<br/>Step 2: 비중 drift]
    F -- Yes --> H[Step 1-3: PnL + drift + rebal sequence]
    H --> I{model age > 63d * 1.5?}
    I -- Yes --> J[train_model 재학습]
    I -- No --> K[predictions.loc t_date]
    J --> L[Production parity 4단계: a~d]
    K --> L
    L --> M[validate_new_weights<br/>+ TC charge to today PnL]
    G --> N[기록 + 다음 날]
    M --> N
\`\`\`

### Production parity 4단계 (`daily_update.py:440-506`)
리밸런싱 일자에 production backtest와 정확히 같은 결과가 나오도록 보장:

| 단계 | 코드 | 함수 |
|---|---|---|
| (a) target_weights | `daily_update.py:457-464` | `optimize_portfolio()` — MVO with cov shrinkage |
| (b) confidence | `daily_update.py:467-479` | `compute_signal_confidence()` — spread × trailing IC |
| (c) candidate_weights | `daily_update.py:482-484` | `apply_dynamic_execution()` — no-trade band + partial η, conf-scaled |
| (d) new_weights | `daily_update.py:487-506` | `project_portfolio_weights()` — TE + sector hard constraints |

**중요한 한계:** `daily_update.py`는 `outputs/daily_state.pkl`만 갱신하고 `outputs/baseline_v4/backtest_result.pkl`을 갱신하지 않는다. 따라서 **`--mode incremental`로만 운영하면 dashboard는 마지막 `--mode full` 시점의 backtest 결과를 계속 보여준다** (`update_and_deploy.py:194-198` 참조).

---

## 7. `src.backtest.run_backtest`의 8 phase

`run_variant.py`와 `daily_update.py --full-init` 모두 결국 `src.backtest.run_backtest`로 위임한다. CLAUDE.md의 Phase 1~8을 코드 기준으로 다시 정리:

| Phase | 모듈 | 역할 |
|---|---|---|
| 1 | `src/data_loader.py` `UniverseData` | xlsx 로드, ticker 매핑, 결측치 처리 (ffill → cross-sectional median) |
| 2 | `src/feature_engine.py` `build_all_features` | ~350 피처 (Accounting, Price, Sentiment, Conditioning, Macro). Cross-sectional Z-score 정규화 |
| 3 | `src/target_engine.py` `build_targets` | 20일 forward Specific Return = PCA(n=5) 잔차 |
| 4 | `src/model_trainer.py` `walk_forward_train` | 3년 rolling LightGBM, 63일마다 재학습, EMA prediction smoothing |
| 5 | `src/backtest.py` 메인 루프 | Walk-forward 백테스트, 21일(또는 yaml override) 리밸런싱 |
| 6 | `src/portfolio_optimizer.py` `optimize_portfolio` | cvxpy MVO. risk_aversion, turnover_penalty, max_TE, sector ±10% |
| 7 | `src/attribution.py` `run_attribution` | SHAP TreeExplainer, feature group 기여도 |
| 8 | `src/backtest.py` `compute_metrics` | IR, TE, turnover, sub-period metrics |

산출물: `BacktestResult` 객체 → pickle → `outputs/<variant>/backtest_result.pkl` (~65MB)

---

## 8. Stage 3 — Dashboard payload 빌드

함수: `stage_build_dashboard(run_dir, data, out_pkl)` (`update_and_deploy.py:269-291`)
실행: `python scripts/build_dashboard_data.py --run --data --out`

### 입력
- `outputs/baseline_v4/backtest_result.pkl` (~65MB — panel 49MB + models 4MB + ...)
- `data/ai_signal_data.xlsx` (CUR_MKT_CAP 시트만 추가로 read)

### 출력
- `outputs/baseline_v4/dashboard_data.pkl` (~3MB)

### 6 산출 항목 (`scripts/build_dashboard_data.py:73-216`)
| 항목 | 설명 |
|---|---|
| feature_importance_pct | LightGBM gain importance를 walk-forward 평균 → % |
| ic_table | feature × Spearman IC mean/std/IR + bucket 매핑 |
| group_pnl | 7개 feature group (Growth/Quality/Value/Revision/Momentum/Low-vol/Macro) long-short PnL |
| score_breakdowns | 리밸런싱일별 종목 × group z-score 매트릭스 |
| rebal_predictions | 리밸런싱일별 raw prediction |
| bm_weights / portfolio_weights | 비중 이력 |

---

## 9. Stage 4 — Local smoke (optional)

함수: `stage_smoke()` (`update_and_deploy.py:297-306`)
- `--smoke` 옵션 시에만 실행
- `streamlit run streamlit_mobile.py` 로컬 8501 포트에서 blocking
- 사용자가 결과 확인 → Ctrl+C → `--no-smoke`로 다시 실행해서 deploy 진행

기본은 OFF.

---

## 10. Stage 5 — cc2-dashboard repo 배포

함수: `stage_deploy(dashboard_repo, run_dir, dashboard_pkl, push)` (`update_and_deploy.py:327-373`)

### SYNC_FILES (`update_and_deploy.py:71-75`)
- `streamlit_mobile.py`
- `requirements_dashboard.txt`

### 추가 sync
- `outputs/baseline_v4/dashboard_data.pkl` (relative path 보존)

### Git 흐름
1. `_git(["status", "--porcelain"])` — 변경 없으면 commit/push 건너뜀
2. `_git(["add", *SYNC_FILES])`
3. `_git(["add", "-f", str(rel.as_posix())])` — pkl은 .gitignore에 있으므로 -f
4. `_git(["commit", "-m", f"update: dashboard data {timestamp}"])`
5. `_git(["push", "origin", "main"])` (`--no-push` 옵션 시 생략)

### Streamlit Cloud 자동 재배포
push 감지 → ~30초 안에 재빌드 + 재배포 → 휴대폰에서 `https://<user>-cc2-dashboard.streamlit.app` 접속 시 최신 데이터.

상세 setup: `DEPLOY.md` 참조.

---

## 11. 산출물 매핑 표 (전체 stage 통합)

| Stage | Input 파일 | 호출 스크립트 | Output 파일 |
|---|---|---|---|
| 1 | `data/ai_signal_data.xlsx` (Daily_Returns 시트만 peek) | (없음 — pd.read_excel) | (없음 — console만) |
| 2a (full) | 위 + `variants/iter15_65tkr_reb21_vtg.yaml` | `run_variant.py` → `src.backtest.run_backtest` | `outputs/iter15_65tkr_reb21_vtg/{metrics.json, backtest_result.pkl, experiment_manifest.json}` → promote → `outputs/baseline_v4/` |
| 2b (incr) | 위 + `outputs/daily_state.pkl` | `daily_update.py` | `outputs/daily_state.pkl` (갱신) + `outputs/csv/*.csv` |
| 3 | `outputs/baseline_v4/backtest_result.pkl` + xlsx의 CUR_MKT_CAP 시트 | `scripts/build_dashboard_data.py` | `outputs/baseline_v4/dashboard_data.pkl` |
| 4 | `outputs/baseline_v4/dashboard_data.pkl` | `streamlit run streamlit_mobile.py` | (none — blocking server) |
| 5 | 위 + `streamlit_mobile.py` + `requirements_dashboard.txt` | `git add/commit/push` | cc2-dashboard repo의 같은 경로 |

---

## 12. 매일 운영 워크플로우

### A. 매일 (가벼운 갱신, dashboard 미반영)
```
1. data/ai_signal_data.xlsx 새 영업일 데이터 추가
2. update_and_deploy.bat --mode incremental  # ~30초~1분
3. 콘솔 출력 확인 (오늘 PnL, 누적, 현재 비중 Top 5)
```
주의: dashboard pkl은 갱신되지 않으므로 휴대폰 대시보드는 마지막 full 시점을 계속 표시.

### B. 매주~격주 (정식 갱신, dashboard 자동 재배포)
```
1. data/ai_signal_data.xlsx 최신 데이터 확인
2. update_and_deploy.bat   ← 인자 없이 = --mode full
3. ~3-4분 대기 (백테스트 진행 콘솔 모니터)
4. "[OK] Pipeline finished. Streamlit Cloud will redeploy in ~30s." 메시지 확인
5. 휴대폰에서 https://<user>-cc2-dashboard.streamlit.app 접속 → 새 데이터 확인
```

### C. 응급 (push 보류하고 로컬에서만 확인)
```
update_and_deploy.bat --no-push --smoke
# 로컬 streamlit 띄움 → 결과 확인 → Ctrl+C
# OK면 다시 update_and_deploy.bat (push 포함)
```

---

## 13. 트러블슈팅

| 증상 | 원인 | 복구 |
|---|---|---|
| Stage 1 "data file not found" | xlsx 경로 오류 | `--data <abs_path>` 명시 또는 `update_and_deploy.py:57` `DEFAULT_DATA_FILE` 수정 |
| Stage 1 "data is Nd old" WARN | xlsx에 새 영업일 미반영 | xlsx refresh 후 재실행. 7일 이내면 진행 가능 (warn은 fail이 아님) |
| Stage 2 full "variant yaml not found" | yaml 경로 오류 | `--variant <abs_path>` 명시 |
| Stage 2 full backtest 중 OOM | 데이터/피처 메모리 부족 | venv에 충분한 RAM 확보, 다른 프로세스 종료 |
| Stage 2 incr "daily_state.pkl 없음" | 첫 실행 | 자동으로 `--full-init` 실행됨 |
| Stage 2 incr "schema v1 < v2" | 옛 state pkl | `outputs/daily_state.pkl` 삭제 후 `--mode incremental` 재실행 |
| Stage 2 incr 비중 검증 실패 | 제약 위반 | 콘솔 메시지 확인. 자동으로 이전 비중 유지 + state_backup.pkl로 롤백 시도 |
| Stage 3 "backtest_result.pkl not found" | full 모드 한 번도 안 돌림 | `update_and_deploy.bat` (full mode 한 번) 실행 |
| Stage 5 "dashboard repo not found" | repo path 오류 | `CC2_DASHBOARD_REPO` env var 또는 `--dashboard-repo` 옵션 |
| Stage 5 git push 실패 | 인증 / 충돌 | 수동으로 `cd <dashboard_repo> && git pull --rebase && git push` |
| Streamlit Cloud "Module Not Found" | requirements 누락 | `requirements_dashboard.txt`에 추가 → 재배포 |
| 휴대폰에서 dashboard에 옛 날짜 표시 | --mode incremental만 돌림 | `--mode full` 실행으로 dashboard pkl 갱신 |

---

## 부록 A: variants/iter15_65tkr_reb21_vtg.yaml 핵심 override

(production manifest의 핵심 파라미터를 표로 정리 — yaml에서 직접 인용)

## 부록 B: 환경 변수
| 환경 변수 | 기본값 | 용도 |
|---|---|---|
| `CC2_DASHBOARD_REPO` | `C:/Users/westl/Documents/cc2-dashboard` | Stage 5 배포 대상 |
| `CC2_RUN_DIR` | `outputs/baseline_v4` | canonical run dir |
```

## Acceptance Criteria

```bash
test -f docs/UPDATE_AND_DEPLOY_FLOW.md

# 모든 주요 섹션 anchor 존재
for h in "개요" "Entry" "Orchestrator" "Stage 1" "Stage 2a" "Stage 2b" "Phase 1" "Phase 8" "Stage 3" "Stage 5" "산출물 매핑" "매일 운영" "트러블슈팅"; do
  grep -q "$h" docs/UPDATE_AND_DEPLOY_FLOW.md || (echo "MISSING anchor: $h"; exit 1)
done

# mermaid 다이어그램 최소 1개
grep -c '```mermaid' docs/UPDATE_AND_DEPLOY_FLOW.md   # >= 1

# DELETE 후보 스크립트가 doc 본문에 등장하지 않는지 (cleanup 후 dead reference 방지)
python -c "
import re, pathlib
proposal = pathlib.Path('phases/update-deploy-cleanup/_research/cleanup_proposal.md').read_text(encoding='utf-8')
doc = pathlib.Path('docs/UPDATE_AND_DEPLOY_FLOW.md').read_text(encoding='utf-8')
# DELETE 행에서 파일명 추출
delete_files = re.findall(r'\| \\?\\\`?([\\w_/]+\\.py)\\\`? \\|.*\\| DELETE \\|', proposal)
# basename only check (path 차이 무시)
leaked = [f for f in delete_files if pathlib.Path(f).name in doc]
assert not leaked, f'doc references DELETE candidates (will be dead refs after step3): {leaked}'
print(f'OK — doc references 0 DELETE-candidate scripts (checked {len(delete_files)})')
"
```

## 검증 절차

1. AC 커맨드 모두 통과 확인
2. 시각 점검:
   - mermaid 다이어그램이 syntactically valid 한가? (https://mermaid.live 에 붙여서 렌더링 확인 가능)
   - "Stage 2a" 와 "Stage 2b" 가 명확히 구분되어 있는가?
   - 산출물 매핑 표에서 input → script → output 체인이 끊어지지 않는가?
3. step 2 status 업데이트:
   - 성공 → `"completed"` + `"summary": "docs/UPDATE_AND_DEPLOY_FLOW.md created (~K lines, M sections, mermaid diagrams)"`
   - 실패 → `"error"` + 메시지

## 금지사항

- **DELETE 분류된 스크립트를 운영 로직의 일부로 설명하지 마라.** 이유: step3에서 삭제되는 순간 dead reference가 됨. 만약 어떤 DELETE 스크립트가 흐름의 일부라고 판단되면 step1으로 돌아가 KEEP으로 재분류.
- **추측으로 동작을 적지 마라.** 모든 함수 / 라인 / 파일 경로는 실제 source에서 가져온 사실. 확신 없으면 anchor를 빼라.
- **CLAUDE.md를 그대로 복붙하지 마라.** Phase 1~8 표는 *코드 기준*으로 재정리한 새 표. CLAUDE.md는 design 의도, 본 doc은 실제 코드 흐름.
- **`outputs/`, `data/`, source 코드를 수정하지 마라.** 이 step의 write는 `docs/UPDATE_AND_DEPLOY_FLOW.md` 단 하나.
