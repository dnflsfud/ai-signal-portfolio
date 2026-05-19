# Script Inventory

`update_and_deploy.bat`이 동작하는 부모 프로젝트 (`C:\Users\westl\PycharmProjects\pythonProject\venv_vf_new\machine\re_study\c2\ai_signal_cc2_harness`)의 모든 root `*.py` + `scripts/*.py`. `src/`, `archive/`, `.claude/`, `phases/` 제외.

생성 시점: 2026-05-15. 모든 파일 정보는 그 시점 기준.

## 중요 컨텍스트

부모 프로젝트는 매우 dirty 상태:
- 대부분의 cleanup 후보 + load-bearing 파일이 **untracked** (`update_and_deploy.{bat,py}`, `run_variant.py`, `daily_update.py` 등 모두)
- 일부 legacy 파일만 tracked (`grid_search.py`, `main.py`, `run_all.py`, `export_csv.py`)
- 따라서 cleanup은 `git rm`이 아닌 `rm`으로 처리 (tracked 파일은 `git rm`)

## Repo root *.py

| File | Tracked | Lines | Docstring (1 line) | has __main__ |
|---|---|---|---|---|
| `compare_improvements.py` | UNTRACKED | 339 | compare_improvements.py - Baseline vs Improved Model Comparison | yes |
| `compare_max_weight.py` | UNTRACKED | 156 | MAX_WEIGHT 비교 스크립트 | yes |
| `compare_pca.py` | UNTRACKED | 228 | compare_pca.py - PCA n_remove 최적화 비교 | yes |
| `daily_update.py` | UNTRACKED | 704 | 일간 증분 업데이트 스크립트 | yes |
| `experiment_monthly_rebal.py` | UNTRACKED | 283 | Quick Test: 월말 1회 리밸런싱 + 일간 weight drift | yes |
| `experiment_te_sensitivity.py` | UNTRACKED | 184 | TE Sensitivity Test: 12.5% vs 15% vs 17.5% vs 20% | yes |
| `export_csv.py` | TRACKED | 520 | 백테스트 결과를 CSV로 내보내기. (main.py가 import) | yes |
| `feature_pnl_attribution.py` | UNTRACKED | 202 | Feature-level PnL Attribution for LightGBM portfolio | yes |
| `grid_search.py` | TRACKED | 334 | Grid Search v2: 옵티마이저 파라미터 최적화 (경량 버전) | yes |
| `main.py` | TRACKED | 652 | AI Signal Portfolio Construction System (legacy entry) | yes |
| `prepare_data.py` | UNTRACKED | 262 | prepare_data.py - Phase 1~3 데이터 준비 (하네스 패턴) | yes |
| `run_all.py` | TRACKED | 116 | run_all.py - One-Shot Pipeline (main.py + streamlit_app 래퍼) | yes |
| `run_backtest_harness.py` | UNTRACKED | 185 | Phase 4~6 백테스트 (하네스 패턴; prepare_data.py와 페어) | yes |
| `run_finalize_iter15.py` | UNTRACKED | 126 | (no docstring) iter15 reproduction; run_variant.py가 대체 | yes |
| `run_selection_bias.py` | UNTRACKED | 637 | Deflated Sharpe Ratio 검증 (Bailey & Lopez de Prado 2014) | yes |
| `run_variant.py` | UNTRACKED | 338 | Unified variant runner — Stage 2 full mode 백엔드 | yes |
| `streamlit_app.py` | UNTRACKED | 869 | Legacy desktop 대시보드 | yes |
| `streamlit_mobile.py` | UNTRACKED | 494 | Mobile/Cloud 배포용 대시보드 (Stage 5 deploy 대상) | no |
| `update_and_deploy.py` | UNTRACKED | 461 | One-shot orchestrator: data → backtest → dashboard build → deploy | yes |

추가:
- `update_and_deploy.bat` — UNTRACKED, 33 lines. venv 활성 + python update_and_deploy.py 호출 래퍼.

## scripts/*.py

| File | Tracked | Lines | Docstring (1 line) | has __main__ |
|---|---|---|---|---|
| `scripts/build_dashboard_data.py` | UNTRACKED | 231 | Precompute lightweight dashboard data (Stage 3에서 호출) | yes |
| `scripts/compare_revision_variants.py` | UNTRACKED | 155 | (no docstring) | yes |
| `scripts/diagnose_revision_spikes.py` | UNTRACKED | 342 | (no docstring) | yes |
| `scripts/regen_csv_from_pkl.py` | UNTRACKED | 172 | Regenerate dashboard CSVs from existing pkl (export_csv 의존) | yes |
| `scripts/render_baseline_metrics.py` | UNTRACKED | 125 | (no docstring) | yes |

## Reference 카운트 (cross-reference grep 결과)

각 파일명의 외부 reference (자기 자신 + experiment_inventory.json 제외):

| File | refs (incl. .claude/) | refs (excl. .claude/) | 해석 |
|---|---|---|---|
| compare_max_weight | 0 | 0 | 완전 dead |
| experiment_monthly_rebal | 0 | 0 | 완전 dead |
| experiment_te_sensitivity | 0 | 0 | 완전 dead |
| feature_pnl_attribution | 0 | 0 | 완전 dead |
| compare_revision_variants | 1 | 1 | 자기 한 번 |
| compare_improvements | 2 | 2 | 자기 + 인근 |
| compare_pca | 2 | 2 | 자기 + 인근 |
| diagnose_revision_spikes | 4 | 4 | self refs |
| run_backtest_harness | 5 | 5 | self + prepare_data 페어 |
| run_finalize_iter15 | 5 | 5 | self refs |
| prepare_data | 6 | 6 | self + run_backtest_harness 페어 |
| render_baseline_metrics | 6 | 6 | self refs |
| regen_csv_from_pkl | 8 | 6 | self + export_csv import |
| grid_search | 25 | 0 | .claude/agents 문서 references만 |
| export_csv | 38 | 5 | main.py가 import + scripts/regen_csv_from_pkl import |
| run_all | 48 | 6 | docs/ROADMAP + self |
| main | 167 | 21 | CLAUDE.md, AGENTS.md, docs/ROADMAP, streamlit_app, run_selection_bias 등 |
