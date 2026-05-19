# Cleanup Proposal

## 요약

| Verdict | 개수 | 비고 |
|---|---|---|
| KEEP | 7 | production deploy flow + adjacent validation |
| DELETE (확신) | 13 | 0~5 외부 refs, 모두 self/peer |
| NEEDS_USER_DECISION | 4 | 옛 entry-point 체인 (`main.py` ↔ `export_csv.py` ↔ `run_all.py` ↔ `streamlit_app.py`) |

---

## KEEP (7개) — 절대 삭제 X

| File | 이유 | 외부 refs |
|---|---|---|
| `update_and_deploy.bat` | production entry-point (사용자 더블클릭) | (orchestrator) |
| `update_and_deploy.py` | 5-stage orchestrator | (orchestrator) |
| `run_variant.py` | Stage 2 full mode 백엔드 | update_and_deploy.py:251 |
| `daily_update.py` | Stage 2 incremental mode 백엔드 | update_and_deploy.py:203,210 |
| `streamlit_mobile.py` | Stage 4/5 dashboard (cc2-dashboard repo로 sync) | update_and_deploy.py:73,303 |
| `scripts/build_dashboard_data.py` | Stage 3 dashboard 페이로드 빌더 | update_and_deploy.py:279 |
| `run_selection_bias.py` | Deflated Sharpe Ratio validation (Bailey & Lopez de Prado 2014) — CLAUDE.md / AGENTS.md / experiment_inventory.json에 운영 워크플로의 일부로 명시 | CLAUDE.md, AGENTS.md, experiment_inventory.json |

---

## DELETE (확신, 13개)

모두 외부 reference가 0~5개 (self + peer + 기타 dead-script 내부 reference만). production 흐름과 무관.

| File | refs (excl .claude/) | 분류 근거 |
|---|---|---|
| `compare_improvements.py` | 2 | self refs only — A/B/C/D 개선 비교 (1회성 스터디) |
| `compare_max_weight.py` | 0 | 완전 dead — MAX_WEIGHT sweep |
| `compare_pca.py` | 2 | self refs — PCA n_remove sweep |
| `experiment_monthly_rebal.py` | 0 | 완전 dead — 월말 1회 리밸런싱 비교 |
| `experiment_te_sensitivity.py` | 0 | 완전 dead — TE 민감도 스터디 |
| `feature_pnl_attribution.py` | 0 | 완전 dead — feature-level PnL 분석 (1회성) |
| `grid_search.py` | 0 (excl .claude) / 25 (incl .claude) | .claude/agents/skills 문서 멘션만, production import 0개. **agent docs 업데이트 필요** |
| `run_finalize_iter15.py` | 5 | self refs — iter15 reproduction (run_variant.py로 대체) |
| `prepare_data.py` | 6 | self + run_backtest_harness 페어. 옛 "harness 패턴" |
| `run_backtest_harness.py` | 5 | self + prepare_data 페어. 옛 "harness 패턴" |
| `scripts/compare_revision_variants.py` | 1 | self ref only |
| `scripts/diagnose_revision_spikes.py` | 4 | self refs — outputs/diagnostics 1회성 audit |
| `scripts/regen_csv_from_pkl.py` | 6 | export_csv import (export_csv도 NEEDS_USER_DECISION 후보) |
| `scripts/render_baseline_metrics.py` | 6 | self refs |

---

## NEEDS_USER_DECISION (4개) — 옛 entry-point 체인

이 4개는 서로 묶여 있어서 **하나를 결정하면 나머지가 자동 결정**된다.

```
main.py ─── imports ──→ export_csv.py
   ↑
   │ wraps
   │
run_all.py ─── checks ──→ streamlit_app.py (legacy desktop dashboard)
```

| File | 외부 refs | 역할 | 대체된 곳 |
|---|---|---|---|
| `main.py` | 21 | "전체 파이프라인" CLI entry-point. CLAUDE.md/AGENTS.md에 documented | `run_variant.py` (via `update_and_deploy.py --mode full`) |
| `run_all.py` | 6 | main.py + streamlit_app 묶어서 한 번에 실행 | `update_and_deploy.bat` (production) |
| `export_csv.py` | 5 | CSV export 함수 14개 (main.py가 import) | `scripts/build_dashboard_data.py` (dashboard용) |
| `streamlit_app.py` | (이전 분석에서 KEEP으로 분류했었지만 위 의존성 제거 시 함께 제거 가능) | legacy desktop dashboard | `streamlit_mobile.py` (cloud) |

### Option A: 4개 모두 KEEP (보수적)
- 장점: 이전 entry-point 보존. ad-hoc 백테스트 / 데스크탑 대시보드 가능
- 단점: 두 가지 entry path가 공존 → 운영자 혼란
- 작업: 없음 (현 상태 유지)

### Option B: 4개 모두 DELETE (공격적, "깔끔한 최종버전" 의도에 부합)
- 장점: 운영 entry path 단일화 (update_and_deploy.bat만), repo 정돈
- 단점: 다음 파일에서 dead reference 정리 필요:
  - `CLAUDE.md` lines 38, 46, 214, 235 (main.py 관련 4곳) + 49 (streamlit_app)
  - `AGENTS.md` lines 38, 46, 214, 235 (main.py 관련 4곳) + 49 (streamlit_app)
  - `docs/ROADMAP.md` lines 242, 246, 247
  - `run_selection_bias.py:112,244` (error message — main.py / run_variant.py 둘 중 하나로 단순화)
  - `streamlit_mobile.py` (혹시 streamlit_app reference 있으면)
  - `.claude/agents/cc2-pipeline-operator.md:26,41` (`python -u main.py` → `python -u update_and_deploy.py --no-deploy`로 변경)
  - `.claude/agents/dashboard-publisher.md:11` (export_csv 관련)
  - `.claude/agents/backtester.md:14` (grid_search 관련 — 이미 위 DELETE로 처리)
  - `.claude/skills/cc2-iterate/references/eval_gates.md:61` (`python main.py`)
  - `.claude/skills/cc2-run-backtest/SKILL.md` (있다면)
  - `.claude/skills/pipeline-orchestrator/SKILL.md` (있다면)

### Option C: 일부만 DELETE
예: main.py + export_csv는 KEEP, run_all.py + streamlit_app은 DELETE.
근거: main.py는 ad-hoc 백테스트로 유용, run_all/streamlit_app는 update_and_deploy로 완전 대체.

---

## 재현 grep 명령

각 DELETE 항목의 reference 0개를 재확인:

```bash
PARENT="C:/Users/westl/PycharmProjects/pythonProject/venv_vf_new/machine/re_study/c2/ai_signal_cc2_harness"
for stem in compare_improvements compare_max_weight compare_pca experiment_monthly_rebal experiment_te_sensitivity feature_pnl_attribution run_finalize_iter15 prepare_data run_backtest_harness compare_revision_variants diagnose_revision_spikes render_baseline_metrics; do
  cnt=$(grep -rEn "(${stem})\.py|^import ${stem}|^from ${stem} " \
      --include="*.py" --include="*.md" --include="*.json" --include="*.bat" --include="*.yaml" --include="*.txt" \
      --exclude-dir=.git --exclude-dir=outputs --exclude-dir=data --exclude-dir=archive --exclude-dir=__pycache__ --exclude-dir=phases --exclude-dir=variants --exclude-dir=.claude --exclude-dir=.agents \
      "$PARENT" 2>/dev/null | grep -v 'experiment_inventory.json' | wc -l)
  echo "$cnt $stem"
done
```
