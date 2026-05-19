# Step 3: subperiod-deprecate

P1/P2/P3 다중 IR을 *진단용 보조 지표*로 격하하고, primary gate를 step2의 rolling IR + SPA p-value로 전환한다. ROADMAP의 "P2 IR을 X로 회복" 같은 다중 목표 표현을 일괄 폐기한다.

## 읽어야 할 파일

- `src/harness.py` `SUB_PERIODS`, `sub_ir`, `sub_period_irs`
- `run_variant.py` `_summarize` (sub-period 출력 위치) — step2에서 갱신됨
- `docs/BASELINE.md` gate criteria (Task A step3에서 갱신됨)
- `docs/ROADMAP.md` Phase 2 (P2 IR 회복 목표 명시 위치)
- `CLAUDE.md` "검증 체크리스트" 섹션 (Task A step3에서 갱신됨)
- `phases/data-leakage-fix/index.json`, `phases/overlay-ablation/index.json`, `phases/selection-bias-discipline/index.json` 모든 이전 step summary
- **이전 step 산출**: `src/analytics.py`, `outputs/iter15_FINAL_postfix/metrics.json` (rolling_ir_*, spa_pvalue 채워짐)

## 작업

### 1. `src/harness.py` `SUB_PERIODS` 주석 변경

코드는 그대로 두되 (제거하면 기존 metrics 보고서가 깨짐), docstring + 모듈 톱 코멘트로 **demoted to diagnostic** 명시:

```python
# DEPRECATED for primary gating (2026-MM-DD, selection-bias-discipline phase 3).
# P1/P2/P3 sub-period IRs remain as DIAGNOSTIC indicators only — the
# canonical performance gate is now rolling-IR distribution + SPA p-value
# (see src/analytics.py and docs/BASELINE.md). Tuning to maximise sub-period
# IRs is an explicit anti-pattern: it amplifies multiple-comparison cost.

SUB_PERIODS = {
    "P1": ("2018-11-23", "2021-05-11"),
    "P2": ("2021-05-12", "2023-10-27"),
    "P3": ("2023-10-30", "2026-04-13"),
}
```

`sub_period_irs`도 docstring에 "diagnostic only" 한 줄 추가.

### 2. `run_variant.py` `_summarize` 출력 순서 변경

primary metric을 먼저, sub-period는 마지막에 [diagnostic] 라벨로:

```python
def _summarize(metrics, baseline_path):
    print("=" * 60)
    print("  Variant summary")
    print("=" * 60)
    # PRIMARY
    print(f"  IR              : {ir:.3f}")
    print(f"  Rolling IR      : mean={metrics.get('rolling_ir_mean', nan):.3f}, "
          f"min={metrics.get('rolling_ir_min', nan):.3f}, "
          f"pos_frac={metrics.get('rolling_ir_pos_frac', nan):.2f}")
    print(f"  SPA p-value     : {metrics.get('spa_pvalue', nan):.4f}")
    print(f"  Active return   : {metrics.get('active_return', 0.0) * 100:.2f}%")
    print(f"  TE              : {metrics.get('tracking_error', 0.0) * 100:.2f}%")
    print(f"  Turnover        : {metrics.get('avg_annual_turnover', 0.0) * 100:.1f}%")
    # DIAGNOSTIC
    if sp:
        print()
        print("  [diagnostic] Sub-period IR (not a primary gate):")
        print(f"    P1 IR         : {sp.get('P1_ir', nan):.3f}")
        print(f"    P2 IR         : {sp.get('P2_ir', nan):.3f}")
        print(f"    P3 IR         : {sp.get('P3_ir', nan):.3f}")
    ...
```

### 3. `docs/BASELINE.md` gate criteria 재작성

primary gate 새 정의:

```markdown
## Gate criteria (2026-MM-DD update)

A new candidate variant promotes to canonical baseline only when, evaluated
under tuning_mode=research (cutoff=2024-12-31):

### Primary (must pass all)
1. **IR ≥ current baseline IR**
2. **rolling_ir_pos_frac ≥ current baseline - 0.05** (≥95% of stability)
3. **rolling_ir_min ≥ -0.20** (no extended regime of severe underperformance)
4. **spa_pvalue ≤ 0.10** (one-sided H0: E[active] ≤ 0)
5. **Turnover ≤ current baseline + 5%-points two-way**

### Secondary (diagnostic only — informs investigation, not gating)
- Sub-period IRs (P1/P2/P3): minimum threshold none. Reported for regime
  inspection only. Tuning to maximise these is an explicit anti-pattern.
- Drawdown profile, IC stability, weight concentration histogram.

### Process
- All exploration runs use tuning_mode=research (cutoff enforced).
- Final OOS verification uses tuning_mode=oos_verify exactly once per
  candidate; the peek is logged to experiment_inventory.json.
- Promote only if oos_verify metrics also satisfy primaries (1)-(5).
```

### 4. `docs/ROADMAP.md` 갱신

Phase 2 P2 IR 목표 ("P2 IR을 +0.107 → +0.40으로") 모두 strikethrough 또는 명시적 제거. 다음 박스로 교체:

```markdown
## Phase 2 — DEPRECATED (2026-MM-DD)

The original "P2 IR floor" target (+0.107 → +0.40 / +0.30 / etc.) is
deprecated. Sub-period IRs are now diagnostic. The new primary gate is
rolling IR distribution + SPA p-value; see docs/BASELINE.md "Gate criteria".

Reason: simultaneously optimising P1, P2, and P3 IR amplified the
multiple-comparison cost. With N_trials = 402 (per experiment_inventory)
and three correlated targets, the effective N_trials is closer to 1200,
which the haircut Sharpe could not absorb (see CLAUDE.md Selection Bias
section). Replacing the multi-target gate with a single-statistic gate
restores discipline.

Phase 2 subsections (2.1 multi-horizon, 2.2 regime-PCA, 2.3 macro-cross,
2.4 revision cleaning) remain as *research candidates* — each must clear
the new primary gate to graduate.
```

### 5. `CLAUDE.md` "검증 체크리스트" 항목 10 갱신

```markdown
10. ⚠️ **Sub-period stability (diagnostic only)** — P1/P3 강함, P2 양수. 단,
    sub-period IR은 더 이상 promotion gate가 아니다. primary gate는
    rolling_ir + spa_pvalue (docs/BASELINE.md 참조).
```

새 항목 추가:

```markdown
12. ✅ **Single-statistic gate** — rolling_ir + spa_pvalue 기반 promotion 정책
    docs/BASELINE.md에 정의. 다중-목표 fitting 방지.
```

### 6. `docs/AI_METHODOLOGY.md` 보강

§"검증 체크리스트" 또는 §"성과 측정" 어디든 적절한 곳에 다음 박스 추가:

```markdown
### 성과 평가 정책 (2026-MM-DD)
1. **Primary**: IR, rolling_ir (mean/min/pos_frac, 252d window), spa_pvalue.
2. **Diagnostic only**: P1/P2/P3 sub-period IRs, drawdown, IC stability.
3. **Anti-pattern**: 단일 sub-period IR을 promotion gate로 쓰는 것. 다중 비교
   비용 증가로 selection bias 확산.
```

## Acceptance Criteria

```bash
# 1. SUB_PERIODS 코멘트
grep -q "DEPRECATED" src/harness.py
grep -q "diagnostic" src/harness.py

# 2. _summarize 출력 순서 변경 확인
python -c "
import inspect, run_variant
src = inspect.getsource(run_variant._summarize)
# Rolling IR 출력이 P1 IR 출력보다 먼저 나오는가?
idx_ir = src.find('Rolling IR')
idx_p1 = src.find('P1 IR')
assert 0 < idx_ir < idx_p1, f'Rolling IR should appear before P1 IR (got {idx_ir} vs {idx_p1})'
"

# 3. Docs 갱신 확인
grep -q "Primary (must pass all)" docs/BASELINE.md
grep -q "Diagnostic only" docs/BASELINE.md || grep -q "diagnostic only" docs/BASELINE.md
grep -q "spa_pvalue" docs/BASELINE.md
grep -q "DEPRECATED" docs/ROADMAP.md
grep -q "rolling_ir" docs/ROADMAP.md
grep -q "Single-statistic gate" CLAUDE.md
grep -q "성과 평가 정책" docs/AI_METHODOLOGY.md

# 4. summary 출력 실측 (캐시 hit, 빠르게)
python run_variant.py --variant variants/iter15_FINAL_postfix.yaml 2>&1 | tee /tmp/final.log
# Rolling IR 라인이 P1 IR 라인보다 먼저 출력
python -c "
log = open('/tmp/final.log').read()
i = log.find('Rolling IR'); j = log.find('P1 IR')
assert 0 < i < j, f'order wrong: ri={i}, p1={j}'
"

# 5. 기존 sub-period 호출이 깨지지 않았는지
python -c "
from src.harness import sub_period_irs
import pickle
r = pickle.load(open('outputs/iter15_FINAL_postfix/backtest_result.pkl','rb'))
sp = sub_period_irs(r.portfolio_returns.dropna(), r.benchmark_returns.dropna())
assert 'P1_ir' in sp and 'P2_ir' in sp and 'P3_ir' in sp
"

# 6. Task 일관성 — 전 phase가 종료됐는지
python -c "
import json
for d in ['data-leakage-fix','overlay-ablation','selection-bias-discipline']:
    idx = json.load(open(f'phases/{d}/index.json'))
    pending = [s for s in idx['steps'] if s.get('status') != 'completed']
    if d == 'selection-bias-discipline':
        # this step is the last; allow itself to be incomplete at AC check
        pending = [s for s in pending if s.get('step') < 3]
    assert not pending, f'{d}: {pending}'
"
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - `SUB_PERIODS` *값*은 그대로인가? (제거하면 historical metrics가 깨짐)
   - `_summarize`가 새 키 누락 시 NaN으로 안전하게 출력하는가? (old result.pkl 호환)
   - ROADMAP의 Phase 2 sub-항목들이 "research candidate"로 살아 있는가? (모조리 폐기 X)
3. `phases/selection-bias-discipline/index.json` step 3 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "Sub-period IRs demoted to diagnostic. New primary gate: rolling_ir (mean/min/pos_frac) + spa_pvalue. docs/BASELINE.md gate criteria rewritten. docs/ROADMAP.md Phase 2 P2-IR target deprecated. CLAUDE.md checklist updated. _summarize output reordered. All 3 phases (data-leakage-fix, overlay-ablation, selection-bias-discipline) now complete."`
   - 실패/blocked → 사유

또한 이 step이 마지막이므로 top-level `phases/index.json`에서 `selection-bias-discipline`의 status를 `"completed"`로 업데이트한다 (execute.py가 자동 처리 — 수동 편집 금지).

## 금지사항

- **`SUB_PERIODS`를 삭제하거나 빈 dict로 만들지 마라.** 이유: 외부 노트북/`backtest_result.pkl`의 기존 metrics 호환 + 진단용으로 계속 유용.
- **`sub_period_irs`를 deprecation warning으로 감싸지 마라.** 이유: 호출자가 매번 경고 spam. docstring + 모듈 코멘트로 충분.
- **새 primary gate를 *strict* OOS gate로 만들지 마라.** spa_pvalue ≤ 0.10은 *예비* threshold. 더 엄격하게 (0.05) 잡고 싶으면 별도 follow-up에서 데이터로 정당화.
- **`outputs/`의 기존 metrics.json을 뒤로 호환 깨지게 수정하지 마라.** 새 키 추가만, 기존 키 삭제 X.
- **이 step에서 `daily_update.py` 또는 streamlit_mobile.py를 건드리지 마라.** 이유: dashboard cutover는 별도 task.
