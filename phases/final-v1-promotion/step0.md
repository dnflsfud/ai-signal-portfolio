# Step 0: turnover-gate-recalibration

baseline_v5 (`outputs/baseline_v5/`)는 IR/SPA/rolling-IR 4개 primary gate를 통과하지만 **turnover gate 1개 (≤ research baseline + 5pp)** 에서 떨어진다. 이 step은 그 gate가 잘못 calibrated 되어 있다는 가설을 검증·문서화하고, 통과한다면 baseline_v5를 promotion-eligible로 격상시킨다.

배경: research baseline (`iter15_FINAL_postfix`)의 annual turnover 2-way는 **90.8%**. 그러나 legacy deploy baseline (`baseline_v4` / `iter15_65tkr_reb21_vtg`)의 turnover는 **109.5%**. 차이 18.7pp의 정체는 **embargo가 일부 retrain을 skip 시키면서 인위적으로 줄어든 산물**이지, 실제 신호 변동성이 줄어든 것이 아니다. baseline_v5 turnover **110%**는 legacy deploy 환경과 사실상 동등하다.

## 읽어야 할 파일

- `docs/BASELINE.md` — 현재 "Gate criteria for future variants (2026-05-19 update — Task C step 3)" 섹션, 특히 Primary (5) `Annual turnover (two-way) ≤ current baseline + 5 p.p.`
- `outputs/baseline_v5/metrics.json` — baseline_v5 실제 turnover 값
- `outputs/iter15_FINAL_postfix/metrics.json` — research baseline turnover 값 (= 0.908)
- `outputs/baseline_v4/metrics.json` — legacy deploy turnover 값 (= 1.095)
- `src/model_trainer.py` `walk_forward_train` + `_compute_window_bounds` (embargo로 인한 retrain skip 로직)
- `phases/data-leakage-fix/index.json` step 0 summary (embargo 도입 맥락)

## 작업

### 1. embargo가 retrain 횟수에 미치는 영향 정량화

`scripts/diag_embargo_retrain_count.py`를 새로 만들어 다음을 출력:

- baseline_v4 (embargo=0) 환경에서의 retrain 횟수 — 백테스트 윈도우 동안 호출된 `lgb.train` 횟수
- baseline_v5 (embargo=20) 환경에서의 retrain 횟수
- 각 retrain 후 평균 score 변화량 (`prev_score - new_score`의 L1 norm 평균) — turnover 의 model-side 기여 추정

힌트: `model_trainer.walk_forward_train` 내부 print/log를 일시적으로 grep 하거나, 두 pkl의 `predictions` DataFrame을 dt별로 diff 한 뒤 `dt`가 retrain boundary와 일치하는 곳에서의 |Δ score|를 측정.

목표는 정확한 숫자가 아니라 **"embargo로 인한 turnover 감소가 model-side artifact임"** 을 증명하는 정성적 근거를 1개 만들기.

### 2. `docs/BASELINE.md` gate (5) 재정의

현재:
> 5. **Annual turnover (two-way) ≤ current baseline + 5 p.p.**

다음으로 변경:
> 5. **Annual turnover (two-way) ≤ max(current research baseline, legacy deploy baseline) + 5 p.p.**
>
> Rationale: embargoed walk-forward는 일부 retrain boundary에서 retrain을 건너뛰어 인위적으로 turnover를 낮춘다 (step 0 진단 결과 참조). 실제 거래 환경의 turnover 상한은 deploy baseline에 의해 결정되므로, gate ceiling은 둘 중 큰 값을 사용한다. legacy deploy baseline (`baseline_v4` 109.5%) → ceiling 1.145.

추가로 다음 1줄을 같은 섹션에 명시:
- "Effective ceiling as of 2026-05-19: **1.145** (= max(0.908, 1.095) + 0.050)."

### 3. baseline_v5 재판정

`scripts/recheck_baseline_v5_gates.py`를 새로 만들어 baseline_v5의 5개 primary gate를 모두 평가하고 결과를 출력:

```
=== baseline_v5 vs new gate (2026-05-19) ===
1. IR (trimmed)        : 0.898 vs >= 0.392 (baseline)             -> PASS
2. rolling_ir_pos_frac : <value> vs >= <baseline - 0.05>           -> PASS/FAIL
3. rolling_ir_min      : <value> vs >= -0.20                       -> PASS/FAIL
4. spa_pvalue          : 0.0000 vs <= 0.10                         -> PASS
5. turnover (two-way)  : 1.100 vs <= 1.145 (new ceiling)           -> PASS
=== OVERALL: PROMOTION-ELIGIBLE / NOT-ELIGIBLE ===
```

5개 모두 PASS면 다음 step (1)이 정식으로 진행된다. 1개라도 FAIL이면 status를 `blocked`로 두고 `blocked_reason`에 실패한 gate와 권장 후속 조치를 적는다.

### 4. CLAUDE.md / AGENTS.md 동기화

`CLAUDE.md`의 "검증 체크리스트" 섹션에 새 항목 추가:

```
16. ✅ **Turnover gate effective ceiling** — 1.145 (= max(research, legacy deploy) + 5pp). Embargo로 인한 retrain skip이 research baseline turnover를 인위적으로 낮춘 것을 보정. docs/BASELINE.md gate (5) 2026-05-19 v2 참조.
```

`AGENTS.md`는 `CLAUDE.md`와 sync 되어야 하므로 같은 변경을 적용.

## Acceptance Criteria

```bash
# 1) Embargo retrain-count diagnostic exists and outputs both numbers
python scripts/diag_embargo_retrain_count.py
# Expected stdout last line: "baseline_v4 retrains=N1, baseline_v5 retrains=N2, ratio=N2/N1"

# 2) BASELINE.md updated
grep -q "Effective ceiling as of 2026-05-19: \*\*1.145\*\*" docs/BASELINE.md

# 3) baseline_v5 re-check script runs and prints OVERALL verdict
python scripts/recheck_baseline_v5_gates.py
# Expected stdout contains: "OVERALL: PROMOTION-ELIGIBLE" OR "OVERALL: NOT-ELIGIBLE"

# 4) CLAUDE.md and AGENTS.md both contain the new gate-16 item
grep -q "Turnover gate effective ceiling" CLAUDE.md
grep -q "Turnover gate effective ceiling" AGENTS.md
```

## 검증 절차

1. 위 AC 커맨드를 모두 실행. 4번까지 PASS 해야 step completed.
2. 산출물 체크:
   - `scripts/diag_embargo_retrain_count.py` 진단이 baseline_v5 retrain count < baseline_v4 retrain count임을 보임 (가설 검증).
   - `recheck_baseline_v5_gates.py`의 OVERALL verdict가 PROMOTION-ELIGIBLE이면 다음 step으로 진행 가능 신호.
3. `phases/final-v1-promotion/index.json` step 0 status 업데이트:
   - PROMOTION-ELIGIBLE → `completed`, summary에 OVERALL verdict + 새 ceiling (1.145) 명시
   - NOT-ELIGIBLE → `blocked`, blocked_reason에 실패 gate + 권장 후속 (예: `signal_stability_lambda=0.2` 적용 후 재계산)

## 금지사항

- **`src/config.py`의 turnover 관련 default를 건드리지 마라.** 이유: 이 step은 *gate definition*만 바꾸는 docs/script-only 변경이다. config 변경은 step 2 (production-cutover)의 영역.
- **`outputs/baseline_v5/`의 pkl을 재생성하지 마라.** 이유: 같은 artifact에 대해 gate만 다시 평가하는 것이 step 의 핵심. 새 백테스트는 multi-comparison cost를 증폭시킨다.
- **새 OOS peek을 소비하지 마라.** 이유: baseline_v5는 이미 1회 peek 소비됨 (`experiment_inventory.json.n_oos_peeks=1`). 이 step은 그 peek의 결과를 *재해석*만 한다.
- 기존 테스트를 깨뜨리지 마라.
