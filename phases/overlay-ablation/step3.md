# Step 3: production-rebuild

step2의 결정을 반영해 `variants/baseline_v5.yaml`을 작성하고, 단 1회의 `oos_verify` peek로 cutoff 이후 데이터(2025-01-01 ~ 2026-04)에서 OOS 검증한다. 통과 시 production manifest를 promote하고 탈락 overlay를 rollback_log에 기록한다.

## 읽어야 할 파일

- `docs/ABLATION_REPORT.md` (step2 산출 — KEEP/DROP 결정 + 결정 요약 YAML 블록)
- `variants/iter15_FINAL_postfix.yaml` (Task A 산출 — base manifest)
- `outputs/iter15_FINAL_postfix/metrics.json` (research-window baseline)
- `outputs/ablation/summary.csv`
- `docs/BASELINE.md`, `CLAUDE.md` (production 표기 위치)
- `docs/rollback_log.md` (탈락 overlay 기록 형식)
- `experiment_inventory.json` (peek 카운터)
- **이전 step 산출 확인**: `phases/overlay-ablation/index.json` step 0, 1, 2 summary

## 작업

### 1. `variants/baseline_v5.yaml` 작성

step2의 결정 요약 YAML 블록을 base manifest 위에 적용:

```yaml
label: baseline_v5
description: >
  Post-ablation production candidate. Surviving overlays only. Built on
  iter15_FINAL_postfix (embargo + cutoff). KEEP/DROP determined under
  research mode with block-bootstrap CI; see docs/ABLATION_REPORT.md.
out_dir: outputs/baseline_v5
tuning_mode: oos_verify   # 1회 peek — experiment_inventory.json에 기록됨
overrides:
  # === A 단계 산출 (고정) ===
  rebalance_freq: 21
  embargo_days: 20
  # train_cutoff_date 미설정 → enforce_oos_holdout=False 로 전체 윈도우 사용

  # === step2 결정에 따라 ===
  value_trap_gate_enabled: <KEEP/DROP에 따라>
  growth_tilt_enabled: <…>
  pead_boost_enabled: <…>
  mega_cap_funding_mode: <…>
  revision_clean_mode: "<…>"
  feature_mode: "<…>"
```

**`enforce_oos_holdout`은 manifest에 명시하지 마라**. `tuning_mode: oos_verify`가 `compose_config`에서 자동으로 False로 강제하며 동시에 peek 카운터를 증가시킨다 (Task A step1에서 구현).

### 2. 실행 (peek — 1회만)

```bash
python run_variant.py --variant variants/baseline_v5.yaml --no-cache
```

이 실행은 `experiment_inventory.json` `n_oos_peeks`를 정확히 1 증가시킨다. 결과가 마음에 안 들어도 **다시 manifest 만들고 또 peek 하지 마라**. 그건 selection-bias 회계 위반이다.

산출 검증:
- `outputs/baseline_v5/metrics.json`
- `outputs/baseline_v5/backtest_result.pkl`
- `outputs/baseline_v5/experiment_manifest.json`

### 3. OOS 검증 보고: `outputs/baseline_v5/oos_report.md`

```markdown
# baseline_v5 OOS Verification

**Peek date**: <YYYY-MM-DD>
**experiment_inventory.json n_oos_peeks**: <N> (증가량 1)
**git_hash**: <…>

## Configuration delta vs iter15_FINAL_postfix
- value_trap_gate_enabled: <was → now>
- growth_tilt_enabled: <…>
- pead_boost_enabled: <…>
- mega_cap_funding_mode: <…>
- revision_clean_mode: <…>
- feature_mode: <…>

## Metrics

| Metric | iter15_FINAL_postfix (research window) | baseline_v5 (full window, peeked) | Comment |
|---|---|---|---|
| IR | … | … | … |
| P1_ir | … | … | … |
| P2_ir | … | … | … |
| P3_ir (full, incl. 2025+) | — | … | post-cutoff segment |
| TE | … | … | … |
| Turnover | … | … | … |

## Verdict

- [ ] IR ≥ iter15_FINAL_postfix IR (cutoff-window 비교 기준)
- [ ] P2_ir ≥ baseline P2_ir - 0.10
- [ ] turnover ≤ baseline turnover + 5%p
- [ ] DROP된 overlay가 다시 켜지지 않았음을 manifest로 확인

전부 ✓ → PROMOTE. 하나라도 ✗ → DO NOT PROMOTE, 보고서에 사유 + 다음 액션 명시.
```

### 4. PROMOTE 분기

**4a. PROMOTE 가능한 경우**:

i) `docs/BASELINE.md`의 canonical baseline 표기를 `baseline_v5`로 교체. legacy(`iter15_FINAL_postfix`, `iter15_65tkr_reb21_vtg`)는 각각 "research baseline", "pre-leak-fix baseline" 박스에 보존.

ii) `CLAUDE.md` "최종 성과" 표를 baseline_v5 수치로 갱신. 출처 표기 갱신.

iii) `docs/rollback_log.md`에 DROP 항목 일괄 추가 (step2 보고서의 ready-to-paste 블록 사용):
```markdown
## <date> — DROP <overlay>
- Date: <YYYY-MM-DD>
- Reason: ablation under research mode showed ΔIR <X.XXX> [<lo>, <hi>] not distinguishable from zero.
- Original rationale (now superseded): <CLAUDE.md 인용>
- Bootstrap: block_size=10, n_iter=1000, seed=42.
```

iv) **`outputs/baseline_v4/` 처리**: 옵션 A (Task A step3) 정책 유지. `baseline_v5/`는 별도 디렉토리. production 운영 경로(`update_and_deploy.bat`, `daily_update.py`, `streamlit_mobile.py`)는 *이 step에서 건드리지 않는다* — scope 초과. 대신 `outputs/baseline_v5/README.md`에 "다음 deploy 사이클에서 baseline_v4 → baseline_v5 cutover 필요" 한 줄 메모.

**4b. NOT PROMOTE 인 경우**:
- `outputs/baseline_v5/oos_report.md`에 실패 사유 + 분석을 충실히 적는다.
- 어떤 overlay가 cutoff 이후 데이터에서 다르게 작동했는지 (예: P3 강세는 cutoff 이전 데이터로만 측정된 거였고, 2025+ 구간에서 무너짐) 명시.
- production은 그대로 두고 (`outputs/baseline_v4/` = legacy 그대로), 이 task는 *결론 = baseline_v5 promotion 보류*로 close.
- 추가 peek 금지. 새 candidate를 만들고 싶으면 Task B를 재실행해야 한다 (즉 ablation 자체를 다시 — 추가 자유도 N_trials += 6).

### 5. peek 카운터 검증

```python
import json
inv = json.load(open("experiment_inventory.json"))
peeks = inv.get("oos_peeks", [])
# baseline_v5 label이 정확히 1번 등장
labels = [p["label"] for p in peeks]
assert labels.count("baseline_v5") == 1, f"unexpected peek count: {labels}"
```

## Acceptance Criteria

```bash
# 1. manifest + 실행 산출물
test -f variants/baseline_v5.yaml
test -f outputs/baseline_v5/metrics.json
test -f outputs/baseline_v5/backtest_result.pkl
test -f outputs/baseline_v5/oos_report.md

# 2. peek 카운터 +1 (정확히)
python -c "
import json
inv = json.load(open('experiment_inventory.json'))
peeks = inv.get('oos_peeks', [])
assert [p['label'] for p in peeks].count('baseline_v5') == 1, 'baseline_v5 peek count != 1'
print('OOS peeks:', inv['n_oos_peeks'])
"

# 3. PROMOTE 분기 일관성
PROMOTE=$(grep -q "PROMOTE" outputs/baseline_v5/oos_report.md && grep -q "DO NOT" outputs/baseline_v5/oos_report.md; echo $?)
# oos_report에 PROMOTE/DO NOT PROMOTE 결정이 명시되어 있어야 함

# 4. PROMOTE 시 doc 일관성
if grep -q "PROMOTE.*\[x\]" outputs/baseline_v5/oos_report.md 2>/dev/null; then
    grep -q "baseline_v5" docs/BASELINE.md
    grep -q "baseline_v5" CLAUDE.md
    test -f outputs/baseline_v5/README.md
fi

# 5. rollback_log.md 갱신 (DROP된 overlay 수 ≥ 1이면)
DROPS=$(grep -c -E "DROP" docs/ABLATION_REPORT.md || echo 0)
if [ "$DROPS" -gt 0 ]; then
    NEW_ENTRIES=$(grep -c -E "^## .* — DROP" docs/rollback_log.md)
    [ "$NEW_ENTRIES" -gt 0 ] || { echo "rollback_log not updated"; exit 1; }
fi
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - peek은 정확히 1회만 기록됐는가? (`n_oos_peeks` 증가량 = 1)
   - `outputs/baseline_v4/`와 `outputs/baseline_v5/`가 공존하는가? (production cutover는 미수행)
   - DROP된 overlay 모두 rollback_log.md에 사유와 함께 적혔는가?
   - `docs/ROADMAP.md`의 Phase 2 (P2 IR 회복) 목표가 이 step의 결과로 어떻게 갱신/폐기되어야 하는지 한 줄 노트?
3. `phases/overlay-ablation/index.json` step 3 업데이트:
   - 성공 (PROMOTE) → `"status": "completed"`, `"summary": "baseline_v5 promoted. IR=<X.XXX> (vs iter15_FINAL_postfix research <Y.YYY>, vs legacy iter15 1.310). DROPPED: <list>. Rollback log updated. n_oos_peeks=<N> (+1)."`
   - 성공 (NOT PROMOTE) → `"status": "completed"`, `"summary": "baseline_v5 verification failed: <조건 미달 명세>. Production unchanged. Ablation candidates exhausted; further iteration would require a new ablation round (selection-bias cost)."`
   - 실패/blocked → 사유

## 금지사항

- **peek을 2회 이상 하지 마라.** 이유: 이게 selection-bias 회계의 최종 가드. baseline_v5 결과가 마음에 안 들어도 추가 peek 금지. discipline의 핵심.
- **`tuning_mode: oos_verify`를 다른 manifest에 함부로 쓰지 마라.** 이유: 같은 candidate에 대해 1회만 사용. 새 candidate가 등장하면 그건 새 ablation 후 새 manifest.
- **`outputs/baseline_v4/`를 삭제/rename/덮어쓰지 마라.** 이유: production 운영 경로가 참조. cutover는 별도 task.
- **PROMOTE 실패 시 manifest를 조용히 수정해 다시 돌리지 마라.** 이유: 그건 새 peek이고 N_trials += 1. 회계상 정직하게 실패를 기록.
- **DROP된 overlay의 config 디폴트를 이 step에서 False로 바꾸지 마라.** 이유: production cutover 전까지 `daily_update.py` 등이 디폴트를 참조한다. config 디폴트 변경은 별도 후속 task.
