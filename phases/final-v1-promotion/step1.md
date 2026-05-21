# Step 1: selection-bias-recount

현재 `experiment_inventory.json.n_trials_total = 402`는 프로젝트 전 기간 (2024년 후반 ~ 2026-05) 동안 시도된 distinct config 수의 누적치다. **2026-05-19 data-leakage-fix 이후의 모델 클래스는 사실상 다른 평가 환경**이다 (embargo + cutoff). 모든 pre-fix trial을 multi-comparison penalty에 동일하게 부과하면 selection-bias gate가 부당하게 보수적이 된다. 이 step은 N_trials 정의를 "이 baseline 후보군 평가에 직접 기여한 trial"로 좁히고, baseline_v5에 대한 DSR/Haircut을 새 N으로 재측정한다.

## 읽어야 할 파일

- `run_selection_bias.py` (특히 `--auto`, `--n_trials` 인자 처리; 70–88, 133–155 단위 통일 fix 부분)
- `experiment_inventory.json` 전체 — 특히 `scripts[]`, `notes[]`, `oos_peeks[]`
- `docs/BASELINE.md` 의 "Canonical Baseline (Research)" 섹션 + legacy "Selection Bias 검증 (legacy environment — STALE 2026-05-19)" 박스 (`CLAUDE.md` 동일 박스)
- `phases/data-leakage-fix/index.json` summary (어떤 trial이 fix 이후인지 기준선)
- `outputs/baseline_v5/backtest_result.pkl` (재측정 대상 pkl)

## 작업

### 1. N_trials 재정의 정책 결정 + 문서화

`experiment_inventory.json`에 다음 2개 필드 추가 (기존 `n_trials_total` 보존):

```json
"n_trials_pre_leakage_fix": 402,
"n_trials_post_leakage_fix": <count>,
"n_trials_active": <count>,
"n_trials_active_rationale": "Post-2026-05-19 leak-fix trials only. The pre-fix model class used a leaky walk-forward; its trials informed a different alpha surface and are not Pareto-dominant under embargo+cutoff. Counting them inflates the haircut without protecting against the leakage-corrected null."
```

`<count>` 산정 방식: `phases/overlay-ablation/index.json` + `phases/final-v1-promotion/` (현재) + `phases/selection-bias-discipline/` 의 step들에서 호출된 distinct backtest variant 수의 합. 예상치 ~10–15. `outputs/ablation/*/metrics.json` 와 `variants/exp_*.yaml` / `variants/baseline_v5.yaml` 등 fix 이후 산출물을 grep 하여 정확히 세어라.

### 2. DSR/Haircut을 새 N으로 재측정

```bash
python run_selection_bias.py --auto --label baseline_v5 \
    --pkl outputs/baseline_v5/backtest_result.pkl \
    --n_trials <n_trials_active>
```

`--n_trials` 인자가 없으면 추가하라. `run_selection_bias.py`가 `experiment_inventory.json`에서 자동으로 읽도록 하려면 `n_trials_active`를 우선 키로 보고, 없으면 `n_trials_total`로 fallback 하도록 패치한다 (1줄 변경).

산출물:
- `outputs/baseline_v5/selection_bias_report.md` (혹은 `outputs/reports/`)
- 다음 5개 키를 metrics 형식으로 출력:
  - `observed_SR_annualized`
  - `DSR` + `DSR_pvalue`
  - `MinTRL_years_required`
  - `haircut_SR_annualized`
  - `adjusted_SR_annualized`
  - `verdict` ∈ {PASS, FAIL}

### 3. `docs/BASELINE.md` + `CLAUDE.md` 갱신

`docs/BASELINE.md`의 "Canonical Baseline (Research)" 섹션 끝에 새 박스:

```
> ## Selection bias check — baseline_v5 (recount, 2026-05-19 v2)
>
> | Metric | Value | Verdict |
> |---|---:|---|
> | Observed SR (ann.) | <val> | — |
> | DSR | <val> (p=<p>) | PASS/FAIL (p ≤ 0.05?) |
> | MinTRL | <yrs> needed vs <held> held | SUFFICIENT/INSUFFICIENT |
> | Haircut SR (ann.) | <val> | — |
> | Adjusted SR | <val> | PASS/FAIL |
> | **Overall** | | **PASS/FAIL** |
>
> N_trials used: <n> (post-leak-fix trials only; rationale in
> experiment_inventory.json `n_trials_active_rationale`).
```

`CLAUDE.md` 의 기존 "Selection Bias 검증 (legacy environment — STALE 2026-05-19)" 박스 아래에 **새 박스 추가** (legacy 박스는 보존하되 새 박스가 canonical임을 명시).

### 4. step 0이 PROMOTION-ELIGIBLE을 냈을 때만 진행

`phases/final-v1-promotion/index.json` step 0 status가 `completed`이며 summary에 "PROMOTION-ELIGIBLE"이 포함되어야 이 step을 시작한다. 그렇지 않으면 status `blocked`, blocked_reason="step 0 not eligible".

## Acceptance Criteria

```bash
# 1) Inventory has the new keys
python -c "import json; d=json.load(open('experiment_inventory.json')); assert 'n_trials_active' in d and 'n_trials_active_rationale' in d, 'missing keys'; print('n_trials_active =', d['n_trials_active'])"

# 2) Selection-bias rerun produces report file
python run_selection_bias.py --auto --label baseline_v5
test -f outputs/baseline_v5/selection_bias_report.md || test -f outputs/reports/baseline_v5_selection_bias.md

# 3) BASELINE.md has the new recount box
grep -q "Selection bias check — baseline_v5 (recount" docs/BASELINE.md

# 4) CLAUDE.md still has legacy STALE box AND the new recount box (legacy preserved)
grep -q "STALE 2026-05-19" CLAUDE.md
grep -q "recount, 2026-05-19 v2" CLAUDE.md || grep -q "Selection bias check.*baseline_v5" CLAUDE.md
```

## 검증 절차

1. AC 4개 모두 PASS.
2. DSR p-value 와 adjusted_SR을 인간 가독적으로 점검:
   - DSR p < 0.05 → 통계적으로 superior. Verdict PASS.
   - adjusted_SR > 0 → haircut 통과.
3. `phases/final-v1-promotion/index.json` step 1 status 업데이트:
   - DSR PASS AND haircut PASS → `completed`, summary에 verdict + 사용한 N_trials 명시
   - 둘 중 하나라도 FAIL → `completed` (작업은 끝났지만 baseline_v5 promotion 자체는 보류)이되 summary 에 결과 명시. **status를 `blocked`로 두지는 마라** — measurement 자체는 정상 수행된 것이므로.

## 금지사항

- **DSR 계산식 단위를 다시 mismatch 시키지 마라.** 이유: 2026-04-30 이전 결과는 annualized SR vs daily-scale sigma_SR mismatch로 √252 배 부풀어 있었다 (`CLAUDE.md` legacy 박스 참조). `run_selection_bias.py` line 70–88, 133–155 에서 단위가 통일된 것을 확인하고, 그 코드를 건드리지 마라.
- **N_trials를 1로 떨구지 마라.** 이유: baseline_v5 자체가 ablation+baseline_v4 vs postfix vs baseline_v5 비교를 거친 candidate이다. 최소한 `outputs/ablation/` + 본 phase의 candidate set은 모두 counting되어야 한다. 정직한 lower bound는 10 근처일 것.
- **legacy STALE 박스를 지우지 마라.** 이유: 단위 오류 이력 + 기존 `IR=1.31`이 무엇이었는지를 보존하는 audit trail이다.
- 기존 테스트를 깨뜨리지 마라.
