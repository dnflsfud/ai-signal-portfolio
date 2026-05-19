# Step 2: baseline-recompute

embargo + cutoff 적용 후 새로운 **canonical baseline**을 산출한다. 기존 `iter15_65tkr_reb21_vtg` (IR=1.31)는 이 두 가지가 모두 없는 환경의 결과이므로 비교 기준으로 쓸 수 없다.

## 읽어야 할 파일

- `variants/iter15_65tkr_reb21_vtg.yaml` ← 기존 production manifest
- `outputs/iter15_65tkr_reb21_vtg/metrics.json` ← 기존 baseline 수치
- `docs/BASELINE.md` ← canonical baseline 정의 문서
- **이전 step 산출물**:
  - `src/config.py` (embargo_days=20, enforce_oos_holdout=True, train_cutoff_date="2024-12-31")
  - `src/model_trainer.py` (embargo 적용된 walk-forward)
  - `run_variant.py` (research/oos_verify/deploy 모드)
- `phases/data-leakage-fix/index.json` step 0, 1 summary

이전 step에서 embargo와 cutoff가 default ON이 되었는지 직접 확인 (`python -c "from src.config import PipelineConfig; c=PipelineConfig(); print(c.embargo_days, c.enforce_oos_holdout, c.train_cutoff_date)"`).

## 작업

### 1. 새 baseline manifest 작성: `variants/iter15_FINAL_postfix.yaml`

```yaml
label: iter15_FINAL_postfix
description: >
  iter15_65tkr_reb21_vtg + walk-forward embargo(=forward_horizon=20) +
  OOS hold-out (train_cutoff_date=2024-12-31). New canonical baseline after
  data-leakage-fix phase. Compare against legacy iter15_65tkr_reb21_vtg
  to quantify the leakage premium that was previously baked into IR=1.31.
out_dir: outputs/iter15_FINAL_postfix
tuning_mode: research
overrides:
  # 기존 production overrides 그대로 복제 (rebal_freq + VTG)
  rebalance_freq: 21
  value_trap_gate_enabled: true
  vtg_pe_z_threshold: -0.5
  vtg_momentum_threshold: -0.5
  vtg_accel_threshold: 0.5
  vtg_scale: 0.0
  # 명시적 cutoff (config default와 동일하지만 manifest에 박아 재현성 확보)
  train_cutoff_date: "2024-12-31"
  embargo_days: 20
```

### 2. 실행

```bash
python run_variant.py --variant variants/iter15_FINAL_postfix.yaml --no-cache
```

`--no-cache`는 필수: 이전 캐시는 누수 환경에서 학습된 모델이므로 재사용 금지.

실행 시간 ~3-5분. 종료 후 다음 파일 생성을 확인:
- `outputs/iter15_FINAL_postfix/metrics.json`
- `outputs/iter15_FINAL_postfix/backtest_result.pkl`
- `outputs/iter15_FINAL_postfix/experiment_manifest.json`

### 3. 비교 보고서 작성: `outputs/iter15_FINAL_postfix/comparison.md`

다음 표를 직접 산출해 채워라:

```markdown
# iter15_FINAL_postfix vs iter15_65tkr_reb21_vtg

**Generated**: <YYYY-MM-DD>
**git_hash**: <from experiment_manifest.json>

## Methodology delta
- legacy: no embargo (label leak), no cutoff (full sample)
- postfix: embargo_days=20, train_cutoff_date=2024-12-31

> Note: cutoff means predictions are generated only through 2024-12-31.
> 2025-01-01 이후 P3 후반부는 reserved OOS이므로 metrics에 미포함.

## Headline metrics (cutoff 이전 구간만 비교)

| Metric              | legacy iter15 | postfix | Δ |
|---------------------|---------------|---------|---|
| Annual Return       | …             | …       | … |
| Active Return       | …             | …       | … |
| Tracking Error      | …             | …       | … |
| Sharpe              | …             | …       | … |
| Information Ratio   | 1.310         | …       | … |
| Max Drawdown        | …             | …       | … |
| Annual Turnover 2w  | …             | …       | … |
| Avg IC              | …             | …       | … |

## Sub-period IR

| Window | legacy | postfix | Δ |
|--------|--------|---------|---|
| P1 (2018-11~2021-05) | 1.54 | … | … |
| P2 (2021-05~2023-10) | 0.17 | … | … |
| P3 (2023-10~2024-12, cutoff trimmed) | (recomputed on same window) | … | … |

## 해석

- ΔIR의 부호와 크기는?
- P1/P2/P3 중 어디가 가장 큰 충격?
- 라벨 누수가 만들던 "프리미엄"의 추정치는?
```

표 채우기 자동화 (참고용):

```python
import json
legacy = json.load(open("outputs/iter15_65tkr_reb21_vtg/metrics.json"))["metrics"]
postfix = json.load(open("outputs/iter15_FINAL_postfix/metrics.json"))["metrics"]
# legacy의 sub-period IR도 동일 cutoff 윈도우에서 재계산해야 fair comparison.
# 즉 legacy backtest_result.pkl을 로드해서 portfolio_returns / benchmark_returns를
# 2024-12-31에서 잘라 sub_ir 재계산.
```

P3 IR 비교는 양쪽 모두 cutoff(2024-12-31)에서 잘라야 공정하다. legacy IR도 그 윈도우로 재계산하라.

### 4. (선택) sanity log

`outputs/iter15_FINAL_postfix/sanity.txt`에 다음을 1줄씩 기록:
- `embargo applied: True (embargo_days=20)`
- `oos cutoff applied: True (train_cutoff_date=2024-12-31)`
- `walk_forward last predict date: <YYYY-MM-DD>` (응당 2024-12-31 이전)
- `legacy last predict date: <YYYY-MM-DD>` (cutoff 없으므로 2026-04 어딘가)

`backtest_result.pkl`의 `portfolio_returns.index[-1]`로 확인.

## Acceptance Criteria

```bash
# 1. 산출 파일 존재
test -f outputs/iter15_FINAL_postfix/metrics.json
test -f outputs/iter15_FINAL_postfix/backtest_result.pkl
test -f outputs/iter15_FINAL_postfix/experiment_manifest.json
test -f outputs/iter15_FINAL_postfix/comparison.md

# 2. cutoff 실제 적용 확인 (예측이 2024-12-31에서 끝나야 함)
python -c "
import pickle, pandas as pd
r = pickle.load(open('outputs/iter15_FINAL_postfix/backtest_result.pkl','rb'))
last = r.portfolio_returns.dropna().index[-1]
assert pd.Timestamp(last) <= pd.Timestamp('2024-12-31'), f'cutoff not enforced: last={last}'
print(f'OK — last predict date: {last.date()}')
"

# 3. comparison.md에 핵심 섹션 존재
grep -c '^## ' outputs/iter15_FINAL_postfix/comparison.md   # >= 3
grep -q 'Headline metrics' outputs/iter15_FINAL_postfix/comparison.md
grep -q 'Sub-period' outputs/iter15_FINAL_postfix/comparison.md
grep -q '해석' outputs/iter15_FINAL_postfix/comparison.md

# 4. peek counter 증가하지 않았어야 함 (research 모드는 peek가 아님)
python -c "
import json
inv = json.load(open('experiment_inventory.json'))
# step 1에서 dryrun으로 1번 증가한 상태가 baseline. 이 step에서는 그대로여야 함.
print('OOS peeks now:', inv.get('n_oos_peeks', 0))
"
```

## 검증 절차

1. AC 커맨드 통과.
2. 아키텍처 체크리스트:
   - `outputs/iter15_FINAL_postfix/experiment_manifest.json`의 config snapshot에 `embargo_days=20`, `enforce_oos_holdout=true`, `train_cutoff_date="2024-12-31"`가 모두 박혀 있는가?
   - `comparison.md`의 sub-period IR이 legacy 쪽도 동일 cutoff 윈도우에서 재계산됐는가? (단순히 기존 metrics.json 값을 베껴 쓰면 안 됨)
3. `phases/data-leakage-fix/index.json` step 2 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "iter15_FINAL_postfix executed. New baseline IR=<X.XXX> (legacy 1.310, Δ=<+/-Y.YYY>). P1=<>, P2=<>, P3=<cutoff-trimmed>. Last predict date confirmed <YYYY-MM-DD>. comparison.md written."`
   - 실패/blocked → 사유

## 금지사항

- **legacy iter15 metrics.json을 그대로 비교에 쓰지 마라.** 이유: 윈도우가 다르다 (legacy=2026-04까지, postfix=2024-12-31까지). 같은 윈도우에서 재계산해야 한다.
- **--no-cache를 빠뜨리지 마라.** 이유: 이전 캐시는 누수 환경 모델이라 재사용하면 변경 효과가 가려진다.
- **결과 IR이 낮게 나왔다고 cutoff/embargo를 풀지 마라.** 이유: 이것이 핵심 진단 수치. "IR이 떨어진다는 사실 자체"가 Task B/C의 입력이다.
- **이 step에서 production manifest를 promote하지 마라.** 이유: production 갱신은 Task B step3의 일이다.
