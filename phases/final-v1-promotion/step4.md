# Step 4: live-delta-monitor

Production deploy 가 끝나면 (step 2 완료 후) **live 실행 vs backtest 재계산 간 drift** 를 매일 측정해야 한다. 둘은 이론적으로 같은 결과를 내야 하지만 실무에서는 다음으로 어긋난다:

- 새 데이터 한 줄이 들어오면서 walk-forward window 가 한 칸 밀려 모델/예측이 약간 달라짐
- ffill 동작 or NaN propagation 차이
- 부동소수 누적
- 코드 변경 (의도치 않은 회귀)

이 step 은 daily target weights 를 **live snapshot 으로 저장** 하고, 백테스트가 그 날짜를 다시 계산했을 때의 weights 와 L1 차이를 추적한다. 큰 drift 가 발생하면 alarming.

step 2 (production-cutover) 가 `completed` 인 상태에서만 시작.

## 읽어야 할 파일

- `daily_update.py` — incremental 일간 갱신 로직, target weights 가 어디서 계산되어 어디로 가는지
- `update_and_deploy.py` — full 모드에서 daily_update 와의 관계
- `src/backtest.py` `BacktestResult` 의 `portfolio_weights` dict 구조
- `outputs/baseline_v4/backtest_result.pkl` — 비교 대상이 될 backtest pkl
- `docs/UPDATE_AND_DEPLOY_FLOW.md` (step 2 에서 갱신된 cutover history)

## 작업

### 1. Live snapshot 저장 hook

`daily_update.py` 에 일간 실행 마지막 단계로 다음 동작 추가:

```python
def _persist_live_snapshot(target_weights: pd.Series, asof: pd.Timestamp,
                           out_dir: Path = Path("outputs/live_log")) -> Path:
    """Persist today's target weights as the 'live decision' snapshot.

    Idempotent: if a file for this asof already exists, do NOT overwrite —
    the first decision of the day is what matters; subsequent runs of
    daily_update.py within the same date are intra-day re-runs and must
    not modify the audit trail.

    Returns: path written (or existing if already present).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{asof.strftime('%Y-%m-%d')}.csv"
    if fname.exists():
        return fname
    df = target_weights.rename("target_weight").to_frame().reset_index()
    df.columns = ["ticker", "target_weight"]
    df.insert(0, "asof", asof.strftime("%Y-%m-%d"))
    df.to_csv(fname, index=False, encoding="utf-8")
    return fname
```

호출 위치: `daily_update.py` 가 target weights 를 산출해 `outputs/baseline_v4/portfolio_weights.csv` 등에 기록하는 직후. snapshot 은 별도 `outputs/live_log/<date>.csv` 로 저장 (한 파일 / 한 영업일).

### 2. Live-vs-backtest delta 계산 script

`scripts/compute_live_delta.py`:

```python
"""Compute L1 drift between live snapshots and the latest backtest's
portfolio_weights for the same dates.

Output: outputs/live_delta_log.csv with rows
    (asof, n_tickers, l1_drift, max_drift_ticker, max_drift_value, n_missing_tickers)

L1 drift definition:
    drift_t = sum_i |w_live[i] - w_bt[i]|   on union of tickers
    (missing tickers treated as 0 on the missing side)

Reads:
    outputs/live_log/*.csv  (one file per day)
    outputs/baseline_v4/backtest_result.pkl  (latest production backtest)

Append-only: existing rows in outputs/live_delta_log.csv are preserved;
new (asof) rows are appended for any date that has a live snapshot but
no log entry yet.
"""
```

힌트: `BacktestResult.portfolio_weights` dict 는 `{rebal_date: pd.Series}` 형태. `asof` 와 정확히 일치하는 rebal_date 가 없으면 "asof <= rebal_date" 중 가장 가까운 것을 비교 기준으로 삼고, `n_missing_tickers` 로 explicit 표시.

### 3. `update_and_deploy.bat` / `update_and_deploy.py` 에 monitor 호출 추가

full mode 마지막 단계 (dashboard build 후) 와 incremental 모드 모두에서 `scripts/compute_live_delta.py` 를 호출. 실패해도 deploy 자체는 중단되지 않도록 (try/except, log only).

### 4. drift threshold 알람

`outputs/live_delta_log.csv` 의 가장 최근 row 의 `l1_drift` 가 **0.10 (즉 10pp 총 weight drift)** 를 초과하면 stdout 에 명시적 경고 print:

```
[LIVE-DELTA WARN] asof=2026-MM-DD l1_drift=<val> exceeds 0.10 threshold. Investigate.
```

threshold 는 `docs/UPDATE_AND_DEPLOY_FLOW.md` 에 명문화.

### 5. 문서화

`docs/UPDATE_AND_DEPLOY_FLOW.md` 끝에 "Live drift monitor" 섹션 추가:

```markdown
## Live drift monitor (added 2026-05-XX)

매 daily_update / update_and_deploy 실행 마지막에 다음이 자동 수행된다:

1. `outputs/live_log/<date>.csv` 에 그 날의 target weights 가 한 번 저장
   (intra-day 재실행 시 덮어쓰지 않음).
2. `scripts/compute_live_delta.py` 가 `outputs/live_delta_log.csv` 를 갱신.
3. `l1_drift > 0.10` 일 경우 stdout 에 [LIVE-DELTA WARN] 출력.

`l1_drift` 가 큰 날은 (a) 코드 변경 (b) 데이터 정합성 문제 (c) walk-forward
재훈련 boundary 의 의도된 reshuffle 중 하나. 0.10 초과가 2영업일 연속 발생하면
backtest vs live divergence 를 root-cause 분석한다.
```

## Acceptance Criteria

```bash
# 1) Live snapshot helper exists and is called from daily_update.py
grep -q "_persist_live_snapshot" daily_update.py

# 2) Delta script exists
test -f scripts/compute_live_delta.py

# 3) Run daily_update.py once and check both artifacts appear
python daily_update.py
ls outputs/live_log/*.csv | head -1
test -s outputs/live_delta_log.csv

# 4) Re-run idempotency: second daily_update.py same day must not modify
#    today's live_log file
TODAY=$(date +%Y-%m-%d)
MTIME_BEFORE=$(stat -c %Y outputs/live_log/$TODAY.csv 2>/dev/null || stat -f %m outputs/live_log/$TODAY.csv)
python daily_update.py
MTIME_AFTER=$(stat -c %Y outputs/live_log/$TODAY.csv 2>/dev/null || stat -f %m outputs/live_log/$TODAY.csv)
test "$MTIME_BEFORE" = "$MTIME_AFTER"

# 5) Threshold warning prints when injected
python -c "
import pandas as pd
df = pd.DataFrame([{'asof':'2099-01-01','n_tickers':65,'l1_drift':0.5,'max_drift_ticker':'X','max_drift_value':0.1,'n_missing_tickers':0}])
df.to_csv('outputs/live_delta_log.csv', mode='a', index=False, header=False)
" && python scripts/compute_live_delta.py 2>&1 | grep -q "LIVE-DELTA WARN"

# 6) Docs updated
grep -q "Live drift monitor" docs/UPDATE_AND_DEPLOY_FLOW.md
```

## 검증 절차

1. AC 1–6 모두 PASS.
2. 1주일 (5영업일) 의 데이터를 모은 후 `outputs/live_delta_log.csv` 가 5 행 누적되는지를 확인 (이 step 의 실행 시점에는 충족 불가능. step 검증 후 후속 운영 항목으로 남긴다).
3. `phases/final-v1-promotion/index.json` step 4 status `completed`. summary 에 첫 row 의 `l1_drift` 값 + threshold 0.10 + 첫 snapshot 파일 경로 명시.

## 금지사항

- **`outputs/live_log/<date>.csv` 가 이미 존재하면 덮어쓰지 마라.** 이유: 이 파일은 "그 날 첫 결정" 의 audit trail 이다. intra-day 재실행으로 변경되면 drift 측정 자체가 의미를 잃는다.
- **drift threshold 위반 시 production 을 중단시키지 마라.** 이유: 첫 도입 단계에서는 monitoring only. 자동 중단은 false positive 가 production 가용성을 해칠 수 있다. 경고 출력 + 인간 검토가 올바른 경로.
- **`outputs/live_delta_log.csv` 를 매번 처음부터 다시 작성하지 마라.** 이유: append-only 의 audit trail 가치를 보존. 기존 행을 갱신하려면 `recompute_live_delta_history.py` 같은 별도 명시적 script 로만 가능해야 한다.
- 기존 테스트를 깨뜨리지 마라.
