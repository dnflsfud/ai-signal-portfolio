# Step 1: ablation-runner

step0에서 작성한 7개 ablation manifest를 일괄 실행하고 결과를 단일 summary CSV로 모은다.

## 읽어야 할 파일

- `variants/ablation_*.yaml` (step0 산출 7개)
- `run_variant.py` (CLI 시그니처 + metrics.json 구조)
- `src/harness.py` `sub_period_irs` (sub-period IR 계산 helper)
- **이전 task 산출**: `outputs/iter15_FINAL_postfix/metrics.json` (비교 기준)
- `phases/overlay-ablation/index.json` step 0 summary 확인

## 작업

### `scripts/run_ablation.py` 작성

요구사항:
- 입력: `variants/ablation_*.yaml` glob (또는 명시 리스트 인자)
- 각 manifest를 `python run_variant.py --variant <path> --no-cache` subprocess로 순차 실행 (parallel 금지 — GIL 무관하지만 LightGBM/cvxpy가 CPU 포화)
- 각 실행 후 `<out_dir>/metrics.json` 파싱 → summary row 한 줄
- baseline (`outputs/iter15_FINAL_postfix/metrics.json`)도 첫 row로 포함
- 종합 결과를 `outputs/ablation/summary.csv`로 저장
- 진행 상황을 stdout에 1줄/variant로 보고 (예: `[3/7] ablation_no_pead → IR=1.18 (Δ=-0.06) ✓`)
- 1개 실패해도 나머지 진행, 실패 사유를 row의 `error` 컬럼에 기록
- 종료 코드: 모든 실행 성공 시 0, 부분 실패 시 1

### 인터페이스

```python
# scripts/run_ablation.py

def run_one(manifest_path: Path) -> dict:
    """Run a single variant, return summary row dict.

    On failure, returns dict with 'label', 'error', other fields NaN.
    """

def collect_metrics(metrics_path: Path) -> dict:
    """Load metrics.json and flatten to a single row dict.

    Returns keys:
      label, ir, active_return, tracking_error, sharpe, max_drawdown,
      annual_turnover_2way, avg_ic,
      P1_ir, P2_ir, P3_ir,
      delta_ir (filled later vs baseline)
    """

def delta_vs_baseline(rows: list[dict], baseline_label: str) -> list[dict]:
    """Append delta_* columns for each metric vs the row matching baseline_label."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics",
                        default="outputs/iter15_FINAL_postfix/metrics.json")
    parser.add_argument("--pattern", default="variants/ablation_*.yaml")
    parser.add_argument("--out", default="outputs/ablation/summary.csv")
    parser.add_argument("--skip-baseline-rerun", action="store_true",
                        help="Don't re-execute baseline; just read existing metrics.json")
    args = parser.parse_args()
    ...
```

핵심 동작:
1. baseline metrics.json이 없으면 (step A2 미수행) abort.
2. 각 ablation variant 실행 (subprocess) → metrics.json 로드 → row
3. baseline row를 맨 위에 추가
4. delta_vs_baseline 적용
5. CSV 저장

### subprocess 호출 패턴

```python
import subprocess, sys
cmd = [sys.executable, "run_variant.py", "--variant", str(manifest_path), "--no-cache"]
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
if proc.returncode != 0:
    return {"label": label, "error": proc.stderr[-500:], ...}
```

타임아웃 600초/variant (~10분). 전체 7개 + baseline 재실행 안 함 (= 7회) → 최대 ~70분.

### `--no-cache` 강제 이유

각 ablation은 `feature_mode` 또는 `revision_clean_mode` 등 Phase 1/2/4 차원을 건드린다. 캐시 재사용은 silent 결과 오염을 만든다 (Task C step0에서 fingerprint로 막을 예정이지만 그 전까지는 --no-cache 강제).

### CSV 컬럼 순서

```
label, error,
ir, delta_ir,
active_return, delta_active_return,
tracking_error, delta_tracking_error,
sharpe, delta_sharpe,
max_drawdown, delta_max_drawdown,
annual_turnover_2way, delta_annual_turnover_2way,
avg_ic, delta_avg_ic,
P1_ir, delta_P1_ir,
P2_ir, delta_P2_ir,
P3_ir, delta_P3_ir
```

소수점 4자리 고정.

## Acceptance Criteria

```bash
# 1. 스크립트 존재 + 임포트
test -f scripts/run_ablation.py
python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('ra', 'scripts/run_ablation.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert hasattr(m, 'run_one') and hasattr(m, 'collect_metrics') and hasattr(m, 'delta_vs_baseline')"

# 2. 실제 실행 (~ 60분 — 백그라운드로 띄우고 종료 확인 권장)
python scripts/run_ablation.py --skip-baseline-rerun

# 3. summary CSV 존재 + 형태 검증
test -f outputs/ablation/summary.csv
python -c "
import pandas as pd
df = pd.read_csv('outputs/ablation/summary.csv')
expected_labels = {'iter15_FINAL_postfix', 'ablation_no_vtg', 'ablation_no_growth_tilt', 'ablation_no_pead', 'ablation_no_mega_funding', 'ablation_revision_down_only', 'ablation_feature_mode_lean', 'ablation_all_overlays_off'}
got = set(df['label'].tolist())
missing = expected_labels - got
assert not missing, f'missing rows: {missing}'
assert 'delta_ir' in df.columns
assert df.loc[df['label'] == 'iter15_FINAL_postfix', 'delta_ir'].iloc[0] == 0.0
print(f'CSV OK — {len(df)} rows, {len(df.columns)} cols')
print(df[['label','ir','delta_ir','P2_ir','delta_P2_ir']].to_string(index=False))
"

# 4. 각 variant 산출물 존재
for v in no_vtg no_growth_tilt no_pead no_mega_funding revision_down_only feature_mode_lean all_overlays_off; do
    test -f "outputs/ablation/ablation_${v}/metrics.json" || { echo "MISSING: $v"; exit 1; }
done
```

## 검증 절차

1. AC 통과. 60-70분 소요 예상이므로 background 권장.
2. 아키텍처 체크리스트:
   - `experiment_inventory.json` `n_oos_peeks`가 증가하지 않았어야 (research 모드만 사용)
   - 각 variant의 `experiment_manifest.json`이 `embargo_days=20`, `enforce_oos_holdout=true`, `train_cutoff_date="2024-12-31"`를 일관되게 기록하는가?
3. `phases/overlay-ablation/index.json` step 1 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "scripts/run_ablation.py written. 7 ablation variants executed under research mode (~XX min). summary.csv: baseline IR=<X.XXX>; deltas — no_vtg=<>, no_growth_tilt=<>, no_pead=<>, no_mega_funding=<>, revision_down_only=<>, feature_mode_lean=<>, all_off=<>. K runs returned non-zero (see error column)."`
   - 실패/blocked → 사유

## 금지사항

- **변형들을 병렬 실행하지 마라** (multiprocessing.Pool 등). 이유: cvxpy/LightGBM이 단일 변형 안에서 이미 CPU를 포화. 병렬은 thrash → 노이즈가 측정 noise로 들어옴.
- **`--no-cache`를 빼지 마라.** 이유: feature_mode 변경 시 캐시는 오염됐다 (Task C step0 전까지 fingerprint 가드 없음).
- **실패한 variant를 silent skip하지 마라.** 이유: error 컬럼에 stderr 끝 500자 기록 필수. 추후 디버깅의 출발점.
- **baseline을 재실행하지 마라** (`--skip-baseline-rerun` 기본). 이유: Task A step2 산출물이 정확한 비교 기준. 재실행하면 timestamp drift + 동일 시드 재현성에 노이즈.
- **CSV에서 행을 정렬하지 마라** (baseline + variant 순). 이유: 비교 직관성. label 알파벳 정렬은 step2 보고서에서.
