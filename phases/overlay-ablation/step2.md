# Step 2: ablation-report

step1의 `outputs/ablation/summary.csv`를 가공해 사람이 판단할 수 있는 `docs/ABLATION_REPORT.md`를 작성한다. 핵심은 "어떤 overlay가 진짜 alpha이고 어떤 게 in-sample fit인가"의 결정표 산출이다.

## 읽어야 할 파일

- `outputs/ablation/summary.csv` (step1 산출)
- `outputs/ablation/ablation_*/backtest_result.pkl` (필요시 daily active return 추출용)
- `outputs/iter15_FINAL_postfix/backtest_result.pkl` (baseline)
- `docs/BASELINE.md` (gate criteria 정의)
- **이전 step 산출 확인**: `phases/overlay-ablation/index.json` step 0, 1 summary
- `docs/rollback_log.md` (탈락 overlay 기록 형식 참고)

## 작업

### 1. 통계적 신뢰구간 계산

summary.csv의 `delta_ir`만 보고 결정하면 노이즈를 신호로 오인할 수 있다. 각 ablation variant에 대해 다음을 계산:

a) **block bootstrap CI for ΔIR**
- baseline 일간 active return 시계열 `a_base`와 variant 일간 active return `a_var`를 동일 기간으로 정렬
- 일별 difference `d_t = a_base_t - a_var_t` 계산
- block bootstrap (block_size=10일, n_iter=1000)으로 `mean(d) / std(d) * sqrt(252)`의 분포 추정 → ΔIR 95% CI
- 영(zero) 포함 여부가 "통계적으로 0과 구분 가능한지"의 판정

b) **P2 비악화 검정**
- variant의 P2_ir이 baseline P2 IR 대비 -0.10 이하면 자동 탈락 (CLAUDE.md 기준선의 P2 IR 안정성 요구)

위 둘을 묶는 helper를 `scripts/ablation_bootstrap.py` (신규)에 두라:

```python
def block_bootstrap_delta_ir(
    base_active: pd.Series,  # baseline daily active return
    var_active: pd.Series,   # variant daily active return
    block_size: int = 10,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Return {'delta_ir_mean', 'delta_ir_lo95', 'delta_ir_hi95', 'p_value_two_sided'}."""
```

### 2. 결정 규칙 (gate)

각 overlay의 KEEP/DROP을 다음 규칙으로 판정:

| 조건 | 판정 |
|---|---|
| `delta_ir_hi95 < 0` AND `P2 비악화 만족` | **KEEP** (overlay 끄면 IR 유의 하락 → overlay가 alpha) |
| `delta_ir_lo95 > 0` | **DROP** (overlay 끄면 IR 유의 상승 → overlay가 손해) |
| CI가 0을 포함하면 (`lo95 < 0 < hi95`) | **DROP** (유의하지 않은 overlay는 자유도만 잡아먹음 — 제거 시 변동 없음을 신뢰구간이 입증) |
| `P2 악화` (delta_P2_ir < -0.10) | **DROP** (regime stability 우선) |

복수 조건이 충돌하면 **DROP을 우선**한다 (parsimony 원칙).

특별 케이스:
- `ablation_all_overlays_off`는 결정 대상이 아니라 **honest baseline** 참고용. 그 IR이 iter15_FINAL_postfix와 얼마나 다른지가 *모든 overlay 합산 효과*다. 보고서에 별도 박스로.
- `ablation_feature_mode_lean`은 overlay가 아닌 *feature panel* 차원이다. 별도 박스에 두고 KEEP/DROP은 따로 판정.

### 3. 작성: `docs/ABLATION_REPORT.md`

다음 섹션 구조로 작성하라:

```markdown
# Overlay Ablation Report

**Generated**: <YYYY-MM-DD>
**Baseline**: outputs/iter15_FINAL_postfix (IR=<X.XXX>, P1/P2/P3=<.../.../...>)
**Environment**: embargo_days=20, train_cutoff_date=2024-12-31, tuning_mode=research

## TL;DR
- KEEP: <list>
- DROP: <list>
- Aggregate IR if all dropped overlays were already off (counterfactual): <X.XXX> (from ablation_all_overlays_off)

## Methodology

- Each variant disables a single production overlay vs iter15_FINAL_postfix baseline.
- Block bootstrap (block_size=10, n_iter=1000, seed=42) on daily active return
  differences → 95% CI for ΔIR.
- Decision rule: <copy 결정 규칙 표>

## Per-overlay results

### value_trap_gate
| Metric | baseline | ablation_no_vtg | Δ | 95% CI |
|---|---|---|---|---|
| IR | … | … | … | […, …] |
| P1_ir | … | … | … | — |
| P2_ir | … | … | … | — |
| P3_ir | … | … | … | — |
| turnover | … | … | … | — |

**Verdict**: KEEP / DROP — <한 줄 사유 + CI 인용>

### growth_tilt
(동일 양식)

### pead_boost
(동일 양식)

### mega_cap_funding
(동일 양식)

### revision_cleaning (reversion_gated → down_only)
(동일 양식)

## Feature panel ablation (별도 카테고리)

### feature_mode (core → lean)
(동일 양식)

## Honest baseline (모든 overlay OFF)

| Metric | iter15_FINAL_postfix | ablation_all_overlays_off | Δ |
|---|---|---|---|
| IR | … | … | … |
| …

**Interpretation**: <overlay 6종이 합쳐 만들고 있는 IR 차이가 X.XXX. 이 중 DROP 판정된 overlay의 합산 기여도가 in-sample fit으로 의심되는 부분.>

## 결정 요약 (next step input)

production rebuild (step3)에 반영할 overlay 설정:
\`\`\`yaml
overrides:
  value_trap_gate_enabled: <true|false>
  growth_tilt_enabled: <true|false>
  pead_boost_enabled: <true|false>
  mega_cap_funding_mode: <true|false>
  revision_clean_mode: "<reversion_gated|down_only>"
  feature_mode: "<core|lean>"
\`\`\`
```

### 4. 탈락 overlay → `docs/rollback_log.md` 항목 추가 준비

각 DROP 판정에 대해 다음 항목을 rollback_log에 추가할 수 있도록 보고서 끝에 ready-to-paste 블록 둠 (실제 추가는 step3에서):

```markdown
## (pending step3) Rollback entries

### <date> — DROP <overlay_name>
- Reason: ablation under research mode (cutoff=2024-12-31) showed ΔIR <X.XXX> [<lo>, <hi>], not distinguishable from zero (or hi95 < 0 contradicted prior assumption).
- Original rationale: <CLAUDE.md에서 인용>
- Bootstrap: block_size=10, n_iter=1000.
```

## Acceptance Criteria

```bash
# 1. 스크립트 + 산출 존재
test -f scripts/ablation_bootstrap.py
test -f docs/ABLATION_REPORT.md

# 2. 보고서 핵심 섹션
grep -c "^## " docs/ABLATION_REPORT.md         # >= 5
grep -q "TL;DR" docs/ABLATION_REPORT.md
grep -q "Methodology" docs/ABLATION_REPORT.md
grep -q "Per-overlay results" docs/ABLATION_REPORT.md
grep -q "Honest baseline" docs/ABLATION_REPORT.md
grep -q "결정 요약" docs/ABLATION_REPORT.md

# 3. 각 overlay 결정이 KEEP/DROP 둘 중 하나로 명시됐는지
for kw in value_trap_gate growth_tilt pead_boost mega_cap_funding revision_cleaning feature_mode; do
    grep -q "$kw" docs/ABLATION_REPORT.md || { echo "MISSING overlay: $kw"; exit 1; }
done
grep -c -E "Verdict.*: (KEEP|DROP)" docs/ABLATION_REPORT.md   # >= 6

# 4. 부트스트랩 reproducibility
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('ab', 'scripts/ablation_bootstrap.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# minimal smoke: function exists and returns dict with expected keys
import pandas as pd, numpy as np
np.random.seed(0)
a = pd.Series(np.random.randn(500)*0.01, index=pd.date_range('2020-01-01', periods=500, freq='B'))
b = pd.Series(np.random.randn(500)*0.01, index=a.index)
r = m.block_bootstrap_delta_ir(a, b, n_iter=200)
assert {'delta_ir_mean','delta_ir_lo95','delta_ir_hi95'} <= set(r.keys())
"

# 5. 결정 요약 블록이 YAML로 parse 가능한지
python -c "
import re, yaml
body = open('docs/ABLATION_REPORT.md', encoding='utf-8').read()
m = re.search(r'overrides:.*?(?=\`\`\`)', body, re.DOTALL)
assert m, 'no overrides block'
yaml.safe_load(m.group(0))   # raises on invalid yaml
"
```

## 검증 절차

1. AC 통과.
2. 아키텍처 체크리스트:
   - 결정 요약 블록의 키가 모두 `src/config.py` `PipelineConfig`에 실존하는가?
   - CI의 양쪽 끝이 합리적 범위인가 (예: |ΔIR| < 1.0)? 비정상이면 block_size/n_iter 점검.
   - P2 IR이 단일 변수에 의해 ±0.5 이상 흔들리면 → bootstrap 노이즈가 아니라 진짜 신호이거나 코드 버그. 사람이 한 번 더 봐야 함 (manual review note 작성).
3. `phases/overlay-ablation/index.json` step 2 업데이트:
   - 성공 → `"status": "completed"`, `"summary": "docs/ABLATION_REPORT.md written. scripts/ablation_bootstrap.py for block-bootstrap ΔIR CI. Decisions — KEEP: <list>; DROP: <list>. Honest baseline (all overlays off) IR=<X.XXX> (Δ vs production -<Y.YYY>)."`
   - 실패/blocked → 사유

## 금지사항

- **결정 규칙에서 'P2 비악화'를 빼지 마라.** 이유: 단일 IR 최대화만 보면 P1/P3 강한 구간으로 쏠려 regime risk 누적. 통계적 유의성 + 안정성 두 조건 모두 필요.
- **단일 random seed로 부트스트랩을 끝내지 마라.** seed=42로 고정하되, `n_iter >= 1000` 보장. CI가 너무 좁으면 (예: 너비 < 0.05) seed 의심.
- **rollback_log.md에 이 step에서 직접 쓰지 마라.** 이유: step3의 production rebuild가 결정을 *집행*할 때 함께 기록한다 (rollback과 promotion은 묶음).
- **`outputs/ablation/summary.csv`를 수정하지 마라.** 이유: step1의 raw 산출. 가공은 보고서 안에서만.
- **결정 모호 시 KEEP을 디폴트로 하지 마라.** 이유: parsimony 원칙. 유의하지 않은 overlay는 자유도만 잡아먹는다.
