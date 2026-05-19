# iter15_FINAL_postfix vs iter15_65tkr_reb21_vtg

**Generated**: 2026-05-19
**Postfix predict window end**: 2024-12-31
**Legacy predict window end (full)**: 2026-05-15
**Cutoff used for fair compare**: 2024-12-31

## Methodology delta
- legacy: no embargo (label leak), no cutoff (full sample through 2026-04)
- postfix: `embargo_days=20`, `train_cutoff_date=2024-12-31`

## Headline metrics (cutoff-trimmed window, fair comparison)

| Metric              | legacy iter15 (trimmed) | postfix | Δ |
|---------------------|-------------------------|---------|---|
| First / Last date   | 2018-11-26 → 2024-12-31 | 2018-11-26 → 2024-12-31 | — |
| N days              | 1592 | 1592 | +0 |
| Annual Return       | 26.05% | 24.58% | -1.48p |
| Active Return       | 2.73% | 1.26% | -1.48p |
| Annual Vol          | 22.67% | 22.47% | -0.20p |
| Tracking Error      | 2.93% | 2.88% | -0.05p |
| Sharpe              | 1.149 | 1.094 | -0.055 |
| **Information Ratio** | **0.804** | **0.392** | **-0.412** |
| Max Drawdown        | -30.34% | -29.95% | +0.39p |
| Annual Turnover 2w  | 113.6% | 90.8% | -22.8p |
| Avg IC              | 0.0450 | 0.0463 | +0.0013 |

## Sub-period IR

| Window | legacy (same window) | postfix | Δ |
|--------|----------------------|---------|---|
| P1 | 0.844 | 1.287 | +0.444 |
| P2 | 0.783 | -0.497 | -1.280 |
| P3 (cutoff-trimmed) | 0.780 | 0.390 | -0.389 |

## Full-window context (NOT used as gate — drift after cutoff included)

| Metric | legacy iter15 (full) | postfix (full, drift after cutoff) |
|--------|----------------------|------------------------------------|
| First → Last date | 2018-11-26 → 2026-05-15 | 2018-11-26 → 2026-05-15 |
| IR | 1.294 | 1.017 |
| Annual Return | 29.96% | 29.12% |

> Postfix full-window includes ~360 days of pure drift (no new predictions) after 2024-12-31. Not a fair alpha measure.

## 해석

- **ΔIR (cutoff-trimmed)**: -0.412. Negative = leakage premium that the embargo removed.
- **P2 collapse**: 가장 큰 충격은 P2 (rate-hike regime). legacy(trimmed)=0.783 → postfix=-0.497, Δ=-1.280. 라벨 누수가 P2 IR을 인위적으로 부양하고 있었음을 시사.
- **P1/P3**는 상대적으로 견고 (둘 다 positive). 본질적 alpha는 살아 있음.
- **누수 프리미엄 추정치**: cutoff-trimmed window 기준 ΔIR=-0.412. 이 차이의 상당 부분이 early_stopping이 forward 20일 라벨을 통해 미래를 본 효과.

## Next steps

- Task A step 3: BASELINE.md/CLAUDE.md/ROADMAP.md 갱신 (postfix가 new canonical)
- Task B: 이 postfix 위에서 in-sample fit 의심 overlay 6종 ablation
- 우선순위: P2 negative IR 원인을 ablation으로 분리 (특히 value_trap_gate, growth_tilt)