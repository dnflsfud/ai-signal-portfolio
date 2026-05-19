# Revision Spike Asymmetry Audit

_Generated for threshold=15.0, extreme=50.0, reversion=0.5._

This diagnostic quantifies whether the baseline `down_only` revision
cleaner (src/features/sellside.py) misses symmetric UP-side Factset
rollover artifacts. See docs/ROADMAP.md § Phase 2.4.

---

## Factset_EPS_Revision

- Dates: 2014-01-27 → 2026-04-16  (3189 rows, 60 tickers, 191,340 non-NaN cells)
- Params: threshold=15.0, extreme_threshold=50.0, reversion_ratio=0.5

### 1. Raw single-day spikes (`|diff| > threshold`)

| Direction | Count | % of non-NaN cells |
|---|---|---|
| DOWN (current cleaner) | 2,693 | 1.407% |
| UP   (currently IGNORED) | 2,697 | 1.410% |

**Asymmetry ratio (UP / DOWN) = 1.001**. If ≥ 0.5, up-side artifact is material.

### 2. Reversion-gated spikes (`|prev|>extreme & today collapses`)

| Direction | Count | % of non-NaN cells |
|---|---|---|
| rollover DOWN (high→neutral) | 795 | 0.415% |
| rollover UP   (low→neutral) | 388 | 0.203% |

### 3. Earnings proximity (±5 trading days)

| Mask | Total | Near earnings | % near |
|---|---|---|---|
| down_simple | 2,693 | 832 | 30.9% |
| up_simple | 2,697 | 1,724 | 63.9% |
| rollover_down | 795 | 45 | 5.7% |
| rollover_up | 388 | 91 | 23.5% |

High % near-earnings for rollover_up confirms these are Factset window artifacts rather than real upgrades.

### 4. Top tickers by rollover-UP count (what the new mode would catch)

| Ticker | Rollover-UP count |
|---|---|
| EQIX | 16 |
| DE | 15 |
| AMD | 15 |
| AMZN | 14 |
| ISRG | 13 |
| LITE | 13 |
| TER | 12 |
| GLW | 12 |
| MU | 10 |
| GS | 10 |
| TSLA | 10 |
| GOOGL | 10 |
| AMAT | 10 |
| MPC | 10 |
| WMT | 9 |

### 5. Downstream impact on 63d MA (crude bound)

If rollover_up cells were ffilled, the 63d moving average would shift by:
- mean |ΔMA|: **0.129** points
- max  |ΔMA|: **3.11** points
- cells with |ΔMA|>1.0: **6.1%**

> Note: this is a crude magnitude bound. What actually feeds the model is the CROSS-SECTIONAL z-score of these values, so a 1-2 point mean shift can still translate to a meaningful rank change. Rank-based sensitivity needs a separate experiment (compare CS rank before/after).


---

## Factset_Sales_Revision

- Dates: 2014-01-27 → 2026-04-16  (3189 rows, 60 tickers, 191,340 non-NaN cells)
- Params: threshold=15.0, extreme_threshold=50.0, reversion_ratio=0.5

### 1. Raw single-day spikes (`|diff| > threshold`)

| Direction | Count | % of non-NaN cells |
|---|---|---|
| DOWN (current cleaner) | 2,332 | 1.219% |
| UP   (currently IGNORED) | 2,358 | 1.232% |

**Asymmetry ratio (UP / DOWN) = 1.011**. If ≥ 0.5, up-side artifact is material.

### 2. Reversion-gated spikes (`|prev|>extreme & today collapses`)

| Direction | Count | % of non-NaN cells |
|---|---|---|
| rollover DOWN (high→neutral) | 635 | 0.332% |
| rollover UP   (low→neutral) | 318 | 0.166% |

### 3. Earnings proximity (±5 trading days)

| Mask | Total | Near earnings | % near |
|---|---|---|---|
| down_simple | 2,332 | 753 | 32.3% |
| up_simple | 2,358 | 1,439 | 61.0% |
| rollover_down | 635 | 44 | 6.9% |
| rollover_up | 318 | 53 | 16.7% |

High % near-earnings for rollover_up confirms these are Factset window artifacts rather than real upgrades.

### 4. Top tickers by rollover-UP count (what the new mode would catch)

| Ticker | Rollover-UP count |
|---|---|
| DE | 15 |
| LITE | 13 |
| TER | 13 |
| MU | 12 |
| AMD | 12 |
| GLW | 10 |
| AMZN | 10 |
| FN | 9 |
| BLK | 9 |
| AAPL | 9 |
| HON | 9 |
| LRCX | 8 |
| CAT | 7 |
| MA | 7 |
| SPGI | 7 |

### 5. Downstream impact on 63d MA (crude bound)

If rollover_up cells were ffilled, the 63d moving average would shift by:
- mean |ΔMA|: **0.104** points
- max  |ΔMA|: **3.17** points
- cells with |ΔMA|>1.0: **4.6%**

> Note: this is a crude magnitude bound. What actually feeds the model is the CROSS-SECTIONAL z-score of these values, so a 1-2 point mean shift can still translate to a meaningful rank change. Rank-based sensitivity needs a separate experiment (compare CS rank before/after).
