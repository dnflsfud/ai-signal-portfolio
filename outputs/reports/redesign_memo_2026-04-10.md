# AI Signal Redesign Memo

Generated: 2026-04-10

## 1. Executive Summary

- Post-fix strategy CAGR is 27.71% vs EW benchmark 26.73%.
- Active CAGR is only about 1.1% and active IR is about 0.13.
- Average IC is 0.0052, below the usual 0.01 threshold for a durable cross-sectional signal.
- Selection Bias gate is `FAIL` even after using `N_trials = 396`.
- The strategy is not broken because of one single bug anymore. The remaining problem is structural:
  weak alpha is being translated into full active-share and near-cap turnover almost all the time.

## 2. What Failed In 2023-2026

### 2.1 Period 3 is the real problem

- Sub-period IR:
  - Period 1: 0.238
  - Period 2: 0.374
  - Period 3: -0.121
- Annual active returns:
  - 2023: -9.3%
  - 2024: -11.7%
  - 2025: +12.7%
  - 2026 YTD: +2.5%
- 2020 contributes most of the full-sample active payoff. That is not a healthy profile.

### 2.2 The strategy leaned into the wrong style mix

Average active tilts since 2023-02-28:

- Overweights:
  - `style_Cyclical`: +8.13%
  - `sector_Semiconductors`: +4.04%
  - `sector_Industrials`: +4.00%
  - `sector_Technology`: +1.93%
- Underweights:
  - `style_Quality Growth`: -3.30%
  - `style_Quality`: -2.89%
  - `style_Growth`: -2.60%
  - `sector_Consumer Disc.`: -2.40%
  - `sector_Real Estate`: -1.99%
  - `sector_Financials`: -1.89%

Interpretation:

- The model was not simply "long AI". It was long the cyclical / second-derivative expression of the theme.
- It systematically underweighted quality-growth compounders and parts of the mega-cap leadership complex.
- That is exactly the kind of positioning that can work during one phase of the cycle and then break badly when leadership narrows or rotates.

### 2.3 Ticker-level positioning confirms the style diagnosis

Average active weights in Period 3:

- Top OW:
  - `MU` +2.86%
  - `BE` +2.26%
  - `000660` +2.06%
  - `LITE` +1.80%
  - `ABBV` +1.68%
  - `VRT` +1.36%
  - `PLTR` +1.23%
- Top UW:
  - `EQIX` -1.61%
  - `AMZN` -1.36%
  - `ISRG` -1.13%
  - `LIN` -0.98%
  - `AMD` -0.85%
  - `META` -0.80%
  - `SPGI` -0.78%
  - `TSLA` -0.75%

Interpretation:

- The portfolio kept paying for cyclical semiconductor / industrial / hardware sensitivity.
- At the same time it underweighted several quality-growth or platform names that mattered in the AI-led regime.
- This is not just a bad month or two. It is a persistent portfolio construction bias.

### 2.4 The model's information content is narrow

Feature importance concentration:

- Accounting: 44.3% of total importance across 87 features
- Sellside: 24.7% across 79 features
- Price: 17.7% across 45 features
- Factor: 9.4% across 80 features
- Conditioning: 3.9% across 54 features

Top 30 features contain:

- Accounting: 16
- Sellside: 8
- Price: 5
- Conditioning: 1
- Factor: 0

Interpretation:

- The model is telling us very clearly that Factor and most Conditioning features are not pulling their weight.
- The feature set is too wide relative to the signal breadth.
- We are paying complexity cost without receiving out-of-sample alpha.

## 3. Feature Reduction Priority

### 3.1 First cuts: move these out of the main alpha model

1. Factor block
- 80 features, zero representation in the top 30.
- Recommendation: remove the whole Factor block from the main alpha model.
- Keep at most 3-5 regime descriptors outside the alpha model as risk-budget controls.

2. Most Conditioning features
- 54 features but only one truly high-value feature appears near the very top: `earn_cycle_pos`.
- Recommendation: cut calendar one-hots, most sector one-hots, and most broad regime flags from the alpha model.

### 3.2 Keep only the strongest Conditioning features

Keep list:

- `earn_cycle_pos`
- `size_rank`
- `regime_avg_vol_63d`
- `regime_avg_rec`
- `regime_mkt_ret_63d`
- optional:
  - `regime_rev_breadth_eps`
  - `regime_rev_breadth_sales`

Target size: 4-6 features.

### 3.3 Compress Accounting hard

Current issue:

- The model uses many variants of the same concept at multiple horizons.
- `oper_margin_*`, `eps_*`, `sales_*`, `capex_*`, valuation volatility variants are highly redundant.

Keep concepts, not ladders:

- Profitability trend:
  - `oper_margin_chg_63d`
  - `oper_margin_chg_252d`
  - `oper_margin_chg_vol`
- Cash efficiency:
  - `cash_conversion_z`
  - `capex_intensity_z`
  - `op_leverage_63d`
- Valuation / quality:
  - `best_pe_ratio_level_z`
  - `best_peg_ratio_level_z`
  - `best_px_bps_ratio_level_z`
  - `roe_pe_z`
- Fundamental stability:
  - `best_eps_chg_vol`
  - `best_sales_chg_vol`
  - `best_calculated_fcf_chg_vol`

Target size: 12-18 features.

### 3.4 Compress Sellside into persistence and breadth

Keep:

- `analyst_rec_level`
- `analyst_rec_rank`
- `analyst_rec_vs_median`
- `eps_rev_ma_63d`
- `sales_rev_ma_63d`
- `eps_rev_time_high`
- `sales_rev_time_low`
- `eps_rev_vol`
- `sales_rev_vol`
- `tg_upside_vol`
- `tg_conviction`

Drop:

- many duplicated bounded / adjusted / directional revision transforms that represent the same underlying revision persistence.

Target size: 10-12 features.

### 3.5 Price block should become a pure trend-risk sleeve

Keep:

- `ma_cross_50_200`
- `beta_63d`
- `realized_vol_126d`
- `downside_vol_63d`
- `dist_52w_low`
- `range_position_52w`
- `max_ret_63d`
- `min_ret_63d`

Target size: 8-10 features.

### 3.6 Suggested new total feature count

- Accounting: 14-18
- Sellside: 10-12
- Price: 8-10
- Conditioning: 4-6
- Factor in alpha model: 0-3

Target total: 36-49 features.

## 4. Turnover Reduction Design

### 4.1 The current portfolio is constraint-driven, not alpha-driven

Observed from outputs:

- Average active share: 24.68%
- Active share is near the 25% cap in 96.7% of rebalances.
- Average turnover per rebalance: 14.84%
- Turnover is near the 15% cap in 56.6% of rebalances.

Interpretation:

- The optimizer is almost always using all available risk budget.
- The alpha is too weak to justify that.
- This is the main reason realized active IR is poor even when gross signal is not completely random.

### 4.2 Immediate portfolio design changes

1. Dynamic active-share budget
- Do not always allow the portfolio to use the full `max_active_share`.
- Tie active-share budget to signal strength:
  - low forecast dispersion or low recent IC -> cut active-share limit
  - high forecast dispersion and stable IC -> allow more active risk

2. Trade buffer / no-trade band
- Only trade if proposed weight change exceeds a threshold.
- Example:
  - do not trade if `abs(target - current) < 0.5%`
  - or use threshold proportional to estimated trading cost / forecast confidence

3. Partial execution toward target
- Replace full jump to target with:
  - `w_exec = w_prev + eta * (w_target - w_prev)`
- Start with `eta = 0.25 ~ 0.50`.
- This alone can materially reduce turnover with weak loss in gross alpha.

4. Rank hysteresis
- Entry rule should be stricter than exit rule.
- Example:
  - enter only if stock is in top decile
  - hold until it drops below the 30th or 40th percentile

5. Confidence-weighted alpha shrinkage
- Send optimizer a shrunk forecast:
  - `alpha_used = raw_score * confidence_scale * regime_scale`
- If cross-sectional dispersion is narrow, shrink everyone toward zero.

6. Rebalance redesign
- Move from single-frequency "always optimize" to:
  - monthly core rebalance
  - optional weekly micro-overlay only for very large forecast changes

## 5. Structural Redesign Recommendation

### 5.1 Separate the model into sleeves

Current issue:

- Everything is thrown into one LightGBM.
- Weak factor / regime / calendar signals can still perturb the model and increase turnover.

Recommended architecture:

1. Core alpha sleeve
- Accounting + Sellside + a compact Price block
- This should drive stock selection

2. Regime sleeve
- Small set of macro / volatility / breadth descriptors
- This should not directly rank stocks
- It should scale risk budget, alpha shrinkage, and possibly style caps

3. Portfolio sleeve
- Converts desired alpha into target weights
- Applies dynamic active-share budget
- Applies no-trade band and partial execution

4. Execution sleeve
- Keeps realized weights separate from desired weights
- This is where turnover control should live

### 5.2 What to change in code structure

- `src/features/factor.py`
  - stop using this as a large alpha feature source
  - repurpose a minimal subset into a regime-state object

- `src/features/conditioning.py`
  - reduce to a compact structural context set
  - remove bulky calendar and one-hot blocks from the alpha model

- `src/model_trainer.py`
  - train on the compact core feature set
  - optionally expose forecast confidence:
    - forecast dispersion
    - rank stability across retrains
    - prediction magnitude after shrinkage

- `src/portfolio_optimizer.py`
  - add dynamic active-share cap
  - add optional dead-band logic
  - add partial-step execution toward target

- `src/backtest.py`
  - keep both:
    - desired target weights
    - executed weights
  - report how often active-share and turnover constraints are binding

### 5.3 The design target should change

Target design is no longer:

- "maximize raw in-sample alpha subject to constraints"

Target design should become:

- "maximize robust alpha conversion per unit of turnover and per unit of active risk"

That means the model should be judged on:

- recent-period IR
- adjusted SR after realistic trial count
- turnover-adjusted IR
- hit rate by regime
- constraint-binding frequency

## 6. Practical Next Build

Version B1:

- remove most Factor + Conditioning alpha features
- cut feature count to about 45
- keep current LightGBM but retrain on compact features only

Version B2:

- add dynamic active-share cap
- add 0.5%-1.0% no-trade band
- add partial target execution with `eta = 0.35`

Version B3:

- re-run only a very small experiment budget
- compare against current fixed architecture on:
  - full sample
  - 2023-2026 only
  - turnover-adjusted active IR

## 7. Bottom Line

- The strategy still contains signal, but not enough robust signal to support the current breadth, turnover, and full-risk-budget portfolio mapping.
- The right fix is not "more tuning".
- The right fix is:
  - smaller alpha model
  - regime as a risk gate, not as a large feature dump
  - execution-aware portfolio construction
  - stronger discipline on active-share usage
