# Feature Catalog (core mode, 2026-04-13)

This document explains every feature currently active in the production `core` mode — 56 features from the `CORE_FEATURE_WHITELIST` in `src/features/assembly.py`, plus the universe-level preprocessing that feeds them.

The pipeline walks these phases:

```
UniverseData → accounting/price/sellside/factor/conditioning raw features
            → lean momentum composites + growth composites + financials block
            → apply_core_filter (keep only CORE_FEATURE_WHITELIST)
            → CS z-score (except conditioning/factor broadcasts)
            → fillna with per-feature cross-sectional median
            → 56 features × tickers × dates panel → LightGBM
```

Every feature ends the pipeline as a `(dates × tickers)` DataFrame, aligned to the same universe and business-day calendar. Rolling windows use trading days, not calendar days.

---

## 0. Shared helpers (`src/features/utils.py`)

These appear on almost every line below — read them once before the catalog.

| Helper | Formula | Purpose |
|---|---|---|
| `cross_sectional_zscore(df)` | `(df - df.mean(axis=1)) / df.std(axis=1)` | Standardize each day across the ~60 tickers. "How extreme is this ticker **today**". |
| `cs_rank(df)` | `df.rank(axis=1, pct=True)` | Percentile rank in [0, 1]; outlier-robust cousin of z-score. |
| `safe_pct_change(df, n)` | `(df - df.shift(n)) / abs(df.shift(n))` | `pct_change` that uses `abs(denominator)` to avoid sign flips when the base is negative (FCF, ROE). |
| `rolling_tsz(df, w=756, mp=252)` | `(df - rolling_mean(w)) / rolling_std(w)` | **Per-ticker** rolling 3Y time-series z-score. Each column standardized against its OWN history, not today's cross-section. Used for valuation features (see §4). |
| `clip_outliers(df, n=5)` | `df.clip(-n, n)` | Post-z-score winsorization, applied after CS z-score. |

---

## 1. Accounting / Quality block (12 features)

Source sheets: `BEST_EPS`, `BEST_SALES`, `BEST_GROSS_MARGIN`, `OPER_MARGIN`, `BEST_CALCULATED_FCF`, `BEST_CAPEX`, `BEST_ROE`. Built in `src/features/accounting.py` via `build_accounting_features` and `_add_cross_ratios`.

### Margin / trend features

| Feature | Formula | What it captures |
|---|---|---|
| `oper_margin_chg_63d` | `pct_change(oper_margin, 63)` | 3-month operating-margin trend. |
| `oper_margin_chg_252d` | `pct_change(oper_margin, 252)` | 1-year operating-margin trend. |
| `oper_margin_accel` | `chg_21d − chg_63d` | Margin acceleration (short vs medium trend). |
| `best_gross_margin_chg_63d` | `pct_change(gm, 63)` | 3-month gross-margin trend. |
| `best_gross_margin_chg_252d` | `pct_change(gm, 252)` | 1-year gross-margin trend. |
| `op_leverage_63d` | `pct_change(oper_margin, 63) − pct_change(gm, 63)` | Positive → operating leverage (OM growing faster than GM). |
| `earnings_quality_252d` | `pct_change(eps, 252) − pct_change(sales, 252)` | EPS growth > sales growth ⇒ margin expansion / buybacks. |

### Level & composite features

| Feature | Formula | What it captures |
|---|---|---|
| `best_roe_level_z` | `cross_sectional_zscore(BEST_ROE)` | Cross-section ROE level z-score (raw level, no self-history normalization — ROE is naturally cross-sectional). |
| `best_calculated_fcf_level_z` | `cross_sectional_zscore(FCF)` | Cross-section FCF level. |
| `best_capex_level_z` | `cross_sectional_zscore(CAPEX)` | Cross-section CAPEX level. |
| `cash_conversion_z` | `cross_sectional_zscore(FCF / abs(EPS))` | FCF-to-earnings ratio, z-scored. High ⇒ quality of reported earnings. |
| `capex_intensity_z` | `cross_sectional_zscore(CAPEX / abs(SALES))` | Capital intensity (asset heavy vs light). |

**Why these exist**: Quality has historically been the strongest single-style block in SHAP importance. 12 features cover the three quality pillars — margin stability/growth (6), profitability level (3), capital efficiency (3).

---

## 2. Growth block (3 features)

Source sheets: `BEST_EPS`, `BEST_SALES`. Built in `accounting.py`.

| Feature | Formula | Purpose |
|---|---|---|
| `best_sales_chg_252d` | `pct_change(BEST_SALES, 252)` | 1-year sales growth. |
| `best_sales_accel` | `chg_21d − chg_63d` | Sales growth acceleration. |
| `best_eps_chg_252d` | `pct_change(BEST_EPS, 252)` | 1-year EPS growth. |

**Why only 3**: Growth is intentionally a smaller block than Quality (6 → 3 after REDESIGN H pruning). EPS/Sales growth are the only true growth signals; short-horizon (21/63d) growth is noisy and was dropped.

---

## 3. Value block — **REDESIGN M (2026-04-13)** (3 features)

Source sheets: `BEST_PEG_RATIO`, `BEST_EV_TO_BEST_EBITDA`, `BEST_PX_BPS_RATIO`. Built in `accounting.py` via the `VALUATION_SHEETS` loop.

**Normalization change** (critical): each valuation level metric now goes through **per-ticker rolling 3-year time-series z-score first**, then cross-sectional z-score:

```python
tsz = rolling_tsz(raw, window=756, min_periods=252)   # per-ticker history
feature = cross_sectional_zscore(tsz)                  # cross-section on TSZ values
```

Why: raw valuation cross-section treats every ticker as if drawn from the same distribution, which penalizes structurally high-multiple names (NVDA, LLY, COST) even when they're cheap vs their own 3Y norm. The TSZ step answers "how expensive is ticker *relative to its own history*", and the cross-section step ranks those history-relative scores.

| Feature | Formula | Interpretation |
|---|---|---|
| `best_peg_ratio_level_z` | `cs_zscore(rolling_tsz(PEG, 756))` | + ⇒ PEG above ticker's own 3Y norm vs peers. High ⇒ "historically expensive for growth". |
| `best_ev_to_best_ebitda_level_z` | `cs_zscore(rolling_tsz(EV/EBITDA, 756))` | + ⇒ EV/EBITDA expensive vs own history. |
| `best_px_bps_ratio_level_z` | `cs_zscore(rolling_tsz(PBR, 756))` | + ⇒ P/B expensive vs own history. |

(Forward `BEST_PE_RATIO` is intentionally NOT in this block — it enters via the Financials block's `fin_pe_level_z` instead, where the TSZ treatment + upper clip also apply.)

**Why only 3 level-z features**: REDESIGN H dropped the `_chg_21d / _chg_63d / _accel / _vs_median / _vol / _rank` variants of every valuation sheet. Importance analysis found they were < 0.3% each.

---

## 4. Momentum block (7 features)

Source: `Daily_Returns`, `PX_LAST`. Built in `src/features/price.py` + lean composites in `assembly.py`.

| Feature | Formula | Purpose |
|---|---|---|
| `momentum_252d` | `rolling_sum(returns, 252)` | 1-year price momentum. |
| `risk_adj_mom_252d` | `momentum_252d / rolling_std(returns, 252)` | Sharpe-style momentum. |
| `ma_cross_21_50` | `(MA21 / MA50) − 1` | Short-term trend vs medium-term. |
| `ma_cross_50_200` | `(MA50 / MA200) − 1` | Medium vs long-term trend (golden/death cross). |
| `max_ret_63d` | `rolling_max(returns, 63)` | Largest single-day gain in last 3 months (lottery / upside skew). |
| `min_ret_63d` | `rolling_min(returns, 63)` | Largest single-day loss (tail risk). |
| `mom_accel_63_252` | `cs_rank(mom_63) − cs_rank(mom_252)` | Short-vs-long momentum spread rank. |

---

## 5. Price / Risk block (5 features)

Source: `Daily_Returns`. Built in `price.py`.

| Feature | Formula | Purpose |
|---|---|---|
| `beta_63d` | Rolling OLS β to equal-weight market (63d) | Systematic risk. |
| `idio_vol_63d` | `std(return − β·market)` × √252 | Stock-specific volatility. |
| `realized_vol_21d` | `std(returns, 21) × √252` | Short-term total volatility. |
| `realized_vol_126d` | `std(returns, 126) × √252` | Medium-term volatility. |
| `dist_52w_high` | `(price / rolling_max(252)) − 1` | Drawdown from 52-week high (negative value; closer to 0 = near high). |

---

## 6. Sellside block (8 features)

Source sheets: `EQY_REC_CONS` (analyst recommendation 1-5), `Factset_TG_Price`, `Factset_EPS_Revision`, `Factset_Sales_Revision`. Built in `src/features/sellside.py`.

Important: revision sheets are pre-cleaned by `clean_revision_spikes()` to remove false "drops" caused by consensus period rollovers around earnings dates (the cleaning uses `data.earnings_timeline` when available).

| Feature | Formula | Purpose |
|---|---|---|
| `analyst_rec_level` | Raw `EQY_REC_CONS` (1-5, 5 = Strong Buy) | Current analyst consensus level. |
| `analyst_rec_stability` | `rolling_std(rec, 63)` | Consensus stability; low ⇒ agreement. |
| `tg_mom_63d` | `pct_change(target_price, 63)` | Target-price momentum. |
| `tg_upside` | `(target_price / px_last) − 1` | Implied upside to consensus target. |
| `eps_rev_ma_63d` | `rolling_mean(cleaned_eps_rev, 63)` | Average EPS revision signal over 3 months. |
| `eps_rev_trend` | `rolling_mean(rev, 21) − rolling_mean(rev, 63)` | Short vs medium revision trend. |
| `eps_rev` | Cleaned `Factset_EPS_Revision` level | Raw bounded [-100, 100] revision score. |
| `sales_rev_ma_63d` | `rolling_mean(cleaned_sales_rev, 63)` | Sales-revision 3-month moving average. |

---

## 7. Macro / Factor block (5 features)

Source: `Factor_PX_LAST`, `Factor_Returns` sheets (SPX, NDX, UST yields, F_Quality/F_Growth/F_Value ETFs, etc.). Built in `src/features/factor.py`. **These features broadcast one macro time series to every ticker**, so they serve the model as regime variables (conditioning), not stock-specific alpha.

| Feature | Formula | Purpose |
|---|---|---|
| `fac_yield_slope` | `UST_10Y − UST_2Y` (broadcast) | Yield curve slope; + = steep, − = inverted. |
| `fac_F_Quality_mom_63d` | `rolling_sum(F_Quality_etf_returns, 63)` | Quality-factor 3-month momentum. |
| `fac_F_Growth_mom_63d` | `rolling_sum(F_Growth_etf_returns, 63)` | Growth-factor momentum. |
| `fac_F_Value_mom_63d` | `rolling_sum(F_Value_etf_returns, 63)` | Value-factor momentum. |
| `fac_value_growth_63d` | `sum(F_Value, 63) − sum(F_Growth, 63)` | Value-vs-growth spread; + = value outperforming. |

Broadcasting trick: each feature is `(dates × 1)` then tiled across tickers, so LightGBM can use them in tree splits alongside stock-specific variables (e.g. "when yield curve inverted AND momentum high AND high ROE ⇒ OW").

---

## 8. Regime / Conditioning block (3 features)

Built in `src/features/conditioning.py`. Like factor features, these are broadcast across tickers.

| Feature | Formula | Purpose |
|---|---|---|
| `earn_cycle_pos` | `days_since_earnings / (days_since + days_to_next)` | Position in earnings cycle, 0 = day after report, 1 = day before next. |
| `regime_mkt_ret_21d` | `rolling_sum(EW_market_return, 21)` broadcast | 1-month market return (regime proxy). |
| `cal_is_Q1` | Binary {0,1} broadcast | January-March indicator; captures calendar effects. |

---

## 9. Financials block — **REDESIGN K + M** (11 features)

Source sheets: `BEST_ROE`, `BEST_PX_BPS_RATIO`, `BEST_PE_RATIO`, `BEST_EPS`, `BEST_SALES`. Built in `src/features/assembly.py:build_financials_features`.

This block was ported from codex_v2 to give the model bank-specific drivers (ROE, P/B, P/E gaps) for banks that have no FCF/CAPEX/EBITDA. It's computed for ALL tickers so the model can learn "bank-like" patterns wherever they appear.

**REDESIGN M (2026-04-13)**: `fin_pe_level_z` and `fin_pb_level_z` now also use per-ticker rolling 3Y TSZ before cross-section (same as §3). `fin_pe_level_z` retains its upper-tail clip at z=1.5 applied AFTER both normalization steps.

| Feature | Formula | Purpose |
|---|---|---|
| `fin_roe_level_z` | `cross_sectional_zscore(ROE)` | Quality: current ROE level rank. |
| `fin_roe_chg_63d` | `pct_change(ROE, 63)` | ROE 3-month change. |
| `fin_roe_chg_252d` | `pct_change(ROE, 252)` | ROE 1-year change. |
| `fin_pb_level_z` | `cs_zscore(rolling_tsz(P/B, 756))` | P/B vs own 3Y norm, cross-sectionally ranked. |
| `fin_pb_chg_63d` | `pct_change(P/B, 63)` | P/B 3-month change. |
| `fin_pe_level_z` | `cs_zscore(rolling_tsz(P/E, 756)).clip(upper=1.5)` | P/E vs own 3Y norm; upper clip prevents extreme-PE mega-caps from dominating. |
| `fin_pe_chg_63d` | `pct_change(P/E, 63)` | P/E 3-month re-rating / de-rating. |
| `fin_eps_chg_63d` | `pct_change(EPS, 63)` | EPS 3-month trend (Financials block copy). |
| `fin_sales_chg_63d` | `pct_change(SALES, 63)` | Sales 3-month trend. |
| `fin_roe_pb_gap` | `cs_rank(ROE) − cs_rank(P/B)` | Quality-at-discount: high ROE + low P/B ⇒ undervalued quality bank. |
| `fin_roe_pe_gap` | `cs_rank(ROE) − cs_rank(P/E)` | Same as above for P/E (quality at discount). |

**Why the gap features matter**: a stock where `cs_rank(ROE) ≈ top` AND `cs_rank(valuation) ≈ bottom` gets a high gap score — this is the "Buffett zone". NVDA/LLY tend to have matching high ranks for both ROE and valuation, so the gap ≈ 0 (no special signal). Banks with high ROE but low P/B light up the gap.

---

## 10. Post-processing (all features)

After category blocks assemble, `build_all_features()` does:

1. **Core filter** (`apply_core_filter`): keep only features in `CORE_FEATURE_WHITELIST` (56 entries). Dropped features are silently ignored if missing from the panel.
2. **Cross-sectional z-score**: every accounting / price / sellside / financials feature gets `cross_sectional_zscore` applied (skipping conditioning and factor broadcasts which are intentionally cross-section invariant).
3. **NaN fill**: per-feature cross-sectional median for any remaining NaN cells, with `0.0` as ultimate fallback.
4. **Panel shape**: `(dates × tickers × features)` = `(~3200 × 60 × 56)` floats, ~43 MB.

The resulting panel feeds the walk-forward LightGBM trainer (`model_trainer.py`), where each retrain window produces per-feature EWMA importance for the next cycle's feature gating.

---

## Feature count by block (core mode)

| Block | Count | Share |
|---|---|---|
| Accounting / Quality | 12 | 21% |
| Growth | 3 | 5% |
| Value (TSZ-normalized) | 3 | 5% |
| Momentum | 7 | 13% |
| Price / Risk | 5 | 9% |
| Sellside | 8 | 14% |
| Macro / Factor | 5 | 9% |
| Regime / Conditioning | 3 | 5% |
| Financials (TSZ-normalized on PE/PB) | 11 | 20% |
| **Total** | **57** | **100%** |

(Actual runtime count may show ~56 if one whitelist entry is missing from the source panel; warnings print at core-filter time.)

---

## Changelog

- **REDESIGN M (2026-04-13)**: valuation level features (`best_peg_ratio_level_z`, `best_ev_to_best_ebitda_level_z`, `best_px_bps_ratio_level_z`, `fin_pe_level_z`, `fin_pb_level_z`) switched from pure cross-sectional z-score on raw levels to per-ticker rolling 3Y TSZ → cross-section. `rolling_tsz` helper added to `src/features/utils.py`. `fin_pe_level_z` upper clip at z=1.5 retained.
- **REDESIGN L (2026-04-13)**: `fin_pe_level_z` upper-tail clipped at z=1.5 (later supplemented by TSZ in REDESIGN M). `peg_growth_spread` removed from growth composites — dead in core mode anyway, `best_peg_ratio_level_z` is the active PEG feature.
- **REDESIGN K (2026-04-12)**: Financials block ported from codex_v2 (11 features).
- **REDESIGN H (2026-04-12)**: feature count trimmed from 81 to 46 based on feature_importance ranking. Removed valuation `_chg_21d / _chg_63d / _accel / _vs_median / _vol / _rank` variants.
- **REDESIGN C++ (2026-04-11)**: core whitelist introduced (~85 target, eventually pruned to 46/56/57 depending on Financials block status).
