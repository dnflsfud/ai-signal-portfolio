# Selection Bias Analysis Report

Generated: 2026-04-10 16:08:19

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0000, Adjusted SR=0.10, MinTRL=88.6yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.174
- Number of trials (N): 396
- Expected max SR under null: 0.070
- sigma(SR): 0.0203
- Deflated SR: 5.143 (p-value: 0.0000)
- Skewness: 0.493, Kurtosis: 8.209
- Observations: 2424 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 88.6 years (22335 trading days)
- Available: 9.6 years (2424 trading days)
- Verdict: **INSUFFICIENT -- 데이터 부족**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 396
- Observed SR: 0.174
- Haircut: 0.070
- Adjusted SR: 0.104
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2016-12-19
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2016-12-19 ~ 2020-01-22): IR = 0.238 [PASS]
- Period 2 (2020-01-23 ~ 2023-02-27): IR = 0.374 [PASS]
- Period 3 (2023-02-28 ~ 2026-04-02): IR = -0.121 [FAIL]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
