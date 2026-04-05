# Selection Bias Analysis Report

Generated: 2026-04-06 00:43:10

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0000, Adjusted SR=0.37, MinTRL=20.0yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.365
- Number of trials (N): 1
- Expected max SR under null: 0.000
- sigma(SR): 0.0202
- Deflated SR: 18.122 (p-value: 0.0000)
- Skewness: 0.669, Kurtosis: 9.038
- Observations: 2424 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 20.0 years (5033 trading days)
- Available: 9.6 years (2424 trading days)
- Verdict: **INSUFFICIENT -- 데이터 부족**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 1
- Observed SR: 0.365
- Haircut: 0.000
- Adjusted SR: 0.365
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2016-12-19
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2016-12-19 ~ 2020-01-22): IR = 0.517 [PASS]
- Period 2 (2020-01-23 ~ 2023-02-27): IR = 0.517 [PASS]
- Period 3 (2023-02-28 ~ 2026-04-02): IR = 0.024 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
