# Selection Bias Analysis Report

Generated: 2026-04-04 23:46:49

## 1. Summary Verdict
- **PASS** -- DSR p=0.0000, Adjusted SR=0.63, MinTRL=6.4yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.632
- Number of trials (N): 1
- Expected max SR under null: 0.000
- sigma(SR): 0.0199
- Deflated SR: 31.810 (p-value: 0.0000)
- Skewness: 1.375, Kurtosis: 15.699
- Observations: 2408 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 6.4 years (1623 trading days)
- Available: 9.6 years (2408 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 1
- Observed SR: 0.632
- Haircut: 0.000
- Adjusted SR: 0.632
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2016-12-19
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2016-12-19 ~ 2020-01-14): IR = 0.625 [PASS]
- Period 2 (2020-01-15 ~ 2023-02-09): IR = 0.380 [PASS]
- Period 3 (2023-02-10 ~ 2026-03-11): IR = 0.907 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
