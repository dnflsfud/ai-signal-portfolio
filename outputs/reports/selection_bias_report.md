# Selection Bias Analysis Report

Generated: 2026-04-11 23:32:47

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0000, Adjusted SR=0.42, MinTRL=10.3yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.502
- Number of trials (N): 399
- Expected max SR under null: 0.077
- sigma(SR): 0.0223
- Deflated SR: 19.007 (p-value: 0.0000)
- Skewness: 1.497, Kurtosis: 19.573
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 10.3 years (2594 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **INSUFFICIENT -- 데이터 부족**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 399
- Observed SR: 0.502
- Haircut: 0.077
- Adjusted SR: 0.424
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.681 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = -0.488 [FAIL]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 1.034 [PASS]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
