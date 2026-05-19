# Selection Bias Analysis Report

Generated: 2026-05-02 21:22:24

## 1. Summary Verdict
- **FAIL** -- DSR p=0.4309, Adjusted SR=0.06, MinTRL=1.6yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.299
- Number of trials (N): 402
- Expected max SR under null: 1.237
- sigma(SR): 0.3572
- Deflated SR: 0.174 (p-value: 0.4309)
- Skewness: 0.352, Kurtosis: 6.505
- Observations: 1936 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.6 years (397 trading days)
- Available: 7.7 years (1936 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 402
- Observed SR: 1.299
- Haircut: 1.237
- Adjusted SR: 0.062
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-26
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-26 ~ 2021-05-14): IR = 1.534 [PASS]
- Period 2 (2021-05-17 ~ 2023-11-03): IR = 0.232 [PASS]
- Period 3 (2023-11-06 ~ 2026-04-23): IR = 1.907 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
