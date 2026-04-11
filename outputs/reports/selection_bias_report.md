# Selection Bias Analysis Report

Generated: 2026-04-11 22:58:41

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0000, Adjusted SR=0.19, MinTRL=38.2yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.266
- Number of trials (N): 398
- Expected max SR under null: 0.079
- sigma(SR): 0.0228
- Deflated SR: 8.208 (p-value: 0.0000)
- Skewness: 0.358, Kurtosis: 8.950
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 38.2 years (9616 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **INSUFFICIENT -- 데이터 부족**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 398
- Observed SR: 0.266
- Haircut: 0.079
- Adjusted SR: 0.187
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.170 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = -0.534 [FAIL]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 0.963 [PASS]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
