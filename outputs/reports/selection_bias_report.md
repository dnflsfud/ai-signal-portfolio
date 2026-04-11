# Selection Bias Analysis Report

Generated: 2026-04-11 23:46:08

## 1. Summary Verdict
- **FAIL** -- DSR p=0.0000, Adjusted SR=0.28, MinTRL=20.3yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.359
- Number of trials (N): 400
- Expected max SR under null: 0.078
- sigma(SR): 0.0224
- Deflated SR: 12.552 (p-value: 0.0000)
- Skewness: 1.570, Kurtosis: 24.083
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 20.3 years (5106 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **INSUFFICIENT -- 데이터 부족**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 400
- Observed SR: 0.359
- Haircut: 0.078
- Adjusted SR: 0.282
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.010 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = -0.266 [FAIL]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 1.098 [PASS]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
