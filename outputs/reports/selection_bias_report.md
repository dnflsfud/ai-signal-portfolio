# Selection Bias Analysis Report

Generated: 2026-04-12 23:41:58

## 1. Summary Verdict
- **WARN** -- DSR p=0.0000, Adjusted SR=0.67, MinTRL=4.7yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.751
- Number of trials (N): 402
- Expected max SR under null: 0.078
- sigma(SR): 0.0226
- Deflated SR: 29.820 (p-value: 0.0000)
- Skewness: 0.600, Kurtosis: 10.139
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 4.7 years (1183 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 402
- Observed SR: 0.751
- Haircut: 0.078
- Adjusted SR: 0.673
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.734 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = -0.130 [FAIL]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 1.533 [PASS]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
