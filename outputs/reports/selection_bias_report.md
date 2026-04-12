# Selection Bias Analysis Report

Generated: 2026-04-12 00:41:53

## 1. Summary Verdict
- **WARN** -- DSR p=0.0000, Adjusted SR=0.60, MinTRL=5.4yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.679
- Number of trials (N): 401
- Expected max SR under null: 0.076
- sigma(SR): 0.0219
- Deflated SR: 27.533 (p-value: 0.0000)
- Skewness: 2.220, Kurtosis: 36.444
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 5.4 years (1364 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 401
- Observed SR: 0.679
- Haircut: 0.076
- Adjusted SR: 0.603
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.787 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = -0.338 [FAIL]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 1.421 [PASS]
- Verdict: **UNSTABLE -- 시기 의존적 성과**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
