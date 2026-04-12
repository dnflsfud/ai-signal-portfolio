# Selection Bias Analysis Report

Generated: 2026-04-12 16:02:33

## 1. Summary Verdict
- **PASS** -- DSR p=0.0000, Adjusted SR=0.51, MinTRL=7.5yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 0.582
- Number of trials (N): 402
- Expected max SR under null: 0.077
- sigma(SR): 0.0222
- Deflated SR: 22.797 (p-value: 0.0000)
- Skewness: 1.796, Kurtosis: 25.709
- Observations: 1920 trading days
- Verdict: **PASS**

## 3. Minimum Track Record Length
- Required: 7.5 years (1899 trading days)
- Available: 7.6 years (1920 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 402
- Observed SR: 0.582
- Haircut: 0.077
- Adjusted SR: 0.505
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-23
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-23 ~ 2021-05-06): IR = 0.080 [PASS]
- Period 2 (2021-05-07 ~ 2023-10-19): IR = 0.354 [PASS]
- Period 3 (2023-10-20 ~ 2026-04-02): IR = 1.169 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
