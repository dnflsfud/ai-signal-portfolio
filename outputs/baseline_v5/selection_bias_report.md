# Selection Bias Analysis Report

Generated: 2026-05-19 22:17:40

## 1. Summary Verdict
- **WARN** -- DSR p=0.0708, Adjusted SR=0.52, MinTRL=1.6yr

## 2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
- Observed SR: 1.289
- Number of trials (N): 10
- Expected max SR under null: 0.765
- sigma(SR): 0.3565
- Deflated SR: 1.470 (p-value: 0.0708)
- Skewness: 0.293, Kurtosis: 5.947
- Observations: 1952 trading days
- Verdict: **FAIL -- 다중 비교 보정 후 유의하지 않음**

## 3. Minimum Track Record Length
- Required: 1.6 years (405 trading days)
- Available: 7.7 years (1952 trading days)
- Verdict: **SUFFICIENT**

## 4. Grid Search Bias (Haircut)
- Combinations tested: 10
- Observed SR: 1.289
- Haircut: 0.765
- Adjusted SR: 0.524
- Verdict: **PASS**

## 5. Universe Survivorship
- Backtest start: 2018-11-26
- Late entrants (data starts >30d after backtest): None
- Verdict: **CLEAN**

## 6. Sub-period Stability
- Period 1 (2018-11-26 ~ 2021-05-21): IR = 0.721 [PASS]
- Period 2 (2021-05-24 ~ 2023-11-17): IR = 1.256 [PASS]
- Period 3 (2023-11-20 ~ 2026-05-15): IR = 1.861 [PASS]
- Verdict: **STABLE**

## References
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting" (Haircut Sharpe Ratio)
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism"
