# DATA MINING REPORT

## 4.1 Lag Structure Mining (CCF)
Analyzing which lags of exogenous features have the highest correlation with RON 95.

- **brent_oil**: Optimal Lag = 7 days (Corr: 0.122)
- **usd_vnd**: Optimal Lag = 21 days (Corr: 0.051)
- **gold_sell**: Optimal Lag = 15 days (Corr: 0.046)
- **baltic_dry**: Optimal Lag = 29 days (Corr: 0.043)
- **fed_funds**: Optimal Lag = 28 days (Corr: 0.156)

## 4.2 Relationship Mining (Granger Causality)
Checking if 'A' helps predict 'B' beyond 'B's own history.

### Testing: brent_oil -> gas_ron95
- Minimum p-value across 7 lags: 0.00000
- Result: **CAUSAL RELATIONSHIP DETECTED** (p < 0.05)
### Testing: usd_vnd -> gas_ron95
- Minimum p-value across 7 lags: 0.38218
- Result: No significant causality detected at 5% level.

## 4.3 Structural Change Point Detection
Detected 5 major volatility shift points:
- 2020-03-29
- 2020-05-28
- 2022-07-21
- 2022-09-19
- 2022-07-11

## 4.4 Synthesis for Feature Engineering
1. **Lag Selection**: Features should be lagged by discovered optimal lags (mostly 0-5 days).
2. **Causality**: Brent Oil is a verified driver; USD/VND requires non-linear treatment.
3. **Volatility**: Models must account for regime shifts detected at identified change points.
