# FEATURE ENGINEERING REPORT

Total features created: 13 (excluding date and target)
## Feature List:
- date
- price_lag_0
- price_lag_1
- price_lag_6
- brent_lag_0
- brent_lag_6
- price_mean_7
- price_std_7
- brent_mean_30
- price_return_1
- brent_return_1
- regime
- day_of_week
- month
- target_next_day

## Safety Check:
- ALL features are based on data available at time 't' to predict 't+1'.
- Time gaps handled: No gaps in daily data.
- Final Dataset Shape: (2657, 15)
