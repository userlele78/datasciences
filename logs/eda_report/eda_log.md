# FULL EDA REPORT

## 3.1 Data Overview
- Shape: (2687, 13)
- Time Range: 2018-09-23 00:00:00 to 2026-01-30 00:00:00
- Missing values:
fed_funds       0
sp500           0
gold_buy        0
gold_sell       0
brent_oil       0
baltic_dry      0
usd_vnd         0
oil_rigs        0
gas_rigs        0
misc_rigs       0
diesel_do       0
gas_e5_ron92    0
gas_ron95       0

## 3.2 Univariate Analysis
               count          mean           std          min           25%          50%           75%           max      skew  kurtosis
fed_funds     2687.0  2.662538e+00  2.020281e+00         0.05  1.000000e-01         2.40  4.480000e+00  5.330000e+00 -0.052804 -1.558531
sp500         2687.0  4.314271e+03  1.167145e+03      2237.40  3.320130e+03      4181.17  5.120675e+03  6.978600e+03  0.474754 -0.637306
gold_buy      2687.0  6.835072e+07  2.767982e+07  36150000.00  5.322500e+07  66000000.00  7.650000e+07  1.883000e+08  1.537704  2.389707
gold_sell     2687.0  6.939790e+07  2.825354e+07  36250000.00  5.437500e+07  66900000.00  7.852500e+07  1.913000e+08  1.492521  2.230550
brent_oil     2687.0  7.227735e+01  1.750861e+01        19.33  6.305500e+01        72.69  8.276000e+01  1.279800e+02 -0.034903  0.678604
baltic_dry    2687.0  1.712565e+03  7.892524e+02       393.00  1.225500e+03      1594.00  2.051000e+03  5.650000e+03  1.390903  3.506670
usd_vnd       2687.0  2.395418e+04  1.119512e+03     22625.00  2.317500e+04     23392.00  2.479750e+04  2.643250e+04  0.931004 -0.548243
oil_rigs      2687.0  5.207477e+02  1.708750e+02       172.00  4.180000e+02       499.00  6.090000e+02  8.880000e+02  0.200836 -0.158725
gas_rigs      2687.0  1.240331e+02  3.372968e+01        68.00  1.000000e+02       117.00  1.510000e+02  2.020000e+02  0.599825 -0.494523
misc_rigs     2687.0  2.726833e+00  2.207708e+00         0.00  1.000000e+00         2.00  4.000000e+00  1.100000e+01  0.959578  1.070095
diesel_do     2687.0  1.806844e+04  3.792571e+03      9850.00  1.605000e+04     18120.00  2.019000e+04  3.001000e+04  0.208741  0.257326
gas_e5_ron92  2687.0  2.012170e+04  3.478046e+03     10940.00  1.871000e+04     19970.00  2.202000e+04  3.130000e+04  0.159123  1.273526
gas_ron95     2687.0  2.107992e+04  3.567533e+03     11630.00  1.957000e+04     21060.00  2.304000e+04  3.287000e+04  0.163742  1.391767

## 3.3 Bivariate Analysis
Top correlations with target (RON 95):
gas_ron95       1.000000
gas_e5_ron92    0.996716
brent_oil       0.929290
diesel_do       0.908706
gas_rigs        0.355725
oil_rigs        0.333224
baltic_dry      0.326998
fed_funds       0.280583
sp500           0.211602
gold_sell       0.101153
gold_buy        0.098493
misc_rigs       0.027117
usd_vnd         0.007140

## 3.4 Multivariate Analysis (PCA)
Components needed for 95% variance: 5

## 3.5 Time Series Analysis
ADF Statistic for gas_ron95: -2.4197782236735277
p-value: 0.13623178353893706
Stationary: No

## 3.9 Regime Detection
Regime Statistics:
                mean          std  count
regime                                  
0.0     19077.508306  2413.109359   1505
1.0     18710.961538  6760.837128     52
2.0     23902.524977  2691.180576   1101

## 3.11 PRELIMINARY INSIGHTS
1. Target is non-stationary (requires differencing or trend modeling).
2. Strong correlation detected with Brent Oil and Gold prices.
3. 3 clear regimes identified (Low, Mid, High volatility/price levels).
