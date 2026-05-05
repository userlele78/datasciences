export interface PriceFeatures {
  price_lag_0: number;
  brent_lag_0: number;
  brent_lag_7: number;
  brent_mean_30: number;
  regime: number;
  brent_diff_1: number;
  brent_diff_7: number;
  days_since_last_change: number;
  model_name: string;
}

export interface PredictionResponse {
  model_used: string;
  predicted_delta: number;
  predicted_price_tomorrow: number;
  current_price: number;
}
