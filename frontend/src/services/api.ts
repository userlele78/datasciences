import axios from 'axios';
import { PriceFeatures, PredictionResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const getPrediction = async (features: PriceFeatures): Promise<PredictionResponse> => {
  const response = await axios.post<PredictionResponse>(`${API_BASE_URL}/predict`, features);
  return response.data;
};
