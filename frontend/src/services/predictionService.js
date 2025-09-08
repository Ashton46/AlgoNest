import axios from 'axios';

const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  ENDPOINTS: {
    PREDICT: '/predict',
    HISTORICAL: '/historical',
    METRICS: '/metrics'
  }
};

class PredictionService {
  constructor(baseURL = API_CONFIG.BASE_URL) {
    this.api = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    this.api.interceptors.request.use(
      (config) => {
        console.log(`Making ${config.method?.toUpperCase()} request to:`, config.url);
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  async predict(gameData) {
    try {
      const backendData = this.convertToBackendFormat(gameData);
      const response = await this.api.post(API_CONFIG.ENDPOINTS.PREDICT, backendData);
      return response.data;
    } catch (error) {
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  convertToBackendFormat(frontendData) {
    const { sport, score_home, score_away, time_minutes, time_seconds, down, distance, yard_line } = frontendData;
    
    return {
      sport,
      score_home: parseInt(score_home) || 0,
      score_away: parseInt(score_away) || 0,
      time_minutes: parseInt(time_minutes) || 0,
      time_seconds: parseInt(time_seconds) || 0,
      down: parseInt(down) || 1,
      distance: parseInt(distance) || 10,
      yard_line: parseInt(yard_line) || 50
    };
  }

  async getHistoricalData(params) {
    try {
      const response = await this.api.get('/history', { params });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch historical data: ${error.message}`);
    }
  }

  async getMetrics() {
    try {
      const response = await this.api.get(API_CONFIG.ENDPOINTS.METRICS);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch metrics: ${error.message}`);
    }
  }
}

// Placeholder Service (Returns null/empty values until backend is ready)
class PlaceholderPredictionService {
  async predict(gameData) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Return null structure - will be replaced with real data
    return {
      predictions: null,
      confidence: null,
      processingTime: null,
      modelVersion: null
    };
  }

  async getHistoricalData() {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return null;
  }

  async getMetrics() {
    await new Promise(resolve => setTimeout(resolve, 500));
    return null;
  }
}

export const realPredictionService = new PredictionService();
export const placeholderPredictionService = new PlaceholderPredictionService();

// Placeholder until backend is ready
export const predictionService = realPredictionService;