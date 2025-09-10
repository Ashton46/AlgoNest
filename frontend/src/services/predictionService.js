import axios from 'axios';

const API_CONFIG = {
  BASE_URL: 'http://localhost:8000/api',
  ENDPOINTS: {
    PREDICT: '/predict',
    HISTORICAL: '/history',
    METRICS: '/metrics',
    HEALTH: '/health',
    TRAIN: '/models/{sport}/train',
    STATS: '/models/{sport}/stats'
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
    
    const score_differential = parseInt(score_home) - parseInt(score_away);
    const time_remaining = (parseInt(time_minutes) || 0) * 60 + (parseInt(time_seconds) || 0);
    
    return {
      sport,
      score_home: parseInt(score_home) || 0,
      score_away: parseInt(score_away) || 0,
      time_minutes: parseInt(time_minutes) || 0,
      time_seconds: parseInt(time_seconds) || 0,
      down: parseInt(down) || 1,
      distance: parseInt(distance) || 10,
      yard_line: parseInt(yard_line) || 50,
      score_differential,
      time_remaining,
      quarter: frontendData.quarter || 1,
      shot_clock: frontendData.shot_clock || 24
    };
  }

  async getHistoricalData(params) {
    try {
      const response = await this.api.get(API_CONFIG.ENDPOINTS.HISTORICAL, { params });
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

  async healthCheck() {
    try {
      const response = await this.api.get(API_CONFIG.ENDPOINTS.HEALTH);
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  async trainModel(sport, forceRetrain = false, background = true) {
    try {
      const response = await this.api.get(
        API_CONFIG.ENDPOINTS.TRAIN.replace('{sport}', sport),
        { params: { force_retrain: forceRetrain, background } }
      );
      return response.data;
    } catch (error) {
      throw new Error(`Model training failed: ${error.message}`);
    }
  }

  async getModelStats(sport) {
    try {
      const response = await this.api.get(
        API_CONFIG.ENDPOINTS.STATS.replace('{sport}', sport)
      );
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get model stats: ${error.message}`);
    }
  }
}

class PlaceholderPredictionService {
  async predict(gameData) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    return null;
  }

  async getHistoricalData() {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return null;
  }

  async getMetrics() {
    await new Promise(resolve => setTimeout(resolve, 500));
    return null;
  }

  async healthCheck() {
    await new Promise(resolve => setTimeout(resolve, 300));
    return null;
  }

  async trainModel(sport, forceRetrain = false, background = true) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    return null;
  }

  async getModelStats(sport) {
    await new Promise(resolve => setTimeout(resolve, 800));
    return null;
  }
}

export const realPredictionService = new PredictionService();
export const placeholderPredictionService = new PlaceholderPredictionService();
export const predictionService = realPredictionService;