import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts';
import { TrendingUp, BarChart3, Target, Activity } from 'lucide-react';
import { predictionService } from '../services/predictionService';

const TrendsAndProbability = ({ prediction }) => {
const normalizeHistory = (historicalData) => {
    if (!historicalData) return [];
    if (Array.isArray(historicalData)) return historicalData;
    if (historicalData.predictions) {
      return historicalData.predictions.map((item, index) => {
        const confidence = Number(item.confidence || 0);
        return {
          game: index + 1,
          accuracy: Math.round(confidence * 100),
          confidence: Math.round(confidence * 100)
        };
      });
    }
    return [];
  };

  const normalizeMetrics = (metricsData) => {
    if (!metricsData) return null;
    if (metricsData.model_accuracy) {
      return {
        modelAccuracy: metricsData.model_accuracy,
        confidence: metricsData.average_confidence,
        predictionTime: metricsData.average_processing_time,
        gamesAnalyzed: metricsData.total_predictions,
        uptime: metricsData.uptime
      };
    }
    return metricsData;
  };

  const [trendData, setTrendData] = useState([]);
  const [winProbData, setWinProbData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadAnalyticsData = async () => {
      setLoading(true);
      try {
        const [historicalData, metricsData] = await Promise.all([
          predictionService.getHistoricalData(),
          predictionService.getMetrics()
        ]);
       
        setTrendData(normalizeHistory(historicalData));
        setMetrics(normalizeMetrics(metricsData));
        
      } catch (error) {
        console.error('Failed to load analytics data:', error);
        setTrendData([]);
        setMetrics(null);
      } finally {
        setLoading(false);
      }
    };

    loadAnalyticsData();
  }, []);

  useEffect(() => {
    if (prediction && prediction.win_probability) {
      setWinProbData([
        {
          time: 'Now',
          homeWinProb: Math.round((prediction.win_probability.home_win_prob || 0) * 100),
          awayWinProb: Math.round((prediction.win_probability.away_win_prob || 0) * 100)
        }
      ]);
    } else {
      setWinProbData([]);
    }
  }, [prediction]);


  const EmptyChart = ({ title, message = "No data available" }) => (
    <div className="empty-chart">
      <h4>{title}</h4>
      <div className="empty-chart-content">
        <p>{message}</p>
      </div>
    </div>
  );

  return (
    <div className="trends-container">
      <div className="trends-grid">
        <div className="glass-card chart-card">
          <div className="chart-header">
            <TrendingUp size={20} />
            <h3>Prediction Accuracy Trend</h3>
          </div>
          {trendData && trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="game" stroke="#ccc" />
                <YAxis stroke="#ccc" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(0,255,136,0.3)'
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#00ff88"
                  fill="url(#accuracyGradient)"
                  strokeWidth={2}
                />
                <defs>
                  <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00ff88" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00ff88" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart title="" message="Historical accuracy data will appear here once available" />
          )}
        </div>

        <div className="glass-card chart-card">
          <div className="chart-header">
            <BarChart3 size={20} />
            <h3>Model Confidence</h3>
          </div>
          {trendData && trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="game" stroke="#ccc" />
                <YAxis stroke="#ccc" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(0,255,136,0.3)'
                  }}
                />
                <Bar dataKey="confidence" fill="#4ecdc4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart title="" message="Model confidence data will appear here once available" />
          )}
        </div>

        <div className="glass-card chart-card wide-chart">
          <div className="chart-header">
            <Target size={20} />
            <h3>Live Win Probability</h3>
          </div>
          {winProbData && winProbData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={winProbData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="#ccc" />
                <YAxis stroke="#ccc" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    border: '1px solid rgba(0,255,136,0.3)'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="homeWinProb"
                  stroke="#00ff88"
                  strokeWidth={3}
                  dot={{ fill: '#00ff88', r: 4 }}
                  name="Home Team"
                />
                <Line
                  type="monotone"
                  dataKey="awayWinProb"
                  stroke="#ff6b6b"
                  strokeWidth={3}
                  dot={{ fill: '#ff6b6b', r: 4 }}
                  name="Away Team"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart title="" message="Win probability tracking will appear here once games are in progress" />
          )}
        </div>

        <div className="glass-card metrics-card">
          <div className="metrics-header">
            <Activity size={20} />
            <h3>Live Metrics</h3>
          </div>
          <div className="metrics-grid">
            <div className="metric-item">
              <div className="metric-value">
                {metrics?.modelAccuracy || '--'}
              </div>
              <div className="metric-label">Model Accuracy</div>
            </div>
            <div className="metric-item">
              <div className="metric-value">
                {metrics?.confidence || '--'}
              </div>
              <div className="metric-label">Confidence</div>
            </div>
            <div className="metric-item">
              <div className="metric-value">
                {metrics?.predictionTime || '--'}
              </div>
              <div className="metric-label">Prediction Time</div>
            </div>
            <div className="metric-item">
              <div className="metric-value">
                {metrics?.gamesAnalyzed || '--'}
              </div>
              <div className="metric-label">Games Analyzed</div>
            </div>
          </div>
          {loading && (
            <div className="metrics-loading">
              <div className="chart-loading-spinner"></div>
              <p>Loading analytics...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const EmptyChart = ({ title, message = "No data available" }) => (
  <div className="empty-chart">
    <h4>{title}</h4>
    <div className="empty-chart-content">
      <p>{message}</p>
    </div>
  </div>
);

export { EmptyChart };
export default TrendsAndProbability;