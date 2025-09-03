export const generateHeatmapData = (yardLine = 50) => {
  // Return empty zones structure - will be populated by backend
  return [
    { name: 'Run Left', x: 0.2, y: 0.5, probability: null, color: '#ff6b6b', id: 0 },
    { name: 'Run Center', x: 0.5, y: 0.5, probability: null, color: '#4ecdc4', id: 1 },
    { name: 'Run Right', x: 0.8, y: 0.5, probability: null, color: '#45b7d1', id: 2 },
    { name: 'Pass Short Left', x: 0.2, y: 0.3, probability: null, color: '#96ceb4', id: 3 },
    { name: 'Pass Short Right', x: 0.8, y: 0.3, probability: null, color: '#feca57', id: 4 },
    { name: 'Pass Deep', x: 0.5, y: 0.2, probability: null, color: '#ff9ff3', id: 5 },
  ];
};

// Generate empty historical trend structure
export const generateTrendData = () => {
  // Return empty array - will be populated by backend
  return [];
};

// Generate empty win probability structure
export const generateWinProbabilityData = () => {
  // Return empty array - will be populated by backend
  return [];
};

export const generateShotChartData = () => {
  return [
    { name: 'Paint', x: 50, y: 15, probability: null, color: '#ff6b6b' },
    { name: 'Mid-Range Left', x: 30, y: 35, probability: null, color: '#4ecdc4' },
    { name: 'Mid-Range Right', x: 70, y: 35, probability: null, color: '#45b7d1' },
    { name: '3PT Corner Left', x: 15, y: 85, probability: null, color: '#96ceb4' },
    { name: '3PT Corner Right', x: 85, y: 85, probability: null, color: '#feca57' },
    { name: '3PT Top', x: 50, y: 75, probability: null, color: '#ff9ff3' },
  ];
};

export const generateRealTimeMetrics = () => {
  return {
    modelAccuracy: null,
    confidence: null,
    predictionTime: null,
    gamesAnalyzed: null,
    uptime: null
  };
};

export const COLOR_PALETTES = {
  primary: ['#00ff88', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
  secondary: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
  gradient: {
    green: 'linear-gradient(45deg, #00ff88, #0066ff)',
    blue: 'linear-gradient(45deg, #4ecdc4, #45b7d1)',
    red: 'linear-gradient(45deg, #ff6b6b, #ff9ff3)'
  }
};

export const ANIMATIONS = {
  duration: {
    fast: 300,
    medium: 500,
    slow: 1000
  },
  easing: {
    smooth: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)'
  }
};