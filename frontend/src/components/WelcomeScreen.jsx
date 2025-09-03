import React from 'react';
import { Brain } from 'lucide-react';

const WelcomeScreen = () => {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <Brain size={80} className="welcome-icon" />
        <h2>Advanced Sports AI</h2>
        <p>Configure your game situation and experience cutting-edge sports analytics with:</p>
        <ul>
          <li>Interactive 3D field visualizations</li>
          <li>Real-time probability heatmaps</li>
          <li>Historical trend analysis</li>
          <li>Win probability tracking</li>
          <li>AI-powered play predictions</li>
        </ul>
      </div>
    </div>
  );
};

export default WelcomeScreen;
