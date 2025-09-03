import React, { useState } from 'react';
import { Brain } from 'lucide-react';
import GameForm from './components/GameForm';
import InteractiveFootballField from './components/InteractiveFootballField';
import InteractiveBasketballCourt from './components/InteractiveBasketballCourt';
import TrendsAndProbability from './components/TrendsAndProbability';
import WelcomeScreen from './components/WelcomeScreen';
import LoadingScreen from './components/LoadingScreen';
import './styles/App.css';

function App() {
  const [formData, setFormData] = useState({
    sport: 'football',
    score_home: '',
    score_away: '',
    time_minutes: '',
    time_seconds: '',
    down: '',
    distance: '',
    yard_line: '50'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [animateField, setAnimateField] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);
    setAnimateField(false);

    try {
      const result = await predictionService.predict(formData);
      setPrediction(result);
      
      if (result && result.predictions) {
        setAnimateField(true);
        setTimeout(() => setAnimateField(false), 3000);
      }
    } catch (err) {
      setError("Unable to connect to prediction service. Please try again later.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="background-animation">
        <div className="floating-shape shape-1"></div>
        <div className="floating-shape shape-2"></div>
        <div className="floating-shape shape-3"></div>
        <div className="floating-shape shape-4"></div>
      </div>

      <header className="app-header">
        <div className="header-content">
          <div className="logo-container">
            <Brain className="logo-icon" />
            <h1>AI Sports Analytics</h1>
          </div>
          <p className="subtitle">Advanced machine learning with 3D visualizations and real-time analytics</p>
        </div>
      </header>

      <main className="main-content">
        <div className="content-layout">
          <div className="sidebar">
            <GameForm 
              formData={formData} 
              handleChange={handleChange} 
              handleSubmit={handleSubmit}
              loading={loading}
            />
            {error && (
              <div className="error-message">
                <span>⚠️ {error}</span>
              </div>
            )}
          </div>
          
          <div className="main-visualization">
            {prediction && prediction.predictions && (
              <>
                <div className="visualization-tabs">
                  <div className="tab active">Field Analysis</div>
                  <div className="tab">Trends & Probability</div>
                </div>
                
                {formData.sport === 'football' ? (
                  <InteractiveFootballField 
                    prediction={prediction} 
                    formData={formData}
                    animateField={animateField}
                  />
                ) : (
                  <InteractiveBasketballCourt 
                    prediction={prediction} 
                    formData={formData}
                  />
                )}
                
                <TrendsAndProbability prediction={prediction} />
              </>
            )}
            
            {!prediction && !loading && (
              <WelcomeScreen />
            )}

            {loading && (
              <LoadingScreen />
            )}

            {prediction && !prediction.predictions && !loading && (
              <div className="no-data-screen">
                <div className="no-data-content">
                  <Brain size={60} className="no-data-icon" />
                  <h3>Prediction Service Connected</h3>
                  <p>Backend service is running but no prediction data available yet.</p>
                  <p>Please ensure your AI model is properly trained and configured.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;