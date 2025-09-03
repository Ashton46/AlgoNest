import React, { useState } from 'react';
import axios from 'axios';
import GameForm from './components/GameForm';
import PredictionDisplay from './PredictionDisplay';
import './App.css';

const API_URL = 'http://localhost:8000/predict';

function App() {
  const [formData, setFormData] = useState({
    sport: 'football',
    score_home: '',
    score_away: '',
    time_minutes: '',
    time_seconds: '',
    down: '1',
    distance: '',
    yard_line: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      const response = await axios.post(API_URL, formData);
      setPrediction(response.data);
    } catch (err) {
      setError("An error occurred. Please check your data and try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Play Predictor 🧠🏈</h1>
        <p>Input a game situation and predict the next play.</p>
      </header>
      <main>
        <div className="content-container">
          <GameForm formData={formData} handleChange={handleChange} handleSubmit={handleSubmit} />
          <div className="output-container">
            {loading && <p>Predicting play...</p>}
            {error && <p className="error">{error}</p>}
            {prediction && <PredictionDisplay prediction={prediction} />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;