import React from 'react';

const PredictionDisplay = ({ prediction }) => {
  if (!prediction || !prediction.predictions) return null;

  const topPrediction = prediction.predictions[0];
  const otherPredictions = prediction.predictions.slice(1);

  const renderFootballField = () => (
    <div className="football-field">
      <p className="field-title">Predicted Play</p>
      <div className="predicted-play-path">
      </div>
      <div className="yard-line yard-line-50"></div>
      <div className="yard-line yard-line-10"></div>
    </div>
  );

  return (
    <div className="prediction-display">
      <h2>Top Prediction</h2>
      <div className="prediction-box">
        <h3>{topPrediction.play_type}</h3>
        <p>{(topPrediction.probability * 100).toFixed(1)}% Chance</p>
      </div>

      {renderFootballField()}

      {otherPredictions.length > 0 && (
        <>
          <h3>Alternative Plays</h3>
          <ul className="alternative-plays-list">
            {otherPredictions.map((p, index) => (
              <li key={index}>
                {p.play_type}: {(p.probability * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

export default PredictionDisplay;