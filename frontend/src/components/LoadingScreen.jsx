import React from 'react';

const LoadingScreen = () => {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-spinner-large"></div>
        <h3>Processing Game Data...</h3>
        <p>Running advanced AI algorithms</p>
        <div className="loading-steps">
          <div className="step active">Analyzing field position</div>
          <div className="step">Calculating probabilities</div>
          <div className="step">Generating 3D visualization</div>
          <div className="step">Preparing insights</div>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;