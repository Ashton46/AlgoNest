import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { generateShotChartData } from '../utils/dataGenerators';

const InteractiveBasketballCourt = ({ prediction, formData }) => {
  const [shotChart, setShotChart] = useState([]);

  useEffect(() => {
    const baseZones = generateShotChartData();
    
    // Update zones with actual prediction data when ready 
    if (prediction && prediction.predictions) {
      const updatedZones = baseZones.map(zone => {
        const matchingPrediction = prediction.predictions.find(p => 
          p.play_type && p.play_type.toLowerCase().includes('shot') || 
          p.play_type && p.play_type.toLowerCase().includes(zone.name.toLowerCase())
        );
        
        return {
          ...zone,
          probability: matchingPrediction ? matchingPrediction.probability : null
        };
      });
      setShotChart(updatedZones);
    } else {
      setShotChart(baseZones);
    }
  }, [prediction]);

  return (
    <div className="basketball-court-container">
      <div className="court-header">
        <Activity size={24} />
        <h3>Shot Probability Chart</h3>
      </div>
      
      <div className="court-svg-container">
        <svg viewBox="0 0 500 470" className="basketball-court">
          {/* Court outline */}
          <rect x="25" y="25" width="450" height="420" fill="#D4A574" stroke="#000" strokeWidth="3"/>
          
          {/* Three-point line */}
          <path d="M 25 95 Q 250 200 475 95" fill="none" stroke="#000" strokeWidth="2"/>
          
          {/* Free throw circle */}
          <circle cx="250" cy="95" r="60" fill="none" stroke="#000" strokeWidth="2"/>
          
          {/* Key/Paint */}
          <rect x="190" y="25" width="120" height="190" fill="rgba(255,255,255,0.1)" stroke="#000" strokeWidth="2"/>
          
          {/* Basketball hoop */}
          <circle cx="250" cy="45" r="12" fill="none" stroke="#FF6600" strokeWidth="4"/>
          
          {/* Probability zones - only show if data exists */}
          {shotChart.map((zone, index) => (
            zone.probability !== null && zone.probability > 0 && (
              <g key={index}>
                <circle 
                  cx={zone.x * 4.5 + 25} 
                  cy={zone.y * 4.2 + 25} 
                  r={zone.probability * 40}
                  fill={zone.color}
                  opacity="0.4"
                  className="shot-zone"
                />
                <text 
                  x={zone.x * 4.5 + 25} 
                  y={zone.y * 4.2 + 30}
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="12"
                  fontWeight="bold"
                >
                  {(zone.probability * 100).toFixed(0)}%
                </text>
              </g>
            )
          ))}
          
          {/* Player position */}
          <circle cx="250" cy="200" r="8" fill="#00ff88" stroke="#fff" strokeWidth="2"/>
          
          {/* Show placeholder if no data */}
          {shotChart.every(zone => zone.probability === null) && (
            <text 
              x="250" 
              y="250"
              textAnchor="middle"
              fill="#ccc"
              fontSize="16"
              opacity="0.7"
            >
              Waiting for prediction data...
            </text>
          )}
        </svg>
      </div>
      
      <div className="shot-zones-legend">
        {shotChart.map((zone, index) => (
          <div key={index} className="shot-zone-item">
            <div className="zone-color" style={{ backgroundColor: zone.color }}></div>
            <span>
              {zone.name}: {
                zone.probability !== null 
                  ? `${(zone.probability * 100).toFixed(1)}%`
                  : 'No data'
              }
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default InteractiveBasketballCourt;